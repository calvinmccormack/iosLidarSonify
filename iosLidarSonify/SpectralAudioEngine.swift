import Foundation
import AVFoundation
import Accelerate
import Combine

final class SpectralAudioEngine: ObservableObject {

    // UI
    @Published var pan: Float = 0
    @Published var enableEdgeClicks: Bool = false
    @Published var outputGainDB: Float = 6.0

    // Constants
    private let sampleRate: Double
    private let fftSize: Int = 1024
    private let hop: Int = 256

    // AGC
    private var agcGain: Float = 1.0
    private let agcAlpha: Float = 0.95
    private let agcTargetRMS: Float = 0.20

    // Engine
    private let engine = AVAudioEngine()
    private var srcNode: AVAudioSourceNode!
    private let reverb = AVAudioUnitReverb()

    // Window / DFT
    private var window: [Float]
    private var invWindowEnergy: Float
    private let forwardDFT: vDSP.DFT<Float>
    private let inverseDFT: vDSP.DFT<Float>

    // Buffers
    private var timeBlock = [Float]()
    private var winTime   = [Float]()
    private var freqReal  = [Float]()
    private var freqImag  = [Float]()
    private var ifftReal  = [Float]()
    private var zeroImag  = [Float]()
    private var scratchImagTime = [Float]()

    // Target spectral boost from scan
    private var targetBoostBands = [Float](repeating: 0, count: 40)
    private var targetBoostLin: Float = powf(10.0, 12.0/20.0)
    private var targetBoostSmooth: Float = 0.6

    // ===== Simple comb filters (time domain) =====
    struct FFComb {
        var alpha: Float
        var M: Int
        var buf: [Float]; var i = 0
        init(alpha: Float, M: Int) { self.alpha = alpha; self.M = max(1, M); self.buf = .init(repeating: 0, count: max(1, M)) }
        mutating func reinit(alpha: Float, M: Int) { self.alpha = alpha; self.M = max(1, M); self.buf = .init(repeating: 0, count: self.M); self.i = 0 }
        mutating func process(_ x: Float) -> Float {
            let y = x + alpha * buf[i]
            buf[i] = x
            i = (i + 1) % M
            return y
        }
    }
    struct FBComb {
        var g: Float
        var M: Int
        var buf: [Float]; var i = 0
        init(g: Float, M: Int) { self.g = g; self.M = max(1, M); self.buf = .init(repeating: 0, count: max(1, M)) }
        mutating func reinit(g: Float, M: Int) { self.g = g; self.M = max(1, M); self.buf = .init(repeating: 0, count: self.M); self.i = 0 }
        mutating func process(_ x: Float) -> Float {
            let y = x + g * buf[i]
            buf[i] = y
            i = (i + 1) % M
            return y
        }
    }
    struct APComb {
        var a: Float
        var M: Int
        var xbuf: [Float], ybuf: [Float]; var i = 0
        init(a: Float, M: Int) { self.a = a; self.M = max(1, M); xbuf = .init(repeating: 0, count: max(1, M)); ybuf = xbuf }
        mutating func reinit(a: Float, M: Int) { self.a = a; self.M = max(1, M); xbuf = .init(repeating: 0, count: self.M); ybuf = xbuf; i = 0 }
        mutating func process(_ x: Float) -> Float {
            // M is always >= 1; i is always < M
            let xm = xbuf[i], ym = ybuf[i]
            let y = -a * x + xm + a * ym
            xbuf[i] = x
            ybuf[i] = y
            i = (i + 1) % M
            return y
        }
    }

    // Live instances
    private var sphereComb = FBComb(g: 0.80, M: 200)
    private var tetraComb  = FFComb(alpha: 1.0, M: 120)
    private var cubeComb   = APComb(a: 0.85, M: 96)
    private var cubeComb2  = APComb(a: 0.60, M: 48) // extra stage to make cube audible

    private var sphereActive = false
    private var tetraActive  = false
    private var cubeActive   = false

    private var sphereLevel: Float = 0.0
    private var tetraLevel:  Float = 0.0
    private var cubeLevel:   Float = 0.0
    private var cubeMixBeta: Float = 0.60  // stronger dry+AP mix for audibility

    // Pending params for lock-free swap at block boundary
    private struct PendingFB { var need=false; var g:Float=0.8;  var M:Int=64; var level:Float=0; var active=false }
    private struct PendingFF { var need=false; var a:Float=1.0;  var M:Int=64; var level:Float=0; var active=false }
    private struct PendingAP { var need=false; var a:Float=0.85; var M:Int=64; var level:Float=0; var active=false }
    private var pendSphere = PendingFB()
    private var pendTetra  = PendingFF()
    private var pendCube1  = PendingAP()
    private var pendCube2  = PendingAP()

    // OLA ring
    private var olaL: [Float]
    private var olaR: [Float]
    private var olaWrite = 0
    private var olaRead  = 0

    // 40-band mapping
    private let numBands = 40
    private var bandEdges = [Float]()
    private var binToBand = [Int]()

    // Distance (kept but not coloring timbre now)
    private var z01: Float = 0
    private var zSlew: Float = 0
    private let zSlewA: Float = 0.15

    // Pink state (unused for now)
    private var p0: Float = 0, p1: Float = 0, p2: Float = 0

    // Edge click
    private var clickEnv: Float = 0

    // Band gain smoothing
    private var currentGains = [Float](repeating: 1, count: 40)
    private let gainSmooth: Float = 0.85

    init() {
        let session = AVAudioSession.sharedInstance()
        self.sampleRate = session.sampleRate

        window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        var sum: Float = 0
        vDSP_sve(window, 1, &sum, vDSP_Length(fftSize))
        invWindowEnergy = sum > 0 ? 1.0 / sum : 1.0

        timeBlock = [Float](repeating: 0, count: fftSize)
        winTime   = [Float](repeating: 0, count: fftSize)
        freqReal  = [Float](repeating: 0, count: fftSize)
        freqImag  = [Float](repeating: 0, count: fftSize)
        ifftReal  = [Float](repeating: 0, count: fftSize)
        zeroImag  = [Float](repeating: 0, count: fftSize)
        scratchImagTime = [Float](repeating: 0, count: fftSize)
        olaL = [Float](repeating: 0, count: fftSize * 2)
        olaR = [Float](repeating: 0, count: fftSize * 2)

        forwardDFT = vDSP.DFT<Float>(count: fftSize, direction: .forward, transformType: .complexComplex, ofType: Float.self)!
        inverseDFT = vDSP.DFT<Float>(count: fftSize, direction: .inverse, transformType: .complexComplex, ofType: Float.self)!

        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        srcNode = AVAudioSourceNode { [weak self] _, _, frameCount, ablPtr -> OSStatus in
            guard let self = self else { return noErr }
            let n = Int(frameCount)
            let abl = UnsafeMutableAudioBufferListPointer(ablPtr)
            let outL = abl[0].mData!.assumingMemoryBound(to: Float.self)
            let outR = abl[1].mData!.assumingMemoryBound(to: Float.self)

            var produced = 0
            while produced < n {
                self.processBlock() // renders hop samples
                let toCopy = min(self.hop, n - produced)
                self.copyFromOLA(to: outL + produced, outR: outR + produced, count: toCopy)
                produced += toCopy
            }
            return noErr
        }

        engine.attach(srcNode)
        engine.attach(reverb)
        engine.connect(srcNode, to: reverb, format: fmt)
        engine.connect(reverb, to: engine.mainMixerNode, format: fmt)
        reverb.loadFactoryPreset(.largeRoom)
        reverb.wetDryMix = 0
        engine.mainMixerNode.outputVolume = 1.0

        try? session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try? session.setActive(true)
    }

    func start() { try? engine.start() }
    func stop()  { engine.stop() }

    // MARK: Controls from pipeline/UI

    func configureBands(fMin: Double, fMax: Double) {
        let fm = Float(max(20, min(fMin, fMax - 10)))
        let fM = Float(max(fm + 10, Float(fMax)))
        bandEdges = [Float](repeating: 0, count: numBands + 1)
        let ratio: Float = fM / fm
        for i in 0...numBands {
            let t: Float = Float(i) / Float(numBands)
            bandEdges[i] = fm * powf(ratio, t)
        }
        let nyq = Float(sampleRate / 2)
        let binHz = nyq / Float(fftSize/2)
        binToBand = [Int](repeating: numBands - 1, count: fftSize/2 + 1)
        for b in 0..<numBands {
            let f0 = bandEdges[b]
            let f1 = bandEdges[b+1]
            let i0 = max(0, Int(floorf(f0 / binHz)))
            let i1 = min(fftSize/2, Int(ceilf(f1 / binHz)))
            if i0 <= i1 {
                for i in i0...i1 { binToBand[i] = b }
            }
        }
    }

    func updateEnvelope(_ gains40: [Float]) {
        var g = gains40
        if g.count < numBands { g += Array(repeating: 1, count: numBands - g.count) }
        for i in 0..<numBands { g[i] = min(1.0, max(0.05, g[i])) }
        var s = [Float](repeating: 1, count: numBands)
        for i in 0..<numBands {
            let a = (i > 0) ? g[i-1] : g[i]
            let b = g[i]
            let c = (i+1 < numBands) ? g[i+1] : g[i]
            s[i] = 0.2*a + 0.6*b + 0.2*c
        }
        for i in 0..<numBands {
            currentGains[i] = gainSmooth * currentGains[i] + (1 - gainSmooth) * s[i]
        }
    }

    func updateDistance(_ z: Float) {
        z01 = max(0, min(1, z))
        zSlew = zSlewA * zSlew + (1 - zSlewA) * z01
        DispatchQueue.main.async { [weak self] in self?.reverb.wetDryMix = 0 }
    }

    /// Called from the pipeline each scan step.
    /// `mask`: 40-length 0…1; `shape`: 1 sphere, 2 tetra, 3 cube.
    func setTargetBands(_ mask: [Float], shape: Int, boostDB: Float = 12.0) {
        var m = mask
        if m.count < numBands { m += Array(repeating: 0, count: numBands - m.count) }
        if m.count > numBands { m = Array(m.prefix(numBands)) }
        for i in 0..<numBands {
            targetBoostBands[i] = targetBoostSmooth * targetBoostBands[i]
                                 + (1 - targetBoostSmooth) * max(0, min(1, m[i]))
        }

        var localBoostDB = boostDB
        var lvl: Float = 0.6
        switch shape {
        case 1: localBoostDB = max(6.0, boostDB - 3.0); lvl = 0.40  // sphere softer
        case 2: localBoostDB = boostDB + 6.0;           lvl = 0.95  // tetra forward
        case 3: localBoostDB = boostDB + 8.0;           lvl = 1.10  // cube most forward
        default: localBoostDB = 0; lvl = 0
        }
        targetBoostLin = powf(10.0, localBoostDB / 20.0)

        // Spectral centroid of active bands → f0
        var sumW: Float = 0, sumF: Float = 0
        for b in 0..<numBands {
            let w = targetBoostBands[b]
            if w > 1e-3 {
                let f0 = bandEdges[b], f1 = bandEdges[min(b+1, numBands)]
                let fc = sqrtf(max(10, f0) * max(10, f1))
                sumF += w * fc
                sumW += w
            }
        }
        let fc = (sumW > 0) ? (sumF / sumW) : 600.0
        setTargetComb(shape: shape, on: shape != 0, f0: fc, level: lvl)
    }

    func clearTarget() {
        for i in 0..<numBands { targetBoostBands[i] = 0 }
        setTargetComb(shape: 1, on: false, f0: 600, level: 0)
        setTargetComb(shape: 2, on: false, f0: 600, level: 0)
        setTargetComb(shape: 3, on: false, f0: 600, level: 0)
    }

    func triggerEdge(_ strength: Float) {
        clickEnv = max(clickEnv, max(0, min(1, strength)) * 0.6)
    }

    /// UI thread safe: only sets pending params. Actual reinit occurs on audio thread.
    func setTargetComb(shape: Int, on: Bool, f0: Float, level: Float) {
        let sr = Float(sampleRate)
        let fc = max(40, min(8000, f0))
        let M  = max(1, Int(round(sr / fc)))

        switch shape {
        case 1: // feedback comb for sphere
            pendSphere.g = 0.82
            pendSphere.M = M
            pendSphere.level = max(0, min(1, level))
            pendSphere.active = on
            pendSphere.need = true
        case 2: // feed-forward comb for tetra
            pendTetra.a = 1.0 // alpha
            pendTetra.M = M
            pendTetra.level = max(0, min(1, level))
            pendTetra.active = on
            pendTetra.need = true
        case 3: // dual all-pass for cube
            pendCube1.a = 0.85; pendCube1.M = M
            pendCube2.a = 0.60; pendCube2.M = max(1, M/2)
            pendCube1.level = max(0, min(1, level))
            pendCube1.active = on
            pendCube1.need = true
            pendCube2.need = true
        default: break
        }
    }

    // MARK: - DSP

    private func applyPendingCombUpdates() {
        if pendSphere.need {
            sphereActive = pendSphere.active
            sphereLevel  = pendSphere.level
            sphereComb.reinit(g: pendSphere.g, M: pendSphere.M)
            pendSphere.need = false
        }
        if pendTetra.need {
            tetraActive = pendTetra.active
            tetraLevel  = pendTetra.level
            tetraComb.reinit(alpha: pendTetra.a, M: pendTetra.M)
            pendTetra.need = false
        }
        if pendCube1.need || pendCube2.need {
            cubeActive = pendCube1.active
            cubeLevel  = pendCube1.level
            cubeComb.reinit(a: pendCube1.a, M: pendCube1.M)
            cubeComb2.reinit(a: pendCube2.a, M: pendCube2.M)
            pendCube1.need = false
            pendCube2.need = false
        }
    }

    private func processBlock() {
        // Commit any pending comb changes on the audio thread
        applyPendingCombUpdates()

        // Excitation
        genWhite()

        // window → FFT
        vDSP_vmul(timeBlock, 1, window, 1, &winTime, 1, vDSP_Length(fftSize))
        forwardDFT.transform(inputReal: winTime,
                             inputImaginary: zeroImag,
                             outputReal: &freqReal,
                             outputImaginary: &freqImag)

        applySpectralShaping()

        // iDFT
        inverseDFT.transform(inputReal: freqReal,
                             inputImaginary: freqImag,
                             outputReal: &ifftReal,
                             outputImaginary: &scratchImagTime)

        // Normalize block energy
        var scale = 1.0 / Float(fftSize)
        vDSP_vsmul(ifftReal, 1, &scale, &ifftReal, 1, vDSP_Length(fftSize))
        var gain = invWindowEnergy * 4
        vDSP_vsmul(ifftReal, 1, &gain, &ifftReal, 1, vDSP_Length(fftSize))

        // Add shape signatures
        if sphereActive || tetraActive || cubeActive {
            for i in 0..<fftSize {
                let exc = Float.random(in: -1...1)
                var add: Float = 0
                if sphereActive { add += sphereLevel * sphereComb.process(exc) }
                if tetraActive  { add += tetraLevel  * tetraComb.process(exc) }
                if cubeActive {
                    // dual AP + dry for strong phasing
                    let y1 = cubeComb.process(exc)
                    let y2 = cubeComb2.process(y1)
                    add += cubeLevel * ((1 - cubeMixBeta) * exc + cubeMixBeta * y2)
                }
                ifftReal[i] += add
            }
        }

        // Edge click
        if enableEdgeClicks, clickEnv > 1e-4 {
            let len = min(24, fftSize)
            for i in 0..<len {
                let env = (clickEnv * 0.2) * expf(-Float(i) / 6.0)
                ifftReal[i] += env
            }
            clickEnv *= 0.2
        } else {
            clickEnv *= 0.1
        }

        // AGC
        var rms: Float = 0
        vDSP_rmsqv(ifftReal, 1, &rms, vDSP_Length(fftSize))
        if rms > 1e-6 {
            var target = agcTargetRMS / rms
            target = max(0.25, min(4.0, target))
            agcGain = agcAlpha * agcGain + (1 - agcAlpha) * target
            vDSP_vsmul(ifftReal, 1, &agcGain, &ifftReal, 1, vDSP_Length(fftSize))
        }

        // Makeup + limiter
        var makeup: Float = powf(10.0, outputGainDB / 20.0)
        vDSP_vsmul(ifftReal, 1, &makeup, &ifftReal, 1, vDSP_Length(fftSize))
        var peak: Float = 0
        vDSP_maxmgv(ifftReal, 1, &peak, vDSP_Length(fftSize))
        if peak > 0.95 {
            var s: Float = 0.95 / peak
            vDSP_vsmul(ifftReal, 1, &s, &ifftReal, 1, vDSP_Length(fftSize))
        }

        // Constant-power pan
        let theta = (pan + 1) * Float.pi * 0.25
        let gL = sin(theta)
        let gR = cos(theta)

        // OLA write
        for i in 0..<fftSize {
            let idx = (olaWrite + i) % olaL.count
            olaL[idx] += ifftReal[i] * gL
            olaR[idx] += ifftReal[i] * gR
        }
        olaWrite = (olaWrite + hop) % olaL.count
    }

    private func copyFromOLA(to outL: UnsafeMutablePointer<Float>,
                             outR: UnsafeMutablePointer<Float>,
                             count: Int) {
        for i in 0..<count {
            outL[i] = olaL[olaRead]
            outR[i] = olaR[olaRead]
            olaL[olaRead] = 0
            olaR[olaRead] = 0
            olaRead = (olaRead + 1) % olaL.count
        }
    }

    private func genWhite() {
        for i in 0..<fftSize {
            timeBlock[i] = Float.random(in: -1...1) * 0.35
        }
    }

    private func applySpectralShaping() {
        guard bandEdges.count == numBands + 1, !binToBand.isEmpty else { return }
        let half = fftSize / 2
        let nyq  = Float(sampleRate / 2)
        let binHz = nyq / Float(half)

        // DC
        do {
            let band = max(0, min(binToBand[0], numBands - 1))
            let f0 = bandEdges[band], f1 = bandEdges[min(band + 1, numBands)]
            let fc = sqrtf(max(10, f0) * max(10, f1))
            let tri = max(0, 1 - (fc / max(60, (f1 - f0)) * 2)) // small weight
            let gAtt = min(1.0, max(0.05, currentGains[band]))
            let boost = 1.0 + (targetBoostLin - 1.0) * targetBoostBands[band]
            let w = gAtt * tri * boost
            freqReal[0] *= w; freqImag[0] *= w
        }

        // 1 … half-1 with mirror
        if half > 1 {
            for i in 1..<(half) {
                let band = max(0, min(binToBand[i], numBands - 1))
                let gBand = currentGains[band]
                let f = Float(i) * binHz
                let f0 = bandEdges[band]
                let f1 = bandEdges[min(band + 1, numBands)]
                let fcBand = sqrtf(max(10, f0) * max(10, f1))
                let bw = max(60, (f1 - f0))
                let tri = max(0, 1 - abs(f - fcBand) / (bw * 0.5))
                let gAtt = min(1.0, max(0.05, gBand))
                let boost = 1.0 + (targetBoostLin - 1.0) * targetBoostBands[band]
                let w = gAtt * tri * boost

                freqReal[i] *= w; freqImag[i] *= w
                let j = fftSize - i
                freqReal[j] *= w; freqImag[j] *= w
            }
        }

        // Nyquist
        let bandN = max(0, min(binToBand[half], numBands - 1))
        let gN = min(1.0, max(0.05, currentGains[bandN]))
        let boostN = 1.0 + (targetBoostLin - 1.0) * targetBoostBands[half < binToBand.count ? bandN : (numBands - 1)]
        let wN = gN * boostN
        freqReal[half] *= wN; freqImag[half] *= wN
    }
}
