import Foundation
import AVFoundation
import Accelerate
import Combine

final class SpectralAudioEngine: ObservableObject {

    enum SourceMode: String, CaseIterable, Identifiable {
        case square, pinkNoise, gammatoneNoise
        var id: String { rawValue }
        var label: String {
            switch self {
            case .square: return "Square"
            case .pinkNoise: return "Pink noise"
            case .gammatoneNoise: return "Gammatone noise"
            }
        }
    }

    // UI controls
    @Published var pan: Float = 0                 // -1…+1
    @Published var sourceMode: SourceMode = .square
    @Published var enableEdgeClicks: Bool = false
    // Master output gain (dB)
    @Published var outputGainDB: Float = 6.0

    // Constants
    private let sampleRate: Double
    private let fftSize: Int = 1024
    private let hop: Int = 256 // 4× overlap

    // AGC (post-shaping, pre-pan)
    private var agcGain: Float = 1.0
    private let agcAlpha: Float = 0.95     // smoothing (0.95 = slow)
    private let agcTargetRMS: Float = 0.20 // ~ -14 dBFS (slightly louder baseline)

    // Engine
    private let engine = AVAudioEngine()
    private var srcNode: AVAudioSourceNode!
    private let reverb = AVAudioUnitReverb()

    // Square osc
    private var phase: Float = 0
    private var freq: Float = 220

    // Window / DFT
    private var window: [Float]
    private var invWindowEnergy: Float
    private let forwardDFT: vDSP.DFT<Float>
    private let inverseDFT: vDSP.DFT<Float>

    // Buffers
    private var timeBlock = [Float]()
    private var winTime   = [Float]()
    private var freqReal  = [Float]()  // N
    private var freqImag  = [Float]()  // N
    private var ifftReal  = [Float]()  // N
    private var zeroImag  = [Float]()
    private var scratchImagTime = [Float]()

    // Overlap-add ring
    private var olaL: [Float]
    private var olaR: [Float]
    private var olaWrite = 0
    private var olaRead  = 0

    // 40-band mapping
    private let numBands = 40
    private var bandEdges = [Float]()     // Hz (numBands+1)
    private var binToBand = [Int]()       // 0…N/2 -> band index

    // Distance & LPF
    // z01: 0 = near, 1 = far
    private var z01: Float = 0
    private var zSlew: Float = 0          // smoothed z for LPF/Reverb
    private let zSlewA: Float = 0.15

    // Pink noise IIR state (Paul Kellet style approx)
    private var p0: Float = 0, p1: Float = 0, p2: Float = 0

    // Edge clicks
    private var clickEnv: Float = 0        // injected short click per block

    // Gain smoothing
    private var currentGains = [Float](repeating: 1, count: 40)
    private let gainSmooth: Float = 0.85

    init() {
        let session = AVAudioSession.sharedInstance()
        self.sampleRate = session.sampleRate

        // Hann window normalized
        window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        var sum: Float = 0
        vDSP_sve(window, 1, &sum, vDSP_Length(fftSize))
        invWindowEnergy = sum > 0 ? 1.0 / sum : 1.0

        // Buffers
        timeBlock = [Float](repeating: 0, count: fftSize)
        winTime   = [Float](repeating: 0, count: fftSize)
        freqReal  = [Float](repeating: 0, count: fftSize)
        freqImag  = [Float](repeating: 0, count: fftSize)
        ifftReal  = [Float](repeating: 0, count: fftSize)
        zeroImag  = [Float](repeating: 0, count: fftSize)
        scratchImagTime = [Float](repeating: 0, count: fftSize)
        olaL = [Float](repeating: 0, count: fftSize * 2)
        olaR = [Float](repeating: 0, count: fftSize * 2)

        // DFTs (complex<->complex)
        forwardDFT = vDSP.DFT<Float>(count: fftSize, direction: .forward, transformType: .complexComplex, ofType: Float.self)!
        inverseDFT = vDSP.DFT<Float>(count: fftSize, direction: .inverse, transformType: .complexComplex, ofType: Float.self)!

        // Source node
        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        srcNode = AVAudioSourceNode { [weak self] _, _, frameCount, ablPtr -> OSStatus in
            guard let self = self else { return noErr }
            let n = Int(frameCount)
            let abl = UnsafeMutableAudioBufferListPointer(ablPtr)
            let outL = abl[0].mData!.assumingMemoryBound(to: Float.self)
            let outR = abl[1].mData!.assumingMemoryBound(to: Float.self)

            var produced = 0
            while produced < n {
                self.processBlock()
                let toCopy = min(self.hop, n - produced)
                self.copyFromOLA(to: outL + produced, outR: outR + produced, count: toCopy)
                produced += toCopy
            }
            return noErr
        }

        // Graph: Source -> Reverb -> MainMixer
        engine.attach(srcNode)
        engine.attach(reverb)
        engine.connect(srcNode, to: reverb, format: fmt)
        engine.connect(reverb, to: engine.mainMixerNode, format: fmt)
        reverb.loadFactoryPreset(.largeRoom)
        reverb.wetDryMix = 0 // start dry
        engine.mainMixerNode.outputVolume = 1.0

        try? session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try? session.setActive(true)
    }

    func start() { try? engine.start() }
    func stop()  { engine.stop() }

    // MARK: Public controls from pipeline/UI

    func configureBands(fMin: Double, fMax: Double) {
        let fm = Float(max(20, min(fMin, fMax - 10)))
        let fM = Float(max(fm + 10, Float(fMax)))
        bandEdges = [Float](repeating: 0, count: numBands + 1)
        let ratio: Float = fM / fm
        for i in 0...numBands {
            let t: Float = Float(i) / Float(numBands)
            bandEdges[i] = fm * powf(ratio, t)
        }
        // Bin → band map (0…N/2)
        let nyq = Float(sampleRate / 2)
        let binHz = nyq / Float(fftSize/2)
        binToBand = [Int](repeating: numBands - 1, count: fftSize/2 + 1)
        for b in 0..<numBands {
            let f0 = bandEdges[b]
            let f1 = bandEdges[b+1]
            let i0 = max(0, Int(floorf(f0 / binHz)))
            let i1 = min(fftSize/2, Int(ceilf(f1 / binHz)))
            for i in i0...i1 { binToBand[i] = b }
        }
    }

    func updateEnvelope(_ gains40: [Float]) {
        // 1) Clamp to attenuation‑only (0.05…1.0)
        var g = gains40
        if g.count < numBands { g += Array(repeating: 1, count: numBands - g.count) }
        for i in 0..<numBands { g[i] = min(1.0, max(0.05, g[i])) }
        // 2) Light spectral smoothing across neighbors (0.2, 0.6, 0.2)
        var s = [Float](repeating: 1, count: numBands)
        for i in 0..<numBands {
            let a = (i > 0) ? g[i-1] : g[i]
            let b = g[i]
            let c = (i+1 < numBands) ? g[i+1] : g[i]
            s[i] = 0.2*a + 0.6*b + 0.2*c
        }
        // 3) Temporal smoothing
        for i in 0..<numBands {
            currentGains[i] = gainSmooth * currentGains[i] + (1 - gainSmooth) * s[i]
        }
    }

    /// 0 (near) … 1 (far) — also drives LPF & reverb
    func updateDistance(_ z: Float) {
        z01 = max(0, min(1, z))
        // Smooth for stability; keep for future use but do not color the timbre now
        zSlew = zSlewA * zSlew + (1 - zSlewA) * z01
        // Disable distance-based reverb while evaluating pure filter-bank sonification
        DispatchQueue.main.async { [weak self] in self?.reverb.wetDryMix = 0 }
    }

    /// Edge strength 0…1 → inject short click
    func triggerEdge(_ strength: Float) {
        let s = max(0, min(1, strength))
        // accumulate a bit so rapid edges pop
        clickEnv = max(clickEnv, s * 0.6)
    }

    // MARK: - DSP

    private func processBlock() {
        // Generate excitation
        switch sourceMode {
        case .square:
            genSquare()
            // window → FFT
            vDSP_vmul(timeBlock, 1, window, 1, &winTime, 1, vDSP_Length(fftSize))
            forwardDFT.transform(inputReal: winTime,
                                 inputImaginary: zeroImag,
                                 outputReal: &freqReal,
                                 outputImaginary: &freqImag)

        case .pinkNoise:
            genPink()
            vDSP_vmul(timeBlock, 1, window, 1, &winTime, 1, vDSP_Length(fftSize))
            forwardDFT.transform(inputReal: winTime,
                                 inputImaginary: zeroImag,
                                 outputReal: &freqReal,
                                 outputImaginary: &freqImag)

        case .gammatoneNoise:
            genHermitianNoiseSpectrum() // fills freqReal/freqImag (N) as Hermitian
        }

        // Apply 40-band envelope, LPF (2–12 kHz with Z), and mirror
        applySpectralShaping()

        // iDFT
        inverseDFT.transform(inputReal: freqReal,
                             inputImaginary: freqImag,
                             outputReal: &ifftReal,
                             outputImaginary: &scratchImagTime)

        // Normalize block energy (FFT scale + OLA compensation)
        var scale = 1.0 / Float(fftSize)
        vDSP_vsmul(ifftReal, 1, &scale, &ifftReal, 1, vDSP_Length(fftSize))
        var gain = invWindowEnergy * 4
        vDSP_vsmul(ifftReal, 1, &gain, &ifftReal, 1, vDSP_Length(fftSize))

        // Optional edge click overlay (short decay at start of block)
        if enableEdgeClicks, clickEnv > 1e-4 {
            let len = min(24, fftSize)
            for i in 0..<len {
                // softer, shorter click
                let env = (clickEnv * 0.2) * expf(-Float(i) / 6.0)
                ifftReal[i] += env
            }
            clickEnv *= 0.2 // faster decay per block
        } else {
            // decay even if disabled so queued clicks drain
            clickEnv *= 0.1
        }

        // AGC: normalize block RMS to a gentle target, smoothed to avoid pumping
        var rms: Float = 0
        vDSP_rmsqv(ifftReal, 1, &rms, vDSP_Length(fftSize))
        if rms > 1e-6 {
            var target = agcTargetRMS / rms
            // constrain instantaneous correction
            target = max(0.25, min(4.0, target))
            agcGain = agcAlpha * agcGain + (1 - agcAlpha) * target
            vDSP_vsmul(ifftReal, 1, &agcGain, &ifftReal, 1, vDSP_Length(fftSize))
        }

        // Master makeup gain (post-AGC, pre-limiter)
        var makeup: Float = powf(10.0, outputGainDB / 20.0)
        vDSP_vsmul(ifftReal, 1, &makeup, &ifftReal, 1, vDSP_Length(fftSize))

        // Safety limiter (pre-pan)
        var peak: Float = 0
        vDSP_maxmgv(ifftReal, 1, &peak, vDSP_Length(fftSize))
        if peak > 0.95 {
            var s: Float = 0.95 / peak
            vDSP_vsmul(ifftReal, 1, &s, &ifftReal, 1, vDSP_Length(fftSize))
        }

        // Constant-power pan
        let theta = (pan + 1) * Float.pi * 0.25
        let gL = sin(theta) // flipped earlier so L→R matches visual
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

    // MARK: Generators

    private func genSquare() {
        let sr = Float(sampleRate)
        let dphi = 2 * Float.pi * freq / sr
        var ph = phase
        for i in 0..<fftSize {
            ph += dphi; if ph >= 2 * .pi { ph -= 2 * .pi }
            timeBlock[i] = (ph < .pi) ? 1 : -1
        }
        phase = ph
    }

    private func genPink() {
        // Filtered white noise (approx 1/f)
        for i in 0..<fftSize {
            let w: Float = Float.random(in: -1...1)
            p0 = 0.99886 * p0 + 0.0555179 * w
            p1 = 0.99332 * p1 + 0.0750759 * w
            p2 = 0.96900 * p2 + 0.1538520 * w
            // final mix; small white to flatten top
            let y = p0 + p1 + p2 + 0.3104856 * w
            timeBlock[i] = y * 0.25 // tame level
        }
    }

    private func genHermitianNoiseSpectrum() {
        // Create complex spectrum with Hermitian symmetry for real time signal
        let half = fftSize / 2
        let nyq  = Float(sampleRate / 2)
        let binHz = nyq / Float(half)

        // DC and Nyquist real, imag=0
        freqReal[0] = 0;      freqImag[0] = 0
        freqReal[half] = 0;   freqImag[half] = 0

        for i in 1..<half {
            // gaussian-ish noise via Box-Muller
            let u1 = max(1e-12, Float.random(in: 0..<1))
            let u2 = Float.random(in: 0..<1)
            let mag = sqrtf(-2 * logf(u1))
            var re = mag * cosf(2 * .pi * u2)
            var im = mag * sinf(2 * .pi * u2)

            // keep flat spectrum here; shaping happens later
            _ = Float(i) * binHz // f (unused)
            // no shelf

            freqReal[i] = re
            freqImag[i] = im
            // Mirror
            let j = fftSize - i
            freqReal[j] =  re
            freqImag[j] = -im
        }
    }

    private func applySpectralShaping() {
        let half = fftSize / 2
        let nyq  = Float(sampleRate / 2)
        let binHz = nyq / Float(half)

        for i in 0...half {
            // Choose gain for this bin
            let band = binToBand[i]
            let gBand = currentGains[min(max(band, 0), numBands - 1)]

            // Simple triangular "gammatone-ish" weighting around band center
            let f = Float(i) * binHz
            let f0 = bandEdges[band]
            let f1 = bandEdges[min(band + 1, numBands)]
            let fcBand = sqrtf(f0 * f1)           // geometric mean
            let bw = max(60, (f1 - f0))           // Hz
            let tri = max(0, 1 - abs(f - fcBand) / (bw * 0.5))

            // No LPF for now; rely purely on the 40-band envelope
            let gAtt = min(1.0, max(0.05, gBand))
            let w = gAtt * tri

            // +freq
            freqReal[i] *= w
            freqImag[i] *= w
            // mirror (avoid double at DC/Nyq)
            if i != 0 && i != half {
                let j = fftSize - i
                freqReal[j] *= w
                freqImag[j] *= w
            }
        }
    }
}
