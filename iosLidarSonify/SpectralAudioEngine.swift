import Foundation
import AVFoundation
import Accelerate
import Combine

final class SpectralAudioEngine: ObservableObject {
    // Public control from UI
    @Published var pan: Float = 0 // -1..+1

    // Constants
    private let sampleRate: Double
    private let fftSize: Int = 1024
    private let hop: Int = 256 // 4x overlap

    // Engine
    private let engine = AVAudioEngine()
    private var srcNode: AVAudioSourceNode!

    // Source osc
    private var phase: Float = 0
    private var freq: Float = 220

    // Window
    private var window: [Float]
    private var invWindowEnergy: Float

    // DFTs (complex <-> complex for widest SDK support)
    private let forwardDFT: vDSP.DFT<Float>
    private let inverseDFT: vDSP.DFT<Float>

    // Buffers
    private var timeBlock = [Float]()
    private var winTime   = [Float]()
    private var freqReal  = [Float]()  // N bins
    private var freqImag  = [Float]()  // N bins
    private var ifftReal  = [Float]()  // N time

    // Convenience for DFT API
    private var zeroImag: [Float]
    private var scratchImagTime: [Float]

    // Overlap-add ring
    private var olaL: [Float]
    private var olaR: [Float]
    private var olaWrite = 0
    private var olaRead  = 0

    // 40-band mapping
    private let numBands = 40
    private var bandEdges = [Float]()     // Hz edges (numBands+1)
    private var binToBand = [Int]()       // map 0...N/2 to band

    // Envelope smoothing
    private var currentGains = [Float](repeating: 1, count: 40)
    private let gainSmooth: Float = 0.2

    init() {
        let session = AVAudioSession.sharedInstance()
        self.sampleRate = session.sampleRate

        // Hann window (normalized for overlap-add)
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

        zeroImag         = [Float](repeating: 0, count: fftSize)
        scratchImagTime  = [Float](repeating: 0, count: fftSize)

        olaL = [Float](repeating: 0, count: fftSize * 2)
        olaR = [Float](repeating: 0, count: fftSize * 2)

        // DFTs
        forwardDFT = vDSP.DFT<Float>(count: fftSize,
                                     direction: .forward,
                                     transformType: .complexComplex,
                                     ofType: Float.self)!
        inverseDFT = vDSP.DFT<Float>(count: fftSize,
                                     direction: .inverse,
                                     transformType: .complexComplex,
                                     ofType: Float.self)!

        // Source node
        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        srcNode = AVAudioSourceNode { [weak self] _, _, frameCount, ablPtr -> OSStatus in
            guard let self = self else { return noErr }
            let n = Int(frameCount)
            let abl = UnsafeMutableAudioBufferListPointer(ablPtr)
            let outL = abl[0].mData!.assumingMemoryBound(to: Float.self)
            let outR = abl[1].mData!.assumingMemoryBound(to: Float.self)

            // Produce before consume so the ring always has data
            var produced = 0
            while produced < n {
                self.processBlock()
                let toCopy = min(self.hop, n - produced)
                self.copyFromOLA(to: outL + produced, outR: outR + produced, count: toCopy)
                produced += toCopy
            }
            return noErr
        }

        engine.attach(srcNode)
        engine.connect(srcNode, to: engine.mainMixerNode, format: fmt)

        try? session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try? session.setActive(true)
    }

    func start() { try? engine.start() }
    func stop()  { engine.stop() }

    // Configure log-spaced bands (call on start & when min/max change)
    func configureBands(fMin: Double, fMax: Double) {
        let fm = Float(max(20, min(fMin, fMax - 10)))
        let fM = Float(max(fm + 10, Float(fMax)))
        bandEdges = [Float](repeating: 0, count: numBands + 1)

        let ratio: Float = fM / fm
        for i in 0...numBands {
            let t: Float = Float(i) / Float(numBands)
            bandEdges[i] = fm * powf(ratio, t)   // Float pow
        }

        // Map only non-negative frequencies (0…N/2)
        let nyquist = Float(sampleRate / 2)
        let binHz   = nyquist / Float(fftSize/2)
        binToBand   = [Int](repeating: numBands - 1, count: fftSize/2 + 1)

        for b in 0..<numBands {
            let f0 = bandEdges[b]
            let f1 = bandEdges[b+1]
            let i0 = max(0, Int(floorf(f0 / binHz)))
            let i1 = min(fftSize/2, Int(ceilf(f1 / binHz)))
            for i in i0...i1 { binToBand[i] = b }
        }
    }

    // Update 40-band envelope
    func updateEnvelope(_ gains40: [Float]) {
        let n = min(gains40.count, numBands)
        for i in 0..<n {
            let tgt = max(0.001, min(4.0, gains40[i]))
            currentGains[i] = gainSmooth * currentGains[i] + (1 - gainSmooth) * tgt
        }
    }

    // MARK: - DSP

    private func processBlock() {
        // 1) Generate square wave
        let sr = Float(sampleRate)
        let twoPi = Float.pi * 2
        let dphi = twoPi * freq / sr
        var ph = phase
        for i in 0..<fftSize {
            ph += dphi; if ph >= twoPi { ph -= twoPi }
            timeBlock[i] = (ph < .pi) ? 1 : -1
        }
        phase = ph

        // 2) Window
        vDSP_vmul(timeBlock, 1, window, 1, &winTime, 1, vDSP_Length(fftSize))

        // 3) Complex spectrum
        forwardDFT.transform(inputReal: winTime,
                             inputImaginary: zeroImag,
                             outputReal: &freqReal,
                             outputImaginary: &freqImag)

        // 4) Apply 40-band gains symmetrically (+/- freqs)
        let half = fftSize / 2
        for i in 0...half { // include DC & Nyquist
            let b = binToBand[i]
            let g = currentGains[b]
            // +freq
            freqReal[i] *= g
            freqImag[i] *= g
            // mirror to -freq
            if i != 0 && i != half {
                let j = fftSize - i
                freqReal[j] *= g
                freqImag[j] *= g
            }
        }

        // 5) Inverse complex DFT (imag time discarded)
        inverseDFT.transform(inputReal: freqReal,
                             inputImaginary: freqImag,
                             outputReal: &ifftReal,
                             outputImaginary: &scratchImagTime)

        // 6) Normalize and constant-power pan
        var scale = 1.0 / Float(fftSize)
        vDSP_vsmul(ifftReal, 1, &scale, &ifftReal, 1, vDSP_Length(fftSize))

        // compensate window energy for 4x overlap
        var gain = invWindowEnergy * 4
        vDSP_vsmul(ifftReal, 1, &gain, &ifftReal, 1, vDSP_Length(fftSize))

        let theta = (pan + 1) * Float.pi * 0.25 // -1..+1 → 0..π/2
        let gL = sin(theta) // swapped to flip L/R
        let gR = cos(theta)

        // 7) Overlap-add write
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
}
