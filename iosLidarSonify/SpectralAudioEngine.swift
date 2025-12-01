import Foundation
import AVFoundation
import Accelerate
import Combine

final class SpectralAudioEngine: ObservableObject {
    
    // UI controls
    @Published var pan: Float = 0                 // -1…+1
    @Published var enableEdgeClicks: Bool = false
    @Published var outputGainDB: Float = 6.0      // Master output gain (dB)
    
    // Constants
    private let sampleRate: Double
    private let fftSize: Int = 1024
    private let hop: Int = 256 // 4× overlap
    
    // AGC (post-shaping, pre-pan)
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
    private var targetBoostBands = [Float](repeating: 0, count: 40)
    private var targetBoostLin: Float = powf(10.0, 12.0/20.0)
    private let targetBoostSmooth: Float = 0.6
    private var targetShape: Int = 0
    
    // Pre-allocated foreground buffer
    private var foregroundBuf = [Float]()
    
    // ===== STABLE Comb filters - FIXED M values per shape =====
    // The M values are now fixed at initialization and don't change based on fc
    // This eliminates pitch drift and the "glitchy" sound
    
    struct FBComb {
        var g: Float
        let M: Int  // Now constant
        var buf: [Float]
        var i = 0
        
        init(g: Float, M: Int) {
            self.g = g
            self.M = max(1, M)
            self.buf = [Float](repeating: 0, count: self.M)
        }
        
        mutating func process(_ x: Float) -> Float {
            let y = x + g * buf[i]
            buf[i] = y
            i = (i + 1) % M
            return y
        }
        
        mutating func clear() {
            for j in 0..<buf.count { buf[j] = 0 }
            i = 0
        }
    }
    
    struct FFComb {
        var alpha: Float
        let M: Int
        var buf: [Float]
        var i = 0
        
        init(alpha: Float, M: Int) {
            self.alpha = alpha
            self.M = max(1, M)
            self.buf = [Float](repeating: 0, count: self.M)
        }
        
        mutating func process(_ x: Float) -> Float {
            let y = x + alpha * buf[i]
            buf[i] = x
            i = (i + 1) % M
            return y
        }
        
        mutating func clear() {
            for j in 0..<buf.count { buf[j] = 0 }
            i = 0
        }
    }
    
    struct APComb {
        var a: Float
        let M: Int
        var xbuf: [Float]
        var ybuf: [Float]
        var i = 0
        
        init(a: Float, M: Int) {
            self.a = a
            self.M = max(1, M)
            self.xbuf = [Float](repeating: 0, count: self.M)
            self.ybuf = [Float](repeating: 0, count: self.M)
        }
        
        mutating func process(_ x: Float) -> Float {
            let xm = xbuf[i], ym = ybuf[i]
            let y = -a * x + xm + a * ym
            xbuf[i] = x
            ybuf[i] = y
            i = (i + 1) % M
            return y
        }
        
        mutating func clear() {
            for j in 0..<xbuf.count { xbuf[j] = 0; ybuf[j] = 0 }
            i = 0
        }
    }
    
    // ===== FIXED M VALUES FOR DISTINCT TIMBRES =====
    // At 48kHz: M=200 → 240Hz, M=100 → 480Hz, M=50 → 960Hz
    // These are tuned for distinct, musical timbres
    
    // Sphere: warm, droning, low fundamental (around 200Hz)
    private var sphereComb = FBComb(g: 0.92, M: 240)  // ~200Hz @ 48kHz
    
    // Tetra: bright, metallic, inharmonic (two combs at non-integer ratio)
    private var tetraComb1 = FFComb(alpha: 0.85, M: 89)   // ~540Hz @ 48kHz
    private var tetraComb2 = FFComb(alpha: 0.75, M: 144)  // ~333Hz @ 48kHz (golden ratio-ish)
    
    // Cube: percussive, clicky, short decay
    private var cubeComb = APComb(a: 0.6, M: 37)  // ~1300Hz @ 48kHz
    
    // Shape activation states
    private var sphereActive = false
    private var tetraActive  = false
    private var cubeActive   = false
    
    // Current levels (smoothed)
    private var sphereLevel: Float = 0.0
    private var tetraLevel:  Float = 0.0
    private var cubeLevel:   Float = 0.0
    
    // Target levels for ramping
    private var sphereTargetLevel: Float = 0.0
    private var tetraTargetLevel:  Float = 0.0
    private var cubeTargetLevel:   Float = 0.0
    
    // Smoothing factor for level ramping (higher = slower/smoother)
    private let levelRampAlpha: Float = 0.95
    
    // Shape mask modulation - use the actual 40-band mask to modulate the sound
    private var currentMask = [Float](repeating: 0, count: 40)
    private var smoothedMask = [Float](repeating: 0, count: 40)
    private let maskSmoothAlpha: Float = 0.85
    
    // Overlap-add ring
    private var olaL: [Float]
    private var olaR: [Float]
    private var olaWrite = 0
    private var olaRead  = 0
    
    // 40-band mapping
    private let numBands = 40
    private var bandEdges = [Float]()
    private var binToBand = [Int]()
    
    // Distance & LPF
    private var z01: Float = 0
    private var zSlew: Float = 0
    private let zSlewA: Float = 0.15
    
    // Edge clicks
    private var clickEnv: Float = 0
    
    // Gain smoothing
    private var currentGains = [Float](repeating: 1, count: 40)
    private let gainSmooth: Float = 0.85
    
    // ===== DEBUG =====
    private let DEBUG_AUDIO = false  // Reduced logging
    private var lastShapeReported: Int = -1
    private var framesSinceLastLog = 0
    
    private func shapeName(_ s: Int) -> String {
        switch s { case 1: return "sphere"; case 2: return "tetra"; case 3: return "cube"; default: return "none" }
    }
    
    // ===== init =====
    
    init() {
        let session = AVAudioSession.sharedInstance()
        self.sampleRate = session.sampleRate
        
        // Re-initialize combs with correct sample rate
        // Sphere: ~180Hz fundamental for warmth
        let sphereM = max(100, Int(sampleRate / 180))
        sphereComb = FBComb(g: 0.92, M: sphereM)
        
        // Tetra: ~400Hz and ~650Hz for inharmonic metallic sound
        let tetra1M = max(50, Int(sampleRate / 400))
        let tetra2M = max(50, Int(sampleRate / 650))
        tetraComb1 = FFComb(alpha: 0.85, M: tetra1M)
        tetraComb2 = FFComb(alpha: 0.75, M: tetra2M)
        
        // Cube: ~1000Hz for percussive attack
        let cubeM = max(20, Int(sampleRate / 1000))
        cubeComb = APComb(a: 0.6, M: cubeM)
        
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
        foregroundBuf = [Float](repeating: 0, count: fftSize)
        olaL = [Float](repeating: 0, count: fftSize * 2)
        olaR = [Float](repeating: 0, count: fftSize * 2)
        
        // DFTs
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
        
        // Graph
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
    
    // MARK: Public controls
    
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
            for i in i0...i1 { binToBand[i] = b }
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
    }
    
    /// Set target bands and shape. The mask encodes the vertical extent of the shape.
    func setTargetBands(_ mask: [Float], shape: Int, boostDB: Float = 12.0) {
        // Store mask for modulation
        for i in 0..<min(numBands, mask.count) {
            currentMask[i] = mask[i]
        }
        
        // Smooth the target boost bands
        for i in 0..<numBands {
            let target = i < mask.count ? max(0, min(1, mask[i])) : 0
            targetBoostBands[i] = targetBoostSmooth * targetBoostBands[i] + (1 - targetBoostSmooth) * target
        }
        
        // Calculate coverage (what % of bands are active)
        let coverage = mask.reduce(0, +) / Float(numBands)
        
        // Set target levels based on shape
        // Coverage modulates the level - larger shapes = louder
        let baseLvl = 0.5 + coverage * 0.4  // 0.5 to 0.9 based on coverage
        
        targetBoostLin = powf(10.0, boostDB / 20.0)
        targetShape = shape
        
        switch shape {
        case 1: // sphere
            sphereTargetLevel = baseLvl * 0.7
            tetraTargetLevel = 0
            cubeTargetLevel = 0
        case 2: // tetra
            sphereTargetLevel = 0
            tetraTargetLevel = baseLvl * 0.85
            cubeTargetLevel = 0
        case 3: // cube
            sphereTargetLevel = 0
            tetraTargetLevel = 0
            cubeTargetLevel = baseLvl * 1.0
        default:
            sphereTargetLevel = 0
            tetraTargetLevel = 0
            cubeTargetLevel = 0
        }
        
        // Activate shapes as needed (but don't clear buffers - just set active flag)
        if shape == 1 && !sphereActive { sphereActive = true }
        if shape == 2 && !tetraActive { tetraActive = true }
        if shape == 3 && !cubeActive { cubeActive = true }
        
        // Logging (throttled)
        if DEBUG_AUDIO && shape != lastShapeReported {
            lastShapeReported = shape
            print("[Audio] Shape: \(shapeName(shape)), coverage: \(String(format: "%.2f", coverage))")
        }
    }
    
    func clearTarget() {
        for i in 0..<numBands {
            targetBoostBands[i] *= 0.9  // Gradual decay
            currentMask[i] = 0
        }
        sphereTargetLevel = 0
        tetraTargetLevel = 0
        cubeTargetLevel = 0
    }
    
    func triggerEdge(_ strength: Float) {
        let s = max(0, min(1, strength))
        clickEnv = max(clickEnv, s * 0.6)
    }
    
    // MARK: - DSP
    
    private func processBlock() {
        // Smooth level transitions
        sphereLevel = levelRampAlpha * sphereLevel + (1 - levelRampAlpha) * sphereTargetLevel
        tetraLevel  = levelRampAlpha * tetraLevel  + (1 - levelRampAlpha) * tetraTargetLevel
        cubeLevel   = levelRampAlpha * cubeLevel   + (1 - levelRampAlpha) * cubeTargetLevel
        
        // Smooth mask for modulation
        for i in 0..<numBands {
            smoothedMask[i] = maskSmoothAlpha * smoothedMask[i] + (1 - maskSmoothAlpha) * currentMask[i]
        }
        
        // Deactivate and clear shapes that have fully faded out
        if sphereActive && sphereTargetLevel == 0 && sphereLevel < 0.001 {
            sphereActive = false
            sphereComb.clear()
        }
        if tetraActive && tetraTargetLevel == 0 && tetraLevel < 0.001 {
            tetraActive = false
            tetraComb1.clear()
            tetraComb2.clear()
        }
        if cubeActive && cubeTargetLevel == 0 && cubeLevel < 0.001 {
            cubeActive = false
            cubeComb.clear()
        }
        
        // Generate background noise
        genWhite()
        
        // Window → FFT
        vDSP_vmul(timeBlock, 1, window, 1, &winTime, 1, vDSP_Length(fftSize))
        forwardDFT.transform(inputReal: winTime,
                             inputImaginary: zeroImag,
                             outputReal: &freqReal,
                             outputImaginary: &freqImag)
        
        // Apply spectral shaping (depth envelope)
        applySpectralShaping()
        
        // iDFT
        inverseDFT.transform(inputReal: freqReal,
                             inputImaginary: freqImag,
                             outputReal: &ifftReal,
                             outputImaginary: &scratchImagTime)
        
        // Normalize
        var scale = 1.0 / Float(fftSize)
        vDSP_vsmul(ifftReal, 1, &scale, &ifftReal, 1, vDSP_Length(fftSize))
        var gain = invWindowEnergy * 4
        vDSP_vsmul(ifftReal, 1, &gain, &ifftReal, 1, vDSP_Length(fftSize))
        
        // === Generate foreground shape signal ===
        let hasActiveSignal = (sphereActive && sphereLevel > 0.005) ||
                              (tetraActive && tetraLevel > 0.005) ||
                              (cubeActive && cubeLevel > 0.005)
        
        // Calculate mask-based modulation
        // Higher bands = higher frequency modulation
        var maskMod: Float = 0.5
        var maskWeightSum: Float = 0
        for i in 0..<numBands {
            let w = smoothedMask[i]
            maskMod += w * Float(i) / Float(numBands)
            maskWeightSum += w
        }
        if maskWeightSum > 0.01 {
            maskMod /= maskWeightSum
        }
        // maskMod now ranges 0-1, with higher = shape is higher in the frame
        
        // Clear foreground buffer
        for i in 0..<fftSize { foregroundBuf[i] = 0 }
        
        if hasActiveSignal {
            // Use mask modulation to affect timbre slightly
            // Higher shapes get brighter excitation
            let excBrightness = 0.3 + maskMod * 0.7  // 0.3 to 1.0
            
            for i in 0..<fftSize {
                // Shaped noise excitation - higher maskMod = brighter
                var exc = Float.random(in: -1...1) * 0.5
                if excBrightness > 0.5 {
                    // Add some high frequency content for higher shapes
                    exc += Float.random(in: -0.3...0.3)
                }
                
                var add: Float = 0
                
                // SPHERE: warm, droning, sustained
                if sphereActive && sphereLevel > 0.005 {
                    // Modulate feedback based on mask (larger = more sustain)
                    sphereComb.g = 0.88 + maskWeightSum * 0.06  // 0.88 to 0.94
                    let out = sphereComb.process(exc)
                    add += sphereLevel * out * 0.8
                }
                
                // TETRA: metallic, inharmonic, bright
                if tetraActive && tetraLevel > 0.005 {
                    let t1 = tetraComb1.process(exc)
                    let t2 = tetraComb2.process(exc * 0.8)
                    let out = t1 * 0.55 + t2 * 0.45
                    add += tetraLevel * out * 0.9
                }
                
                // CUBE: percussive, clicky, short
                if cubeActive && cubeLevel > 0.005 {
                    let wet = cubeComb.process(exc)
                    // More dry = more attack
                    let dryMix: Float = 0.5 - maskMod * 0.2  // Less dry for higher shapes
                    let out = dryMix * exc + (1 - dryMix) * wet
                    add += cubeLevel * out * 1.0
                }
                
                foregroundBuf[i] = add
            }
        }
        
        // Mix background + foreground with ducking
        var fgSumSq: Float = 0
        vDSP_svesq(foregroundBuf, 1, &fgSumSq, vDSP_Length(fftSize))
        let fgRMS = sqrtf(fgSumSq / Float(fftSize))
        let hasForeground = fgRMS > 1e-4
        
        // Duck background when shape present
        let bgDuck: Float = hasForeground ? 0.35 : 1.0
        let fgGain: Float = 0.8
        
        for i in 0..<fftSize {
            ifftReal[i] = ifftReal[i] * bgDuck + foregroundBuf[i] * fgGain
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
        
        // Makeup gain
        var makeup: Float = powf(10.0, outputGainDB / 20.0)
        vDSP_vsmul(ifftReal, 1, &makeup, &ifftReal, 1, vDSP_Length(fftSize))
        
        // Safety limiter
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
    
    // MARK: Generators
    
    private func genWhite() {
        for i in 0..<fftSize {
            timeBlock[i] = Float.random(in: -1...1) * 0.35
        }
    }
    
    // MARK: Spectral shaping
    
    private func applySpectralShaping() {
        guard bandEdges.count == numBands + 1, !binToBand.isEmpty else { return }
        let half = fftSize / 2
        let nyq  = Float(sampleRate / 2)
        let binHz = nyq / Float(half)
        
        for i in 0...half {
            let bRaw = binToBand[i]
            let band = max(0, min(bRaw, numBands - 1))
            let gBand = currentGains[band]
            
            let f = Float(i) * binHz
            let f0 = bandEdges[band]
            let f1 = bandEdges[min(band + 1, numBands)]
            let fcBand = sqrtf(max(10, f0) * max(10, f1))
            let bw = max(60, (f1 - f0))
            let tri = max(0, 1 - abs(f - fcBand) / (bw * 0.5))
            
            let gAtt = min(1.0, max(0.05, gBand))
            
            // Target boost
            let tBoost = targetBoostBands[band]
            let boostFactor = 1.0 + (targetBoostLin - 1.0) * tBoost
            
            let w = gAtt * tri * boostFactor
            
            freqReal[i] *= w
            freqImag[i] *= w
            
            if i > 0 && i < half {
                let mirrorIdx = fftSize - i
                freqReal[mirrorIdx] *= w
                freqImag[mirrorIdx] *= w
            }
        }
    }
}
