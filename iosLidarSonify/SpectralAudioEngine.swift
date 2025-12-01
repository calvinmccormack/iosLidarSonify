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
    private var freqReal  = [Float]()  // N
    private var freqImag  = [Float]()  // N
    private var ifftReal  = [Float]()  // N
    private var zeroImag  = [Float]()
    private var scratchImagTime = [Float]()
    private var targetBoostBands = [Float](repeating: 0, count: 40)
    private var targetBoostLin: Float = powf(10.0, 12.0/20.0)
    private var targetBoostSmooth: Float = 0.6
    private var targetShape: Int = 0
    
    // Pre-allocated foreground buffer to avoid per-block allocation
    private var foregroundBuf = [Float]()
    
    // ===== Comb filters for target-object sonification =====
    // Lightweight time-domain implementations with DISTINCT timbres
    
    /// Feedback comb: creates pitched resonance at f0 = sr/M
    /// Higher g = longer decay, more tonal. Used for SPHERE (warm, sustained)
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
        mutating func clear() { buf = [Float](repeating: 0, count: M); i = 0 }
    }
    
    /// Feed-forward comb: creates notches, brighter/harsher. Used for TETRA (metallic)
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
        mutating func clear() { buf = [Float](repeating: 0, count: M); i = 0 }
    }
    
    /// Allpass comb: preserves magnitude, shifts phase. Used for CUBE (diffuse, percussive)
    struct APComb {
        var a: Float
        var M: Int
        var xbuf: [Float], ybuf: [Float]; var i = 0
        init(a: Float, M: Int) { self.a = a; self.M = max(1, M); xbuf = .init(repeating: 0, count: max(1, M)); ybuf = xbuf }
        mutating func reinit(a: Float, M: Int) { self.a = a; self.M = max(1, M); xbuf = .init(repeating: 0, count: self.M); ybuf = xbuf; i = 0 }
        mutating func process(_ x: Float) -> Float {
            let xm = xbuf[i], ym = ybuf[i]
            let y = -a * x + xm + a * ym
            xbuf[i] = x
            ybuf[i] = y
            i = (i + 1) % M
            return y
        }
        mutating func clear() { xbuf = [Float](repeating: 0, count: M); ybuf = xbuf; i = 0 }
    }
    
    // ===== ENHANCED COMB PARAMETERS FOR CLEAR DIFFERENTIATION =====
    // Sphere: warm, sustained, low-mid emphasis (feedback comb, high g, longer M)
    private var sphereComb = FBComb(g: 0.94, M: 220)
    
    // Tetra: bright, metallic, harsh edge (feed-forward, shorter M for higher harmonics)
    private var tetraComb = FFComb(alpha: 0.92, M: 85)
    // Add second tetra comb for richer metallic texture
    private var tetraComb2 = FFComb(alpha: 0.80, M: 127)
    
    // Cube: percussive, diffuse, rhythmic (allpass, very short for transients)
    private var cubeComb = APComb(a: 0.65, M: 48)
    
    private var sphereActive = false
    private var tetraActive  = false
    private var cubeActive   = false
    
    private var sphereLevel: Float = 0.0
    private var tetraLevel:  Float = 0.0
    private var cubeLevel:   Float = 0.0
    
    // Target levels for smooth ramping (avoids clicks)
    private var sphereTargetLevel: Float = 0.0
    private var tetraTargetLevel:  Float = 0.0
    private var cubeTargetLevel:   Float = 0.0
    private let levelRampAlpha: Float = 0.92  // Smoothing factor
    
    // Hold current M to avoid per-column reinitialization
    private var sphereM: Int = 220
    private var tetraM:  Int = 85
    private var cubeM:   Int = 48
    
    // Dry/wet for allpass branch (cube) - more dry = more percussive attack
    private var cubeMixBeta: Float = 0.50
    
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
    private var z01: Float = 0
    private var zSlew: Float = 0
    private let zSlewA: Float = 0.15
    
    // Pink noise IIR state
    private var p0: Float = 0, p1: Float = 0, p2: Float = 0
    
    // Edge clicks
    private var clickEnv: Float = 0
    
    // Gain smoothing
    private var currentGains = [Float](repeating: 1, count: 40)
    private let gainSmooth: Float = 0.85
    
    // ===== DEBUG aids =====
    private let DEBUG_AUDIO = true
    
    private func shapeName(_ s: Int) -> String {
        switch s {
        case 1: return "sphere"
        case 2: return "tetra"
        case 3: return "cube"
        default: return "none"
        }
    }
    
    private var lastReportedShape: Int = -999
    private var lastMaskCount: Int = -1
    private var lastFcReported: Int = -1
    private var lastCombState: String = ""
    
    private func activeCombSummary() -> String {
        var parts: [String] = []
        if sphereActive { parts.append("sphere(M:\(sphereM), lvl:\(String(format: "%.2f", sphereLevel)))") }
        if tetraActive  { parts.append("tetra(M:\(tetraM), lvl:\(String(format: "%.2f", tetraLevel)))") }
        if cubeActive   { parts.append("cube(M:\(cubeM), lvl:\(String(format: "%.2f", cubeLevel)))") }
        if parts.isEmpty { return "none" }
        return parts.joined(separator: " | ")
    }
    
    private func reportIfChanged(context: String,
                                 shape: Int,
                                 maskCount: Int,
                                 fcHz: Float,
                                 boostDB: Float) {
        guard DEBUG_AUDIO else { return }
        let fcInt = Int(round(fcHz))
        let comb = activeCombSummary()
        if shape != lastReportedShape || maskCount != lastMaskCount || fcInt != lastFcReported || comb != lastCombState {
            print("[Audio]", context,
                  "shape=\(shapeName(shape))",
                  "maskBands=\(maskCount)",
                  "fc≈\(fcInt)Hz",
                  "boost=\(String(format: "%.1f", boostDB))dB",
                  "| active:", comb)
            lastReportedShape = shape
            lastMaskCount = maskCount
            lastFcReported = fcInt
            lastCombState = comb
        }
    }
    
    // ===== init =====
    
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
        foregroundBuf = [Float](repeating: 0, count: fftSize)
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
        reverb.wetDryMix = 0
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
    
    /// 0 (near) … 1 (far)
    func updateDistance(_ z: Float) {
        z01 = max(0, min(1, z))
        zSlew = zSlewA * zSlew + (1 - zSlewA) * z01
        DispatchQueue.main.async { [weak self] in self?.reverb.wetDryMix = 0 }
    }
    
    /// Provide a per-band 0…1 mask for the target under the scan line and pick a comb timbre.
    /// - mask: length-40, 1.0 = "inside target bands"; 0.0 = background
    /// - shape: 1=sphere, 2=tetra, 3=cube, 0=none
    /// - boostDB: how much louder than background; default +12 dB
    func setTargetBands(_ mask: [Float], shape: Int, boostDB: Float = 12.0) {
        var m = mask
        if m.count < numBands { m += Array(repeating: 0, count: numBands - m.count) }
        if m.count > numBands { m = Array(m.prefix(numBands)) }
        for i in 0..<numBands {
            targetBoostBands[i] = targetBoostSmooth * targetBoostBands[i]
            + (1 - targetBoostSmooth) * max(0, min(1, m[i]))
        }
        
        // Shape-aware target boost and level - ENHANCED DIFFERENTIATION
        var localBoostDB = boostDB
        var lvl: Float = 0.6
        switch shape {
        case 1: // sphere (warm, sustained, lower in mix to avoid masking)
            localBoostDB = boostDB
            lvl = 0.50
        case 2: // tetra (bright, metallic - push forward)
            localBoostDB = boostDB + 8.0
            lvl = 1.0
        case 3: // cube (percussive, very forward)
            localBoostDB = boostDB + 10.0
            lvl = 1.1
        default:
            localBoostDB = 0
            lvl = 0
        }
        targetBoostLin = powf(10.0, localBoostDB / 20.0)
        targetShape = shape
        
        // Center frequency from spectral centroid of active bands
        var sumW: Float = 0, sumF: Float = 0
        for b in 0..<numBands {
            let w = targetBoostBands[b]
            if w > 1e-3 {
                let f0 = bandEdges[b], f1 = bandEdges[min(b+1, numBands)]
                let fc = sqrtf(f0 * f1)
                sumF += w * fc
                sumW += w
            }
        }
        let fc = (sumW > 0) ? (sumF / sumW) : 600.0
        
        // DEBUG: how many bands are actually lit after smoothing?
        let activeBands = targetBoostBands.reduce(0) { $0 + ($1 > 0.5 ? 1 : 0) }
        reportIfChanged(context: "setTargetBands",
                        shape: shape,
                        maskCount: activeBands,
                        fcHz: fc,
                        boostDB: localBoostDB)
        
        // IMPORTANT: solo the active shape; explicitly disable the others
        switch shape {
        case 1:
            setSphere(on: true,  f0: fc, level: lvl)
            setTetra(on:  false, f0: fc, level: 0)
            setCube(on:   false, f0: fc, level: 0)
        case 2:
            setSphere(on: false, f0: fc, level: 0)
            setTetra(on:  true,  f0: fc, level: lvl)
            setCube(on:   false, f0: fc, level: 0)
        case 3:
            setSphere(on: false, f0: fc, level: 0)
            setTetra(on:  false, f0: fc, level: 0)
            setCube(on:   true,  f0: fc, level: lvl)
        default:
            setSphere(on: false, f0: fc, level: 0)
            setTetra(on:  false, f0: fc, level: 0)
            setCube(on:   false, f0: fc, level: 0)
        }
    }
    
    func clearTarget() {
        for i in 0..<numBands { targetBoostBands[i] = 0 }
        if DEBUG_AUDIO { print("[Audio] clearTarget (all shapes OFF)") }
        setSphere(on: false, f0: 600, level: 0)
        setTetra(on:  false, f0: 600, level: 0)
        setCube(on:   false, f0: 600, level: 0)
    }
    
    func triggerEdge(_ strength: Float) {
        let s = max(0, min(1, strength))
        clickEnv = max(clickEnv, s * 0.6)
    }
    
    // MARK: Shape controls - ENHANCED PARAMETERS
    
    func setTargetComb(shape: Int, on: Bool, f0: Float, level: Float) {
        let sr = Float(sampleRate)
        let lv = max(0, min(1, level)) * 0.7  // Slightly higher base level
        
        switch shape {
        case 1: // sphere → feedback comb (warm, sustained, tonal)
            if on {
                sphereTargetLevel = lv
                if !sphereActive {
                    sphereActive = true
                    sphereLevel = 0  // Start at 0, ramp up
                    // Sphere: longer delays for lower, warmer pitch
                    let desiredM = max(1, Int(round(sr / max(80, min(4000, f0)))))
                    sphereM = max(100, min(500, desiredM))  // Lower range = warmer
                    sphereComb.reinit(g: 0.94, M: sphereM)  // High feedback for sustain
                    if DEBUG_AUDIO { print("[Audio] sphere ON  M=\(sphereM) g=0.94 target_lvl=\(String(format: "%.2f", lv))") }
                } else {
                    sphereComb.g = 0.94
                }
            } else {
                sphereTargetLevel = 0
                if sphereActive && sphereLevel < 0.001 {
                    sphereActive = false
                    sphereComb.clear()
                    if DEBUG_AUDIO { print("[Audio] sphere OFF") }
                }
            }
            
        case 2: // tetra → dual feed-forward combs (bright, metallic, harsh)
            if on {
                tetraTargetLevel = lv
                if !tetraActive {
                    tetraActive = true
                    tetraLevel = 0
                    // Tetra: shorter delays for brighter, more metallic sound
                    let desiredM = max(1, Int(round(sr / max(200, min(8000, f0)))))
                    tetraM = max(40, min(160, desiredM))  // Shorter = brighter
                    tetraComb.reinit(alpha: 0.92, M: tetraM)
                    // Second comb at inharmonic ratio for metallic quality
                    tetraComb2.reinit(alpha: 0.80, M: max(30, Int(Float(tetraM) * 1.618)))
                    if DEBUG_AUDIO { print("[Audio] tetra  ON  M=\(tetraM)/\(tetraComb2.M) alpha=0.92/0.80 target_lvl=\(String(format: "%.2f", lv))") }
                } else {
                    tetraComb.alpha = 0.92
                    tetraComb2.alpha = 0.80
                }
            } else {
                tetraTargetLevel = 0
                if tetraActive && tetraLevel < 0.001 {
                    tetraActive = false
                    tetraComb.clear()
                    tetraComb2.clear()
                    if DEBUG_AUDIO { print("[Audio] tetra  OFF") }
                }
            }
            
        case 3: // cube → allpass comb + heavy dry mix (percussive, diffuse, rhythmic)
            if on {
                cubeTargetLevel = lv * 1.3  // Extra boost for cube
                if !cubeActive {
                    cubeActive = true
                    cubeLevel = 0
                    // Cube: very short for percussive transients
                    let desiredM = max(1, Int(round(sr / max(400, min(12000, f0)))))
                    cubeM = max(20, min(100, desiredM))  // Very short = percussive
                    cubeComb.reinit(a: 0.65, M: cubeM)
                    cubeMixBeta = 0.40  // More dry signal for attack
                    if DEBUG_AUDIO { print("[Audio] cube   ON  M=\(cubeM) a=0.65 mix=0.40 target_lvl=\(String(format: "%.2f", cubeTargetLevel))") }
                } else {
                    cubeComb.a = 0.65
                    cubeMixBeta = 0.40
                }
            } else {
                cubeTargetLevel = 0
                if cubeActive && cubeLevel < 0.001 {
                    cubeActive = false
                    cubeComb.clear()
                    if DEBUG_AUDIO { print("[Audio] cube   OFF") }
                }
            }
            
        default: break
        }
    }
    
    func setSphere(on: Bool, f0: Float, level: Float) { setTargetComb(shape: 1, on: on, f0: f0, level: level) }
    func setTetra(on: Bool,  f0: Float, level: Float) { setTargetComb(shape: 2, on: on, f0: f0, level: level) }
    func setCube(on: Bool,   f0: Float, level: Float) { setTargetComb(shape: 3, on: on, f0: f0, level: level) }
    
    // MARK: - DSP - IMPROVED MIXING WITH LEVEL RAMPING
    
    private func processBlock() {
        // Smooth level transitions to avoid clicks
        sphereLevel = levelRampAlpha * sphereLevel + (1 - levelRampAlpha) * sphereTargetLevel
        tetraLevel  = levelRampAlpha * tetraLevel  + (1 - levelRampAlpha) * tetraTargetLevel
        cubeLevel   = levelRampAlpha * cubeLevel   + (1 - levelRampAlpha) * cubeTargetLevel
        
        // Deactivate shapes that have ramped down
        if sphereActive && sphereTargetLevel == 0 && sphereLevel < 0.001 {
            sphereActive = false
            sphereComb.clear()
        }
        if tetraActive && tetraTargetLevel == 0 && tetraLevel < 0.001 {
            tetraActive = false
            tetraComb.clear()
            tetraComb2.clear()
        }
        if cubeActive && cubeTargetLevel == 0 && cubeLevel < 0.001 {
            cubeActive = false
            cubeComb.clear()
        }
        
        // Generate excitation (white noise)
        genWhite()
        
        // window → FFT
        vDSP_vmul(timeBlock, 1, window, 1, &winTime, 1, vDSP_Length(fftSize))
        forwardDFT.transform(inputReal: winTime,
                             inputImaginary: zeroImag,
                             outputReal: &freqReal,
                             outputImaginary: &freqImag)
        
        // Apply envelope and target emphasis
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
        
        // === Generate foreground shape signal separately ===
        let hasActiveSignal = (sphereActive && sphereLevel > 0.005) ||
                              (tetraActive && tetraLevel > 0.005) ||
                              (cubeActive && cubeLevel > 0.005)
        
        // Clear foreground buffer
        for i in 0..<fftSize { foregroundBuf[i] = 0 }
        
        if hasActiveSignal {
            for i in 0..<fftSize {
                // Independent noise source for foreground (decorrelated from background)
                let exc = Float.random(in: -1...1) * 0.6
                var add: Float = 0
                
                // SPHERE: feedback comb for warm, sustained, tonal sound
                if sphereActive && sphereLevel > 0.005 {
                    let sphereOut = sphereComb.process(exc)
                    add += sphereLevel * sphereOut
                }
                
                // TETRA: dual feed-forward combs for bright, metallic, inharmonic sound
                if tetraActive && tetraLevel > 0.005 {
                    let t1 = tetraComb.process(exc)
                    let t2 = tetraComb2.process(exc)
                    // Mix two combs for richer metallic texture
                    let tetraOut = t1 * 0.6 + t2 * 0.4
                    add += tetraLevel * tetraOut
                }
                
                // CUBE: allpass + dry for percussive, diffuse, rhythmic sound
                if cubeActive && cubeLevel > 0.005 {
                    let wet = cubeComb.process(exc)
                    let cubeOut = (1 - cubeMixBeta) * exc + cubeMixBeta * wet
                    add += cubeLevel * cubeOut
                }
                
                foregroundBuf[i] = add
            }
        }
        
        // Mix background + foreground with automatic ducking
        var fgSumSq: Float = 0
        vDSP_svesq(foregroundBuf, 1, &fgSumSq, vDSP_Length(fftSize))
        let fgRMS = sqrtf(fgSumSq / Float(fftSize))
        let hasForeground = fgRMS > 1e-4
        
        // Duck background more aggressively when shape present
        let bgDuck: Float = hasForeground ? 0.30 : 1.0
        let fgGain: Float = 0.85  // Foreground level
        
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
    
    // MARK: - FIXED applySpectralShaping
    
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
            
            // Target boost for bins in emphasized bands
            let tBoost = targetBoostBands[band]
            let boostFactor = 1.0 + (targetBoostLin - 1.0) * tBoost
            
            // Combined weight: base gain × triangular filter × boost
            let w = gAtt * tri * boostFactor
            
            // Apply to frequency domain (both real and imag)
            freqReal[i] *= w
            freqImag[i] *= w
            
            // Mirror for negative frequencies (except DC and Nyquist)
            if i > 0 && i < half {
                let mirrorIdx = fftSize - i
                freqReal[mirrorIdx] *= w
                freqImag[mirrorIdx] *= w
            }
        }
    }
}
