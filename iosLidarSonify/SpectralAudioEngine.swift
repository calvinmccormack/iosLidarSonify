import Foundation
import AVFoundation
import Accelerate
import Combine

final class SpectralAudioEngine: ObservableObject {
    
    // UI controls
    @Published var pan: Float = 0
    @Published var enableEdgeClicks: Bool = false
    @Published var outputGainDB: Float = 6.0
    
    // Constants
    private let sampleRate: Double
    private let blockSize: Int = 512
    private let hop: Int = 128
    
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
    private var targetBoostBands = [Float](repeating: 0, count: 40)
    private var targetBoostLin: Float = 1.0
    private let targetBoostSmooth: Float = 0.6
    
    // Simple comb that doesnt crash
    struct SimpleComb {
        var g: Float
        var M: Int
        var buf: [Float]
        var writeIdx = 0
        
        init(g: Float, M: Int) {
            self.g = max(0.5, min(0.95, g))
            self.M = max(50, min(M, 1000))
            self.buf = [Float](repeating: 0, count: self.M)
        }
        
        @inline(__always)
        mutating func tick(_ x: Float, delay: Float) -> Float {
            let d = max(10.0, min(delay, Float(M - 1)))
            let di = Int(d)
            let frac = d - Float(di)
            
            let ri0 = (writeIdx - di + M) % M
            let ri1 = (ri0 - 1 + M) % M
            
            let y0 = buf[ri0]
            let y1 = buf[ri1]
            let yOut = y0 + frac * (y1 - y0)
            
            let out = x + g * yOut
            buf[writeIdx] = out
            writeIdx = (writeIdx + 1) % M
            
            // Stability check
            if !out.isFinite {
                for i in 0..<M { buf[i] = 0 }
                writeIdx = 0
                return 0
            }
            
            return out
        }
        
        mutating func clear() {
            for i in 0..<M { buf[i] = 0 }
            writeIdx = 0
        }
    }
    
    // One comb per shape
    private var sphereComb: SimpleComb!
    private var tetraComb: SimpleComb!
    private var cubeComb: SimpleComb!
    
    // Frequency ranges (1 octave each)
    private let sphereFreqLow: Float = 60.0
    private let sphereFreqHigh: Float = 120.0
    private let tetraFreqLow: Float = 150.0
    private let tetraFreqHigh: Float = 300.0
    private let cubeFreqLow: Float = 250.0
    private let cubeFreqHigh: Float = 500.0
    
    // Shape state
    private var sphereActive = false
    private var tetraActive  = false
    private var cubeActive   = false
    
    private var sphereLevel: Float = 0.0
    private var tetraLevel:  Float = 0.0
    private var cubeLevel:   Float = 0.0
    
    private var sphereTargetLevel: Float = 0.0
    private var tetraTargetLevel:  Float = 0.0
    private var cubeTargetLevel:   Float = 0.0
    
    private let levelRampAlpha: Float = 0.93
    
    // Y-axis (centroid)
    private var currentCentroid: Float = 0.5
    private var smoothedCentroid: Float = 0.5
    private let centroidSmooth: Float = 0.90
    
    // OLA
    private var olaL: [Float]
    private var olaR: [Float]
    private var olaWrite = 0
    private var olaRead  = 0
    
    // Band mapping
    private let numBands = 40
    private var bandEdges = [Float]()
    private var binToBand = [Int]()
    private var currentGains = [Float](repeating: 1, count: 40)
    private let gainSmooth: Float = 0.85
    
    // Distance
    private var z01: Float = 0
    
    // Edge clicks
    private var clickEnv: Float = 0
    
    // Large pre-generated noise buffer with pseudo-random access
    // Uses prime number indexing to avoid repetitive patterns, solved problem of random num generators crushing cpu
    private var noiseBuffer = [Float]()
    private var noiseIdx = 0
    private let noisePrimeStep = 7919  // Prime number for pseudo-random access
    
    init() {
        let session = AVAudioSession.sharedInstance()
        self.sampleRate = session.sampleRate
        
        // comb setup
        let sr = Float(sampleRate)
        sphereComb = SimpleComb(g: 0.92, M: Int(sr / 60) + 50)
        tetraComb = SimpleComb(g: 0.88, M: Int(sr / 150) + 50)
        cubeComb = SimpleComb(g: 0.80, M: Int(sr / 250) + 50)
        
        // Window
        window = [Float](repeating: 0, count: blockSize)
        vDSP_hann_window(&window, vDSP_Length(blockSize), Int32(vDSP_HANN_NORM))
        var sum: Float = 0
        vDSP_sve(window, 1, &sum, vDSP_Length(blockSize))
        invWindowEnergy = sum > 0 ? 1.0 / sum : 1.0
        
        // Buffers
        timeBlock = [Float](repeating: 0, count: blockSize)
        winTime   = [Float](repeating: 0, count: blockSize)
        freqReal  = [Float](repeating: 0, count: blockSize)
        freqImag  = [Float](repeating: 0, count: blockSize)
        ifftReal  = [Float](repeating: 0, count: blockSize)
        zeroImag  = [Float](repeating: 0, count: blockSize)
        scratchImagTime = [Float](repeating: 0, count: blockSize)
        
        forwardDFT = vDSP.DFT(previous: nil, count: blockSize, direction: .forward, transformType: .complexReal, ofType: Float.self)!
        inverseDFT = vDSP.DFT(previous: nil, count: blockSize, direction: .inverse, transformType: .complexReal, ofType: Float.self)!
        
        // generate noise buffer
        noiseBuffer = (0..<65536).map { _ in Float.random(in: -0.35...0.35) }
        
        // OLA
        let olaLen = 4 * blockSize
        olaL = [Float](repeating: 0, count: olaLen)
        olaR = [Float](repeating: 0, count: olaLen)
        
        // Audio setup
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [])
        try? AVAudioSession.sharedInstance().setActive(true)
        
        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        
        srcNode = AVAudioSourceNode(format: fmt) { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self else { return noErr }
            let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
            guard let left = ablPointer[0].mData?.assumingMemoryBound(to: Float.self),
                  let right = ablPointer[1].mData?.assumingMemoryBound(to: Float.self) else {
                return noErr
            }
            var remaining = Int(frameCount)
            var offset = 0
            while remaining > 0 {
                self.synthesizeBlock()
                let chunk = min(remaining, self.hop)
                self.copyFromOLA(to: left.advanced(by: offset), outR: right.advanced(by: offset), count: chunk)
                remaining -= chunk
                offset += chunk
            }
            return noErr
        }
        
        engine.attach(srcNode)
        // Reverb removed - it was causing spatial confusion for segmented objects
        // engine.attach(reverb)
        // reverb.loadFactoryPreset(.mediumHall)
        // reverb.wetDryMix = 15
        
        // Connect directly to output
        engine.connect(srcNode, to: engine.mainMixerNode, format: fmt)
    }
    
    func start() {
        if !engine.isRunning {
            try? engine.start()
        }
    }
    
    func stop() {
        engine.stop()
    }
    
    func configureBands(fMin: Double, fMax: Double) {
        let fm = Float(fMin), fM = Float(fMax)
        bandEdges = stride(from: 0, through: numBands, by: 1).map { i -> Float in
            let r = Float(i) / Float(numBands)
            return powf(fm / fM, 1 - r) * fM
        }
        
        let half = blockSize / 2
        let nyq  = Float(sampleRate / 2)
        let binHz = nyq / Float(half)
        binToBand.removeAll(keepingCapacity: true)
        for i in 0...half {
            let f = Float(i) * binHz
            var band = 0
            for b in 0..<numBands {
                if f < bandEdges[b+1] { band = b; break }
                if b == numBands - 1 { band = numBands - 1 }
            }
            binToBand.append(band)
        }
    }
    
    func updateEnvelope(_ env: [Float]) {
        guard env.count == numBands else { return }
        for i in 0..<numBands {
            currentGains[i] = gainSmooth * currentGains[i] + (1 - gainSmooth) * env[i]
        }
    }
    
    func setTargetBands(_ mask: [Float], shape: Int, boostDB: Float) {
        guard mask.count == numBands else { return }
        
        targetBoostLin = powf(10.0, boostDB / 20.0)
        for i in 0..<numBands {
            targetBoostBands[i] = targetBoostSmooth * targetBoostBands[i] + (1 - targetBoostSmooth) * mask[i]
        }
        
        // Calculate centroid
        var weightedSum: Float = 0
        var totalWeight: Float = 0
        for i in 0..<numBands {
            let w = mask[i]
            if w > 0.01 {
                weightedSum += w * Float(i) / Float(numBands - 1)
                totalWeight += w
            }
        }
        
        if totalWeight > 0.01 {
            currentCentroid = weightedSum / totalWeight
        } else {
            currentCentroid = 0.5
        }
        
        // Set targets
        switch shape {
        case 1:
            sphereTargetLevel = 1.0
            tetraTargetLevel = 0.0
            cubeTargetLevel = 0.0
            if !sphereActive { sphereActive = true }
        case 2:
            sphereTargetLevel = 0.0
            tetraTargetLevel = 1.0
            cubeTargetLevel = 0.0
            if !tetraActive { tetraActive = true }
        case 3:
            sphereTargetLevel = 0.0
            tetraTargetLevel = 0.0
            cubeTargetLevel = 1.0
            if !cubeActive { cubeActive = true }
        default:
            sphereTargetLevel = 0.0
            tetraTargetLevel = 0.0
            cubeTargetLevel = 0.0
        }
    }
    
    func clearTarget() {
        for i in 0..<numBands {
            targetBoostBands[i] *= targetBoostSmooth
        }
        sphereTargetLevel = 0.0
        tetraTargetLevel = 0.0
        cubeTargetLevel = 0.0
        currentCentroid = 0.5
    }
    
    func updateDistance(_ z: Float) {
        z01 = max(0, min(1, z))
    }
    
    func triggerEdge(_ e: Float) {
        clickEnv = max(clickEnv, e)
    }
    
    private func synthesizeBlock() {
        // Smooth levels
        sphereLevel = levelRampAlpha * sphereLevel + (1 - levelRampAlpha) * sphereTargetLevel
        tetraLevel  = levelRampAlpha * tetraLevel  + (1 - levelRampAlpha) * tetraTargetLevel
        cubeLevel   = levelRampAlpha * cubeLevel   + (1 - levelRampAlpha) * cubeTargetLevel
        
        // Smooth centroid
        smoothedCentroid = centroidSmooth * smoothedCentroid + (1 - centroidSmooth) * currentCentroid
        
        // Deactivate
        if sphereActive && sphereTargetLevel == 0 && sphereLevel < 0.001 {
            sphereActive = false
            sphereComb.clear()
        }
        if tetraActive && tetraTargetLevel == 0 && tetraLevel < 0.001 {
            tetraActive = false
            tetraComb.clear()
        }
        if cubeActive && cubeTargetLevel == 0 && cubeLevel < 0.001 {
            cubeActive = false
            cubeComb.clear()
        }
        
        // Generate noise using prime-stepped indexing for organic variation
        for i in 0..<blockSize {
            timeBlock[i] = noiseBuffer[noiseIdx]
            noiseIdx = (noiseIdx + noisePrimeStep) % noiseBuffer.count
        }
        
        // FFT
        vDSP_vmul(timeBlock, 1, window, 1, &winTime, 1, vDSP_Length(blockSize))
        forwardDFT.transform(inputReal: winTime, inputImaginary: zeroImag,
                             outputReal: &freqReal, outputImaginary: &freqImag)
        
        // Spectral shaping
        applySpectralShaping()
        
        // iFFT
        inverseDFT.transform(inputReal: freqReal, inputImaginary: freqImag,
                             outputReal: &ifftReal, outputImaginary: &scratchImagTime)
        
        var scale = 1.0 / Float(blockSize)
        vDSP_vsmul(ifftReal, 1, &scale, &ifftReal, 1, vDSP_Length(blockSize))
        var gain = invWindowEnergy * 4
        vDSP_vsmul(ifftReal, 1, &gain, &ifftReal, 1, vDSP_Length(blockSize))
        
        // === Foreground shapes ===
        let hasShape = (sphereActive && sphereLevel > 0.01) ||
                       (tetraActive && tetraLevel > 0.01) ||
                       (cubeActive && cubeLevel > 0.01)
        
        if hasShape {
            let sr = Float(sampleRate)
            
            // Calculate delays from centroid
            let sFreq = sphereFreqLow + smoothedCentroid * (sphereFreqHigh - sphereFreqLow)
            let tFreq = tetraFreqLow + smoothedCentroid * (tetraFreqHigh - tetraFreqLow)
            let cFreq = cubeFreqLow + smoothedCentroid * (cubeFreqHigh - cubeFreqLow)
            
            let sDelay = sr / sFreq
            let tDelay = sr / tFreq
            let cDelay = sr / cFreq
            
            let excScale = 0.25 + smoothedCentroid * 0.25
            
            for i in 0..<blockSize {
                
                let excIdx = (noiseIdx + i * noisePrimeStep) % noiseBuffer.count
                let exc = noiseBuffer[excIdx] * excScale * 2.85
                var add: Float = 0
                
                if sphereActive && sphereLevel > 0.01 {
                    let out = sphereComb.tick(exc * 0.5, delay: sDelay)
                    add += sphereLevel * out * 0.6
                }
                
                if tetraActive && tetraLevel > 0.01 {
                    let out = tetraComb.tick(exc * 0.55, delay: tDelay)
                    add += tetraLevel * out * 0.7
                }
                
                if cubeActive && cubeLevel > 0.01 {
                    let out = cubeComb.tick(exc * 0.6, delay: cDelay)
                    add += cubeLevel * out * 0.8
                }
                
                ifftReal[i] = ifftReal[i] * 0.35 + add
            }
        }
        
        // Edge click
        if enableEdgeClicks && clickEnv > 0.001 {
            for i in 0..<min(16, blockSize) {
                ifftReal[i] += clickEnv * 0.15 * expf(-Float(i) / 4.0)
            }
            clickEnv *= 0.3
        } else {
            clickEnv *= 0.1
        }
        
        // AGC
        var rms: Float = 0
        vDSP_rmsqv(ifftReal, 1, &rms, vDSP_Length(blockSize))
        if rms > 1e-6 {
            var target = agcTargetRMS / rms
            target = max(0.3, min(3.0, target))
            agcGain = agcAlpha * agcGain + (1 - agcAlpha) * target
            vDSP_vsmul(ifftReal, 1, &agcGain, &ifftReal, 1, vDSP_Length(blockSize))
        }
        
        // Makeup
        var makeup = powf(10.0, outputGainDB / 20.0)
        vDSP_vsmul(ifftReal, 1, &makeup, &ifftReal, 1, vDSP_Length(blockSize))
        
        // Limiter
        var peak: Float = 0
        vDSP_maxmgv(ifftReal, 1, &peak, vDSP_Length(blockSize))
        if peak > 0.95 {
            var s = 0.95 / peak
            vDSP_vsmul(ifftReal, 1, &s, &ifftReal, 1, vDSP_Length(blockSize))
        }
        
        // Pan (negated so left is left, right is right)
        let theta = (-pan + 1) * Float.pi * 0.25
        let gL = sin(theta)
        let gR = cos(theta)
        
        // OLA
        for i in 0..<blockSize {
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
    
    private func applySpectralShaping() {
        guard bandEdges.count == numBands + 1, !binToBand.isEmpty else { return }
        let half = blockSize / 2
        let nyq  = Float(sampleRate / 2)
        let binHz = nyq / Float(half)
        
        for i in 0...half {
            let band = max(0, min(binToBand[i], numBands - 1))
            let gBand = currentGains[band]
            
            let f = Float(i) * binHz
            let f0 = bandEdges[band]
            let f1 = bandEdges[min(band + 1, numBands)]
            let fcBand = sqrtf(max(10, f0) * max(10, f1))
            let bw = max(60, f1 - f0)
            let tri = max(0, 1 - abs(f - fcBand) / (bw * 0.5))
            
            let gAtt = min(1.0, max(0.05, gBand))
            
            // Target boost for magical emphasis
            // sorry, magical sounding noise was cut due to cpu constraints
            let tBoost = targetBoostBands[band]
            let boostFactor = 1.0 + (targetBoostLin - 1.0) * tBoost
            
            let w = gAtt * tri * boostFactor
            
            freqReal[i] *= w
            freqImag[i] *= w
            
            if i > 0 && i < half {
                let mirrorIdx = blockSize - i
                freqReal[mirrorIdx] *= w
                freqImag[mirrorIdx] *= w
            }
        }
    }
}
