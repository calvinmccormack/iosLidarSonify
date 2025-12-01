import Foundation
import ARKit
import UIKit
import Accelerate
import Combine
import Vision
import CoreML

final class DepthPipeline: NSObject, ObservableObject, ARSessionDelegate {
    static let gridWidth = 60
    static let gridHeight = 40

    // Public outputs
    @Published var debugImage: UIImage?
    @Published var fps: Double = 0
    @Published var scanColumn: Int = 0
    @Published var classHistogramText: String = "classGrid histogram: []"

    // Mapping
    var nearMeters: Float = 0.30
    var farMeters: Float = 4.00
    var gainRangeDB: Float = 24

    // AR
    private let session = ARSession()

    // Downsampled depth buffer (60x40)
    private var grid = [Float](repeating: 0, count: gridWidth * gridHeight)
    private let gridLock = NSLock()

    // Segmentation
    private let segmentationModel = ObjectSegmentation()
    private var lastSegmentationTime: TimeInterval = 0
    private let segmentationInterval: TimeInterval = 0.15
    private var isSegmentationRunning: Bool = false

    // Class IDs per 60x40 cell
    private var classGrid = [UInt8](repeating: 0, count: gridWidth * gridHeight)
    
    // Track classGrid version for debugging
    private var classGridVersion: UInt64 = 0

    private let rotateSegMask90CW: Bool = true
    private let mirrorSegX: Bool = true
    private let mirrorSegY: Bool = true

    var targetClass: UInt8 = 1

    // Sweep
    private var displayLink: CADisplayLink?
    private var sweepSeconds: Double = 2.0
    private var sweepStart = Date()

    private var onColumn: ((Int, [Float], [Float], Int, Float, Float, Float) -> Void)?

    // FPS
    private var lastTime = Date()
    private var frameCount = 0
    
    // Throttling
    private var lastUIUpdateTime: TimeInterval = 0
    private let uiUpdateInterval: TimeInterval = 0.1
    private var lastDebugPrintTime: TimeInterval = 0
    private let debugPrintInterval: TimeInterval = 1.0

    func attach(to container: UIView) {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics.insert(.sceneDepth)
        config.environmentTexturing = .none
        session.delegate = self
        session.run(config)
    }

    func start(sweepSeconds: Double, onColumn: @escaping (Int, [Float], [Float], Int, Float, Float, Float) -> Void) {
        self.onColumn = onColumn
        self.sweepSeconds = sweepSeconds
        self.sweepStart = Date()

        displayLink?.invalidate()
        let dl = CADisplayLink(target: self, selector: #selector(step))
        dl.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        dl.add(to: .main, forMode: .common)
        displayLink = dl
    }

    func stop() {
        displayLink?.invalidate(); displayLink = nil
        onColumn = nil
    }

    @objc private func step() {
        let t = -sweepStart.timeIntervalSinceNow
        let period = max(0.2, sweepSeconds)
        let phase = (t.truncatingRemainder(dividingBy: period) + period)
                     .truncatingRemainder(dividingBy: period) / period

        // FIX: LEFT → RIGHT scan (column 0 to 59)
        let maxCol = Double(Self.gridWidth - 1)
        let col = Int(round(phase * maxCol))  // 0 → 59 as phase goes 0 → 1
        scanColumn = col

        // Get depth envelope for current column (real-time)
        let env = columnEnvelope(col: col)
        
        // Get object mask for current column from classGrid
        let (targetMask, shapeId) = columnTargetMaskAndShape(col: col)

        // Pan: LEFT=-1, RIGHT=+1 (matches visual: left side of screen = left ear)
        let norm = Double(col) / maxCol  // 0 → 1 as we scan left → right
        let pan = Float(norm * 2 - 1)    // -1 → +1

        // Depth-based parameters from real-time data
        var z01 = columnZ01(col: col)
        let edge01 = columnEdge01(col: col)

        // Target coverage boost
        let targetCov = columnTargetCoverage(col: col)
        let boost: Float = targetCov * 0.3
        z01 = clamp01(z01 - boost)

        // Send to audio
        onColumn?(col, env, targetMask, Int(shapeId), pan, z01, edge01)

        // Throttled debug
        let now = CACurrentMediaTime()
        if now - lastDebugPrintTime > debugPrintInterval {
            lastDebugPrintTime = now
            print("sweep col=\(col), shape=\(shapeId), gridVer=\(classGridVersion)")
        }

        // Throttled single UI update
        if now - lastUIUpdateTime > uiUpdateInterval {
            lastUIUpdateTime = now
            if let img = debugImageFromGrid(highlightCol: col) {
                DispatchQueue.main.async {
                    self.debugImage = img
                }
            }
        }
    }
    
    private func columnTargetMaskAndShape(col: Int, bands: Int = DepthPipeline.gridHeight) -> ([Float], UInt8) {
        let W = Self.gridWidth
        let H = Self.gridHeight
        var mask = [Float](repeating: 0, count: bands)
        var hist = [Int](repeating: 0, count: 4)

        gridLock.lock(); defer { gridLock.unlock() }

        for y in 0..<H {
            let cls = Int(classGrid[y * W + col])
            if cls >= 0 && cls < hist.count { hist[cls] += 1 }
        }

        var shapeId: UInt8 = 0
        var bestCount = 0
        for c in 1..<hist.count {
            if hist[c] > bestCount {
                bestCount = hist[c]
                shapeId = UInt8(c)
            }
        }
        
        if shapeId != 0 {
            for y in 0..<H {
                if classGrid[y * W + col] == shapeId {
                    let rBand = (H - 1 - y)
                    if rBand >= 0 && rBand < bands { mask[rBand] = 1.0 }
                }
            }
        }

        return (mask, shapeId)
    }
    
    // MARK: ARSessionDelegate

    private func handleFrame(_ frame: ARFrame) {
        guard let depthPB = frame.sceneDepth?.depthMap else { return }
        updateGrid(from: depthPB)

        let nowTime = frame.timestamp
        if nowTime - lastSegmentationTime > segmentationInterval && !isSegmentationRunning {
            lastSegmentationTime = nowTime
            runSegmentation(on: frame)
        }

        frameCount += 1
        let now = Date()
        let dt = now.timeIntervalSince(lastTime)
        if dt > 0.5 {
            fps = Double(frameCount) / dt
            frameCount = 0
            lastTime = now
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        handleFrame(frame)
    }

    func process(frame: ARFrame) {
        handleFrame(frame)
    }

    // MARK: - Segmentation

    private func runSegmentation(on frame: ARFrame) {
        guard !isSegmentationRunning else { return }
        isSegmentationRunning = true
        
        segmentationModel?.predictMask(from: frame) { [weak self] mask, width, height in
            guard let self = self else { return }
            defer { self.isSegmentationRunning = false }

            guard let mask = mask, width > 0, height > 0 else { return }
            self.updateClassGrid(from: mask, maskWidth: width, maskHeight: height)
        }
    }

    private func updateClassGrid(from mask: [UInt8], maskWidth W: Int, maskHeight H: Int) {
        guard mask.count == W * H else { return }

        var newGrid = [UInt8](repeating: 0, count: Self.gridWidth * Self.gridHeight)

        for gy in 0..<Self.gridHeight {
            for gx in 0..<Self.gridWidth {

                var counts = [Int](repeating: 0, count: 4)

                if rotateSegMask90CW {
                    let rx0 = gx * H / Self.gridWidth
                    let rx1 = min((gx + 1) * H / Self.gridWidth, H)
                    let ry0 = gy * W / Self.gridHeight
                    let ry1 = min((gy + 1) * W / Self.gridHeight, W)

                    var ry = ry0
                    while ry < ry1 {
                        var rx = rx0
                        while rx < rx1 {
                            let xOrig = ry
                            let yOrig = (H - 1 - rx)
                            let idx = yOrig * W + xOrig
                            let cls = Int(mask[idx])
                            if cls >= 0 && cls < counts.count {
                                counts[cls] += 1
                            }
                            rx += 1
                        }
                        ry += 1
                    }
                } else {
                    let x0 = gx * W / Self.gridWidth
                    let x1 = min((gx + 1) * W / Self.gridWidth, W)
                    let y0 = gy * H / Self.gridHeight
                    let y1 = min((gy + 1) * H / Self.gridHeight, H)

                    var yy = y0
                    while yy < y1 {
                        var xx = x0
                        while xx < x1 {
                            let cls = Int(mask[yy * W + xx])
                            if cls >= 0 && cls < counts.count {
                                counts[cls] += 1
                            }
                            xx += 1
                        }
                        yy += 1
                    }
                }

                let (bestClass, _) = counts.enumerated().max(by: { $0.element < $1.element }) ?? (0, 0)
                let dstX = mirrorSegX ? (Self.gridWidth  - 1 - gx) : gx
                let dstY = mirrorSegY ? (Self.gridHeight - 1 - gy) : gy
                newGrid[dstY * Self.gridWidth + dstX] = UInt8(bestClass)
            }
        }

        gridLock.lock()
        classGrid = newGrid
        classGridVersion += 1
        gridLock.unlock()

        var hist = [Int](repeating: 0, count: 4)
        for v in newGrid {
            let idx = Int(v)
            if idx >= 0 && idx < hist.count {
                hist[idx] += 1
            }
        }
        DispatchQueue.main.async {
            self.classHistogramText = "bg: \(hist[0])  sp: \(hist[1])  te: \(hist[2])  cu: \(hist[3])"
        }
    }

    private func updateGrid(from pb: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: Float32.self) else { return }

        gridLock.lock(); defer { gridLock.unlock() }

        for gy in 0..<Self.gridHeight {
            let y0 = gy * h / Self.gridHeight
            let y1 = min((gy + 1) * h / Self.gridHeight, h)
            for gx in 0..<Self.gridWidth {
                let x0 = gx * w / Self.gridWidth
                let x1 = min((gx + 1) * w / Self.gridWidth, w)

                var acc: Float = 0
                var n = 0
                var yy = y0
                while yy < y1 {
                    var xx = x0
                    let row = yy * w
                    while xx < x1 {
                        acc += base[row + xx]
                        n += 1
                        xx += 1
                    }
                    yy += 1
                }
                let mean = n > 0 ? acc / Float(n) : 0
                grid[gy * Self.gridWidth + gx] = mean
            }
        }
    }

    func columnEnvelope(col: Int) -> [Float] {
        gridLock.lock(); defer { gridLock.unlock() }
        var env = [Float](repeating: 1, count: Self.gridHeight)
        for r in 0..<Self.gridHeight {
            let gy = (Self.gridHeight - 1 - r)
            let d = grid[gy * Self.gridWidth + col]
            let t = clamp01(1 - (d - nearMeters) / max(0.001, (farMeters - nearMeters)))
            let gDB = (0.0 as Float) - gainRangeDB * (1 - t)
            env[r] = pow(10.0, gDB / 20.0)
        }
        smoothInPlace(&env, a: 0.6)
        return env
    }

    private func columnTargetCoverage(col: Int) -> Float {
        let W = Self.gridWidth
        let H = Self.gridHeight

        gridLock.lock(); defer { gridLock.unlock() }

        var countTarget = 0
        for y in 0..<H {
            let idx = y * W + col
            if classGrid[idx] != 0 {
                countTarget += 1
            }
        }
        return Float(countTarget) / Float(H)
    }

    private func columnZ01(col: Int) -> Float {
        let W = Self.gridWidth
        let H = Self.gridHeight
        gridLock.lock(); defer { gridLock.unlock() }
        var acc: Float = 0
        let range = max(0.001, (farMeters - nearMeters))
        for y in 0..<H {
            let d = grid[y * W + col]
            let t = clamp01((d - nearMeters) / range)
            acc += t
        }
        return acc / Float(H)
    }

    private func columnEdge01(col: Int) -> Float {
        let W = Self.gridWidth
        let H = Self.gridHeight
        let c1 = min(W - 1, col + 1)
        gridLock.lock(); defer { gridLock.unlock() }
        var acc: Float = 0
        for y in 0..<H {
            let a = grid[y * W + col]
            let b = grid[y * W + c1]
            acc += abs(b - a)
        }
        let mean = acc / Float(H)
        let norm = mean / max(0.001, (farMeters - nearMeters))
        return clamp01(norm * 4)
    }

    private func smoothInPlace(_ x: inout [Float], a: Float) {
        guard x.count > 1 else { return }
        var prev = x[0]
        for i in 1..<x.count {
            prev = a * prev + (1 - a) * x[i]
            x[i] = prev
        }
        prev = x.last ?? 0
        for i in (0..<(x.count-1)).reversed() {
            prev = a * prev + (1 - a) * x[i]
            x[i] = prev
        }
    }

    /// Render single debug image with depth, segmentation overlay, AND scan line
    private func debugImageFromGrid(highlightCol: Int) -> UIImage? {
        let W = Self.gridWidth
        let H = Self.gridHeight

        var rgba = [UInt8](repeating: 0, count: W * H * 4)

        gridLock.lock()
        let range = max(0.001, (farMeters - nearMeters))
        for y in 0..<H {
            for x in 0..<W {
                let idx = y * W + x
                let d = grid[idx]
                let t = clamp01(1 - (d - nearMeters) / range)
                let gray = UInt8(t * 255)

                let cls = classGrid[idx]
                let base = idx * 4

                // Check if this is the scan line column
                let isScanLine = (x == highlightCol)

                if isScanLine {
                    // Draw scan line in red
                    rgba[base + 0] = 255  // R
                    rgba[base + 1] = 0    // G
                    rgba[base + 2] = 0    // B
                    rgba[base + 3] = 255  // A
                } else if cls == 0 {
                    rgba[base + 0] = gray
                    rgba[base + 1] = gray
                    rgba[base + 2] = gray
                    rgba[base + 3] = 255
                } else {
                    let overlayAlpha: Float = 0.55
                    let grayF = Float(gray)
                    var r: Float = grayF, g: Float = grayF, b: Float = grayF
                    switch cls {
                    case 1:
                        r = max(grayF, 200)
                        g = grayF * (1.0 - overlayAlpha)
                        b = grayF * (1.0 - overlayAlpha)
                    case 2:
                        r = grayF * (1.0 - overlayAlpha)
                        g = max(grayF, 200)
                        b = grayF * (1.0 - overlayAlpha)
                    case 3:
                        r = grayF * (1.0 - overlayAlpha)
                        g = grayF * (1.0 - overlayAlpha)
                        b = max(grayF, 200)
                    default:
                        break
                    }
                    rgba[base + 0] = UInt8(min(255, r))
                    rgba[base + 1] = UInt8(min(255, g))
                    rgba[base + 2] = UInt8(min(255, b))
                    rgba[base + 3] = 255
                }
            }
        }
        gridLock.unlock()

        let bytesPerRow = W * 4
        let cfData = CFDataCreate(nil, rgba, rgba.count)!
        let provider = CGDataProvider(data: cfData)!
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        if let cg = CGImage(width: W,
                            height: H,
                            bitsPerComponent: 8,
                            bitsPerPixel: 32,
                            bytesPerRow: bytesPerRow,
                            space: cs,
                            bitmapInfo: bitmapInfo,
                            provider: provider,
                            decode: nil,
                            shouldInterpolate: false,
                            intent: .defaultIntent) {
            return UIImage(cgImage: cg, scale: UIScreen.main.scale, orientation: .up)
        }
        return nil
    }
}

@inline(__always) private func clamp01(_ x: Float) -> Float { max(0, min(1, x)) }
