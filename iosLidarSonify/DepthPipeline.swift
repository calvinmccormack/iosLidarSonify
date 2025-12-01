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
    @Published var segOverlay: UIImage? = nil
    @Published var fps: Double = 0
    @Published var scanColumn: Int = 0
    @Published var classHistogramText: String = "classGrid histogram: []"

    // Mapping
    var nearMeters: Float = 0.30
    var farMeters: Float = 4.00
    var gainRangeDB: Float = 24 // 0 dB to -gainRangeDB

    // AR
    private let session = ARSession()

    // Downsampled buffer (60x40)
    private var grid = [Float](repeating: 0, count: gridWidth * gridHeight)
    private let gridLock = NSLock()

    // Segmentation: downsampled class grid aligned with depth grid
    private let segmentationModel = ObjectSegmentation()
    private var lastSegmentationTime: TimeInterval = 0
    private let segmentationInterval: TimeInterval = 0.15 // seconds, ~6–7 Hz
    private var isSegmentationRunning: Bool = false

    // Class IDs per 60x40 cell (from segmentation)
    private var classGrid = [UInt8](repeating: 0, count: gridWidth * gridHeight)

    private let rotateSegMask90CW: Bool = true
    // Mirror the segmentation grid writeout so classGrid aligns with camera view.
    // This affects BOTH the debug overlay and all downstream logic (sweep, target coverage).
    private let mirrorSegX: Bool = true   // horizontal mirror
    private let mirrorSegY: Bool = true   // vertical mirror

    // Leave overlay flips off to avoid double mirroring in the rendered image.
    private let overlayFlipX: Bool = false
    private let overlayFlipY: Bool = false

    // Target class for object-aware sonification (placeholder)
    var targetClass: UInt8 = 1

    // Sweep
    private var displayLink: CADisplayLink?
    private var sweepTimer: Timer?
    private var sweepSeconds: Double = 2.0
    private var sweepStart = Date()

    // Callback: column envelope + target mask + shape id + pan + z (0 near…1 far) + edge strength (0…1)
    private var onColumn: ((Int, [Float], [Float], Int, Float, Float, Float) -> Void)?

    // FPS measure
    private var lastTime = Date()
    private var frameCount = 0

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

        // Use display link to drive sweep independent of AR frame rate
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

        // RIGHT → LEFT across columns
        let maxCol = Double(Self.gridWidth - 1)
        let col = (Self.gridWidth - 1) - Int(round(phase * maxCol))
        scanColumn = col

        let env = columnEnvelope(col: col)
        let (targetMask, shapeId) = columnTargetMaskAndShape(col: col)

        // Pan: +1 at right → -1 at left to match visual
        let norm = Double(col) / maxCol       // 1 → 0 over sweep
        let pan = Float(norm * 2 - 1)         // +1 → -1

        // Distance & edge strength for this (mirrored) column
        var z01 = columnZ01(col: col)
        let edge01 = columnEdge01(col: col)

        // Target coverage in this column (0..1 of vertical cells belonging to targetClass)
        let targetCov = columnTargetCoverage(col: col)
        print("scan col \(col), targetCov = \(targetCov)")

        // Optional debug print to show when a target is active in this column
        if shapeId != 0 {
            print("target active col \(col): shape \(shapeId), bands set: \(targetMask.enumerated().filter{ $0.element > 0 }.map{ $0.offset })")
        }

        // Simple target-based boost: when target coverage is high, pull z01 slightly "closer"
        // so the target feels more foreground in the sonification.
        let boost: Float = targetCov * 0.3
        z01 = clamp01(z01 - boost)

        onColumn?(col, env, targetMask, Int(shapeId), pan, z01, edge01)

        if let img = debugImageFromGrid() { debugImage = img }
    }
    /// Build a per-band (40) mask for targetClass coverage in this column,
    /// and return the dominant non-background class id seen in the column (0 if none).
    private func columnTargetMaskAndShape(col: Int, bands: Int = DepthPipeline.gridHeight) -> ([Float], UInt8) {
        let W = Self.gridWidth
        let H = Self.gridHeight
        var mask = [Float](repeating: 0, count: bands)
        var hist = [Int](repeating: 0, count: 4)

        gridLock.lock(); defer { gridLock.unlock() }

        for y in 0..<H {
            let cls = Int(classGrid[y * W + col])
            if cls >= 0 && cls < hist.count { hist[cls] += 1 }
            if classGrid[y * W + col] == targetClass {
                // map image row y to band index rBand where 0 = bottom
                let rBand = (H - 1 - y)
                if rBand >= 0 && rBand < bands { mask[rBand] = 1.0 }
            }
        }

        // Choose dominant non-background class id
        var shapeId: UInt8 = 0
        var bestCount = 0
        for c in 1..<hist.count {
            if hist[c] > bestCount {
                bestCount = hist[c]
                shapeId = UInt8(c)
            }
        }

        // If there is effectively no target coverage, zero the mask so audio can clear target
        if !mask.contains(where: { $0 > 0 }) { shapeId = 0 }

        return (mask, shapeId)
    }


    // MARK: ARSessionDelegate

    private func handleFrame(_ frame: ARFrame) {
        guard let depthPB = frame.sceneDepth?.depthMap else { return }
        updateGrid(from: depthPB)

        // Run segmentation at a lower rate than AR frame rate
        let nowTime = frame.timestamp
        if nowTime - lastSegmentationTime > segmentationInterval {
            lastSegmentationTime = nowTime
            runSegmentation(on: frame)
        }

        // FPS estimate from AR callback
        frameCount += 1
        let now = Date()
        let dt = now.timeIntervalSince(lastTime)
        if dt > 0.5 {
            fps = Double(frameCount) / dt
            frameCount = 0
            lastTime = now
        }

        // Update published overlay from the current debug grid render
        if let img = debugImageFromGrid() {
            DispatchQueue.main.async {
                self.segOverlay = img
                self.debugImage = img
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        handleFrame(frame)
    }

    /// Public entry for external ARSession delegates to forward frames here.
    func process(frame: ARFrame) {
        handleFrame(frame)
    }

    // MARK: - Segmentation helpers

    private func runSegmentation(on frame: ARFrame) {
        guard !isSegmentationRunning else { return }
        isSegmentationRunning = true
        segmentationModel?.predictMask(from: frame) { [weak self] mask, width, height in
            guard let self = self else { return }
            defer { self.isSegmentationRunning = false }

            guard let mask = mask,
                  width > 0, height > 0 else { return }

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
                    // --- Downsample a 90°-CW-rotated view of the mask (do not rotate the final image) ---
                    // Rotated view has width H and height W.
                    let rx0 = gx * H / Self.gridWidth
                    let rx1 = min((gx + 1) * H / Self.gridWidth, H)
                    let ry0 = gy * W / Self.gridHeight
                    let ry1 = min((gy + 1) * W / Self.gridHeight, W)

                    var ry = ry0
                    while ry < ry1 {
                        var rx = rx0
                        while rx < rx1 {
                            // pure 90° CW rotation (no extra horizontal flip)
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
                    // --- No rotation: original orientation ---
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
                // Write into possibly mirrored destination cell so the 60x40 class grid matches the camera view.
                let dstX = mirrorSegX ? (Self.gridWidth  - 1 - gx) : gx
                let dstY = mirrorSegY ? (Self.gridHeight - 1 - gy) : gy
                newGrid[dstY * Self.gridWidth + dstX] = UInt8(bestClass)
            }
        }

        gridLock.lock()
        classGrid = newGrid
        gridLock.unlock()

        // DEBUG: class distribution in the 60x40 grid (4 classes)
        var hist = [Int](repeating: 0, count: 4)
        for v in newGrid {
            let idx = Int(v)
            if idx >= 0 && idx < hist.count {
                hist[idx] += 1
            }
        }
        print("classGrid histogram (bg, sphere, tetra, cube):", hist)
        
        // Update published text for UI
        DispatchQueue.main.async {
            self.classHistogramText = "bg: \(hist[0])  sp: \(hist[1])  te: \(hist[2])  cu: \(hist[3])"
        }
    }

    // Downsample depth to 60x40 by average pooling per cell
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

    // Build 40-band envelope (bottom row → lowest freq). Depth → gain mapping.
    func columnEnvelope(col: Int) -> [Float] {
        gridLock.lock(); defer { gridLock.unlock() }
        var env = [Float](repeating: 1, count: Self.gridHeight)
        for r in 0..<Self.gridHeight {
            // flip vertically so index 0 = bottom
            let gy = (Self.gridHeight - 1 - r)
            let d = grid[gy * Self.gridWidth + col]
            // Normalize depth: near→1, far→0 (invert because nearer = louder)
            let t = clamp01(1 - (d - nearMeters) / max(0.001, (farMeters - nearMeters)))
            // Map to linear gain via dB range (0 dB down to -gainRangeDB)
            let gDB = (0.0 as Float) - gainRangeDB * (1 - t)
            env[r] = pow(10.0, gDB / 20.0)
        }
        // Optional smoothing across bands to reduce combing
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
            if classGrid[idx] == targetClass {
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
            let t = clamp01((d - nearMeters) / range) // 0 near → 1 far
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
        return clamp01(norm * 4) // amplify a bit
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

    private func debugImageFromGrid() -> UIImage? {
        // Render 60x40: grayscale depth, with non-background segmentation cells tinted blue
        let W = Self.gridWidth
        let H = Self.gridHeight

        // Build RGBA buffer
        var rgba = [UInt8](repeating: 0, count: W * H * 4)

        gridLock.lock()
        let range = max(0.001, (farMeters - nearMeters))
        for y in 0..<H {
            for x in 0..<W {
                let idx = y * W + x
                let d = grid[idx]
                // Normalize depth: near→1, far→0 (closer=brighter)
                let t = clamp01(1 - (d - nearMeters) / range)
                let gray = UInt8(t * 255)

                let cls = classGrid[idx]
                let base = idx * 4

                if cls == 0 {
                    // Background: grayscale
                    rgba[base + 0] = gray       // R
                    rgba[base + 1] = gray       // G
                    rgba[base + 2] = gray       // B
                    rgba[base + 3] = 255        // A
                } else {
                    // Foreground classes with color coding
                    // sphere=1 → red, tetra=2 → green, cube=3 → blue
                    let overlayAlpha: Float = 0.55
                    let grayF = Float(gray)
                    var r: Float = grayF, g: Float = grayF, b: Float = grayF
                    switch cls {
                    case 1: // sphere → red
                        r = max(grayF, 200)
                        g = grayF * (1.0 - overlayAlpha)
                        b = grayF * (1.0 - overlayAlpha)
                    case 2: // tetra → green
                        r = grayF * (1.0 - overlayAlpha)
                        g = max(grayF, 200)
                        b = grayF * (1.0 - overlayAlpha)
                    case 3: // cube → blue
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

        // Apply requested flips to the overlay buffer so it aligns with the UI camera view
        if overlayFlipX || overlayFlipY {
            var flipped = [UInt8](repeating: 0, count: rgba.count)
            for y in 0..<H {
                for x in 0..<W {
                    let sx = overlayFlipX ? (W - 1 - x) : x
                    let sy = overlayFlipY ? (H - 1 - y) : y
                    let sIdx = (sy * W + sx) * 4
                    let dIdx = (y * W + x) * 4
                    flipped[dIdx + 0] = rgba[sIdx + 0] // R
                    flipped[dIdx + 1] = rgba[sIdx + 1] // G
                    flipped[dIdx + 2] = rgba[sIdx + 2] // B
                    flipped[dIdx + 3] = rgba[sIdx + 3] // A
                }
            }
            rgba = flipped
        }

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
            // No longer rotate overlay when displayed
            return UIImage(cgImage: cg, scale: UIScreen.main.scale, orientation: .up)
        }
        return nil
    }
}

@inline(__always) private func clamp01(_ x: Float) -> Float { max(0, min(1, x)) }
