import Foundation
import ARKit
import UIKit
import Accelerate
import Combine

final class DepthPipeline: NSObject, ObservableObject, ARSessionDelegate {
    static let gridWidth = 60
    static let gridHeight = 40

    // Public outputs
    @Published var debugImage: UIImage?
    @Published var fps: Double = 0
    @Published var scanColumn: Int = 0

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

    // Target class for object-aware sonification (placeholder)
    var targetClass: UInt8 = 1

    // Sweep
    private var displayLink: CADisplayLink?
    private var sweepTimer: Timer?
    private var sweepSeconds: Double = 2.0
    private var sweepStart = Date()

    // Callback: column envelope + pan + z (0 near…1 far) + edge strength (0…1)
    private var onColumn: ((Int, [Float], Float, Float, Float) -> Void)?

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

    func start(sweepSeconds: Double, onColumn: @escaping (Int, [Float], Float, Float, Float) -> Void) {
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

        // Pan: +1 at right → -1 at left to match visual
        let norm = Double(col) / maxCol       // 1 → 0 over sweep
        let pan = Float(norm * 2 - 1)         // +1 → -1

        // Distance & edge strength for this (mirrored) column
        var z01 = columnZ01(col: col)
        let edge01 = columnEdge01(col: col)

        // Target coverage in this column (0..1 of vertical cells belonging to targetClass)
        let targetCov = columnTargetCoverage(col: col)
        print("scan col \(col), targetCov = \(targetCov)")

        // Simple target-based boost: when target coverage is high, pull z01 slightly "closer"
        // so the target feels more foreground in the sonification.
        let boost: Float = targetCov * 0.3
        z01 = clamp01(z01 - boost)

        onColumn?(col, env, pan, z01, edge01)

        if let img = debugImageFromGrid() { debugImage = img }
    }


    // MARK: ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
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
            let y0 = gy * H / Self.gridHeight
            let y1 = min((gy + 1) * H / Self.gridHeight, H)

            for gx in 0..<Self.gridWidth {
                let x0 = gx * W / Self.gridWidth
                let x1 = min((gx + 1) * W / Self.gridWidth, W)

                // Majority vote for class in this cell
                var counts = [Int](repeating: 0, count: 6) // 6 classes in MiniUNetHW6
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

                let (bestClass, _) = counts.enumerated().max(by: { $0.element < $1.element }) ?? (0, 0)
                newGrid[gy * Self.gridWidth + gx] = UInt8(bestClass)
            }
        }

        gridLock.lock()
        classGrid = newGrid
        gridLock.unlock()

        // DEBUG: print class distribution in the 60x40 grid
        var hist = [Int](repeating: 0, count: 6)
        for v in newGrid {
            let idx = Int(v)
            if idx >= 0 && idx < hist.count {
                hist[idx] += 1
            }
        }
        print("classGrid histogram:", hist)
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
            let d = grid[gy * Self.gridWidth + (Self.gridWidth - 1 - col)]
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
        // Mirror horizontally to match columnEnvelope’s mirroring
        let W = Self.gridWidth
        let H = Self.gridHeight
        let cm = (W - 1 - col)

        gridLock.lock(); defer { gridLock.unlock() }

        var countTarget = 0
        for y in 0..<H {
            let idx = y * W + cm
            if classGrid[idx] == targetClass {
                countTarget += 1
            }
        }
        return Float(countTarget) / Float(H)
    }

    private func columnZ01(col: Int) -> Float {
        // Mirror horizontally to match columnEnvelope’s mirroring
        let W = Self.gridWidth
        let H = Self.gridHeight
        let cm = (W - 1 - col)
        gridLock.lock(); defer { gridLock.unlock() }
        var acc: Float = 0
        let range = max(0.001, (farMeters - nearMeters))
        for y in 0..<H {
            let d = grid[y * W + cm]
            let t = clamp01((d - nearMeters) / range) // 0 near → 1 far
            acc += t
        }
        return acc / Float(H)
    }

    private func columnEdge01(col: Int) -> Float {
        // Mirror horizontally to match columnEnvelope’s mirroring
        let W = Self.gridWidth
        let H = Self.gridHeight
        let cm = (W - 1 - col)
        let c1 = min(W - 1, cm + 1)
        gridLock.lock(); defer { gridLock.unlock() }
        var acc: Float = 0
        for y in 0..<H {
            let a = grid[y * W + cm]
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
        // Render 60x40 as grayscale for dev UI
        let W = Self.gridWidth
        let H = Self.gridHeight
        let scale: CGFloat = 4
        let size = CGSize(width: CGFloat(W), height: CGFloat(H))

        // Normalize grid to 0..1 from near..far (closer=brighter)
        var pixels = [UInt8](repeating: 0, count: W * H)
        gridLock.lock()
        for y in 0..<H {
            for x in 0..<W {
                let d = grid[y * W + x]
                let t = clamp01(1 - (d - nearMeters) / max(0.001, (farMeters - nearMeters)))
                pixels[y * W + x] = UInt8(t * 255)
            }
        }
        gridLock.unlock()

        let cfData = CFDataCreate(nil, pixels, pixels.count)!
        let provider = CGDataProvider(data: cfData)!
        let cs = CGColorSpaceCreateDeviceGray()
        if let cg = CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: W, space: cs, bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) {
            return UIImage(cgImage: cg, scale: 1.0/scale, orientation: .up)
        }
        return nil
    }
}

@inline(__always) private func clamp01(_ x: Float) -> Float { max(0, min(1, x)) }
