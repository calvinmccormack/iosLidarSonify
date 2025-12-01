import Foundation
import Vision
import CoreML
import UIKit
import ARKit

private let _segCIContext = CIContext(options: nil)
private let _forceGrayscaleInput = false  // feed original BGRA; model likely expects 3‑ch input

// Class ids from training
enum SegClass: UInt8 { case bg = 0, sphere = 1, tetra = 2, cube = 3 }

// Post-processing / tuning knobs
struct SegTuning {
    var enabled: Bool = true
    /// 3x3 median-style majority pass; 0 = disabled
    var medianPasses: Int = 1

    /// Global morphology defaults (used as fallback if per-class not set differently)
    var dilateIters: Int = 0
    var erodeIters: Int = 1

    /// Per-class morphology overrides
    var dilateItersSphere: Int = 3
    var dilateItersTetra:  Int = 3
    var dilateItersCube:   Int = 1

    var erodeItersSphere: Int = 2
    var erodeItersTetra:  Int = 2
    var erodeItersCube:   Int = 1

    /// Keep only components >= minAreaFrac * (W*H) for each class
    var minAreaFracSphere: Float = 0.001
    var minAreaFracTetra:  Float = 0.001
    var minAreaFracCube:   Float = 0.015

    /// Per-class caps (0 = unlimited). Falls back to keepTopPerClass when 0.
    var keepTopPerClass: Int = 1
    var keepTopSphere:   Int = 0
    var keepTopTetra:    Int = 0
    var keepTopCube:     Int = 0

    /// Debug printouts
    var debug: Bool = true

    /// Priority when classes overlap after morphology (first wins)
    var priority: [SegClass] = [.tetra, .sphere, .cube, .bg]
}

// Overlay colors: red=sphere, green=tetra, blue=cube
private let segColors: [SegClass: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)] = [
    .sphere: (255,   0,   0, 140),
    .tetra:  (  0, 255,   0, 140),
    .cube:   (  0,   0, 255, 140)
]

final class ObjectSegmentation {
    // Global tuning (can be changed at runtime)
    static var tuning = SegTuning()
    static func setTuning(_ t: SegTuning) { tuning = t }

    // MARK: - Model & init

    private func makeGrayscaleBuffer(from src: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(src)
        let h = CVPixelBufferGetHeight(src)
        let ci = CIImage(cvPixelBuffer: src)
            .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.0])
        var dst: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8,
            kCVPixelBufferWidthKey: w,
            kCVPixelBufferHeightKey: h,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_OneComponent8, attrs as CFDictionary, &dst)
        guard status == kCVReturnSuccess, let out = dst else { return nil }
        _segCIContext.render(ci, to: out)
        return out
    }

    private let vnModel: VNCoreMLModel

    init?() {
        // Load compiled Core ML model by file name, not by generated Swift class
        let url =
            Bundle.main.url(forResource: "ShapeSeg_unet_mbv2_best", withExtension: "mlmodelc", subdirectory: "Models")
            ?? Bundle.main.url(forResource: "ShapeSeg_unet_mbv2_best", withExtension: "mlmodelc")

        guard let modelURL = url else {
            print("ObjectSegmentation: could not find ShapeSeg_unet_mbv2_best.mlmodelc in bundle")
            return nil
        }
        do {
            let coreML = try MLModel(contentsOf: modelURL)
            // Debug: model I/O description
            print("[Seg] Loaded MLModel:", modelURL.lastPathComponent)
            let md = coreML.modelDescription
            for (name, d) in md.inputDescriptionsByName {
                if let ic = d.imageConstraint {
                    print("[Seg] Input", name, "imageConstraint:", ic)
                } else if let mc = d.multiArrayConstraint {
                    print("[Seg] Input", name, "multiArray shape:", mc.shape)
                } else {
                    print("[Seg] Input", name, "(unknown type)")
                }
            }
            for (name, d) in md.outputDescriptionsByName {
                if let ic = d.imageConstraint {
                    print("[Seg] Output", name, "imageConstraint:", ic)
                } else if let mc = d.multiArrayConstraint {
                    print("[Seg] Output", name, "multiArray shape:", mc.shape)
                } else {
                    print("[Seg] Output", name, "(unknown type)")
                }
            }
            self.vnModel = try VNCoreMLModel(for: coreML)
            // Important for segmentation, keep full frame and scale
            self.vnModel.inputImageFeatureName = self.vnModel.inputImageFeatureName
        } catch {
            print("ObjectSegmentation: failed to load model: \(error)")
            return nil
        }
    }

    // MARK: - Prediction

    // New helper to summarize logits per channel
    private static func summarizeLogits(_ arr: MLMultiArray) {
        let (flat, shape) = readMultiArrayFloat(arr)
        guard !flat.isEmpty else { return }
        let (C, H, W): (Int, Int, Int) = {
            if shape.count == 3 { return (shape[0], shape[1], shape[2]) }
            if shape.count == 4 { return (shape[1], shape[2], shape[3]) }
            return (0,0,0)
        }()
        guard C > 0, H > 0, W > 0 else { return }
        let HW = H * W
        let sample = min(HW, 50_000)
        let step = max(1, HW / sample)
        var means = [Float](repeating: 0, count: C)
        for c in 0..<C {
            var acc: Float = 0
            var n = 0
            var i = 0
            while i < HW && n < sample {
                acc += flat[c*HW + i]
                i += step; n += 1
            }
            means[c] = acc / Float(max(1,n))
        }
        print("[Seg] channel means (C,H,W=\(C),\(H),\(W)):", means)
    }

    /// Returns raw class mask as [UInt8] plus width and height
    func predictMask(from frame: ARFrame,
                     completion: @escaping (_ mask: [UInt8]?, _ width: Int, _ height: Int) -> Void)
    {
        // AR rear camera in portrait yields .right orientation
        let req = VNCoreMLRequest(model: vnModel) { req, _ in
            // Two possibilities: a pixel buffer mask, or a multi-array of logits
            if let pix = (req.results as? [VNPixelBufferObservation])?.first?.pixelBuffer {
                var (mask, w, h) = Self.pixelBufferToMask(pix)
                if Self.tuning.enabled {
                    mask = Self.tuneMask(mask, w: w, h: h, tuning: Self.tuning)
                }
                // Debug: class histogram over first chunk
                let N = min(mask.count, 10_000)
                var htmp = [Int](repeating: 0, count: 8)
                for i in 0..<N { htmp[Int(mask[i]) % 8] += 1 }
                print("[Seg] mask preview counts (first \(N)):", htmp)
                completion(mask, w, h)
                return
            }

            if let feat = (req.results as? [VNCoreMLFeatureValueObservation])?.first?.featureValue,
               let arr = feat.multiArrayValue {
                print("[Seg] VNCoreMLFeatureValueObservation: dtype=\(arr.dataType.rawValue) shape=\(arr.shape)")

                // Sample min/max of the first 1024 logits for sanity
                if arr.dataType == .float32 || arr.dataType == .float16 {
                    let sample = min(1024, arr.count)
                    var minv: Float = .greatestFiniteMagnitude
                    var maxv: Float = -.greatestFiniteMagnitude
                    if arr.dataType == .float32 {
                        let p = arr.dataPointer.bindMemory(to: Float32.self, capacity: sample)
                        for i in 0..<sample { let v = Float(p[i]); minv = min(minv, v); maxv = max(maxv, v) }
                    } else {
                        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: sample)
                        for i in 0..<sample { let v = Float(Float16(bitPattern: p[i])); minv = min(minv, v); maxv = max(maxv, v) }
                    }
                    print("[Seg] logits sample min/max:", minv, maxv)
                }

                Self.summarizeLogits(arr)

                var (mask, w, h) = Self.multiArrayLogitsToMask(arr)
                var N = min(mask.count, 10_000)
                var htmp = [Int](repeating: 0, count: 8)
                for i in 0..<N { htmp[Int(mask[i]) % 8] += 1 }
                print("[Seg] mask preview counts (first \(N)):", htmp)

                // Fallback: if all bg, try NHWC interpretation
                if htmp[0] == N, arr.shape.count == 4,
                   Int(truncating: arr.shape[1]) > 8, Int(truncating: arr.shape[3]) == 4 {
                    print("[Seg] argmax all-bg; trying NHWC fallback decode…")
                    (mask, w, h) = Self.multiArrayLogitsToMaskNHWC(arr)
                    N = min(mask.count, 10_000)
                    htmp = [Int](repeating: 0, count: 8)
                    for i in 0..<N { htmp[Int(mask[i]) % 8] += 1 }
                    print("[Seg] NHWC mask preview counts (first \(N)):", htmp)
                }

                if Self.tuning.enabled {
                    mask = Self.tuneMask(mask, w: w, h: h, tuning: Self.tuning)
                    let N2 = min(mask.count, 10_000)
                    var h2 = [Int](repeating: 0, count: 8)
                    for i in 0..<N2 { h2[Int(mask[i]) % 8] += 1 }
                    print("[Seg] cleaned mask preview counts (first \(N2)):", h2)
                }
                completion(mask, w, h)
                return
            }

            completion(nil, 0, 0)
        }

        // Use the full frame (desired for segmentation)
        req.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill

        let inputPB = frame.capturedImage  // BGRA from ARKit
        let handler = VNImageRequestHandler(cvPixelBuffer: inputPB,
                                            orientation: .right,
                                            options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do { try handler.perform([req]) }
            catch { print("VN perform error: \(error)"); completion(nil, 0, 0) }
        }
    }

    /// Convenience: returns colored overlay UIImage and 4‑bin histogram [bg, sphere, tetra, cube]
    func predictMaskOverlayAndHistogram(from frame: ARFrame,
                                        completion: @escaping (_ overlay: UIImage?, _ histogram: [Int]) -> Void)
    {
        predictMask(from: frame) { mask, w, h in
            guard let mask = mask, w > 0, h > 0 else { completion(nil, [0,0,0,0]); return }

            // Histogram
            var hist = [Int](repeating: 0, count: 4)
            for v in mask { hist[Int(v)] += 1 }

            // Build RGBA overlay
            let count = w * h
            var rgba = [UInt8](repeating: 0, count: count * 4)
            for i in 0..<count {
                let cls = SegClass(rawValue: mask[i]) ?? .bg
                let base = i * 4
                if let c = segColors[cls] {
                    rgba[base + 0] = c.r
                    rgba[base + 1] = c.g
                    rgba[base + 2] = c.b
                    rgba[base + 3] = c.a
                } else {
                    rgba[base + 3] = 0
                }
            }

            // Make image
            let provider = CGDataProvider(data: Data(rgba) as CFData)!
            let cg = CGImage(width: w,
                             height: h,
                             bitsPerComponent: 8,
                             bitsPerPixel: 32,
                             bytesPerRow: w * 4,
                             space: CGColorSpaceCreateDeviceRGB(),
                             bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                             provider: provider,
                             decode: nil,
                             shouldInterpolate: true,
                             intent: .defaultIntent)
            let overlay = cg.map { UIImage(cgImage: $0) }
            completion(overlay, hist)
        }
    }

    // MARK: - Tuning Helpers (private static)

    // Build 0/1 binary for a single class
    private static func binary(from mask: [UInt8], w: Int, h: Int, for cls: SegClass) -> [UInt8] {
        let v = cls.rawValue
        var out = [UInt8](repeating: 0, count: w*h)
        for i in 0..<w*h { out[i] = (mask[i] == v) ? 1 : 0 }
        return out
    }

    // 3x3 majority (median-like) filter, one pass
    private static func majority3x3(_ bin: [UInt8], w: Int, h: Int) -> [UInt8] {
        var out = bin
        let W = w, H = h
        if W < 3 || H < 3 { return out }
        for y in 1..<H-1 {
            var idx = y*W + 1
            for _ in 1..<W-1 {
                var s = 0
                s += Int(bin[idx - W - 1]); s += Int(bin[idx - W]); s += Int(bin[idx - W + 1])
                s += Int(bin[idx - 1]);     s += Int(bin[idx]);     s += Int(bin[idx + 1])
                s += Int(bin[idx + W - 1]); s += Int(bin[idx + W]); s += Int(bin[idx + W + 1])
                out[idx] = (s >= 5) ? 1 : 0
                idx += 1
            }
        }
        return out
    }

    // 3x3 dilation (iters times)
    private static func dilate3x3(_ bin: [UInt8], w: Int, h: Int, iters: Int) -> [UInt8] {
        var cur = bin
        if iters <= 0 { return cur }
        let W = w, H = h
        if W < 3 || H < 3 { return cur }
        for _ in 0..<iters {
            var out = cur
            for y in 1..<H-1 {
                var idx = y*W + 1
                for _ in 1..<W-1 {
                    if cur[idx] == 0 {
                        if cur[idx - W - 1] | cur[idx - W] | cur[idx - W + 1] |
                           cur[idx - 1]     | cur[idx + 1]                   |
                           cur[idx + W - 1] | cur[idx + W] | cur[idx + W + 1] != 0 {
                            out[idx] = 1
                        }
                    }
                    idx += 1
                }
            }
            cur = out
        }
        return cur
    }

    // 3x3 erosion (iters times)
    private static func erode3x3(_ bin: [UInt8], w: Int, h: Int, iters: Int) -> [UInt8] {
        var cur = bin
        if iters <= 0 { return cur }
        let W = w, H = h
        if W < 3 || H < 3 { return cur }
        for _ in 0..<iters {
            var out = cur
            for y in 1..<H-1 {
                var idx = y*W + 1
                for _ in 1..<W-1 {
                    if cur[idx] == 1 {
                        if (cur[idx - W - 1] & cur[idx - W] & cur[idx - W + 1] &
                            cur[idx - 1]     & cur[idx + 1]                   &
                            cur[idx + W - 1] & cur[idx + W] & cur[idx + W + 1]) == 0 {
                            out[idx] = 0
                        }
                    }
                    idx += 1
                }
            }
            cur = out
        }
        return cur
    }

    // Keep only components above minArea and at most keepTop largest (0 = unlimited)
    private static func filterComponents(_ bin: [UInt8], w: Int, h: Int, minArea: Int, keepTop: Int) -> [UInt8] {
        let N = w*h
        var visited = [UInt8](repeating: 0, count: N)
        var components: [[Int]] = []
        var idx = 0
        while idx < N {
            if bin[idx] == 1 && visited[idx] == 0 {
                // BFS
                var q: [Int] = [idx]
                visited[idx] = 1
                var comp: [Int] = []
                while !q.isEmpty {
                    let p = q.removeLast()
                    comp.append(p)
                    let y = p / w
                    let x = p - y*w
                    if x > 0 {
                        let n = p - 1
                        if bin[n] == 1 && visited[n] == 0 { visited[n] = 1; q.append(n) }
                    }
                    if x + 1 < w {
                        let n = p + 1
                        if bin[n] == 1 && visited[n] == 0 { visited[n] = 1; q.append(n) }
                    }
                    if y > 0 {
                        let n = p - w
                        if bin[n] == 1 && visited[n] == 0 { visited[n] = 1; q.append(n) }
                    }
                    if y + 1 < h {
                        let n = p + w
                        if bin[n] == 1 && visited[n] == 0 { visited[n] = 1; q.append(n) }
                    }
                }
                if comp.count >= minArea { components.append(comp) }
            }
            idx += 1
        }
        if keepTop > 0 && components.count > keepTop {
            components.sort { $0.count > $1.count }
            components = Array(components.prefix(keepTop))
        }
        var out = [UInt8](repeating: 0, count: N)
        for comp in components { for i in comp { out[i] = 1 } }
        return out
    }

    // Merge class binaries back into a single label mask following priority
    private static func mergeClasses(w: Int, h: Int, bins: [SegClass: [UInt8]], priority: [SegClass]) -> [UInt8] {
        let N = w*h
        var out = [UInt8](repeating: SegClass.bg.rawValue, count: N)
        for i in 0..<N {
            var assigned = false
            for cls in priority {
                if let b = bins[cls], i < b.count, b[i] == 1 {
                    out[i] = cls.rawValue
                    assigned = true
                    break
                }
            }
            if !assigned { out[i] = SegClass.bg.rawValue }
        }
        return out
    }

    // Full tuning pipeline: majority filter - dilate - erode - component filter - merge by priority
    private static func tuneMask(_ mask: [UInt8], w: Int, h: Int, tuning: SegTuning) -> [UInt8] {
        if !tuning.enabled { return mask }
        let N = w*h

        // build binaries
        var bSphere = binary(from: mask, w: w, h: h, for: .sphere)
        var bTetra  = binary(from: mask, w: w, h: h, for: .tetra)
        var bCube   = binary(from: mask, w: w, h: h, for: .cube)

        // majority passes
        for _ in 0..<max(0, tuning.medianPasses) {
            bSphere = majority3x3(bSphere, w: w, h: h)
            bTetra  = majority3x3(bTetra,  w: w, h: h)
            bCube   = majority3x3(bCube,   w: w, h: h)
        }

        // morphology (per-class with fallback to global)
        let dS = max(0, tuning.dilateItersSphere)
        let dT = max(0, tuning.dilateItersTetra)
        let dC = max(0, tuning.dilateItersCube)

        let eS = max(0, tuning.erodeItersSphere)
        let eT = max(0, tuning.erodeItersTetra)
        let eC = max(0, tuning.erodeItersCube)

        if dS > 0 { bSphere = dilate3x3(bSphere, w: w, h: h, iters: dS) }
        if dT > 0 { bTetra  = dilate3x3(bTetra,  w: w, h: h, iters: dT) }
        if dC > 0 { bCube   = dilate3x3(bCube,   w: w, h: h, iters: dC) }

        if eS > 0 { bSphere = erode3x3(bSphere, w: w, h: h, iters: eS) }
        if eT > 0 { bTetra  = erode3x3(bTetra,  w: w, h: h, iters: eT) }
        if eC > 0 { bCube   = erode3x3(bCube,   w: w, h: h, iters: eC) }

        // component filtering per class
        let areaF: (Float) -> Int = { frac in
            let px = Int(Float(N) * max(0, min(1, frac)))
            return max(1, px)
        }
        let kS = (tuning.keepTopSphere > 0) ? tuning.keepTopSphere : tuning.keepTopPerClass
        let kT = (tuning.keepTopTetra  > 0) ? tuning.keepTopTetra  : tuning.keepTopPerClass
        let kC = (tuning.keepTopCube   > 0) ? tuning.keepTopCube   : tuning.keepTopPerClass
        bSphere = filterComponents(bSphere, w: w, h: h,
                                   minArea: areaF(tuning.minAreaFracSphere),
                                   keepTop: kS)
        bTetra  = filterComponents(bTetra,  w: w, h: h,
                                   minArea: areaF(tuning.minAreaFracTetra),
                                   keepTop: kT)
        bCube   = filterComponents(bCube,   w: w, h: h,
                                   minArea: areaF(tuning.minAreaFracCube),
                                   keepTop: kC)

        // Merge by priority (cube and tetra win over sphere by default)
        let merged = mergeClasses(w: w, h: h,
                                  bins: [.sphere: bSphere, .tetra: bTetra, .cube: bCube],
                                  priority: tuning.priority)

        if tuning.debug {
            var hist = [Int](repeating: 0, count: 4)
            for v in merged { hist[Int(v)] += 1 }
            print("[Seg] tuned histogram [bg,sphere,tetra,cube]:", hist)
        }
        return merged
    }

    // MARK: - Output adapters (private static)

    private static func pixelBufferToMask(_ pb: CVPixelBuffer) -> ([UInt8], Int, Int) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return ([], 0, 0) }

        var out = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            let src = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for x in 0..<w {
                out[y * w + x] = src[x]
            }
        }
        return (out, w, h)
    }

    /// Read MLMultiArray as a flat [Float] regardless of .float32 or .float16, and return (buffer, shape)
    private static func readMultiArrayFloat(_ arr: MLMultiArray) -> ([Float], [Int]) {
        let shape = arr.shape.map { Int(truncating: $0) }
        let count = shape.reduce(1, *)
        switch arr.dataType {
        case .float32:
            let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: count)
            let buf = UnsafeBufferPointer(start: ptr, count: count)
            return (Array(buf), shape)
        case .float16:
            let ptrU16 = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            let bufU16 = UnsafeBufferPointer(start: ptrU16, count: count)
            var out = [Float](); out.reserveCapacity(count)
            for v in bufU16 {
                let f16 = Float16(bitPattern: v)
                out.append(Float(f16))
            }
            return (out, shape)
        default:
            return ([], shape)
        }
    }

    private static func multiArrayLogitsToMask(_ arr: MLMultiArray) -> ([UInt8], Int, Int) {
        let (flat, shape) = readMultiArrayFloat(arr)
        guard !flat.isEmpty else { return ([], 0, 0) }

        // Accept shapes: [C,H,W], [1,C,H,W], [H,W] (already argmaxed)
        if shape.count == 2 {
            let H = shape[0], W = shape[1]
            var mask = [UInt8](repeating: 0, count: H*W)
            for i in 0..<(H*W) { mask[i] = UInt8(max(0, min(255, Int(flat[i])))) }
            return (mask, W, H)
        }

        let (C, H, W): (Int, Int, Int) = {
            if shape.count == 3 { return (shape[0], shape[1], shape[2]) }
            if shape.count == 4 { return (shape[1], shape[2], shape[3]) } // assume N=1
            return (0,0,0)
        }()
        guard C > 0, H > 0, W > 0 else { return ([], 0, 0) }

        let HW = H * W
        var mask = [UInt8](repeating: 0, count: HW)
        for i in 0..<HW {
            var bestC = 0
            var bestV = -Float.greatestFiniteMagnitude
            for c in 0..<C {
                let v = flat[c * HW + i]
                if v > bestV { bestV = v; bestC = c }
            }
            mask[i] = UInt8(bestC)
        }
        return (mask, W, H)
    }

    // NHWC fallback decoding
    private static func multiArrayLogitsToMaskNHWC(_ arr: MLMultiArray) -> ([UInt8], Int, Int) {
        let (flat, shape) = readMultiArrayFloat(arr)
        guard shape.count == 4 else { return ([], 0, 0) }
        // Assume [1, H, W, C]
        let H = Int(truncating: shape[1] as NSNumber)
        let W = Int(truncating: shape[2] as NSNumber)
        let C = Int(truncating: shape[3] as NSNumber)
        guard H>0, W>0, C>0 else { return ([], 0, 0) }
        let HW = H * W
        var mask = [UInt8](repeating: 0, count: HW)
        for y in 0..<H {
            for x in 0..<W {
                let base = (y*W + x) * C
                var bestC = 0
                var bestV = -Float.greatestFiniteMagnitude
                for c in 0..<C {
                    let v = flat[base + c]
                    if v > bestV { bestV = v; bestC = c }
                }
                mask[y*W + x] = UInt8(bestC)
            }
        }
        return (mask, W, H)
    }
}
