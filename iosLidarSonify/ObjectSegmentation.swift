import Foundation
import Vision
import CoreML
import UIKit
import ARKit

private let _segCIContext = CIContext(options: nil)
private let _forceGrayscaleInput = false  // feed original BGRA; model likely expects 3-ch input

// Class ids from training
enum SegClass: UInt8 { case bg = 0, sphere = 1, tetra = 2, cube = 3 }

// Overlay colors: red=sphere, green=tetra, blue=cube
private let segColors: [SegClass: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)] = [
    .sphere: (255,   0,   0, 140),
    .tetra:  (  0, 255,   0, 140),
    .cube:   (  0,   0, 255, 140)
]

final class ObjectSegmentation {
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
        // If you added the model inside a “Models” group, try that subdir first.
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

    /// Base predict that returns raw class mask as [UInt8] plus width and height
    func predictMask(from frame: ARFrame,
                     completion: @escaping (_ mask: [UInt8]?, _ width: Int, _ height: Int) -> Void)
    {
        // AR rear camera in portrait yields .right orientation
        let req = VNCoreMLRequest(model: vnModel) { req, _ in
            // Two possibilities: a pixel buffer mask, or a multi-array of logits
            if let pix = (req.results as? [VNPixelBufferObservation])?.first?.pixelBuffer {
                // Assume per-pixel class ids in pix; log basic info
                let w0 = CVPixelBufferGetWidth(pix)
                let h0 = CVPixelBufferGetHeight(pix)
                let pf  = CVPixelBufferGetPixelFormatType(pix)
                print("[Seg] VNPixelBufferObservation:", w0, "x", h0, "pf:", pf)
                // Quick 8-bin rough histogram over first rows (debug only)
                var rough = [Int](repeating: 0, count: 8)
                CVPixelBufferLockBaseAddress(pix, .readOnly)
                if let base = CVPixelBufferGetBaseAddress(pix) {
                    let n = min(w0 * h0, 2048)
                    let p = base.assumingMemoryBound(to: UInt8.self)
                    for i in 0..<n { rough[Int(p[i] >> 5)] += 1 }
                }
                CVPixelBufferUnlockBaseAddress(pix, .readOnly)
                print("[Seg] PB rough hist (8 bins):", rough)

                let (mask, w, h) = Self.pixelBufferToMask(pix)
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
                if htmp[0] == N, arr.shape.count == 4, Int(truncating: arr.shape[1]) > 8, Int(truncating: arr.shape[3]) == 4 {
                    print("[Seg] argmax all-bg; trying NHWC fallback decode…")
                    (mask, w, h) = Self.multiArrayLogitsToMaskNHWC(arr)
                    N = min(mask.count, 10_000)
                    htmp = [Int](repeating: 0, count: 8)
                    for i in 0..<N { htmp[Int(mask[i]) % 8] += 1 }
                    print("[Seg] NHWC mask preview counts (first \(N)):", htmp)
                }

                completion(mask, w, h)
                return
            }

            completion(nil, 0, 0)
        }
        // Confirm: scaleFill uses the full frame (desired for segmentation)
        req.imageCropAndScaleOption = .scaleFill

        let inputPB = frame.capturedImage  // BGRA from ARKit
        let handler = VNImageRequestHandler(cvPixelBuffer: inputPB,
                                            orientation: .right,
                                            options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do { try handler.perform([req]) }
            catch { print("VN perform error: \(error)"); completion(nil, 0, 0) }
        }
    }

    /// Convenience: returns colored overlay UIImage and 4-bin histogram [bg, sphere, tetra, cube]
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

    // MARK: - Output adapters

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
            // Already per-pixel class ids (assumed in flat as Float)
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
        // flat is [C, H, W] contiguous (Core ML uses row-major)
        for i in 0..<HW {
            var bestC = 0
            var bestV = -Float.greatestFiniteMagnitude
            // stride by HW to move across channels at fixed pixel
            for c in 0..<C {
                let v = flat[c * HW + i]
                if v > bestV { bestV = v; bestC = c }
            }
            mask[i] = UInt8(bestC)
        }
        return (mask, W, H)
    }

    // New helper for NHWC fallback decoding
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
