import Foundation
import Vision
import ARKit
import CoreML

final class ObjectSegmentation {
    private let vnModel: VNCoreMLModel
    private let queue = DispatchQueue(label: "ObjectSegmentationQueue")

    init?() {
        do {
            let coreMLModel = try MiniUNetHW6(configuration: MLModelConfiguration()).model
            self.vnModel = try VNCoreMLModel(for: coreMLModel)
        } catch {
            print("Failed to load MiniUNetHW6: \(error)")
            return nil
        }
    }

    /// Run segmentation on the ARFrame's RGB image.
    /// completion(mask, width, height) where mask.count == width * height
    func predictMask(from frame: ARFrame,
                     completion: @escaping (_ mask: [UInt8]?, _ width: Int, _ height: Int) -> Void) {

        let pixelBuffer = frame.capturedImage

        let request = VNCoreMLRequest(model: vnModel) { request, error in
            guard error == nil else {
                completion(nil, 0, 0)
                return
            }
            guard let result = request.results?.first as? VNCoreMLFeatureValueObservation,
                  let multiArray = result.featureValue.multiArrayValue else {
                completion(nil, 0, 0)
                return
            }

            // Handle shape [1,C,H,W] or [C,H,W]
            let shape = multiArray.shape.map { Int(truncating: $0) }
            let (C, H, W): (Int, Int, Int)
            if shape.count == 4 {
                C = shape[1]; H = shape[2]; W = shape[3]
            } else if shape.count == 3 {
                C = shape[0]; H = shape[1]; W = shape[2]
            } else {
                completion(nil, 0, 0)
                return
            }

            var mask = [UInt8](repeating: 0, count: W * H)
            let ptr = multiArray.dataPointer.bindMemory(to: Float.self,
                                                        capacity: C * H * W)

            // Argmax over classes for each pixel
            for y in 0..<H {
                for x in 0..<W {
                    var bestClass = 0
                    var bestVal = -Float.greatestFiniteMagnitude
                    for c in 0..<C {
                        let idx = c * H * W + y * W + x
                        let val = ptr[idx]
                        if val > bestVal {
                            bestVal = val
                            bestClass = c
                        }
                    }
                    mask[y * W + x] = UInt8(bestClass)
                }
            }

            completion(mask, W, H)
        }

        // ARKit camera buffer orientation; .right usually correct for portrait
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .right,
                                            options: [:])

        queue.async {
            do { try handler.perform([request]) }
            catch { completion(nil, 0, 0) }
        }
    }
}
