import SwiftUI
import ARKit

struct ARDepthCaptureView: UIViewRepresentable {
    let pipeline: DepthPipeline

    func makeUIView(context: Context) -> UIView {
        let v = UIView(frame: .zero)
        pipeline.attach(to: v)
        return v
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}
