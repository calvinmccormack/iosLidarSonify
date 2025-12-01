import SwiftUI
import ARKit
import RealityKit

struct ARDepthCaptureView: UIViewRepresentable {
    let pipeline: DepthPipeline

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.automaticallyConfigureSession = false
        arView.isHidden = true // headless

        let config = ARWorldTrackingConfiguration()
        var semantics: ARWorldTrackingConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { semantics.insert(.sceneDepth) }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) { semantics.insert(.smoothedSceneDepth) }
        config.frameSemantics = semantics

        arView.session.delegate = context.coordinator
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) { }

    static func dismantleUIView(_ uiView: ARView, coordinator: Coordinator) {
        uiView.session.delegate = nil
        uiView.session.pause()
    }

    class Coordinator: NSObject, ARSessionDelegate {
        let parent: ARDepthCaptureView
        init(_ parent: ARDepthCaptureView) { self.parent = parent }

        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            parent.pipeline.process(frame: frame)
        }

        // Helpful diagnostics
        func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
            #if DEBUG
            print("[AR] tracking:", camera.trackingState)
            #endif
        }
        func session(_ session: ARSession, didFailWithError error: Error) {
            #if DEBUG
            print("[AR] session error:", error.localizedDescription)
            #endif
        }
    }
}
