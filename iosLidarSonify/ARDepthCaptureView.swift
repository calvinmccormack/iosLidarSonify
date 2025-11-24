import SwiftUI
import ARKit
import RealityKit

struct ARDepthCaptureView: UIViewRepresentable {
    let pipeline: DepthPipeline

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIView {
        let container = UIView(frame: .zero)
        pipeline.attach(to: container)

        // Try to find an embedded AR view immediately
        assignSessionDelegate(in: container, coordinator: context.coordinator)

        // If attach(to:) adds the AR view on the next runloop, try again
        DispatchQueue.main.async { [weak container] in
            if let c = container { self.assignSessionDelegate(in: c, coordinator: context.coordinator) }
        }
        return container
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // Ensure delegate remains set even if the AR view was re-created
        assignSessionDelegate(in: uiView, coordinator: context.coordinator)
    }

    static func dismantleUIView(_ uiView: UIView, coordinator: Coordinator) {
        // Cleanly remove delegates to avoid dangling callbacks
        if let scn = uiView.subviews.compactMap({ $0 as? ARSCNView }).first {
            scn.session.delegate = nil
        }
        if let ar = uiView.subviews.compactMap({ $0 as? ARView }).first {
            ar.session.delegate = nil
        }
    }

    // MARK: - Helper
    private func assignSessionDelegate(in view: UIView, coordinator: Coordinator) {
        if let scn = view.subviews.compactMap({ $0 as? ARSCNView }).first {
            if scn.session.delegate !== coordinator {
                scn.session.delegate = coordinator
            }
        } else if let ar = view.subviews.compactMap({ $0 as? ARView }).first {
            if ar.session.delegate !== coordinator {
                ar.session.delegate = coordinator
            }
        }
    }

    // MARK: - Coordinator
    class Coordinator: NSObject, ARSessionDelegate {
        let parent: ARDepthCaptureView
        init(_ parent: ARDepthCaptureView) { self.parent = parent }

        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            parent.pipeline.process(frame: frame)
        }

        // Optional: useful diagnostics during bring-up
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
