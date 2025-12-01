import SwiftUI
import ARKit

struct ContentView: View {
    @StateObject private var depthPipeline = DepthPipeline()
    @StateObject private var audio = SpectralAudioEngine()

    @State private var isRunning = false
    @State private var sweepSeconds: Double = 2.0
    @State private var fMin: Double = 50
    @State private var fMax: Double = 10050
    @State private var gainRangeDB: Double = 24
    @State private var showSeg: Bool = true
    @State private var segAlpha: Double = 0.55
    

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 16) {
                // LEFT: Preview (≈ 2/3 width)
                ZStack(alignment: .topLeading) {
                    if let image = depthPipeline.debugImage {
                        Image(uiImage: image)
                            .resizable()
                            .interpolation(.none)
                            .scaledToFit()
                            // Keep LiDAR depth ~4:3 in a landscape space
                            .aspectRatio(4.0/3.0, contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .border(Color.gray)

                        if showSeg, let seg = depthPipeline.segOverlay {
                            Image(uiImage: seg)
                                .resizable()
                                .interpolation(.none)
                                .scaledToFit()
                                .aspectRatio(4.0/3.0, contentMode: .fit)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .opacity(segAlpha)
                                .allowsHitTesting(false)
                        }

                        // Full-height scan bar moving LEFT → RIGHT
                        GeometryReader { g in
                            let norm = CGFloat(depthPipeline.scanColumn) / CGFloat(DepthPipeline.gridWidth - 1)
                            let x = (1 - norm) * g.size.width
                            Rectangle()
                                .fill(Color.red.opacity(0.85))
                                .frame(width: 2, height: g.size.height)
                                .position(x: x, y: g.size.height / 2)
                        }
                        .allowsHitTesting(false)
                    } else {
                        Text("Waiting for depth…")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .border(Color.gray)
                    }
                }
                .frame(width: geo.size.width * 0.66, height: geo.size.height)
                .clipped()

                // RIGHT: Controls (≈ 1/3 width)
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Button(isRunning ? "Stop" : "Start") {
                            if isRunning { stop() } else { start() }
                        }
                        .buttonStyle(.borderedProminent)

                        Spacer()
                    }


                    Group {
                        LabeledSlider(title: "Sweep (s)", value: $sweepSeconds, range: 0.5...8.0, format: "%.1f")
                        LabeledSlider(title: "Min Hz", value: $fMin, range: 20...500, format: "%.0f")
                        LabeledSlider(title: "Max Hz", value: $fMax, range: 2000...20000, format: "%.0f")
                        LabeledSlider(title: "Atten (dB)", value: $gainRangeDB, range: 6...48, format: "%.0f")
                    }

                    // Segmentation overlay controls
                    Toggle("Seg Overlay", isOn: $showSeg)

                    HStack {
                        Text("Overlay α").frame(width: 90, alignment: .leading)
                        Slider(value: $segAlpha, in: 0...1)
                        Text(String(format: "%.2f", segAlpha))
                            .monospacedDigit()
                            .frame(width: 60, alignment: .trailing)
                    }

                    // Legend for classes: red=sphere, green=tetra, blue=cube
                    HStack(spacing: 8) {
                        LegendSwatch(color: .red);   Text("Sphere").font(.caption)
                        LegendSwatch(color: .green); Text("Tetra").font(.caption)
                        LegendSwatch(color: .blue);  Text("Cube").font(.caption)
                    }
                    .opacity(showSeg ? 1 : 0.3)

                    Spacer()

                    let colText = "Col \(depthPipeline.scanColumn + 1)/\(DepthPipeline.gridWidth)"
                    let panText = String(format: "%.2f", audio.pan)
                    let fpsText = String(format: "%.1f", depthPipeline.fps)
                    Text("\(colText) • Pan \(panText) • FPS \(fpsText)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    
                    Text(depthPipeline.classHistogramText)
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                    
                }
                .frame(width: geo.size.width * 0.34, height: geo.size.height)
            }
            .padding(16)
        }
        // Headless AR session (hidden view just runs the session)
        .overlay(
            ARDepthCaptureView(pipeline: depthPipeline).frame(width: 0, height: 0)
        )
        .onAppear {
            audio.configureBands(fMin: fMin, fMax: fMax)
        }
        .onChange(of: fMin) { _, v in audio.configureBands(fMin: v, fMax: fMax) }
        .onChange(of: fMax) { _, v in audio.configureBands(fMin: fMin, fMax: v) }
        .onChange(of: gainRangeDB) { _, v in depthPipeline.gainRangeDB = Float(v) }
    }

    private func start() {
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else { return }
        audio.start()
        depthPipeline.start(sweepSeconds: sweepSeconds) { col, envelope, targetMask40, shapeId, pan, z01, edge01 in
            // Background scene envelope (40 bands)
            audio.updateEnvelope(envelope)

            // Foreground: emphasize the object currently under the scan line
            if shapeId != 0 {
                // +12 dB boost over background (adjust to taste)
                audio.setTargetBands(targetMask40, shape: shapeId, boostDB: 12)
            } else {
                audio.clearTarget()
            }

            // Spatial & transient cues
            audio.pan = pan
            audio.updateDistance(z01)   // drives LPF and reverb send
            audio.triggerEdge(edge01)   // short click overlay on edges
        }
        isRunning = true
    }

    private func stop() {
        depthPipeline.stop()
        audio.stop()
        isRunning = false
    }
}

// Small helper for tidy labeled sliders
private struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String

    var body: some View {
        HStack {
            Text(title).frame(width: 90, alignment: .leading)
            Slider(value: $value, in: range)
            Text(String(format: format, value))
                .monospacedDigit()
                .frame(width: 60, alignment: .trailing)
        }
    }
}

private struct LegendSwatch: View {
    let color: Color
    var body: some View {
        Rectangle()
            .fill(color)
            .frame(width: 12, height: 12)
            .cornerRadius(2)
    }
}
