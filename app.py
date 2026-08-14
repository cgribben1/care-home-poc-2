"""Gradio POC: care-home acoustic event detection with YAMNet (no fine-tuning)."""

from __future__ import annotations

import gradio as gr
import numpy as np

from analyze import get_analyzer

CATEGORY_LABELS = {
    "fall": "Fall / impact sounds",
    "distress": "Distress sounds",
    "cough": "Coughing",
}


def _format_results(result) -> tuple[str, str]:
    if result.alert:
        summary = f"**Alert:** `{result.alert}` detected ({result.method})."
    else:
        summary = "**No alert** — nothing crossed the threshold for fall, distress, or cough."

    lines = [
        summary,
        f"Clip length: **{result.duration_sec:.1f}s**",
        "",
        "### Target categories",
        "| Category | Score | Triggered | Top AudioSet match |",
        "|---|---:|:---:|---|",
    ]

    for cat in result.categories:
        title = CATEGORY_LABELS.get(cat.category, cat.category)
        top_match = cat.top_labels[0].label if cat.top_labels else "—"
        top_score = cat.top_labels[0].score if cat.top_labels else 0.0
        triggered = "yes" if cat.triggered else "no"
        lines.append(
            f"| {title} | **{cat.score:.2f}** | {triggered} | {top_match} ({top_score:.2f}) |"
        )

    lines.extend(["", "### Top overall AudioSet labels (foundation model)", ""])
    for item in result.top_overall:
        lines.append(f"- `{item.label}` — {item.score:.2f}")

    lines.extend(
        [
            "",
            "_POC note: scores come from Google's YAMNet (AudioSet). "
            "Fine-tuning on room audio will be needed for production accuracy._",
        ]
    )
    return summary.replace("**", ""), "\n".join(lines)


def analyze_audio(
    audio,
    denoise: bool,
    highpass: bool,
    threshold: float,
    mode: str,
):
    if audio is None:
        return "Upload or record audio first.", "No input."

    sr, waveform = audio
    if waveform is None or len(waveform) == 0:
        return "Empty audio clip.", "No input."

    waveform = waveform.astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    try:
        _, analyze = get_analyzer(mode)
    except FileNotFoundError as exc:
        return str(exc), "Classifier not available."

    result = analyze(
        waveform,
        int(sr),
        denoise=denoise,
        highpass=highpass,
        threshold=threshold if threshold > 0 else None,
    )
    return _format_results(result)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Care Audio POC") as demo:
        gr.Markdown(
            """
            # Care Audio POC
            Detect **falls**, **distress sounds**, and **coughing** using
            [YAMNet](https://tfhub.dev/google/yamnet/1) (pre-trained on AudioSet).

            Upload a `.wav`/`.mp3` file or record from your microphone, then click **Analyze**.
            Optional background noise filtering runs before inference.
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Audio input (upload or microphone)",
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                denoise = gr.Checkbox(value=True, label="Background noise reduction")
                highpass = gr.Checkbox(value=True, label="High-pass filter (remove rumble)")
                threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.60,
                    value=0.15,
                    step=0.01,
                    label="Alert threshold (0 = per-category defaults)",
                )
                mode = gr.Dropdown(
                    choices=["auto", "yamnet", "classifier"],
                    value="auto",
                    label="Detection mode",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Column():
                alert_box = gr.Textbox(label="Summary", lines=2)
                details = gr.Markdown(label="Details")

        analyze_btn.click(
            analyze_audio,
            inputs=[audio_input, denoise, highpass, threshold, mode],
            outputs=[alert_box, details],
        )

        gr.Markdown(
            """
            **Try it:** search YouTube for short clips like *person falling sound effect*,
            *cough sound*, or *distress cry* — download as audio and upload here.
            """
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1")
