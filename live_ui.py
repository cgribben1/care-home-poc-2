"""Gradio UI for two-tier live acoustic monitoring."""

from __future__ import annotations

import gradio as gr

from two_tier_monitor import TwoTierMonitor

_monitor: TwoTierMonitor | None = None


def start_monitor(port: str):
    global _monitor
    if _monitor:
        _monitor.stop()
    _monitor = TwoTierMonitor(port=port.strip() or "COM5")
    _monitor.start()
    return _refresh()


def stop_monitor():
    global _monitor
    if _monitor:
        _monitor.stop()
    return _refresh()


def _refresh():
    if not _monitor:
        return (
            "Not running",
            "No",
            0.0,
            0.0,
            0.0,
            "—",
            "—",
            "—",
            0.0,
            0.0,
            0.0,
            "_No events yet._",
        )

    s = _monitor.get_state()
    if s.error:
        status = f"Error: {s.error}"
    else:
        status = s.status

    triggered = "YES — event!" if s.tier1_triggered else "No"
    alert = s.last_alert.upper() if s.last_alert else "None"

    scores = s.scores
    fall = scores.get("fall", 0.0)
    distress = scores.get("distress", 0.0)
    cough = scores.get("cough", 0.0)

    log_lines = []
    for ev in reversed(s.event_log):
        alert_txt = ev.alert.upper() if ev.alert else "none"
        log_lines.append(
            f"**{ev.timestamp}** — Tier 1: {ev.tier1_reason} (rms={ev.tier1_rms:.4f}) → "
            f"**Tier 2: {alert_txt}** | fall={ev.scores.get('fall', 0):.2f} "
            f"cough={ev.scores.get('cough', 0):.2f} distress={ev.scores.get('distress', 0):.2f}"
        )
    log_md = "\n\n".join(log_lines) if log_lines else "_No events yet._"

    return (
        status,
        triggered,
        s.tier1_rms,
        s.tier1_baseline,
        s.tier1_peak,
        s.tier1_reason,
        s.tier2_status,
        alert,
        fall,
        distress,
        cough,
        log_md,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Care Audio — Two-Tier Monitor") as demo:
        gr.Markdown(
            """
            # Two-tier acoustic monitor

            **Tier 1** — lightweight energy listener (always on, no heavy ML)  
            **Tier 2** — full hybrid model (YAMNet + classifier) runs **only** when Tier 1 detects something

            *Development setup: both tiers run on the PC while the XIAO streams audio.
            In production, Tier 1 can move to the ESP32 firmware; Tier 2 can run on-device (TFLite)
            or on a gateway — the PC is not required long-term.*
            """
        )

        with gr.Row():
            port = gr.Textbox(value="COM5", label="Serial port", scale=1)
            start_btn = gr.Button("Start monitoring", variant="primary", scale=1)
            stop_btn = gr.Button("Stop", scale=1)

        status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Tier 1 — always listening")
        with gr.Row():
            t1_rms = gr.Number(label="Current RMS", precision=4)
            t1_baseline = gr.Number(label="Baseline RMS", precision=4)
            t1_peak = gr.Number(label="Peak", precision=4)
            t1_triggered = gr.Textbox(label="Event suspected?", interactive=False)
        t1_reason = gr.Textbox(label="Tier 1 reason", interactive=False)

        gr.Markdown("### Tier 2 — full model (on demand)")
        t2_status = gr.Textbox(label="Tier 2 result", interactive=False)
        last_alert = gr.Textbox(label="Last alert", interactive=False)
        with gr.Row():
            score_fall = gr.Slider(0, 1, label="Fall confidence", interactive=False)
            score_distress = gr.Slider(0, 1, label="Distress confidence", interactive=False)
            score_cough = gr.Slider(0, 1, label="Cough confidence", interactive=False)

        gr.Markdown("### Event log")
        event_log = gr.Markdown("_No events yet._")

        outputs = [
            status,
            t1_triggered,
            t1_rms,
            t1_baseline,
            t1_peak,
            t1_reason,
            t2_status,
            last_alert,
            score_fall,
            score_distress,
            score_cough,
            event_log,
        ]

        start_btn.click(start_monitor, inputs=[port], outputs=outputs)
        stop_btn.click(stop_monitor, outputs=outputs)
        demo.load(_refresh, outputs=outputs)
        gr.Timer(0.5).tick(_refresh, outputs=outputs)

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1")
