import time
from pathlib import Path
import gradio as gr

from webui.schemas import PRESETS
from webui.manager import EngineManager
from webui.utils import gpu_info, has_spargeattn, check_paths

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MANAGER = EngineManager()

def _status_badge(loaded: bool, msg: str = ""):
    color = "🟢" if loaded else "⚪"
    return f"{color} **Model Loaded**: {loaded}\n\n{msg}".strip()

def load_model(preset_name):
    cfg = PRESETS[preset_name]
    MANAGER.load(cfg)
    info = gpu_info()
    return (
        _status_badge(True, f"Preset: `{preset_name}`\nGPU: {info.get('name','-')}"),
        gr.update(interactive=False),  # disable load while loaded
        gr.update(interactive=True),   # enable unload
    )

def unload_model():
    MANAGER.unload()
    return (
        _status_badge(False, "Unloaded"),
        gr.update(interactive=True),
        gr.update(interactive=False),
    )

def refresh_system():
    info = gpu_info()
    sa = has_spargeattn()
    return info, ("✅ SpargeAttn installed (sagesla available)" if sa else "❌ SpargeAttn not installed (sagesla disabled)")

def validate_paths(preset_name):
    cfg = PRESETS[preset_name]
    missing = check_paths(cfg)
    if missing:
        return "❌ Missing files:\n" + "\n".join(missing)
    return "✅ All checkpoint files exist."

def generate_video(
    preset_name,
    prompt,
    num_steps,
    num_frames,
    num_samples,
    seed_mode,
    seed,
    attention_type,
    sla_topk,
    sigma_max,
    fps,
    keep_dit_on_gpu,
    keep_text_encoder,
    default_norm,
):
    # ---------- pre-check ----------
    if not MANAGER.is_loaded() or MANAGER.cfg.name != preset_name:
        # auto load if not loaded or preset mismatch
        cfg = PRESETS[preset_name]
        MANAGER.load(cfg)

    eng = MANAGER.engine

    # seed handling
    if seed_mode == "random":
        seed = int(time.time()) % 10_000_000
    else:
        seed = int(seed)

    # progress/log buffers
    logs = []
    stage_text = {"text": "starting"}
    progress = {"cur": 0, "total": 1}

    def log_cb(msg):
        logs.append(msg)

    def progress_cb(stage, cur, total):
        stage_text["text"] = stage
        progress["cur"] = cur
        progress["total"] = total

    # file naming
    ts = time.strftime("%Y%m%d-%H%M%S")
    save_path = OUT / f"t2v_{preset_name.replace(' ','_')}_{ts}_seed{seed}.mp4"

    # run inference
    t0 = time.time()
    out_path = eng.generate(
        prompt=prompt,
        num_steps=int(num_steps),
        num_frames=int(num_frames),
        seed=int(seed),
        num_samples=int(num_samples),
        attention_type=attention_type,
        sla_topk=float(sla_topk),
        sigma_max=float(sigma_max),
        default_norm=bool(default_norm),
        save_path=str(save_path),
        fps=int(fps),
        progress_cb=progress_cb,
        log_cb=log_cb,
    )
    t1 = time.time()

    # history row
    meta = {
        "time": ts,
        "preset": preset_name,
        "seed": seed,
        "steps": int(num_steps),
        "frames": int(num_frames),
        "attn": attention_type,
        "topk": float(sla_topk),
        "path": str(out_path),
        "sec": round(t1 - t0, 2),
    }

    # outputs:
    # - video player expects file path
    # - file download uses gr.File
    # - status markdown
    status = f"✅ Done in **{meta['sec']}s** | seed={seed} | saved: `{out_path}`"
    log_text = "\n".join(logs[-200:])  # keep last 200 lines

    return str(out_path), status, log_text, meta

with gr.Blocks(title="TurboDiffusion WebUI (Wan2.1 T2V)") as demo:
    gr.Markdown("# TurboDiffusion WebUI (Engine Mode)\n"
                "✅ Model switch + SageSLA check  →  ✅ Progress & Logs  →  ✅ History & Download  →  ✅ Load/Unload & GPU stats")

    # global states
    history_state = gr.State([])   # list[dict]
    last_meta_state = gr.State({})

    with gr.Tabs():
        # ===================== Generate Tab =====================
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=4):
                    preset = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value=list(PRESETS.keys())[0],
                        label="Model Preset"
                    )
                    prompt = gr.Textbox(lines=3, label="Prompt", value="a cinematic shot of a tiger walking in snow")

                    with gr.Accordion("Basic", open=True):
                        num_steps = gr.Dropdown([1,2,3,4], value=4, label="Steps")
                        num_frames = gr.Slider(17, 81, value=81, step=1, label="Frames")
                        num_samples = gr.Slider(1, 4, value=1, step=1, label="Num Samples")

                        seed_mode = gr.Radio(["fixed", "random"], value="fixed", label="Seed Mode")
                        seed = gr.Number(value=0, precision=0, label="Seed (fixed mode)")

                    with gr.Accordion("Quality & Speed", open=True):
                        # SageSLA may be disabled if SpargeAttn missing
                        attention_type = gr.Dropdown(["sla", "sagesla", "original"], value="sla", label="Attention Type")
                        sla_topk = gr.Slider(0.05, 0.20, value=0.10, step=0.01, label="SLA top-k (sla/sagesla)")
                        sigma_max = gr.Slider(10, 120, value=80, step=1, label="sigma_max")
                        default_norm = gr.Checkbox(value=False, label="default_norm (faster norm)")

                    with gr.Accordion("Output", open=False):
                        fps = gr.Slider(8, 30, value=16, step=1, label="FPS")

                    with gr.Accordion("Advanced", open=False):
                        keep_dit_on_gpu = gr.Checkbox(value=True, label="Keep DiT on GPU (recommended)")
                        keep_text_encoder = gr.Checkbox(value=False, label="Keep UMT5 encoder (if you modify umt5 cache)")

                    run_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=5):
                    stage_md = gr.Markdown("**Stage:** idle")
                    prog = gr.Slider(0, 100, value=0, step=1, label="Progress (%)", interactive=False)
                    status_md = gr.Markdown("")

                    out_video = gr.Video(label="Output Video", interactive=False)

                    log_box = gr.Textbox(label="Logs", lines=18, interactive=False)
                    gr.Markdown("### History")
                    history_df = gr.Dataframe(
                        headers=["time","preset","seed","steps","frames","attn","topk","sec","path"],
                        datatype=["str","str","number","number","number","str","number","number","str"],
                        interactive=False,
                        row_count=10,
                        col_count=(9, "fixed"),
                    )

            def _update_history(meta, history):
                history = [meta] + history
                history = history[:20]
                rows = [[
                    x["time"], x["preset"], x["seed"], x["steps"], x["frames"],
                    x["attn"], x["topk"], x["sec"], x["path"]
                ] for x in history]
                return history, rows

            def _after_gen(video_path, status, logs, meta, history):
                # update progress and stage to done
                stage = "**Stage:** done"
                progress_pct = 100
                history, rows = _update_history(meta, history)
                return (
                    stage, progress_pct,
                    video_path,
                    status, logs,
                    history, rows
                )

            # main click
            run_evt = run_btn.click(
                fn=generate_video,
                inputs=[
                    preset, prompt, num_steps, num_frames, num_samples,
                    seed_mode, seed,
                    attention_type, sla_topk, sigma_max,
                    fps,
                    keep_dit_on_gpu, keep_text_encoder,
                    default_norm
                ],
                outputs=[out_video, status_md, log_box, last_meta_state],
                concurrency_id="gpu",
                concurrency_limit=1,
            )

            run_evt.then(
                fn=_after_gen,
                inputs=[out_video, status_md, log_box, last_meta_state, history_state],
                outputs=[stage_md, prog, out_video, status_md, log_box, history_state, history_df],
            )

        # ===================== Models Tab =====================
        with gr.Tab("Models"):
            with gr.Row():
                with gr.Column(scale=3):
                    preset_m = gr.Dropdown(choices=list(PRESETS.keys()), value=list(PRESETS.keys())[0], label="Preset")
                    validate_btn = gr.Button("Validate Checkpoints")
                    validate_out = gr.Textbox(label="Checkpoint Status", lines=8, interactive=False)

                    load_btn = gr.Button("Load Model", variant="primary")
                    unload_btn = gr.Button("Unload Model", interactive=False)

                with gr.Column(scale=4):
                    status_badge = gr.Markdown(_status_badge(False))
                    sa_text = gr.Markdown("")
                    gpu_json = gr.JSON(label="GPU Info")

                    refresh_btn = gr.Button("Refresh System Info")

            validate_btn.click(validate_paths, inputs=[preset_m], outputs=[validate_out])
            refresh_btn.click(refresh_system, outputs=[gpu_json, sa_text])

            load_btn.click(load_model, inputs=[preset_m], outputs=[status_badge, load_btn, unload_btn], concurrency_id="gpu", concurrency_limit=1)
            unload_btn.click(unload_model, outputs=[status_badge, load_btn, unload_btn], concurrency_id="gpu", concurrency_limit=1)
        # ===================== System Tab =====================
        with gr.Tab("System"):
            gr.Markdown("### Environment & Diagnostics")
            gpu_json2 = gr.JSON(label="GPU Info")
            sa_text2 = gr.Markdown("")
            refresh_btn2 = gr.Button("Refresh")
            refresh_btn2.click(refresh_system, outputs=[gpu_json2, sa_text2])

    # Global queue: safest for GPU workloads
    demo.queue(default_concurrency_limit=1)  # 1 by default anyway, but explicit

demo.launch(server_name="0.0.0.0", server_port=7860)
