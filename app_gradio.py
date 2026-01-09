import os
import time
import traceback
from pathlib import Path
import gradio as gr

from webui.schemas import (
    PRESETS,
    DEFAULT_MODEL_ROOT,
    discoverable_preset_names,
    get_preset,
    preset_details,
    refresh_discovered_presets,
)
from webui.manager import EngineManager
from webui.utils import gpu_info, has_spargeattn, check_paths

_output_dir = os.environ.get("OUTPUT_DIR")
OUT = Path(_output_dir) if _output_dir else (Path.cwd() / "outputs")
OUT.mkdir(parents=True, exist_ok=True)

MANAGER = EngineManager()
PRESET_CHOICES = discoverable_preset_names() or list(PRESETS.keys())
DEFAULT_PRESET = PRESET_CHOICES[0]


def _status_badge(loaded: bool, msg: str = ""):
    color = "🟢" if loaded else "⚪"
    return f"{color} **Model Loaded**: {loaded}\n\n{msg}".strip()


def load_model(preset_name):
    cfg = get_preset(preset_name)
    details = preset_details(preset_name)
    try:
        MANAGER.load(cfg)
    except Exception as exc:  # surface checkpoint errors
        return (
            _status_badge(False, f"❌ Load failed: {exc}"),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    info = gpu_info()
    return (
        _status_badge(True, f"Preset: `{preset_name}`\n{details}\nGPU: {info.get('name','-')}"),
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
    cfg = get_preset(preset_name)
    missing = check_paths(cfg)
    if missing:
        return "❌ Missing files:\n" + "\n".join(missing)
    return "✅ All checkpoint files exist."


def refresh_presets(current_generate_choice, current_models_choice):
    """重新扫描 checkpoints 并刷新预设列表"""
    # Rescan checkpoints so newly mounted/copied weights show up without restart.
    refresh_discovered_presets()
    names = discoverable_preset_names()
    selected_generate = current_generate_choice if current_generate_choice in names else names[0]
    selected_models = current_models_choice if current_models_choice in names else names[0]

    msg = (
        f"✅ discoverd {len(names)} presetconfig\n"
        f"📂 scanning path: {os.environ.get('MODEL_PATHS', DEFAULT_MODEL_ROOT)}"
    )

    return (
        gr.update(choices=names, value=selected_generate),
        gr.update(choices=names, value=selected_models),
        gr.update(value=msg, visible=True),
    )


def refresh_presets_with_feedback(current_generate_choice, current_models_choice):
    """带有加载反馈的预设刷新函数"""
    import time

    # 开始刷新提示
    yield (
        gr.update(),  # preset dropdown
        gr.update(),  # preset_m dropdown
        gr.update(value="🔄 scanning models...", visible=True),  # message
        gr.update(interactive=False),  # 禁用按钮
    )

    # 执行实际刷新
    time.sleep(0.3)  # 短暂延迟让用户看到加载状态
    result = refresh_presets(current_generate_choice, current_models_choice)

    # 完成刷新，恢复按钮状态
    yield (
        result[0],  # preset dropdown
        result[1],  # preset_m dropdown
        result[2],  # message
        gr.update(interactive=True),  # 恢复按钮
    )


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
    logs = []
    try:
        # ---------- pre-check ----------
        if not MANAGER.is_loaded() or MANAGER.cfg.name != preset_name:
            # auto load if not loaded or preset mismatch
            cfg = get_preset(preset_name)
            MANAGER.load(cfg)

        eng = MANAGER.engine
        eng.keep_dit_on_gpu = bool(keep_dit_on_gpu)
        eng.keep_text_encoder = bool(keep_text_encoder)

        # seed handling
        if seed_mode == "random":
            seed = int(time.time()) % 10_000_000
        else:
            seed = int(seed)

        # progress/log buffers
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
        out_path = Path(out_path)
        if out_path.is_dir() or not out_path.exists():
            err = f"❌ invalid output path: {out_path}"
            logs.append(err)
            status = err
            return "", status, "\n".join(logs[-200:]), {}

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
    except Exception as exc:  # capture inference failure
        tb = traceback.format_exc()
        if not logs:
            # ensure the textbox is populated even if the engine raises instantly
            logs.append("❌ No logs were produced before the failure.")
        logs.append(f"❌ Inference failed: {exc}")
        logs.append(tb)
        log_text = "\n".join(logs[-200:])
        status = f"❌ Error during inference: {exc}"
        return None, status, log_text, {}


def create_demo():
    with gr.Blocks(title="TurboDiffusion WebUI (Wan2.1 T2V)") as demo:
        gr.Markdown("# TurboDiffusion WebUI (Engine Mode)\n"
                    "✅ Model switch + SageSLA check  →  ✅ Progress & Logs  →  ✅ History & Download  →  ✅ Load/Unload & GPU stats"
)

        # global states
        history_state = gr.State([])   # list[dict]
        last_meta_state = gr.State({})

        with gr.Tabs():
            # ===================== Generate Tab =====================
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=4):
                        preset = gr.Dropdown(
                            choices=PRESET_CHOICES,
                            value=DEFAULT_PRESET,
                            label="Model Preset"
                        )
                        preset_info = gr.Markdown(preset_details(DEFAULT_PRESET))
                        discover_btn_generate = gr.Button("🔄 Discover models", variant="secondary")
                        discover_msg_generate = gr.Markdown(visible=False)
                        prompt = gr.Textbox(lines=3, label="Prompt", value="a cinematic shot of a tiger walking in snow")

                        with gr.Accordion("Basic", open=True):
                            num_steps = gr.Dropdown([1,2,3,4], value=4, label="Steps")
                            num_frames = gr.Slider(17, 500, value=81, step=1, label="Frames")
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
                    # update progress and stage
                    if meta:
                        stage = "**Stage:** done"
                        progress_pct = 100
                        history, rows = _update_history(meta, history)
                    else:
                        stage = "**Stage:** error"
                        progress_pct = 0
                        rows = [[
                            x["time"], x["preset"], x["seed"], x["steps"], x["frames"],
                            x["attn"], x["topk"], x["sec"], x["path"]
                        ] for x in history]
                    return (
                        stage, progress_pct,
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

                preset.change(
                    lambda name: preset_details(name), inputs=[preset], outputs=[preset_info]
                )

                run_evt.then(
                    fn=_after_gen,
                    inputs=[out_video, status_md, log_box, last_meta_state, history_state],
                    outputs=[stage_md, prog, status_md, log_box, history_state, history_df],
                )

            # ===================== Models Tab =====================
            with gr.Tab("Models"):
                with gr.Row():
                    with gr.Column(scale=3):
                        preset_m = gr.Dropdown(choices=PRESET_CHOICES, value=DEFAULT_PRESET, label="Preset")
                        preset_info_m = gr.Markdown(preset_details(DEFAULT_PRESET))
                        discover_btn_models = gr.Button("🔄 Refresh model list", variant="secondary")
                        discover_msg_models = gr.Markdown(visible=False)
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
                preset_m.change(lambda name: preset_details(name), inputs=[preset_m], outputs=[preset_info_m])
                refresh_btn.click(refresh_system, outputs=[gpu_json, sa_text])

                load_btn.click(load_model, inputs=[preset_m], outputs=[status_badge, load_btn, unload_btn], concurrency_id="gpu", concurrency_limit=1)
                unload_btn.click(unload_model, outputs=[status_badge, load_btn, unload_btn], concurrency_id="gpu", concurrency_limit=1)

                discover_btn_models.click(
                    refresh_presets_with_feedback,
                    inputs=[preset, preset_m],
                    outputs=[preset, preset_m, discover_msg_models, discover_btn_models],
                )
                discover_btn_generate.click(
                    refresh_presets_with_feedback,
                    inputs=[preset, preset_m],
                    outputs=[preset, preset_m, discover_msg_generate, discover_btn_generate],
                )
            # ===================== System Tab =====================
            with gr.Tab("System"):
                gr.Markdown("### Environment & Diagnostics")
                gpu_json2 = gr.JSON(label="GPU Info")
                sa_text2 = gr.Markdown("")
                refresh_btn2 = gr.Button("Refresh")
                refresh_btn2.click(refresh_system, outputs=[gpu_json2, sa_text2])

    # Global queue: safest for GPU workloads
    demo.queue(default_concurrency_limit=1)  # 1 by default anyway, but explicit
    return demo


def main():
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
