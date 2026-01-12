import os
import threading
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
    if str(getattr(cfg, "model", "") or "").startswith("Wan2.2") and not getattr(cfg, "dit_path_high", None):
        missing = [*missing, "dit_path_high (Wan2.2 I2V requires high-noise checkpoint)"]
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


def _progress_percent(stage: str, cur: int, total: int) -> int:
    weights = {
        "embedding": 0.05,
        "setup": 0.05,
        "sampling": 0.80,
        "decode": 0.05,
        "save": 0.05,
    }
    order = ["embedding", "setup", "sampling", "decode", "save"]
    base = 0.0
    if stage in order:
        for item in order:
            if item == stage:
                break
            base += weights[item]
    stage_weight = weights.get(stage, 0.0)
    if not total:
        frac = 0.0
    else:
        frac = float(cur) / float(total)
    pct = int(round((base + stage_weight * frac) * 100))
    return max(0, min(100, pct))


def _format_stage(stage: str, cur: int, total: int) -> str:
    if not stage:
        return "**Stage:** idle"
    if total and total > 1:
        return f"**Stage:** {stage} ({cur}/{total})"
    return f"**Stage:** {stage}"


def _timer_html(elapsed: float, finished: bool = False) -> str:
    label = "Took" if finished else "Elapsed"
    state = "done" if finished else "running"
    return (
        f'<div class="timer {state}">'
        f'<span class="dot"></span>'
        f'<span class="label">{label}</span>'
        f'<span class="time">{elapsed:.1f}s</span>'
        "</div>"
    )


def _generate_video_blocking(
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
    init_image=None,
    i2v_adaptive_resolution: bool = True,
    i2v_boundary: float = 0.9,
    i2v_ode: bool = True,
    progress_cb=None,
    log_cb=None,
):
    logs = []

    def _log(msg):
        logs.append(msg)
        if log_cb:
            log_cb(msg)

    def _progress(stage, cur, total):
        if progress_cb:
            progress_cb(stage, cur, total)

    try:
        cfg = get_preset(preset_name)
        model_name = str(getattr(cfg, "model", "") or "")
        is_i2v = model_name.startswith("Wan2.2")

        if is_i2v and init_image is None:
            status = "I2V preset selected. Please upload an init image."
            _log(status)
            return None, status, "\n".join(logs[-200:]), {}

        # ---------- load (auto reload if load-time options changed) ----------
        MANAGER.load(
            cfg,
            attention_type=attention_type,
            sla_topk=float(sla_topk),
            default_norm=bool(default_norm),
        )

        eng = MANAGER.engine
        if hasattr(eng, "keep_dit_on_gpu"):
            eng.keep_dit_on_gpu = bool(keep_dit_on_gpu)
        if hasattr(eng, "keep_text_encoder"):
            eng.keep_text_encoder = bool(keep_text_encoder)

        # seed handling
        if seed_mode == "random":
            seed = int(time.time()) % 10_000_000
        else:
            seed = int(seed)

        # file naming
        ts = time.strftime("%Y%m%d-%H%M%S")
        mode = "i2v" if is_i2v else "t2v"
        save_path = OUT / f"{mode}_{preset_name.replace(' ','_')}_{ts}_seed{seed}.mp4"

        # run inference
        t0 = time.time()
        if is_i2v:
            out_path = eng.generate(
                prompt=prompt,
                init_image=init_image,
                num_steps=int(num_steps),
                num_frames=int(num_frames),
                seed=int(seed),
                num_samples=int(num_samples),
                adaptive_resolution=bool(i2v_adaptive_resolution),
                boundary=float(i2v_boundary),
                ode=bool(i2v_ode),
                sigma_max=float(sigma_max),
                save_path=str(save_path),
                fps=int(fps),
                progress_cb=_progress,
                log_cb=_log,
            )
        else:
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
                progress_cb=_progress,
                log_cb=_log,
            )
        out_path = Path(out_path)
        if out_path.is_dir() or not out_path.exists():
            err = f"❌ invalid output path: {out_path}"
            _log(err)
            status = err
            return None, status, "\n".join(logs[-200:]), {}

        t1 = time.time()

        # history row
        meta = {
            "time": ts,
            "preset": preset_name,
            "mode": mode,
            "seed": seed,
            "steps": int(num_steps),
            "frames": int(num_frames),
            "attn": attention_type,
            "topk": float(sla_topk),
            "path": str(out_path),
            "sec": round(t1 - t0, 2),
        }
        if is_i2v:
            meta["adaptive_resolution"] = bool(i2v_adaptive_resolution)
            meta["boundary"] = float(i2v_boundary)
            meta["ode"] = bool(i2v_ode)

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
            _log("No logs were produced before the failure.")
        _log(f"Inference failed: {exc}")
        _log(tb)
        log_text = "\n".join(logs[-200:])
        status = f"Error during inference: {exc}"
        return None, status, log_text, {}


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
    init_image=None,
    i2v_adaptive_resolution: bool = True,
    i2v_boundary: float = 0.9,
    i2v_ode: bool = True,
):
    logs = []
    stage_state = {"stage": "starting", "cur": 0, "total": 1}
    result = {"video": None, "status": "", "log_text": "", "meta": {}}
    done = threading.Event()
    lock = threading.Lock()
    start_t = time.time()

    def log_cb(msg):
        with lock:
            logs.append(msg)

    def progress_cb(stage, cur, total):
        with lock:
            stage_state["stage"] = stage
            stage_state["cur"] = cur
            stage_state["total"] = total

    def _run():
        try:
            video, status, log_text, meta = _generate_video_blocking(
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
                init_image,
                i2v_adaptive_resolution,
                i2v_boundary,
                i2v_ode,
                progress_cb=progress_cb,
                log_cb=log_cb,
            )
            result["video"] = video
            result["status"] = status
            result["log_text"] = log_text
            result["meta"] = meta
        except Exception as exc:
            err = f"Error during inference: {exc}"
            with lock:
                logs.append(err)
            result["video"] = None
            result["status"] = err
            result["log_text"] = "\n".join(logs[-200:])
            result["meta"] = {}
        finally:
            done.set()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    last_snapshot = None
    while not done.is_set():
        with lock:
            stage = stage_state["stage"]
            cur = stage_state["cur"]
            total = stage_state["total"]
            log_text = "\n".join(logs[-200:])
        elapsed = time.time() - start_t
        timer_html = _timer_html(elapsed, finished=False)
        snapshot = (stage, cur, total, len(logs), int(elapsed * 10))
        progress_pct = _progress_percent(stage, cur, total)
        stage_md = _format_stage(stage, cur, total)
        if stage and total and total > 1:
            status = f"Running: {stage} ({cur}/{total})"
        elif stage:
            status = f"Running: {stage}"
        else:
            status = "Running..."
        if snapshot != last_snapshot:
            yield (
                status,
                log_text,
                None,
                stage_md,
                progress_pct,
                timer_html,
            )
            last_snapshot = snapshot
        time.sleep(0.2)

    final_log_text = result["log_text"] or "\n".join(logs[-200:])
    meta = result["meta"]
    if meta:
        final_stage = "**Stage:** done"
        final_progress = 100
    else:
        final_stage = "**Stage:** error"
        final_progress = 0
    total_time = time.time() - start_t
    final_timer_html = _timer_html(total_time, finished=True)
    final_status = result["status"]
    if final_status:
        final_status = f"{final_status} | took {total_time:.1f}s"
    else:
        final_status = f"Finished | took {total_time:.1f}s"
    yield (
        final_status,
        final_log_text,
        meta,
        final_stage,
        final_progress,
        final_timer_html,
    )


def create_demo():
    with gr.Blocks(
        title="TurboDiffusion WebUI (Wan2.x T2V / I2V)",
        css="""
        .gradio-container .loading,
        .gradio-container .loading *,
        .gradio-container .block.loading,
        .gradio-container .block:has(.loading) {
          border-color: transparent !important;
          box-shadow: none !important;
          outline: none !important;
        }
        .gradio-container {
          --color-accent: #e5e7eb !important;
          --color-accent-soft: #e5e7eb !important;
          --color-accent-hover: #e5e7eb !important;
          --shadow-1: none !important;
          --shadow-2: none !important;
          --shadow-3: none !important;
        }
        #main-tabs {
          --color-accent: #f97316 !important;
          --color-accent-soft: rgba(249, 115, 22, 0.25) !important;
          --color-accent-hover: #fb923c !important;
        }
        .gradio-container .block,
        .gradio-container .block * {
          --tw-ring-color: transparent !important;
          --tw-ring-shadow: none !important;
          box-shadow: none !important;
          outline: none !important;
        }
        .gradio-container .highlight,
        .gradio-container .gradio-highlight,
        .gradio-container .block.highlight,
        .gradio-container [class*="highlight"] {
          border-color: var(--neutral-200, #e5e7eb) !important;
          box-shadow: none !important;
          outline: none !important;
          --tw-ring-color: transparent !important;
          --tw-ring-shadow: none !important;
        }
        .gradio-container .block:focus-within,
        .gradio-container .block:has(input:focus),
        .gradio-container .block:has(textarea:focus),
        .gradio-container .block:has(.focus) {
          border-color: var(--neutral-200, #e5e7eb) !important;
          box-shadow: none !important;
          outline: none !important;
        }
        .gradio-container [class*="ring-orange"],
        .gradio-container [class*="ring-primary"],
        .gradio-container [class*="border-orange"],
        .gradio-container [class*="border-primary"],
        .gradio-container [class*="outline-orange"],
        .gradio-container [class*="outline-primary"] {
          border-color: var(--neutral-200, #e5e7eb) !important;
          outline-color: var(--neutral-200, #e5e7eb) !important;
          --tw-ring-color: transparent !important;
          --tw-ring-shadow: none !important;
          box-shadow: none !important;
        }
        .gradio-container .gradio-slider,
        .gradio-container .gradio-slider .wrap,
        .gradio-container .gradio-slider input[type="range"],
        .gradio-container .gradio-textbox,
        .gradio-container .gradio-textbox .wrap,
        .gradio-container .gradio-textbox textarea,
        .gradio-container .gradio-textbox input {
          border-color: var(--neutral-200, #e5e7eb) !important;
          box-shadow: none !important;
          outline: none !important;
        }
        .gradio-container .block.output,
        .gradio-container .block.output .wrap,
        .gradio-container .block.output textarea,
        .gradio-container .block.output input {
          border-color: var(--neutral-200, #e5e7eb) !important;
          box-shadow: none !important;
          outline: none !important;
          --tw-ring-color: transparent !important;
          --tw-ring-shadow: none !important;
        }
        #output-panel {
          position: relative;
        }
        #video-timer {
          position: absolute;
          top: 10px;
          right: 12px;
          z-index: 5;
          pointer-events: none;
        }
        #video-timer .timer {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(249, 115, 22, 0.55);
          color: #f97316;
          background: rgba(0, 0, 0, 0.55);
          font-size: 12px;
          letter-spacing: 0.3px;
          backdrop-filter: blur(4px);
          position: relative;
          overflow: hidden;
        }
        #video-timer .timer::after {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(120deg, transparent, rgba(249, 115, 22, 0.22), transparent);
          animation: timer-sweep 2.2s linear infinite;
          opacity: 0.8;
        }
        #video-timer .timer.done::after {
          animation: none;
          opacity: 0.0;
        }
        #video-timer .dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #f97316;
          box-shadow: 0 0 6px rgba(249, 115, 22, 0.7);
          animation: timer-pulse 1.2s ease-in-out infinite;
          flex: 0 0 6px;
        }
        #video-timer .timer.done .dot {
          animation: none;
          box-shadow: none;
        }
        @keyframes timer-sweep {
          0% { transform: translateX(-120%); }
          100% { transform: translateX(120%); }
        }
        @keyframes timer-pulse {
          0% { transform: scale(0.9); opacity: 0.7; }
          50% { transform: scale(1.2); opacity: 1; }
          100% { transform: scale(0.9); opacity: 0.7; }
        }
        """,
    ) as demo:
        gr.Markdown("# TurboDiffusion WebUI (Engine Mode)\n"
                    "✅ Model switch + SageSLA check  →  ✅ Progress & Logs  →  ✅ History & Download  →  ✅ Load/Unload & GPU stats"
)

        # global states
        history_state = gr.State([])   # list[dict]
        last_meta_state = gr.State({})

        with gr.Tabs(elem_id="main-tabs"):
            # ===================== Generate Tab =====================
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=4):
                        default_is_i2v = "WAN2.2" in DEFAULT_PRESET.upper()
                        preset = gr.Dropdown(
                            choices=PRESET_CHOICES,
                            value=DEFAULT_PRESET,
                            label="Model Preset"
                        )
                        preset_info = gr.Markdown(preset_details(DEFAULT_PRESET))
                        discover_btn_generate = gr.Button("🔄 Discover models", variant="secondary")
                        discover_msg_generate = gr.Markdown(visible=False)
                        prompt = gr.Textbox(lines=3, label="Prompt", value="a cinematic shot of a tiger walking in snow")
                        init_image = gr.Image(
                            label="Upload Image (I2V)",
                            type="numpy",
                            sources=["upload"],
                            visible=default_is_i2v,
                        )
                        init_image_hint = gr.Markdown(
                            "Upload an image to enable I2V.",
                            visible=default_is_i2v,
                        )
                        run_btn = gr.Button("Generate", variant="primary")

                        with gr.Accordion("Basic", open=True):
                            num_steps = gr.Dropdown([1,2,3,4], value=4, label="Steps")
                            num_frames = gr.Slider(17, 500, value=81, step=1, label="Frames")
                            num_samples = gr.Slider(1, 4, value=1, step=1, label="Num Samples")

                            seed_mode = gr.Radio(["fixed", "random"], value="fixed", label="Seed Mode")
                            seed = gr.Number(value=0, precision=0, label="Seed (fixed mode)")

                        with gr.Accordion("Quality & Speed", open=True):
                            attention_type = gr.Dropdown(
                                ["sla", "original"],
                                value=("sla"),
                                label="Attention Type",
                            )
                            sla_topk = gr.Slider(0.05, 0.20, value=0.10, step=0.01, label="SLA top-k ")
                            sigma_max = gr.Slider(
                                10,
                                200,
                                value=(200 if default_is_i2v else 80),
                                step=1,
                                label="sigma_max",
                            )
                            default_norm = gr.Checkbox(value=False, label="default_norm (faster norm)")
                            i2v_adaptive_resolution = gr.Checkbox(
                                value=True,
                                label="I2V adaptive resolution (match init image aspect)",
                                visible=default_is_i2v,
                            )
                            i2v_boundary = gr.Slider(
                                0.0,
                                1.0,
                                value=0.9,
                                step=0.01,
                                label="I2V boundary (high→low noise switch)",
                                visible=default_is_i2v,
                            )
                            i2v_ode = gr.Checkbox(
                                value=True,
                                label="I2V ODE sampling (sharper, less robust)",
                                visible=default_is_i2v,
                            )

                        with gr.Accordion("Output", open=False):
                            fps = gr.Slider(8, 30, value=16, step=1, label="FPS")

                        with gr.Accordion("Advanced", open=False):
                            keep_dit_on_gpu = gr.Checkbox(value=True, label="Keep DiT on GPU (recommended)")
                            keep_text_encoder = gr.Checkbox(value=False, label="Keep UMT5 encoder (if you modify umt5 cache)")

                    with gr.Column(scale=5):
                        stage_md = gr.Markdown("**Stage:** idle")
                        prog = gr.Slider(0, 100, value=0, step=1, label="Progress (%)", interactive=False)
                        status_md = gr.Markdown("")

                        with gr.Group(elem_id="output-panel"):
                            out_video = gr.Video(label="Output Video", interactive=False)
                            video_timer = gr.HTML("", elem_id="video-timer")

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

                def _after_gen(status, logs, meta, history):
                    # update progress and stage
                    if meta:
                        stage = "**Stage:** done"
                        progress_pct = 100
                        history, rows = _update_history(meta, history)
                        video_path = meta.get("path")
                    else:
                        stage = "**Stage:** error"
                        progress_pct = 0
                        rows = [[
                            x["time"], x["preset"], x["seed"], x["steps"], x["frames"],
                            x["attn"], x["topk"], x["sec"], x["path"]
                        ] for x in history]
                        video_path = None
                    return (
                        stage, progress_pct,
                        status, logs,
                        history, rows,
                        video_path,
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
                        default_norm,
                        init_image,
                        i2v_adaptive_resolution,
                        i2v_boundary,
                        i2v_ode,
                    ],
                    outputs=[status_md, log_box, last_meta_state, stage_md, prog, video_timer],
                    concurrency_id="gpu",
                    concurrency_limit=1,
                )

                def _on_preset_change(name: str):
                    is_i2v = "WAN2.2" in (name or "").upper()
                    img_update = gr.update(visible=True) if is_i2v else gr.update(visible=False, value=None)
                    hint_update = gr.update(visible=is_i2v)
                    adaptive_update = gr.update(visible=is_i2v, value=True if is_i2v else False)
                    boundary_update = gr.update(visible=is_i2v, value=0.9)
                    ode_update = gr.update(visible=is_i2v, value=True if is_i2v else False)
                    sigma_update = gr.update(value=200 if is_i2v else 80)
                    attn_update = gr.update(value="sla")
                    return (
                        preset_details(name),
                        img_update,
                        hint_update,
                        adaptive_update,
                        boundary_update,
                        ode_update,
                        sigma_update,
                        attn_update,
                    )

                preset.change(
                    _on_preset_change,
                    inputs=[preset],
                    outputs=[
                        preset_info,
                        init_image,
                        init_image_hint,
                        i2v_adaptive_resolution,
                        i2v_boundary,
                        i2v_ode,
                        sigma_max,
                        attention_type,
                    ],
                )

                run_evt.then(
                    fn=_after_gen,
                    inputs=[status_md, log_box, last_meta_state, history_state],
                    outputs=[stage_md, prog, status_md, log_box, history_state, history_df, out_video],
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
