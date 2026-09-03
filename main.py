"""
Combined Dehazing GUI — ICEML (RtxICE-Net) + WLD-Net + YOLO Object Detection
Run: source /path/to/dehaze_env/bin/activate && python main.py
"""

# ── Pure-stdlib / tkinter imports only at module level ────────────────────────
# Heavy C-extension libraries (cv2, numpy, torch, PIL) are imported LAZILY
# inside methods to prevent ARM/Jetson segfaults before Tk is fully running.
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import sys
import threading
import time
from pathlib import Path

# ── Sub-package paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "iceml_src"))
sys.path.insert(0, str(BASE_DIR / "wldnet_src"))

# ── Model weight paths ────────────────────────────────────────────────────────
ICEML_MODEL_PATH = str(BASE_DIR / "models" / "iceml" / "final_dehazing_model.pth")
WLD_MODEL_PATHS = {
    "O-HAZE":  str(BASE_DIR / "models" / "wldnet" / "OH-dehazing_model_final.pth"),
    "RESIDE":  str(BASE_DIR / "models" / "wldnet" / "RD_dehazing_model_final.pth"),
    "NH-HAZE": str(BASE_DIR / "models" / "wldnet" / "NH_dehazing_model_final.pth"),
    "D-HAZE":  str(BASE_DIR / "models" / "wldnet" / "DH_dehazing_model_final.pth"),
}
# YOLO detection: TensorRT engine per resolution, falls back to PyTorch .pt
YOLO_PT_PATH = str(BASE_DIR / "detection_model" / "fyp_test3_best.pt")
YOLO_TRT_DIR = BASE_DIR / "detection_model_TRT"

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#0d0d1a"
PANEL   = "#13132b"
ACCENT  = "#1e1e40"
CYAN    = "#00d4ff"
GREEN   = "#00e676"
RED     = "#ff1744"
ORANGE  = "#ff9100"
TEXT    = "#e0e0e0"
SUBTEXT = "#888899"
FONT    = "DejaVu Sans"


def _lighten(hex_color):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    r, g, b = min(255, r + 40), min(255, g + 40), min(255, b + 40)
    return f"#{r:02x}{g:02x}{b:02x}"


def styled_btn(parent, text, cmd, color=CYAN, **kw):
    # Force a uniform CYAN color scheme to prevent memory corruption/stack smashing
    # in the underling GTK/X11 theme engine on ARM/Jetson Orin Nano when allocating
    # multiple distinct colors/graphics-contexts for Buttons.
    color = CYAN
    import hashlib
    safe_name = "btn_" + hashlib.md5(text.encode()).hexdigest()[:10]
    b = tk.Button(parent, name=safe_name, text=text, command=cmd,
                  bg=color, fg="#0d0d1a", font=(FONT, 10, "bold"),
                  relief="flat",
                  activebackground=_lighten(color), activeforeground="#0d0d1a",
                  bd=0, padx=10, pady=6, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=_lighten(color)))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b


def separator(parent):
    """Horizontal rule — uses Canvas instead of Frame to avoid a Tcl/Tk
    widget-name-counter overflow that causes a segfault on ARM (Jetson)."""
    c = tk.Canvas(parent, height=1, bg=PANEL, bd=0, highlightthickness=0)
    c.pack(fill="x", padx=8, pady=6)
    c.create_line(0, 0, 2000, 0, fill=ACCENT, width=1)


def section_label(parent, text):
    tk.Label(parent, text=text, bg=PANEL, fg=CYAN,
             font=(FONT, 10, "bold")).pack(pady=(8, 2))


# ─────────────────────────────────────────────────────────────────────────────

class DehazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dehazing Suite — ICEML + WLD-Net + Object Detection")
        self.root.configure(bg=BG)
        self.root.geometry("1600x820")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # State
        self.device        = None
        self.model         = None
        self.loaded_engine = None
        self.yolo_model    = None
        self.loaded_yolo_res = None
        self.live_running  = False
        self.cap           = None
        self._latest_frame = None            # newest frame from the grabber thread
        self._frame_lock   = threading.Lock()
        self.recording     = False
        self.vw_orig       = None
        self.vw_dehazed    = None
        self.stop_video    = False
        self.video_path    = None
        self.output_video_path = None

        # Defer UI build until AFTER mainloop() starts — prevents ARM/Jetson segfault
        self.root.after(1, self._build_ui)

    # ── YOLO ──────────────────────────────────────────────────────────────────
    def _load_yolo(self):
        threading.Thread(target=self._ensure_yolo, daemon=True).start()

    def _ensure_yolo(self):
        """(Re)load the YOLO model for the currently selected resolution, preferring
        a matching TensorRT engine and falling back to the PyTorch .pt weights."""
        res = self.res_var.get()
        if self.yolo_model is not None and self.loaded_yolo_res == res:
            return True
        try:
            from ultralytics import YOLO
            eng_path = YOLO_TRT_DIR / f"yolo_fp16_{res}.engine"
            if eng_path.exists():
                self.status(f"Loading YOLO TensorRT engine ({res}px) …")
                self.yolo_model = YOLO(str(eng_path))
                self.status(f"YOLO TensorRT engine loaded ({res}px).")
            else:
                self.status(f"No YOLO TensorRT engine for {res}px, loading PyTorch …")
                self.yolo_model = YOLO(YOLO_PT_PATH)
                self.status("YOLO model loaded (PyTorch).")
            self.loaded_yolo_res = res
            return True
        except Exception as e:
            self.status(f"YOLO load failed: {e}", error=True)
            return False

    def _detect(self, rgb_np, conf=0.4):
        """Run YOLO on an RGB numpy array, return annotated RGB array."""
        if not self.od_var.get():
            return rgb_np
        if not self._ensure_yolo():
            return rgb_np
        import cv2
        cyan_rgb = (0, 212, 255)  # CYAN, in the same RGB channel order as rgb_np
        annotated = rgb_np.copy()
        # imgsz MUST match the resolution the engine was built at — the TRT engines
        # are static-shape (convert_all_trt.py exports with imgsz=res). Without this,
        # ultralytics defaults to imgsz=640 and letterboxes every frame to 640x640,
        # so the 256px engine cost exactly the same as the 512px one.
        results = self.yolo_model(rgb_np, conf=conf, verbose=False,
                                  imgsz=int(self.res_var.get()))
        for r in results:
            for box, score, cls in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                label = self.yolo_model.names[int(cls)]
                text = f"{label} {score:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), cyan_rgb, 2)
                cv2.rectangle(annotated, (x1, y1), (x1 + len(text) * 9 + 6, y1 + 18), cyan_rgb, -1)
                cv2.putText(annotated, text, (x1 + 3, y1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return annotated

    # ── Model loading ─────────────────────────────────────────────────────────
    def _ensure_model(self):
        engine = self.engine_var.get()
        key = "iceml" if engine == "ICEML (RtxICE-Net)" else f"wldnet:{self.wld_model_var.get()}"
        res = self.res_var.get()
        
        # We append resolution or dynamic indicator to the loaded engine key, so if the user
        # changes the resolution settings, the engine is re-loaded at the new resolution!
        key_with_res = f"{key}:{res}"
        if self.loaded_engine == key_with_res:
            return True

        self.status(f"Loading {key} …")
        try:
            import torch
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Try loading TensorRT Engine first from models_TRT folder
            models_trt_dir = BASE_DIR / "models_TRT"

            if key == "iceml":
                eng_path = models_trt_dir / "iceml" / f"iceml_fp16_{res}.engine"
            else:
                wld_key = self.wld_model_var.get()
                wld_prefix = {
                    "O-HAZE": "OH",
                    "RESIDE": "RD",
                    "NH-HAZE": "NH",
                    "D-HAZE": "DH"
                }.get(wld_key, "OH")
                eng_path = models_trt_dir / "wldnet" / f"{wld_prefix}_fp16_{res}.engine"

            loaded_trt = False
            m = None

            try:
                from trt_model import HAS_TRT, TRTModel
                if HAS_TRT and self.device.type == "cuda" and eng_path.exists():
                    self.status(f"Loading TensorRT Engine {eng_path.name} …")
                    m = TRTModel(str(eng_path), device="cuda")
                    loaded_trt = True
                    self.status(f"TensorRT Engine loaded: {eng_path.name}")
            except Exception as trt_err:
                print(f"[-] Failed to load TRT engine: {trt_err}. Falling back to PyTorch.")

            # Fall back to PyTorch if TensorRT load failed or was not available
            if not loaded_trt:
                if key == "iceml":
                    from retinexmodule import dehazing_module as ICEMLModule
                    m = ICEMLModule(3)
                    full_sd = torch.load(ICEML_MODEL_PATH, map_location="cpu")
                    prefix = "retinex_model."
                    new_sd = {k[len(prefix):]: v for k, v in full_sd.items()
                              if k.startswith(prefix)}
                    m.load_state_dict(new_sd)
                    m.to(self.device).eval()
                    if self.device.type == "cuda":
                        m.half()
                else:
                    import dehazing_model as wld_mod
                    wld_key = self.wld_model_var.get()
                    m = wld_mod.Dehazing_Model()
                    m.load_state_dict(torch.load(WLD_MODEL_PATHS[wld_key], map_location="cpu"))
                    m.to(self.device).eval()
                self.status(f"Model loaded: {key} (PyTorch)")

            self.model = m
            self.loaded_engine = key_with_res
            return True
        except Exception as e:
            messagebox.showerror("Model Error", str(e))
            self.status(f"Model load failed: {e}", error=True)
            return False

    # ── Inference helpers ─────────────────────────────────────────────────────
    def _preprocess_iceml(self, frame_bgr, size):
        import cv2, torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, size)
        t = torch.from_numpy(rgb).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        if self.device.type == "cuda":
            t = t.half()
        return t.to(self.device)

    def _postprocess_iceml(self, tensor, out_size):
        import cv2, numpy as np
        t   = tensor.float().squeeze(0).clamp(0, 1)
        out = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return cv2.resize(out, out_size)

    def _preprocess_wld(self, frame_bgr, size):
        import cv2, torch
        import torchvision.transforms as T
        import Feature_Processing
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, size)
        t   = T.ToTensor()(rgb).unsqueeze(0).to(self.device)
        return Feature_Processing.normalize(t)

    def _postprocess_wld(self, tensor, out_size):
        import cv2, numpy as np
        import Feature_Processing
        out    = Feature_Processing.denormalize(tensor)
        out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_np = (out_np * 255).clip(0, 255).astype(np.uint8)
        return cv2.resize(out_np, out_size)

    def _dehaze_frame(self, frame_bgr):
        import torch
        res  = int(self.res_var.get())
        size = (res, res)
        with torch.no_grad():
            if self.loaded_engine.startswith("iceml"):
                inp      = self._preprocess_iceml(frame_bgr, size)
                lr_inv   = self.model(inp)
                dehazed  = (inp.float() * lr_inv.float()).clamp(0, 1)
                return self._postprocess_iceml(dehazed, size)
            else:
                inp = self._preprocess_wld(frame_bgr, size)
                return self._postprocess_wld(self.model(inp), size)

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── Left panel ────────────────────────────────────────────────────────
        left = tk.Frame(self.root, bg=PANEL, width=260)
        left.grid(row=0, column=0, sticky="ns", padx=(10, 0), pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="⚙  SETTINGS", bg=PANEL, fg=CYAN,
                 font=(FONT, 12, "bold")).pack(pady=(14, 4))
        separator(left)

        # Engine selector
        section_label(left, "Dehazing Engine")
        self.engine_var = tk.StringVar(value="ICEML (RtxICE-Net)")
        self.eng_menu = ttk.Combobox(left, textvariable=self.engine_var,
                                     values=["ICEML (RtxICE-Net)", "WLD-Net"],
                                     state="readonly", width=24)
        self.eng_menu.pack(pady=2)
        self.eng_menu.bind("<<ComboboxSelected>>", self._on_engine_change)

        # WLD sub-model (hidden by default)
        self.wld_frame = tk.Frame(left, bg=PANEL)
        self.wld_frame.pack()
        tk.Label(self.wld_frame, text="WLD Dataset", bg=PANEL, fg=SUBTEXT,
                 font=(FONT, 9)).pack(pady=(4, 0))
        self.wld_model_var = tk.StringVar(value="O-HAZE")
        ttk.Combobox(self.wld_frame, textvariable=self.wld_model_var,
                     values=list(WLD_MODEL_PATHS.keys()),
                     state="readonly", width=24).pack(pady=2)
        self.wld_frame.pack_forget()

        # Resolution
        section_label(left, "Resolution")
        self.res_var = tk.StringVar(value="512")
        ttk.Combobox(left, textvariable=self.res_var,
                     values=["256", "512", "1024"],
                     state="readonly", width=24).pack(pady=2)

        separator(left)

        # Object Detection
        section_label(left, "Object Detection")
        self.od_var = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text=" Enable YOLO Detection",
                       variable=self.od_var, bg=PANEL, fg=TEXT,
                       selectcolor=ACCENT, activebackground=PANEL,
                       font=(FONT, 9)).pack()
        tk.Label(left, text="Confidence", bg=PANEL, fg=SUBTEXT,
                 font=(FONT, 9)).pack(pady=(4, 0))
        self.conf_var = tk.DoubleVar(value=0.4)
        tk.Scale(left, from_=0.1, to=0.9, resolution=0.05,
                 orient="horizontal", variable=self.conf_var,
                 bg=PANEL, fg=TEXT, troughcolor=ACCENT,
                 highlightthickness=0, length=210).pack()

        separator(left)

        # Image
        section_label(left, "📷  Image")
        styled_btn(left, "Upload & Dehaze Image",
                   self.dehaze_image, color=CYAN).pack(fill="x", padx=10, pady=3)

        separator(left)

        # Video
        section_label(left, "🎬  Video")
        styled_btn(left, "Upload & Dehaze Video",
                   self.dehaze_video, color=CYAN).pack(fill="x", padx=10, pady=3)
        styled_btn(left, "Play Dehazed Video",
                   self.play_video, color="#7c4dff").pack(fill="x", padx=10, pady=3)

        separator(left)

        # Live
        section_label(left, "📡  Live Webcam")
        styled_btn(left, "▶  Start Live",
                   self.start_live, color=GREEN).pack(fill="x", padx=10, pady=3)
        styled_btn(left, "■  Stop Live",
                   self.stop_live, color=RED).pack(fill="x", padx=10, pady=3)
        styled_btn(left, "⏺  Start Recording",
                   self.start_recording, color=ORANGE).pack(fill="x", padx=10, pady=3)
        styled_btn(left, "⏹  Stop Recording",
                   self.stop_recording, color="#e53935").pack(fill="x", padx=10, pady=3)

        # FPS counter
        self.fps_lbl = tk.Label(left, text="FPS: —", bg=PANEL, fg=CYAN,
                                font=(FONT, 11, "bold"))
        self.fps_lbl.pack(pady=4)

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(self.root, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(1, weight=1)

        for col, title in enumerate(["Original", "Dehazed + Detected"]):
            tk.Label(right, text=title, bg=BG, fg=CYAN,
                     font=(FONT, 12, "bold")).grid(row=0, column=col, pady=(0, 6))

        self.orig_lbl = tk.Label(right, bg=ACCENT, relief="flat")
        self.orig_lbl.grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        self.dehazed_lbl = tk.Label(right, bg=ACCENT, relief="flat")
        self.dehazed_lbl.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready.")
        self._status_label = tk.Label(
            self.root, textvariable=self.status_var,
            bg=ACCENT, fg=TEXT, font=(FONT, 9), anchor="w", padx=8)
        self._status_label.grid(row=1, column=0, columnspan=2,
                                sticky="ew", pady=(0, 4))

        # Start YOLO loading in background
        self.root.after(200, self._load_yolo)

    def _on_engine_change(self, *_):
        if self.engine_var.get() == "WLD-Net":
            self.wld_frame.pack(after=self.eng_menu, pady=2)
        else:
            self.wld_frame.pack_forget()
        self.model         = None
        self.loaded_engine = None

    # ── Status helper ─────────────────────────────────────────────────────────
    def status(self, msg, error=False):
        self.status_var.set(msg)
        color = RED if error else TEXT
        try:
            self._status_label.config(fg=color)
        except Exception:
            pass

    # ── Image preview ─────────────────────────────────────────────────────────
    def _show_pair(self, orig_rgb, dehazed_rgb):
        def _update():
            try:
                from PIL import Image, ImageTk
                lw = max(self.orig_lbl.winfo_width(), 400)
                lh = max(self.orig_lbl.winfo_height(), 400)

                def _fit(rgb):
                    img = Image.fromarray(rgb)
                    # BILINEAR over LANCZOS: LANCZOS is a large separable kernel and
                    # costs several ms per frame on ARM. At preview size the difference
                    # is imperceptible, and this runs on every displayed frame.
                    img.thumbnail((lw, lh), Image.BILINEAR)
                    return ImageTk.PhotoImage(img)

                ph1 = _fit(orig_rgb)
                ph2 = _fit(dehazed_rgb)
                self.orig_lbl.configure(image=ph1)
                self.orig_lbl.image = ph1
                self.dehazed_lbl.configure(image=ph2)
                self.dehazed_lbl.image = ph2
            except Exception:
                pass
        self.root.after(0, _update)

    # ── Image ─────────────────────────────────────────────────────────────────
    def dehaze_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not path:
            return
        if not self._ensure_model():
            return
        try:
            import cv2
            from PIL import Image
            frame_bgr = cv2.imread(path)
            orig_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            dehazed   = self._dehaze_frame(frame_bgr)
            detected  = self._detect(dehazed, self.conf_var.get())
            self._show_pair(orig_rgb, detected)
            name, ext = os.path.splitext(path)
            save_path = f"{name}_dehazed_detected{ext}"
            Image.fromarray(detected).save(save_path)
            self.status(f"Saved: {save_path}")
            messagebox.showinfo("Saved", f"Saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status(str(e), error=True)

    # ── Video ─────────────────────────────────────────────────────────────────
    def dehaze_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        if not self._ensure_model():
            return
        self.video_path = path
        self.stop_video = False
        threading.Thread(target=self._video_loop, daemon=True).start()

    def _video_loop(self):
        import cv2
        cap     = cv2.VideoCapture(self.video_path)
        res     = int(self.res_var.get())
        in_path = Path(self.video_path)
        out_path = str(in_path.with_name(
            in_path.stem + "_dehazed_detected" + in_path.suffix))
        self.output_video_path = out_path

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw     = cv2.VideoWriter(out_path, fourcc, fps, (res, res))

        self.status("Processing video …")
        frame_count = 0
        prev = time.time()
        while not self.stop_video:
            ret, frame = cap.read()
            if not ret:
                break
            orig_rgb = cv2.cvtColor(cv2.resize(frame, (res, res)),
                                    cv2.COLOR_BGR2RGB)
            dehazed  = self._dehaze_frame(frame)
            detected = self._detect(dehazed, self.conf_var.get())
            vw.write(cv2.cvtColor(detected, cv2.COLOR_RGB2BGR))
            frame_count += 1
            if frame_count % 5 == 0:
                self._show_pair(orig_rgb, detected)
                fps_val = 5 / max(time.time() - prev, 1e-9)
                prev = time.time()
                self.root.after(0, self.fps_lbl.config,
                                {"text": f"FPS: {fps_val:.1f}"})

        cap.release()
        vw.release()
        self.status(f"Video saved: {out_path}")
        self.root.after(0, messagebox.showinfo, "Done",
                        f"Video saved:\n{out_path}")

    def play_video(self):
        if not self.video_path or not self.output_video_path:
            messagebox.showwarning("Warning", "No dehazed video available yet.")
            return
        threading.Thread(target=self._play_loop, daemon=True).start()

    def _play_loop(self):
        import cv2, numpy as np
        cap1 = cv2.VideoCapture(self.video_path)
        cap2 = cv2.VideoCapture(self.output_video_path)
        res  = int(self.res_var.get())
        while True:
            r1, f1 = cap1.read()
            r2, f2 = cap2.read()
            if not r1 or not r2:
                break
            f1 = cv2.resize(f1, (res, res))
            f2 = cv2.resize(f2, (res, res))
            cv2.imshow("Original (L)  vs  Dehazed+Detected (R)",
                       np.hstack([f1, f2]))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

    # ── Live ──────────────────────────────────────────────────────────────────
    def start_live(self):
        import cv2
        if self.live_running:
            self.status("Live already running.")
            return
        if not self._ensure_model():
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            return
        # YUYV, not MJPG. MJPG compresses before USB transfer, but noise doesn't
        # JPEG-compress: at exposures below ~360 the grainy frames ballooned past the
        # USB budget and pinned capture at 15 FPS. YUYV frames are a fixed ~18 MB/s at
        # 640x480 regardless of noise, which fits USB 2.0 fine and removes that cliff
        # entirely — measured ~22.3 FPS flat from exposure 40 to 400:
        #   MJPG exp=100 -> 15.3 FPS  |  YUYV exp=100 -> 22.3 FPS   (same brightness)
        #   MJPG exp=330 -> 15.2 FPS  |  YUYV exp=330 -> 22.2 FPS
        # Capture at 640x480 rather than the driver max, since every frame is
        # immediately downscaled to res x res anyway. BUFFERSIZE=1 drops stale queued
        # frames so cap.read() returns the newest one instead of replaying latency.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # With YUYV both knobs are now effectively free below ~400 — tune purely for
        # how the image looks, not for FPS. Measured (YUYV, 640x480):
        #   exp= 40 gain=255 -> 22.2 FPS, brightness  26
        #   exp=200 gain=255 -> 22.3 FPS, brightness  86
        #   exp=300 gain=255 -> 22.5 FPS, brightness 107
        #   exp=400 gain=200 -> 24.9 FPS, brightness 110
        # Past ~440 exposure time itself becomes the frame time (physics: a 60 ms
        # exposure caps you at 16 FPS), so that is the only real ceiling left.
        # Prefer EXPOSURE for brightness and keep GAIN moderate — gain amplifies
        # sensor noise, and that noise feeds the dehazing net.
        EXPOSURE = 200  # units of 100 µs — free up to ~400
        GAIN     = 200  # 0-255, always free

        try:
            os.system(f"v4l2-ctl -d /dev/video0 --set-ctrl="
                      f"exposure_dynamic_framerate=0,auto_exposure=1,"
                      f"exposure_time_absolute={EXPOSURE},gain={GAIN} 2>/dev/null")
        except Exception:
            pass
        self.live_running = True
        self._latest_frame = None
        # Capture (~44 ms/frame) is slower than inference (~21 ms at 256px). Running
        # both on one thread serialises them; a dedicated grabber thread overlaps the
        # USB wait with GPU work, so the pipeline runs at the slower of the two rather
        # than their sum. cap.read() releases the GIL while blocking on the device.
        threading.Thread(target=self._grab_loop, daemon=True).start()
        threading.Thread(target=self._live_loop, daemon=True).start()
        self.status("Live dehazing started.")

    def _grab_loop(self):
        """Continuously pull frames, keeping only the most recent one."""
        while self.live_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self._frame_lock:
                self._latest_frame = frame
        if self.cap:
            self.cap.release()
            self.cap = None

    def _live_loop(self):
        import cv2
        prev = time.time()
        frame_count = 0
        # Redrawing the preview + FPS label every frame competes with the inference
        # thread for the GIL via root.after(); throttling the UI refresh (like the
        # video loop already does every 5 frames) buys real throughput without
        # touching the model or dropping any frame from the FPS measurement.
        UI_UPDATE_EVERY = 3
        last_id = None
        while self.live_running:
            # Take the newest grabbed frame; skip if the grabber hasn't produced a
            # new one yet so we never re-run inference on a frame already processed.
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None or id(frame) == last_id:
                time.sleep(0.002)
                continue
            last_id = id(frame)
            res      = int(self.res_var.get())
            dehazed  = self._dehaze_frame(frame)
            detected = self._detect(dehazed, self.conf_var.get())

            now     = time.time()
            fps_val = 1.0 / max(now - prev, 1e-9)
            prev    = now
            frame_count += 1

            need_orig = (frame_count % UI_UPDATE_EVERY == 0) or self.recording
            orig_rgb = cv2.cvtColor(cv2.resize(frame, (res, res)),
                                    cv2.COLOR_BGR2RGB) if need_orig else None

            if frame_count % UI_UPDATE_EVERY == 0:
                self.root.after(0, self.fps_lbl.config,
                                {"text": f"FPS: {fps_val:.1f}"})
                self._show_pair(orig_rgb, detected)

            if self.recording and self.vw_orig and self.vw_dehazed and orig_rgb is not None:
                self.vw_orig.write(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))
                self.vw_dehazed.write(cv2.cvtColor(detected, cv2.COLOR_RGB2BGR))
        # NOTE: the grabber thread owns self.cap and releases it on exit.

    def stop_live(self):
        self.live_running = False
        self.fps_lbl.config(text="FPS: —")
        self.status("Live dehazing stopped.")

    def start_recording(self):
        import cv2
        if not self.live_running:
            messagebox.showwarning("Warning", "Start live dehazing first.")
            return
        if self.recording:
            return
        res    = int(self.res_var.get())
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._tmp_orig    = str(BASE_DIR / "output" / "tmp_orig.mp4")
        self._tmp_dehazed = str(BASE_DIR / "output" / "tmp_dehazed.mp4")
        os.makedirs(str(BASE_DIR / "output"), exist_ok=True)
        self.vw_orig    = cv2.VideoWriter(self._tmp_orig,    fourcc, 8, (res, res))
        self.vw_dehazed = cv2.VideoWriter(self._tmp_dehazed, fourcc, 8, (res, res))
        self.recording = True
        self.status("Recording started.")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.vw_orig:
            self.vw_orig.release()
            self.vw_orig = None
        if self.vw_dehazed:
            self.vw_dehazed.release()
            self.vw_dehazed = None

        save_dir = filedialog.askdirectory(title="Save recordings to …")
        if not save_dir:
            self.status("Recording saved to output/.")
            return
        try:
            import shutil
            shutil.move(self._tmp_orig,
                        os.path.join(save_dir, "live_original.mp4"))
            shutil.move(self._tmp_dehazed,
                        os.path.join(save_dir, "live_dehazed.mp4"))
            self.status(f"Recordings saved to {save_dir}")
            messagebox.showinfo("Saved", f"Recordings saved to:\n{save_dir}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Close ─────────────────────────────────────────────────────────────────
    def on_closing(self):
        self.live_running = False
        self.recording    = False
        if self.cap:
            self.cap.release()
        if self.vw_orig:
            self.vw_orig.release()
        if self.vw_dehazed:
            self.vw_dehazed.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = DehazeApp(root)
    root.mainloop()
