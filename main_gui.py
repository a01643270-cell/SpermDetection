"""
main_gui.py  –  SpermDetection Pipeline Launcher
=================================================
A professional dark-theme GUI that lets users run every pipeline
step without touching the command line.

Pipeline order (logical step → script file)
-------------------------------------------
Step 1 → 1_extract_frames.py   – Extract Frames
Step 2 → 2_annotation_tool.py  – Annotate Images
Step 3 → 3_convert_to_yolo.py  – Convert to YOLO
Step 4 → 5_train_yolo.py       – Train YOLO Model  (script numbered 5 by convention)
Step 5 → 4_yolo_deepsort.py    – Detect & Track    (script numbered 4 by convention)

Run
---
    python main_gui.py
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Constants / theme
# ---------------------------------------------------------------------------

APP_TITLE = "🔬 SpermDetection – Pipeline Launcher"
STATE_FILE = Path("session_state.json")

# Colours
C_BG = "#1e1e2e"          # main background
C_SURFACE = "#2a2a3e"     # card background
C_BORDER = "#3a3a5c"      # card border
C_ACCENT = "#7c5cbf"      # purple accent
C_ACCENT2 = "#5cbfb0"     # teal accent
C_TEXT = "#cdd6f4"         # main text
C_TEXT_DIM = "#7f849c"    # dimmed text
C_SUCCESS = "#a6e3a1"     # green
C_WARNING = "#f9e2af"     # yellow
C_ERROR = "#f38ba8"       # red
C_RUNNING = "#89dceb"     # cyan

FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_HEADING = ("Segoe UI", 12, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_MONO = ("Courier New", 9)

# Pipeline step definitions
STEPS = [
    {
        "id": "extract_frames",
        "number": 1,
        "name": "Extract Frames",
        "script": "1_extract_frames.py",
        "description": (
            "Extracts individual JPEG frames from raw .avi videos\n"
            "and saves them to datasets/frames/."
        ),
        "prereqs": [],
        "skippable": False,
    },
    {
        "id": "annotate",
        "number": 2,
        "name": "Annotate Images",
        "script": "2_annotation_tool.py",
        "description": (
            "Opens the interactive annotation tool.\n"
            "Draw bounding boxes around every sperm cell."
        ),
        "prereqs": ["extract_frames"],
        "skippable": False,
    },
    {
        "id": "convert_yolo",
        "number": 3,
        "name": "Convert to YOLO",
        "script": "3_convert_to_yolo.py",
        "description": (
            "Converts JSON annotations to YOLO label format\n"
            "and builds train/val/test splits with dataset.yaml."
        ),
        "prereqs": ["annotate"],
        "skippable": False,
    },
    {
        "id": "train_yolo",
        "number": 4,
        "name": "Train YOLO Model",
        "script": "5_train_yolo.py",
        "description": (
            "Trains a YOLOv8 model on the annotated dataset.\n"
            "Best weights are saved to runs/sperm_yolo/."
        ),
        "prereqs": ["convert_yolo"],
        "skippable": False,
    },
    {
        "id": "detect_track",
        "number": 5,
        "name": "Detect & Track",
        "script": "4_yolo_deepsort.py",
        "description": (
            "Runs the trained model + DeepSORT tracker\n"
            "on new videos to produce annotated output."
        ),
        "prereqs": ["train_yolo"],
        "skippable": True,
    },
]

STATUS_ICON = {
    "pending":   "⏳",
    "running":   "⚙️",
    "completed": "✅",
    "skipped":   "⏭️",
    "error":     "❌",
}
STATUS_COLOR = {
    "pending":   C_TEXT_DIM,
    "running":   C_RUNNING,
    "completed": C_SUCCESS,
    "skipped":   C_WARNING,
    "error":     C_ERROR,
}


# ---------------------------------------------------------------------------
# Helper – styled tk widgets
# ---------------------------------------------------------------------------

def _label(parent: tk.Widget, text: str, font=FONT_BODY,
           fg: str = C_TEXT, bg: str = C_BG, **kw) -> tk.Label:
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)


def _btn(parent: tk.Widget, text: str, command, bg: str = C_ACCENT,
         fg: str = "#ffffff", font=FONT_BODY, padx=12, pady=5,
         state=tk.NORMAL, **kw) -> tk.Button:
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, font=font,
        relief=tk.FLAT, cursor="hand2",
        activebackground=C_BORDER, activeforeground=C_TEXT,
        padx=padx, pady=pady, state=state,
        **kw,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class App(tk.Tk):
    """SpermDetection pipeline launcher."""

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.configure(bg=C_BG)
        self.minsize(980, 700)
        self.geometry("1100x800")

        # State – keyed by step id
        self.step_status: dict[str, str] = {s["id"]: "pending" for s in STEPS}
        self.active_proc: subprocess.Popen | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self.config_vars: dict[str, tk.Variable] = {}

        self._load_state()
        self._build_ui()
        self._refresh_all_cards()
        self._poll_log_queue()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Control-l>", lambda _e: self._clear_log())
        self.bind("<Control-q>", lambda _e: self._on_close())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                for step_id, status in data.get("steps", {}).items():
                    if step_id in self.step_status:
                        self.step_status[step_id] = status
            except Exception:
                pass

    def _save_state(self) -> None:
        try:
            STATE_FILE.write_text(
                json.dumps({"steps": self.step_status}, indent=2)
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_header()
        self._build_progress_bar()

        # Main content area
        content = tk.Frame(self, bg=C_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        self._build_steps_panel(content)
        right = tk.Frame(content, bg=C_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)
        self._build_log_panel(right)
        self._build_config_panel(right)

        self._build_footer()

    def _build_header(self) -> None:
        hdr = tk.Frame(self, bg=C_SURFACE, pady=14)
        hdr.pack(fill=tk.X)
        _label(hdr, "🔬  SpermDetection", font=FONT_TITLE,
               bg=C_SURFACE, fg=C_TEXT).pack(side=tk.LEFT, padx=20)
        _label(hdr, "Pipeline Launcher", font=("Segoe UI", 11),
               bg=C_SURFACE, fg=C_TEXT_DIM).pack(side=tk.LEFT, padx=4)
        _label(hdr, "v1.0", font=("Segoe UI", 9),
               bg=C_SURFACE, fg=C_ACCENT).pack(side=tk.RIGHT, padx=20)

    def _build_progress_bar(self) -> None:
        bar_frame = tk.Frame(self, bg=C_BG, pady=6)
        bar_frame.pack(fill=tk.X, padx=16)
        _label(bar_frame, "Pipeline progress:", font=FONT_BODY,
               fg=C_TEXT_DIM).pack(side=tk.LEFT)

        style = ttk.Style(self)
        style.theme_use("default")
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=C_SURFACE,
            background=C_ACCENT2,
            thickness=14,
            bordercolor=C_BORDER,
        )
        self.progress_var = tk.DoubleVar(value=0.0)
        pb = ttk.Progressbar(
            bar_frame,
            variable=self.progress_var,
            maximum=len(STEPS),
            length=260,
            style="Dark.Horizontal.TProgressbar",
        )
        pb.pack(side=tk.LEFT, padx=(10, 6))
        self.progress_label = _label(bar_frame, "0 / 5",
                                     fg=C_TEXT_DIM)
        self.progress_label.pack(side=tk.LEFT)

    def _build_steps_panel(self, parent: tk.Frame) -> None:
        outer = tk.Frame(parent, bg=C_BG)
        outer.grid(row=0, column=0, sticky="nsew")

        canvas = tk.Canvas(outer, bg=C_BG, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview,
                           bg=C_SURFACE, troughcolor=C_BG)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.steps_frame = tk.Frame(canvas, bg=C_BG)
        win_id = canvas.create_window((0, 0), window=self.steps_frame,
                                      anchor="nw")

        def _on_frame_configure(_e: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e: tk.Event) -> None:
            canvas.itemconfig(win_id, width=e.width)

        self.steps_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(
                            int(-1 * (e.delta / 120)), "units"))

        self.step_cards: dict[str, dict] = {}
        for step in STEPS:
            self._build_step_card(self.steps_frame, step)

    def _build_step_card(self, parent: tk.Frame, step: dict) -> None:
        card = tk.Frame(parent, bg=C_SURFACE, bd=1, relief=tk.FLAT,
                        highlightbackground=C_BORDER, highlightthickness=1)
        card.pack(fill=tk.X, padx=4, pady=5)

        # Left strip (accent bar)
        strip = tk.Frame(card, bg=C_ACCENT, width=5)
        strip.pack(side=tk.LEFT, fill=tk.Y)

        body = tk.Frame(card, bg=C_SURFACE, padx=10, pady=8)
        body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Row 1: number + name + status icon
        top = tk.Frame(body, bg=C_SURFACE)
        top.pack(fill=tk.X)

        num_lbl = _label(top, f"Step {step['number']}",
                         font=("Segoe UI", 9, "bold"),
                         fg=C_ACCENT2, bg=C_SURFACE)
        num_lbl.pack(side=tk.LEFT)

        name_lbl = _label(top, f"  {step['name']}",
                          font=FONT_HEADING, fg=C_TEXT, bg=C_SURFACE)
        name_lbl.pack(side=tk.LEFT)

        icon_lbl = _label(top, STATUS_ICON["pending"],
                          font=("Segoe UI", 12), fg=C_TEXT_DIM, bg=C_SURFACE)
        icon_lbl.pack(side=tk.RIGHT)

        status_lbl = _label(top, "pending",
                             font=("Segoe UI", 9),
                             fg=C_TEXT_DIM, bg=C_SURFACE)
        status_lbl.pack(side=tk.RIGHT, padx=(0, 4))

        # Row 2: description
        desc_lbl = _label(body, step["description"],
                          font=("Segoe UI", 9),
                          fg=C_TEXT_DIM, bg=C_SURFACE,
                          justify=tk.LEFT, anchor="w")
        desc_lbl.pack(fill=tk.X, pady=(4, 6))

        # Row 3: script name + buttons
        bot = tk.Frame(body, bg=C_SURFACE)
        bot.pack(fill=tk.X)

        _label(bot, f"📄 {step['script']}",
               font=("Segoe UI", 8), fg=C_TEXT_DIM, bg=C_SURFACE).pack(
            side=tk.LEFT)

        # Skip button (only for skippable steps)
        skip_btn: tk.Button | None = None
        if step["skippable"]:
            skip_btn = _btn(
                bot, "Skip",
                command=lambda sid=step["id"]: self._skip_step(sid),
                bg=C_BORDER, fg=C_WARNING, padx=8, pady=3,
            )
            skip_btn.pack(side=tk.RIGHT, padx=(4, 0))

        launch_btn = _btn(
            bot, "▶  Launch",
            command=lambda s=step: self._launch_step(s),
            bg=C_ACCENT, padx=10, pady=3,
        )
        launch_btn.pack(side=tk.RIGHT)

        self.step_cards[step["id"]] = {
            "card": card,
            "strip": strip,
            "icon_lbl": icon_lbl,
            "status_lbl": status_lbl,
            "launch_btn": launch_btn,
            "skip_btn": skip_btn,
        }

    def _build_log_panel(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=C_SURFACE,
                         highlightbackground=C_BORDER, highlightthickness=1)
        frame.grid(row=0, column=0, sticky="nsew")

        hdr = tk.Frame(frame, bg=C_BORDER, pady=4)
        hdr.pack(fill=tk.X)
        _label(hdr, "📋  Output / Log", font=FONT_HEADING,
               bg=C_BORDER, fg=C_TEXT).pack(side=tk.LEFT, padx=10)

        # Tag-coloured text widget + scrollbar
        self.log_text = tk.Text(
            frame,
            bg="#11111b", fg=C_TEXT,
            font=FONT_MONO,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            insertbackground=C_TEXT,
            selectbackground=C_ACCENT,
        )
        vsb = tk.Scrollbar(frame, command=self.log_text.yview,
                           bg=C_SURFACE, troughcolor=C_BG)
        self.log_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Text tags for colour coding
        self.log_text.tag_configure("info",    foreground=C_TEXT)
        self.log_text.tag_configure("success", foreground=C_SUCCESS)
        self.log_text.tag_configure("warning", foreground=C_WARNING)
        self.log_text.tag_configure("error",   foreground=C_ERROR)
        self.log_text.tag_configure("dim",     foreground=C_TEXT_DIM)
        self.log_text.tag_configure("header",  foreground=C_ACCENT2,
                                    font=("Segoe UI", 9, "bold"))

        # Clear log button
        foot = tk.Frame(frame, bg=C_SURFACE, pady=4)
        foot.pack(fill=tk.X)
        _btn(foot, "🗑  Clear Log", self._clear_log,
             bg=C_BORDER, fg=C_TEXT, padx=8, pady=2).pack(
            side=tk.RIGHT, padx=6)

    def _build_config_panel(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=C_SURFACE,
                         highlightbackground=C_BORDER, highlightthickness=1)
        frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        hdr = tk.Frame(frame, bg=C_BORDER, pady=4)
        hdr.pack(fill=tk.X)
        _label(hdr, "⚙️  Configuration", font=FONT_HEADING,
               bg=C_BORDER, fg=C_TEXT).pack(side=tk.LEFT, padx=10)

        grid = tk.Frame(frame, bg=C_SURFACE, padx=10, pady=6)
        grid.pack(fill=tk.X)

        params = [
            ("Frames per video:", "frames_per_video", tk.IntVar, 5),
            ("Model size (n/s/m/l/x):", "model_size",  tk.StringVar, "s"),
            ("Batch size:", "batch_size",  tk.IntVar, 16),
            ("Epochs:", "epochs",      tk.IntVar, 100),
        ]
        for row_idx, (label, key, VarType, default) in enumerate(params):
            _label(grid, label, fg=C_TEXT_DIM, bg=C_SURFACE,
                   anchor="w").grid(row=row_idx, column=0,
                                    sticky="w", pady=2, padx=(0, 8))
            var = VarType(value=default)
            self.config_vars[key] = var
            entry = tk.Entry(grid, textvariable=var,
                             bg=C_BG, fg=C_TEXT,
                             insertbackground=C_TEXT,
                             relief=tk.FLAT, width=12,
                             font=FONT_BODY)
            entry.grid(row=row_idx, column=1, sticky="w", pady=2)

        _label(grid, "Ctrl+L: clear log   Ctrl+Q: quit",
               font=("Segoe UI", 8), fg=C_TEXT_DIM,
               bg=C_SURFACE).grid(row=len(params), column=0,
                                  columnspan=2, sticky="w", pady=(6, 0))

    def _build_footer(self) -> None:
        foot = tk.Frame(self, bg=C_SURFACE, pady=6)
        foot.pack(fill=tk.X, side=tk.BOTTOM)

        _btn(foot, "❌  Exit", self._on_close,
             bg=C_ERROR, fg="#ffffff", padx=10).pack(side=tk.RIGHT, padx=8)
        _btn(foot, "ℹ  About", self._show_about,
             bg=C_BORDER, fg=C_TEXT, padx=10).pack(side=tk.RIGHT, padx=4)
        _btn(foot, "🔄  Reset State", self._reset_state,
             bg=C_BORDER, fg=C_WARNING, padx=10).pack(side=tk.LEFT, padx=8)

        self.status_bar = _label(foot, "Ready",
                                 font=("Segoe UI", 9),
                                 fg=C_TEXT_DIM, bg=C_SURFACE)
        self.status_bar.pack(side=tk.LEFT, padx=12)

    # ------------------------------------------------------------------
    # Card refresh
    # ------------------------------------------------------------------

    def _refresh_all_cards(self) -> None:
        completed = sum(
            1 for s in STEPS if self.step_status[s["id"]] == "completed"
        )
        self.progress_var.set(completed)
        self.progress_label.configure(text=f"{completed} / {len(STEPS)}")

        for step in STEPS:
            sid = step["id"]
            status = self.step_status[sid]
            widgets = self.step_cards[sid]

            color = STATUS_COLOR[status]
            icon = STATUS_ICON[status]

            widgets["icon_lbl"].configure(text=icon, fg=color)
            widgets["status_lbl"].configure(text=status, fg=color)
            widgets["strip"].configure(
                bg=color if status != "pending" else C_ACCENT
            )

            # Disable launch if prereqs not done or another step is running
            any_running = any(
                self.step_status[s["id"]] == "running" for s in STEPS
            )
            prereqs_ok = all(
                self.step_status[p] in ("completed", "skipped")
                for p in step["prereqs"]
            )
            launch_state = (
                tk.DISABLED
                if any_running or not prereqs_ok
                else tk.NORMAL
            )
            widgets["launch_btn"].configure(state=launch_state)
            if widgets["skip_btn"]:
                widgets["skip_btn"].configure(state=launch_state)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, text: str, tag: str = "info") -> None:
        """Append text to the log widget (must be called from the main thread)."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log_separator(self) -> None:
        self._log("─" * 60, "dim")

    def _poll_log_queue(self) -> None:
        """Drain the thread-safe queue every 100 ms."""
        try:
            while True:
                text, tag = self._log_queue.get_nowait()
                self._log(text, tag)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _q_log(self, text: str, tag: str = "info") -> None:
        """Queue a log message from any thread."""
        self._log_queue.put((text, tag))

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Launching steps
    # ------------------------------------------------------------------

    def _launch_step(self, step: dict) -> None:
        """Validate prereqs, confirm, then launch the step in a thread."""
        sid = step["id"]

        # Check prereqs
        missing = [
            p for p in step["prereqs"]
            if self.step_status[p] not in ("completed", "skipped")
        ]
        if missing:
            messagebox.showwarning(
                "Prerequisites missing",
                f"Please complete the following steps first:\n\n"
                + "\n".join(f"• {p}" for p in missing),
            )
            return

        # Confirm
        if not messagebox.askyesno(
            "Launch step",
            f"Launch  Step {step['number']}: {step['name']}?\n\n"
            f"Script: {step['script']}",
        ):
            return

        script_path = Path(step["script"])
        if not script_path.exists():
            messagebox.showerror(
                "Script not found",
                f"Cannot find '{script_path}'.\n"
                "Make sure you are running from the SpermDetection directory.",
            )
            return

        # Build command – pass config args where relevant
        cmd = self._build_command(step)

        # Log + update status
        self._log_separator()
        ts = datetime.now().strftime("%H:%M:%S")
        self._log(
            f"[{ts}]  ▶  Launching Step {step['number']}: {step['name']}",
            "header",
        )
        self._log(f"  Command: {' '.join(cmd)}", "dim")
        self._log_separator()

        self.step_status[sid] = "running"
        self._refresh_all_cards()
        self.status_bar.configure(
            text=f"Running: Step {step['number']} – {step['name']} …"
        )

        # Run in background thread
        t = threading.Thread(
            target=self._run_step_thread,
            args=(step, cmd),
            daemon=True,
        )
        t.start()

    def _build_command(self, step: dict) -> list[str]:
        """Build the subprocess command for a step, injecting config args."""
        script = step["script"]
        cmd = [sys.executable, script]

        # 5_train_yolo.py accepts --epochs --batch --model
        if step["id"] == "train_yolo":
            try:
                epochs = int(self.config_vars["epochs"].get())
                batch = int(self.config_vars["batch_size"].get())
                model = str(self.config_vars["model_size"].get()).strip()
                cmd += ["--epochs", str(epochs),
                        "--batch", str(batch),
                        "--model", model]
            except (ValueError, KeyError):
                pass

        return cmd

    def _run_step_thread(self, step: dict, cmd: list[str]) -> None:
        """Execute the step subprocess and stream its output to the log."""
        sid = step["id"]
        exit_code = -1
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(Path(__file__).parent),
            )
            self.active_proc = proc

            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                tag = _classify_line(line)
                self._q_log(line, tag)

            proc.wait()
            exit_code = proc.returncode
        except Exception as exc:
            self._q_log(f"ERROR launching process: {exc}", "error")
            exit_code = -1
        finally:
            self.active_proc = None

        # Update state on the main thread
        if exit_code == 0:
            new_status = "completed"
        else:
            new_status = "error"

        def _finish() -> None:
            self.step_status[sid] = new_status
            ts = datetime.now().strftime("%H:%M:%S")
            if new_status == "completed":
                self._log(
                    f"[{ts}]  ✅  Step {step['number']} completed successfully.",
                    "success",
                )
            else:
                self._log(
                    f"[{ts}]  ❌  Step {step['number']} exited with code {exit_code}.",
                    "error",
                )
            self._log_separator()
            self._save_state()
            self._refresh_all_cards()
            self.status_bar.configure(text="Ready")

        self.after(0, _finish)

    def _skip_step(self, step_id: str) -> None:
        step = next(s for s in STEPS if s["id"] == step_id)
        if messagebox.askyesno(
            "Skip step",
            f"Mark Step {step['number']}: {step['name']} as skipped?\n\n"
            "You can still run it later.",
        ):
            self.step_status[step_id] = "skipped"
            ts = datetime.now().strftime("%H:%M:%S")
            self._log(
                f"[{ts}]  ⏭  Step {step['number']}: {step['name']} was skipped.",
                "warning",
            )
            self._save_state()
            self._refresh_all_cards()

    # ------------------------------------------------------------------
    # Misc actions
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        if messagebox.askyesno(
            "Reset pipeline state",
            "Reset all steps to 'pending'?\n\nThis does NOT delete any files.",
        ):
            for sid in self.step_status:
                self.step_status[sid] = "pending"
            ts = datetime.now().strftime("%H:%M:%S")
            self._log(f"[{ts}]  🔄  Pipeline state reset.", "warning")
            self._save_state()
            self._refresh_all_cards()
            self.status_bar.configure(text="State reset.")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About SpermDetection Launcher",
            "SpermDetection Pipeline Launcher\n"
            "Version 1.0\n\n"
            "A GUI front-end for the full sperm-detection pipeline:\n"
            "  1. Extract Frames\n"
            "  2. Annotate Images\n"
            "  3. Convert to YOLO\n"
            "  4. Train YOLO Model\n"
            "  5. Detect & Track\n\n"
            "Keyboard shortcuts:\n"
            "  Ctrl+L  –  clear log\n"
            "  Ctrl+Q  –  quit\n\n"
            "Repository: github.com/a01643270-cell/SpermDetection",
        )

    def _on_close(self) -> None:
        if self.active_proc is not None:
            if not messagebox.askyesno(
                "Process running",
                "A pipeline step is still running.\n"
                "Quit anyway? (The subprocess will be terminated.)",
            ):
                return
            try:
                self.active_proc.terminate()
            except Exception:
                pass
        self._save_state()
        self.destroy()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _classify_line(line: str) -> str:
    """Return a log tag based on heuristic line content."""
    lo = line.lower()
    if any(k in lo for k in ("error", "exception", "traceback", "failed",
                              "critical")):
        return "error"
    if any(k in lo for k in ("warn", "warning")):
        return "warning"
    if any(k in lo for k in ("✅", "success", "done", "complete", "saved",
                              "finished")):
        return "success"
    return "info"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = App()
    # Welcome message
    app._log("Welcome to the SpermDetection Pipeline Launcher!", "header")
    app._log(
        "Select a step on the left and click ▶ Launch to run it.", "dim"
    )
    app._log(
        "Steps without completed prerequisites are automatically disabled.",
        "dim",
    )
    app._log_separator()
    app.mainloop()


if __name__ == "__main__":
    main()
