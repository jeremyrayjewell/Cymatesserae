from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

STYLE_OPTIONS = ("ceramic", "neon", "lava", "glass", "monolith")
REORG_OPTIONS = ("burst", "swirl", "split", "shockwave")
GRAPHIC_OPTIONS = ("voronoi", "circles", "scribbles", "lines", "geometrics")


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.after_id: str | None = None
        self.inside = False
        self.widget.bind("<Enter>", self._schedule, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event[tk.Widget]) -> None:
        self.inside = True
        self._cancel_scheduled()
        self.after_id = self.widget.after(350, self._show)

    def _show(self) -> None:
        self.after_id = None
        if self.tip_window or not self.widget.winfo_exists():
            return
        if not self.text:
            return
        if not self.inside:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip_window,
            text=self.text,
            justify="left",
            background="#fff6cf",
            foreground="#1d1d1d",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            wraplength=280,
        )
        label.pack()

    def _hide(self, _event: tk.Event[tk.Widget] | None = None) -> None:
        self.inside = False
        self._cancel_scheduled()
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None

    def _cancel_scheduled(self) -> None:
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None


class ControlPanel:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cymatesserae Control Panel")
        self.root.geometry("860x720")
        self.root.minsize(760, 660)

        self.audio_path = tk.StringVar()
        self.output_path = tk.StringVar(value="cymatesserae_output.mp4")
        self.width = tk.IntVar(value=1280)
        self.height = tk.IntVar(value=720)
        self.fps = tk.IntVar(value=30)
        self.points = tk.IntVar(value=180)
        self.duration = tk.StringVar()
        self.seed = tk.IntVar(value=7)
        self.cymatic = tk.BooleanVar(value=True)
        self.plate_m = tk.IntVar(value=4)
        self.plate_n = tk.IntVar(value=6)
        self.style_a = tk.StringVar(value="ceramic")
        self.style_b = tk.StringVar(value="neon")
        self.morph_rate = tk.DoubleVar(value=0.18)
        self.layers = tk.IntVar(value=3)
        self.overlap = tk.DoubleVar(value=0.35)
        self.reorg_mode = tk.StringVar(value="burst")
        self.beats_per_switch = tk.IntVar(value=4)
        self.chroma_key_color = tk.StringVar()
        self.graphic_vars = {name: tk.BooleanVar(value=True) for name in GRAPHIC_OPTIONS}
        self.status = tk.StringVar(value="Ready.")
        self.running_process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.log_path: Path | None = None

        self._build()

    def _build(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)

        self._build_file_section(container)
        self._build_render_section(container)
        self._build_style_section(container)
        self._build_graphics_section(container)
        self._build_actions(container)

    def _build_file_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Files", padding=12)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        audio_label = ttk.Label(frame, text="Audio")
        audio_label.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        audio_entry = ttk.Entry(frame, textvariable=self.audio_path)
        audio_entry.grid(row=0, column=1, sticky="ew", pady=4)
        audio_button = ttk.Button(frame, text="Browse", command=self._choose_audio)
        audio_button.grid(row=0, column=2, padx=(10, 0), pady=4)

        output_label = ttk.Label(frame, text="Output MP4")
        output_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        output_entry = ttk.Entry(frame, textvariable=self.output_path)
        output_entry.grid(row=1, column=1, sticky="ew", pady=4)
        output_button = ttk.Button(frame, text="Save As", command=self._choose_output)
        output_button.grid(row=1, column=2, padx=(10, 0), pady=4)

        self._tooltip(audio_label, "Path to the source audio file that will drive the animation.")
        self._tooltip(audio_entry, "You can paste a full path here or use Browse.")
        self._tooltip(audio_button, "Choose a WAV, MP3, FLAC, OGG, or M4A file.")
        self._tooltip(output_label, "Where the MP4 export should be written.")
        self._tooltip(output_entry, "Preview mode ignores this, but export mode uses it.")
        self._tooltip(output_button, "Pick the output MP4 filename and destination folder.")

    def _build_render_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Render", padding=12)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for idx in range(6):
            frame.columnconfigure(idx, weight=1)

        self._spinbox(frame, "Width", self.width, 0, 1, 320, 3840, tooltip="Final video width in pixels.")
        self._spinbox(frame, "Height", self.height, 0, 3, 180, 2160, tooltip="Final video height in pixels.")
        self._spinbox(frame, "FPS", self.fps, 1, 1, 12, 120, tooltip="Frames per second. Higher values look smoother but render slower.")
        self._spinbox(frame, "Points", self.points, 1, 3, 20, 600, tooltip="Base point count for the motion field. More points means denser visuals.")
        self._entry(frame, "Duration", self.duration, 2, 1, tooltip="Optional limit in seconds for quick tests. Leave blank to use the full audio.")
        self._spinbox(frame, "Seed", self.seed, 2, 3, 0, 999999, tooltip="Random seed for repeatable results.")
        self._spinbox(frame, "Plate M", self.plate_m, 3, 1, 1, 16, tooltip="Horizontal cymatic plate mode. Higher values create tighter nodal spacing.")
        self._spinbox(frame, "Plate N", self.plate_n, 3, 3, 1, 16, tooltip="Vertical cymatic plate mode. Pair it with Plate M to change the nodal grid.")
        self._entry(frame, "Chroma Key", self.chroma_key_color, 4, 3, tooltip="Optional solid key color such as 00ff00 or ff00ff. The exact color will be reserved for the background.")
        cymatic_button = ttk.Checkbutton(frame, text="Cymatic mode", variable=self.cymatic)
        cymatic_button.grid(row=4, column=1, sticky="w", pady=(8, 0))
        self._tooltip(cymatic_button, "Pull motion toward vibrating-plate nodal lines for more cymatic behavior.")

    def _build_style_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Style and Motion", padding=12)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for idx in range(6):
            frame.columnconfigure(idx, weight=1)

        self._combo(frame, "Style A", self.style_a, STYLE_OPTIONS, 0, 1, tooltip="Primary color and motion preset.")
        self._combo(frame, "Style B", self.style_b, STYLE_OPTIONS, 0, 3, tooltip="Secondary preset blended with Style A over time.")
        self._spinbox(frame, "Morph Rate", self.morph_rate, 1, 1, 0.0, 2.0, increment=0.05, tooltip="How quickly the renderer drifts between the two styles.")
        self._spinbox(frame, "Layers", self.layers, 1, 3, 1, 8, tooltip="Number of overlapping passes drawn for each graphic family.")
        self._spinbox(frame, "Overlap", self.overlap, 2, 1, 0.0, 1.5, increment=0.05, tooltip="Controls layer spread, echo, and visual density.")
        self._combo(frame, "Reorg", self.reorg_mode, REORG_OPTIONS, 2, 3, tooltip="How strong percussive hits reorganize the motion field.")

    def _build_graphics_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Beat-Driven Graphic Switching", padding=12)
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        for idx in range(5):
            frame.columnconfigure(idx, weight=1)

        graphic_label = ttk.Label(frame, text="Graphic families")
        graphic_label.grid(row=0, column=0, sticky="w")
        helper_label = ttk.Label(frame, text="Enable the families you want in the switch cycle.")
        helper_label.grid(row=1, column=0, columnspan=5, sticky="w", pady=(0, 8))
        self._tooltip(graphic_label, "These are the large visual families the renderer can switch between globally.")
        self._tooltip(helper_label, "The active family changes on the detected beat according to Beats Per Switch.")

        for idx, name in enumerate(GRAPHIC_OPTIONS):
            button = ttk.Checkbutton(frame, text=name, variable=self.graphic_vars[name])
            button.grid(row=2, column=idx, sticky="w")
            tooltip = {
                "voronoi": "Tessellated polygon fields with shard-like cells.",
                "circles": "Beat-reactive bubbles and rings.",
                "scribbles": "Loose gestural strands and hand-drawn motion.",
                "lines": "Directional line fields and vector streaks.",
                "geometrics": "Angular shards and triangular geometric bursts.",
            }[name]
            self._tooltip(button, tooltip)

        self._spinbox(frame, "Beats Per Switch", self.beats_per_switch, 3, 1, 1, 16, tooltip="How many detected beats each graphic family stays active before the next one takes over.")

    def _build_actions(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        frame.grid(row=4, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        status_label = ttk.Label(frame, textvariable=self.status)
        status_label.grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(frame)
        actions.grid(row=0, column=1, sticky="e")
        preview_button = ttk.Button(actions, text="Preview", command=lambda: self._launch(preview=True))
        preview_button.pack(side="left", padx=(0, 8))
        export_button = ttk.Button(actions, text="Export MP4", command=lambda: self._launch(preview=False))
        export_button.pack(side="left")
        self._tooltip(status_label, "Shows whether the renderer is idle, running, or finished.")
        self._tooltip(preview_button, "Open a live render window using the current settings.")
        self._tooltip(export_button, "Render the current settings to an MP4 file.")

    def _entry(self, parent: ttk.Frame, label: str, variable: tk.Variable, row: int, col: int, tooltip: str = "") -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=col - 1, sticky="w", padx=(0, 8), pady=4)
        entry_widget = ttk.Entry(parent, textvariable=variable)
        entry_widget.grid(row=row, column=col, sticky="ew", pady=4)
        self._tooltip(label_widget, tooltip)
        self._tooltip(entry_widget, tooltip)

    def _spinbox(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        row: int,
        col: int,
        min_value: float,
        max_value: float,
        increment: float = 1.0,
        tooltip: str = "",
    ) -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=col - 1, sticky="w", padx=(0, 8), pady=4)
        spinbox_widget = ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=min_value,
            to=max_value,
            increment=increment,
        )
        spinbox_widget.grid(row=row, column=col, sticky="ew", pady=4)
        self._tooltip(label_widget, tooltip)
        self._tooltip(spinbox_widget, tooltip)

    def _combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        row: int,
        col: int,
        tooltip: str = "",
    ) -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=col - 1, sticky="w", padx=(0, 8), pady=4)
        combo_widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo_widget.grid(row=row, column=col, sticky="ew", pady=4)
        self._tooltip(label_widget, tooltip)
        self._tooltip(combo_widget, tooltip)

    def _tooltip(self, widget: tk.Widget, text: str) -> None:
        if text:
            ToolTip(widget, text)

    def _choose_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if path:
            self.audio_path.set(path)
            output_guess = Path(path).with_suffix(".mp4").name
            if not self.output_path.get():
                self.output_path.set(output_guess)

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save MP4 as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if path:
            self.output_path.set(path)

    def _selected_graphics(self) -> list[str]:
        return [name for name, var in self.graphic_vars.items() if var.get()]

    def _build_command(self, preview: bool) -> list[str]:
        audio = self.audio_path.get().strip()
        if not audio:
            raise ValueError("Choose an audio file first.")

        graphics = self._selected_graphics()
        if not graphics:
            raise ValueError("Select at least one graphic family.")

        cmd = [
            sys.executable,
            "-m",
            "cymatesserae",
            audio,
            "--output",
            self.output_path.get().strip() or "cymatesserae_output.mp4",
            "--width",
            str(self.width.get()),
            "--height",
            str(self.height.get()),
            "--fps",
            str(self.fps.get()),
            "--points",
            str(self.points.get()),
            "--plate-mode",
            str(self.plate_m.get()),
            str(self.plate_n.get()),
            "--seed",
            str(self.seed.get()),
            "--style",
            self.style_a.get(),
            "--style-b",
            self.style_b.get(),
            "--morph-rate",
            str(self.morph_rate.get()),
            "--layers",
            str(self.layers.get()),
            "--overlap",
            str(self.overlap.get()),
            "--reorg-mode",
            self.reorg_mode.get(),
            "--graphic-cycle",
            ",".join(graphics),
            "--beats-per-switch",
            str(self.beats_per_switch.get()),
        ]
        duration = self.duration.get().strip()
        if duration:
            cmd.extend(["--duration", duration])
        chroma_key = self.chroma_key_color.get().strip()
        if chroma_key:
            cmd.extend(["--chroma-key-color", chroma_key])
        if preview:
            cmd.append("--preview")
        if self.cymatic.get():
            cmd.append("--cymatic")
        return cmd

    def _launch(self, preview: bool) -> None:
        if self.running_process and self.running_process.poll() is None:
            messagebox.showinfo("Render already running", "Wait for the current run to finish before starting another one.")
            return

        try:
            cmd = self._build_command(preview=preview)
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        try:
            cwd = Path(__file__).resolve().parents[1]
            self.log_path = cwd / "cymatesserae_gui_last_run.log"
            self.log_handle = self.log_path.open("w", encoding="utf-8")
            self.log_handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {' '.join(cmd)}\n\n")
            self.log_handle.flush()
            self.running_process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            messagebox.showerror("Launch failed", str(exc))
            return

        mode = "preview" if preview else "export"
        self.status.set(f"Started {mode} run. Log: {self.log_path.name if self.log_path else 'n/a'}")
        self.root.after(300, self._poll_process)

    def _poll_process(self) -> None:
        if not self.running_process:
            return
        code = self.running_process.poll()
        if code is None:
            self.status.set("Render running...")
            self.root.after(1000, self._poll_process)
            return
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        if code == 0:
            self.status.set("Render finished successfully.")
        else:
            self.status.set(f"Render exited with code {code}.")
            details = self._read_log_tail()
            messagebox.showwarning("Render exited", f"The render process ended with exit code {code}.\n\n{details}")
        self.running_process = None

    def _read_log_tail(self) -> str:
        if self.log_path is None or not self.log_path.exists():
            return "No log file was captured."
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"Could not read log: {exc}"
        lines = [line for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return f"Log saved to {self.log_path}"
        tail = "\n".join(lines[-12:])
        return f"Last log lines:\n{tail}\n\nFull log: {self.log_path}"


def launch_gui() -> None:
    root = tk.Tk()
    ttk.Style(root).theme_use("clam")
    panel = ControlPanel(root)
    root.mainloop()
