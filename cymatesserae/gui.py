from __future__ import annotations

from collections.abc import Callable
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

STYLE_OPTIONS = ("ceramic", "neon", "lava", "glass", "monolith")
REORG_OPTIONS = ("burst", "swirl", "split", "shockwave")
PATTERN_LAYOUT_OPTIONS = ("flow", "grid")
GRID_PATTERN_OPTIONS = ("rect", "brick", "hex", "diamond")
CELL_ALTERNATION_OPTIONS = ("none", "orientation", "color", "both")
GRAPHIC_FAMILIES = (
    {
        "name": "voronoi",
        "title": "Voronoi",
        "summary": "Tessellated shards and cell structures.",
        "details": "Best for mosaic and crystalline fields. Future Voronoi-specific controls can live here without crowding the main panel.",
    },
    {
        "name": "circles",
        "title": "Circles",
        "summary": "Beat-reactive bubbles and rings.",
        "details": "Useful for pulse-driven motion and softer layered looks. This section is ready for future circle-specific controls.",
    },
    {
        "name": "scribbles",
        "title": "Scribbles",
        "summary": "Loose gestural strands and hand-drawn motion.",
        "details": "Good for messy, sketch-like energy. New scribble controls can be added here later.",
    },
    {
        "name": "lines",
        "title": "Lines",
        "summary": "Directional streaks and vector-field sweeps.",
        "details": "Best for flow and velocity-driven scenes. This panel gives lines their own expansion space.",
    },
    {
        "name": "geometrics",
        "title": "Geometrics",
        "summary": "Angular shards and triangular bursts.",
        "details": "Useful for harder-edged forms and crystalline accents. More geometric options can be added here later.",
    },
    {
        "name": "custom",
        "title": "Custom",
        "summary": "Painted sprite stamps from a tiny built-in editor.",
        "details": "Use the mini paint window to sketch simple symbols or shapes, then include them in the beat switch cycle.",
    },
)
GRAPHIC_OPTIONS = tuple(family["name"] for family in GRAPHIC_FAMILIES)
CUSTOM_ELEMENT_KEY = "#ff00ff"
CUSTOM_ELEMENT_PALETTE = (
    "#ffffff",
    "#000000",
    "#ff3b30",
    "#ff9500",
    "#ffcc00",
    "#34c759",
    "#007aff",
    "#5856d6",
    "#ff2d55",
    "#8e8e93",
)


def normalize_mp4_path_text(path_text: str) -> str:
    text = path_text.strip()
    if not text:
        return "cymatesserae_output.mp4"
    path = Path(text)
    if path.suffix.lower() == ".mp4":
        return str(path)
    if path.suffix:
        return str(path.with_suffix(".mp4"))
    return str(path.with_name(f"{path.name}.mp4"))


def normalize_video_dimension(value: int) -> int:
    number = max(2, int(value))
    if number % 2 == 0:
        return number
    return number + 1


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


class CollapsibleSection(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        summary: str,
        expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._expanded = tk.BooleanVar(value=expanded)
        self.columnconfigure(0, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        self.toggle_button = ttk.Button(header, text="", width=2, command=self.toggle)
        self.toggle_button.grid(row=0, column=0, sticky="w")

        title_label = ttk.Label(header, text=title)
        title_label.grid(row=0, column=1, sticky="w")

        summary_label = ttk.Label(header, text=summary, foreground="#4f4f4f")
        summary_label.grid(row=1, column=1, sticky="w", pady=(2, 0))

        self.body = ttk.Frame(self, padding=(28, 8, 0, 4))
        self.body.grid(row=1, column=0, sticky="ew")
        self.body.columnconfigure(0, weight=1)

        self._sync()

    def toggle(self) -> None:
        self._expanded.set(not self._expanded.get())
        self._sync()

    def expand(self) -> None:
        self._expanded.set(True)
        self._sync()

    def collapse(self) -> None:
        self._expanded.set(False)
        self._sync()

    def _sync(self) -> None:
        if self._expanded.get():
            self.toggle_button.configure(text="−")
            self.body.grid()
            return
        self.toggle_button.configure(text="+")
        self.body.grid_remove()


class PaintEditor:
    def __init__(self, parent: tk.Widget, asset_path: Path, on_save: Callable[[Path], None] | None = None) -> None:
        self.asset_path = asset_path
        self.on_save = on_save
        self.grid_size = 128
        self.cell_size = 4
        self.current_color = tk.StringVar(value=CUSTOM_ELEMENT_PALETTE[0])
        self.tool = tk.StringVar(value="pencil")
        self.pixels = [[CUSTOM_ELEMENT_KEY for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.rectangles: list[list[int]] = []
        self.drag_start: tuple[int, int] | None = None

        self.window = tk.Toplevel(parent)
        self.window.title("Mini Paint")
        self.window.resizable(False, False)

        self._build()
        self._load_existing()

    def _build(self) -> None:
        root = ttk.Frame(self.window, padding=12)
        root.pack(fill="both", expand=True)

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Label(toolbar, text="Tool").pack(side="left")
        ttk.Radiobutton(toolbar, text="Pencil", variable=self.tool, value="pencil").pack(side="left", padx=(8, 0))
        ttk.Radiobutton(toolbar, text="Eraser", variable=self.tool, value="eraser").pack(side="left", padx=(8, 0))
        ttk.Radiobutton(toolbar, text="Line", variable=self.tool, value="line").pack(side="left", padx=(8, 0))
        ttk.Radiobutton(toolbar, text="Circle", variable=self.tool, value="circle").pack(side="left", padx=(8, 0))
        ttk.Radiobutton(toolbar, text="Fill", variable=self.tool, value="fill").pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Pick Color", command=self._pick_color).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="Clear", command=self._clear).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Save", command=self._save).pack(side="right")

        palette = ttk.Frame(root)
        palette.pack(fill="x", pady=(0, 10))
        for color in CUSTOM_ELEMENT_PALETTE:
            swatch = tk.Button(
                palette,
                bg=color,
                activebackground=color,
                width=2,
                relief="raised",
                command=lambda selected=color: self.current_color.set(selected),
            )
            swatch.pack(side="left", padx=(0, 6))

        hint = ttk.Label(
            root,
            text="Pencil and eraser draw on drag. Line and circle use click-drag-release. Fill floods the clicked region. Eraser restores transparency for clean sprite stamping.",
            wraplength=420,
            justify="left",
        )
        hint.pack(anchor="w", pady=(0, 8))

        canvas_size = self.grid_size * self.cell_size
        self.canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg=CUSTOM_ELEMENT_KEY, highlightthickness=1, highlightbackground="#6a6a6a")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        for y in range(self.grid_size):
            row: list[int] = []
            for x in range(self.grid_size):
                rect = self.canvas.create_rectangle(
                    x * self.cell_size,
                    y * self.cell_size,
                    (x + 1) * self.cell_size,
                    (y + 1) * self.cell_size,
                    fill=CUSTOM_ELEMENT_KEY,
                    outline="#404040",
                )
                row.append(rect)
            self.rectangles.append(row)

    def _pick_color(self) -> None:
        choice = colorchooser.askcolor(color=self.current_color.get(), parent=self.window)
        if choice[1] and choice[1].lower() != CUSTOM_ELEMENT_KEY:
            self.current_color.set(choice[1].lower())

    def _canvas_to_grid(self, event: tk.Event[tk.Widget]) -> tuple[int, int] | None:
        x = int(event.x // self.cell_size)
        y = int(event.y // self.cell_size)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return x, y
        return None

    def _active_color(self) -> str:
        return CUSTOM_ELEMENT_KEY if self.tool.get() == "eraser" else self.current_color.get()

    def _on_press(self, event: tk.Event[tk.Widget]) -> None:
        point = self._canvas_to_grid(event)
        if point is None:
            return
        tool = self.tool.get()
        if tool in {"pencil", "eraser"}:
            self._set_pixel(point[0], point[1], self._active_color())
            self.drag_start = point
            return
        if tool == "fill":
            self._flood_fill(point[0], point[1], self._active_color())
            self.drag_start = None
            return
        self.drag_start = point

    def _on_drag(self, event: tk.Event[tk.Widget]) -> None:
        point = self._canvas_to_grid(event)
        if point is None:
            return
        tool = self.tool.get()
        if tool in {"pencil", "eraser"}:
            self._set_pixel(point[0], point[1], self._active_color())

    def _on_release(self, event: tk.Event[tk.Widget]) -> None:
        point = self._canvas_to_grid(event)
        if point is None or self.drag_start is None:
            self.drag_start = None
            return
        tool = self.tool.get()
        if tool == "line":
            self._draw_line(self.drag_start, point, self.current_color.get())
        elif tool == "circle":
            self._draw_circle(self.drag_start, point, self.current_color.get())
        self.drag_start = None

    def _set_pixel(self, x: int, y: int, color: str) -> None:
        self.pixels[y][x] = color
        self.canvas.itemconfigure(self.rectangles[y][x], fill=color)

    def _draw_line(self, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self._set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x0 += sx
            if err2 < dx:
                err += dx
                y0 += sy

    def _draw_circle(self, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
        cx = (start[0] + end[0]) / 2.0
        cy = (start[1] + end[1]) / 2.0
        rx = max(0.5, abs(end[0] - start[0]) / 2.0)
        ry = max(0.5, abs(end[1] - start[1]) / 2.0)
        steps = max(24, int(math.tau * max(rx, ry)))
        for step in range(steps):
            theta = math.tau * step / steps
            x = int(round(cx + math.cos(theta) * rx))
            y = int(round(cy + math.sin(theta) * ry))
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self._set_pixel(x, y, color)

    def _flood_fill(self, x: int, y: int, color: str) -> None:
        target = self.pixels[y][x]
        if target == color:
            return
        stack = [(x, y)]
        while stack:
            px, py = stack.pop()
            if not (0 <= px < self.grid_size and 0 <= py < self.grid_size):
                continue
            if self.pixels[py][px] != target:
                continue
            self._set_pixel(px, py, color)
            stack.append((px + 1, py))
            stack.append((px - 1, py))
            stack.append((px, py + 1))
            stack.append((px, py - 1))

    def _clear(self) -> None:
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self._set_pixel(x, y, CUSTOM_ELEMENT_KEY)

    def _save(self) -> None:
        self.asset_path.parent.mkdir(parents=True, exist_ok=True)
        pygame_path = None
        try:
            import pygame

            pygame.init()
            surface = pygame.Surface((self.grid_size, self.grid_size))
            key_rgb = self._hex_to_rgb(CUSTOM_ELEMENT_KEY)
            surface.fill(key_rgb)
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    surface.set_at((x, y), self._hex_to_rgb(self.pixels[y][x]))
            pygame.image.save(surface, self.asset_path.as_posix())
            pygame_path = self.asset_path
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save custom element.\n\n{exc}", parent=self.window)
            return
        finally:
            try:
                import pygame

                pygame.quit()
            except Exception:
                pass

        if self.on_save is not None and pygame_path is not None:
            self.on_save(pygame_path)
        messagebox.showinfo("Saved", f"Custom element saved to:\n{self.asset_path}", parent=self.window)

    def _load_existing(self) -> None:
        if not self.asset_path.exists():
            return
        try:
            import pygame

            pygame.init()
            surface = pygame.image.load(self.asset_path.as_posix())
            width = min(surface.get_width(), self.grid_size)
            height = min(surface.get_height(), self.grid_size)
            key_rgb = self._hex_to_rgb(CUSTOM_ELEMENT_KEY)
            for y in range(height):
                for x in range(width):
                    rgb = tuple(surface.get_at((x, y))[:3])
                    color = CUSTOM_ELEMENT_KEY if rgb == key_rgb else "#{:02x}{:02x}{:02x}".format(*rgb)
                    self._set_pixel(x, y, color)
        except Exception:
            return
        finally:
            try:
                import pygame

                pygame.quit()
            except Exception:
                pass

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple[int, int, int]:
        text = color.lstrip("#")
        return tuple(int(text[idx : idx + 2], 16) for idx in (0, 2, 4))


class ControlPanel:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cymatesserae Control Panel")
        self.root.geometry("860x720")
        self.root.minsize(760, 660)
        self.project_root = Path(__file__).resolve().parents[1]

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
        self.pattern_layout = tk.StringVar(value="flow")
        self.grid_strength = tk.DoubleVar(value=0.82)
        self.geometry_rigidity = tk.DoubleVar(value=0.75)
        self.layer_rigidity = tk.DoubleVar(value=0.55)
        self.tile_overlap = tk.DoubleVar(value=0.25)
        self.grid_columns = tk.IntVar(value=0)
        self.grid_rows = tk.IntVar(value=0)
        self.grid_pattern = tk.StringVar(value="rect")
        self.cell_alternation = tk.StringVar(value="none")
        self.reorg_mode = tk.StringVar(value="burst")
        self.beats_per_switch = tk.IntVar(value=4)
        self.chroma_key_color = tk.StringVar()
        self.custom_element_path = tk.StringVar(value=str(self.project_root / "custom_elements" / "custom_element.bmp"))
        self.custom_element_paths: list[Path] = []
        self.graphic_vars = {name: tk.BooleanVar(value=False) for name in GRAPHIC_OPTIONS}
        self.graphic_sections: dict[str, CollapsibleSection] = {}
        self.status = tk.StringVar(value="Ready.")
        self.running_process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.log_path: Path | None = None

        self._build()

    def _build(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        container = ttk.Frame(canvas, padding=16)
        container.columnconfigure(0, weight=1)
        window_id = canvas.create_window((0, 0), window=container, anchor="nw")

        def _update_scroll_region(_event: tk.Event[tk.Widget]) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _resize_inner_frame(_event: tk.Event[tk.Widget]) -> None:
            canvas.itemconfigure(window_id, width=_event.width)

        container.bind("<Configure>", _update_scroll_region)
        canvas.bind("<Configure>", _resize_inner_frame)

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
        self._combo(frame, "Pattern Layout", self.pattern_layout, PATTERN_LAYOUT_OPTIONS, 2, 3, tooltip="Use flow for freer motion or grid for more stationary placements.")
        self._spinbox(frame, "Grid Strength", self.grid_strength, 3, 1, 0.0, 1.0, increment=0.05, tooltip="How firmly grid layout holds patterns in place.")
        self._spinbox(frame, "Geometry Rigidity", self.geometry_rigidity, 3, 3, 0.0, 1.0, increment=0.05, tooltip="How rigidly tile positions lock to the grid in grid mode.")
        self._spinbox(frame, "Layer Rigidity", self.layer_rigidity, 4, 1, 0.0, 1.0, increment=0.05, tooltip="How much layer shear, offset, and pulse are suppressed in grid mode.")
        self._spinbox(frame, "Tile Overlap", self.tile_overlap, 4, 3, 0.0, 1.0, increment=0.05, tooltip="How tightly stationary tiles and custom sprites overlap.")
        self._spinbox(frame, "Grid Columns", self.grid_columns, 5, 1, 0, 200, tooltip="Explicit grid column count in grid mode. Use 0 to auto-fit.")
        self._spinbox(frame, "Grid Rows", self.grid_rows, 5, 3, 0, 200, tooltip="Explicit grid row count in grid mode. Use 0 to auto-fit.")
        self._combo(frame, "Grid Pattern", self.grid_pattern, GRID_PATTERN_OPTIONS, 6, 1, tooltip="Alternative cell arrangements for grid mode.")
        self._combo(frame, "Cell Alternation", self.cell_alternation, CELL_ALTERNATION_OPTIONS, 6, 3, tooltip="Alternate inverse orientation and/or inverse color between neighboring grid cells.")
        self._combo(frame, "Reorg", self.reorg_mode, REORG_OPTIONS, 7, 1, tooltip="How strong percussive hits reorganize the motion field.")

    def _build_graphics_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Beat-Driven Graphic Switching", padding=12)
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        graphic_label = ttk.Label(frame, text="Graphic families")
        graphic_label.grid(row=0, column=0, sticky="w")
        helper_label = ttk.Label(frame, text="Enable the families you want in the switch cycle.")
        helper_label.grid(row=1, column=0, sticky="w", pady=(0, 6))
        self._tooltip(graphic_label, "These are the large visual families the renderer can switch between globally.")
        self._tooltip(helper_label, "The active family changes on the detected beat according to Beats Per Switch.")

        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, sticky="w", pady=(0, 10))
        ttk.Button(actions, text="Expand all", command=self._expand_all_graphic_sections).pack(side="left")
        ttk.Button(actions, text="Collapse all", command=self._collapse_all_graphic_sections).pack(side="left", padx=(8, 0))

        families = ttk.Frame(frame)
        families.grid(row=3, column=0, sticky="ew")
        families.columnconfigure(0, weight=1)

        for row_idx, family in enumerate(GRAPHIC_FAMILIES):
            section = CollapsibleSection(
                families,
                title=family["title"],
                summary=family["summary"],
                expanded=row_idx == 0,
            )
            section.grid(row=row_idx, column=0, sticky="ew", pady=(0, 8))
            self.graphic_sections[family["name"]] = section

            enabled = ttk.Checkbutton(section.body, text="Include in beat switch cycle", variable=self.graphic_vars[family["name"]])
            enabled.grid(row=0, column=0, sticky="w")
            details = ttk.Label(section.body, text=family["details"], wraplength=680, justify="left")
            details.grid(row=1, column=0, sticky="w", pady=(6, 0))
            self._tooltip(enabled, family["summary"])

            if family["name"] == "custom":
                asset_row = ttk.Frame(section.body)
                asset_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
                asset_row.columnconfigure(1, weight=1)
                ttk.Label(asset_row, text="New element").grid(row=0, column=0, sticky="w", padx=(0, 8))
                asset_entry = ttk.Entry(asset_row, textvariable=self.custom_element_path)
                asset_entry.grid(row=0, column=1, sticky="ew")
                ttk.Button(asset_row, text="Add File", command=self._choose_custom_elements).grid(row=0, column=2, padx=(8, 0))
                ttk.Button(asset_row, text="Open Paint Editor", command=self._open_paint_editor).grid(row=1, column=1, sticky="w", pady=(8, 0))
                self._tooltip(asset_entry, "Type a sprite path here for the paint editor or add it to the list below.")

                list_frame = ttk.Frame(section.body)
                list_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
                list_frame.columnconfigure(0, weight=1)
                ttk.Label(list_frame, text="Custom sprite list").grid(row=0, column=0, sticky="w")
                self.custom_elements_listbox = tk.Listbox(list_frame, height=5)
                self.custom_elements_listbox.grid(row=1, column=0, sticky="ew", pady=(4, 0))
                buttons = ttk.Frame(list_frame)
                buttons.grid(row=1, column=1, sticky="ns", padx=(8, 0))
                ttk.Button(buttons, text="Add Draft", command=self._add_current_custom_path).pack(fill="x")
                ttk.Button(buttons, text="Remove", command=self._remove_selected_custom_element).pack(fill="x", pady=(6, 0))
                ttk.Button(buttons, text="Clear", command=self._clear_custom_elements).pack(fill="x", pady=(6, 0))

        self._spinbox(frame, "Beats Per Switch", self.beats_per_switch, 4, 1, 1, 16, tooltip="How many detected beats each graphic family stays active before the next one takes over.")

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

    def _expand_all_graphic_sections(self) -> None:
        for section in self.graphic_sections.values():
            section.expand()

    def _collapse_all_graphic_sections(self) -> None:
        for section in self.graphic_sections.values():
            section.collapse()

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

    def _choose_custom_elements(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose custom elements",
            filetypes=[("Bitmap images", "*.bmp *.png"), ("All files", "*.*")],
        )
        if not paths:
            return
        for path in paths:
            self._append_custom_element(Path(path))
        self.custom_element_path.set(paths[-1])

    def _open_paint_editor(self) -> None:
        PaintEditor(self.root, Path(self.custom_element_path.get().strip()), on_save=self._handle_custom_element_saved)

    def _handle_custom_element_saved(self, path: Path) -> None:
        self.custom_element_path.set(str(path))
        self._append_custom_element(path)
        self.graphic_vars["custom"].set(True)
        section = self.graphic_sections.get("custom")
        if section is not None:
            section.expand()

    def _append_custom_element(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved not in self.custom_element_paths:
            self.custom_element_paths.append(resolved)
            self.custom_elements_listbox.insert("end", str(resolved))

    def _add_current_custom_path(self) -> None:
        text = self.custom_element_path.get().strip()
        if not text:
            messagebox.showerror("Missing path", "Enter a custom element path first.")
            return
        self._append_custom_element(Path(text))
        self.graphic_vars["custom"].set(True)

    def _remove_selected_custom_element(self) -> None:
        selection = self.custom_elements_listbox.curselection()
        if not selection:
            return
        for index in reversed(selection):
            self.custom_elements_listbox.delete(index)
            del self.custom_element_paths[index]

    def _clear_custom_elements(self) -> None:
        self.custom_elements_listbox.delete(0, "end")
        self.custom_element_paths.clear()

    def _selected_graphics(self) -> list[str]:
        return [name for name, var in self.graphic_vars.items() if var.get()]

    def _build_command(self, preview: bool) -> list[str]:
        audio = self.audio_path.get().strip()
        if not audio:
            raise ValueError("Choose an audio file first.")

        graphics = self._selected_graphics()
        if not graphics:
            raise ValueError("Select at least one graphic family.")
        custom_elements = [path for path in self.custom_element_paths if path.exists()]
        if "custom" in graphics and not custom_elements:
            current = self.custom_element_path.get().strip()
            if current:
                self._append_custom_element(Path(current))
                custom_elements = [path for path in self.custom_element_paths if path.exists()]
        if "custom" in graphics and not custom_elements:
            raise ValueError("Choose or paint at least one custom element before enabling the custom graphic family.")

        cmd = [
            sys.executable,
            "-m",
            "cymatesserae",
            audio,
            "--output",
            normalize_mp4_path_text(self.output_path.get()),
            "--width",
            str(normalize_video_dimension(self.width.get())),
            "--height",
            str(normalize_video_dimension(self.height.get())),
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
            "--pattern-layout",
            self.pattern_layout.get(),
            "--grid-strength",
            str(self.grid_strength.get()),
            "--geometry-rigidity",
            str(self.geometry_rigidity.get()),
            "--layer-rigidity",
            str(self.layer_rigidity.get()),
            "--tile-overlap",
            str(self.tile_overlap.get()),
            "--grid-columns",
            str(self.grid_columns.get()),
            "--grid-rows",
            str(self.grid_rows.get()),
            "--grid-pattern",
            self.grid_pattern.get(),
            "--cell-alternation",
            self.cell_alternation.get(),
            "--reorg-mode",
            self.reorg_mode.get(),
            "--graphic-cycle",
            ",".join(graphics),
            "--beats-per-switch",
            str(self.beats_per_switch.get()),
        ]
        if "custom" in graphics:
            cmd.extend(["--custom-element", *[str(path) for path in custom_elements]])
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
            self.width.set(normalize_video_dimension(self.width.get()))
            self.height.set(normalize_video_dimension(self.height.get()))
            self.output_path.set(normalize_mp4_path_text(self.output_path.get()))
            cmd = self._build_command(preview=preview)
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        try:
            cwd = self.project_root
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
