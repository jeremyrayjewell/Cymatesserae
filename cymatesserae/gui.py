from __future__ import annotations

from collections.abc import Callable
import json
import math
import re
import subprocess
import sys
import tempfile
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

from .config import GraphicLayerConfig, RenderConfig
from .project_io import load_preset_file, load_project_file, save_preset_file, save_project_file
from .shared import get_app_state_dir, normalize_hex_color_text, normalize_mp4_path_text, normalize_video_dimension

STYLE_OPTIONS = ("ceramic", "neon", "lava", "glass", "monolith")
REORG_OPTIONS = ("burst", "swirl", "split", "shockwave")
PATTERN_LAYOUT_OPTIONS = ("flow", "grid")
GRID_PATTERN_OPTIONS = ("rect", "brick", "hex", "diamond")
CELL_ALTERNATION_OPTIONS = ("none", "orientation", "color", "both")
STACK_INTERACTION_OPTIONS = ("none", "crossfade", "shuffle", "pulse", "duck", "spotlight")
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
def _position_toplevel_near_parent(window: tk.Toplevel, parent: tk.Widget, offset_x: int = 40, offset_y: int = 40) -> None:
    try:
        anchor = parent.winfo_toplevel()
        anchor.update_idletasks()
        window.update_idletasks()
        parent_x = anchor.winfo_rootx()
        parent_y = anchor.winfo_rooty()
        parent_width = anchor.winfo_width()
        parent_height = anchor.winfo_height()
        window_width = max(window.winfo_reqwidth(), window.winfo_width())
        window_height = max(window.winfo_reqheight(), window.winfo_height())
        x = parent_x + min(offset_x, max(parent_width - window_width, 0))
        y = parent_y + min(offset_y, max(parent_height - window_height, 0))
        vroot_x = anchor.winfo_vrootx()
        vroot_y = anchor.winfo_vrooty()
        vroot_width = anchor.winfo_vrootwidth()
        vroot_height = anchor.winfo_vrootheight()
        x = max(vroot_x, min(x, vroot_x + max(vroot_width - window_width, 0)))
        y = max(vroot_y, min(y, vroot_y + max(vroot_height - window_height, 0)))
        window.geometry(f"+{x}+{y}")
    except tk.TclError:
        return


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
        self.widget.bind("<Destroy>", self._on_destroy, add="+")

    def _schedule(self, _event: tk.Event[tk.Widget]) -> None:
        if not self.widget.winfo_exists():
            return
        self.inside = True
        self._cancel_scheduled()
        try:
            self.after_id = self.widget.after(350, self._show)
        except tk.TclError:
            self.after_id = None

    def _show(self) -> None:
        self.after_id = None
        try:
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
        except tk.TclError:
            self.tip_window = None

    def _hide(self, _event: tk.Event[tk.Widget] | None = None) -> None:
        self.inside = False
        self._cancel_scheduled()
        if self.tip_window is not None:
            try:
                self.tip_window.destroy()
            except tk.TclError:
                pass
            self.tip_window = None

    def _cancel_scheduled(self) -> None:
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def _on_destroy(self, _event: tk.Event[tk.Widget] | None = None) -> None:
        self._hide()


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
        self.undo_stack: list[list[list[str]]] = []
        self.redo_stack: list[list[list[str]]] = []
        self._history_snapshot: list[list[str]] | None = None

        self.window = tk.Toplevel(parent)
        self.window.title("Mini Paint")
        self.window.resizable(False, False)
        self.window.transient(parent.winfo_toplevel())

        self._build()
        self._load_existing()
        _position_toplevel_near_parent(self.window, parent)

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
        self.undo_button = ttk.Button(toolbar, text="Undo", command=self._undo)
        self.undo_button.pack(side="left", padx=(8, 0))
        self.redo_button = ttk.Button(toolbar, text="Redo", command=self._redo)
        self.redo_button.pack(side="left", padx=(8, 0))
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
        self.window.bind("<Control-z>", self._undo)
        self.window.bind("<Control-y>", self._redo)
        self.window.bind("<Control-Z>", self._undo)
        self.window.bind("<Control-Y>", self._redo)

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
        self._sync_history_buttons()

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

    def _snapshot_pixels(self) -> list[list[str]]:
        return [row.copy() for row in self.pixels]

    def _begin_history_action(self) -> None:
        self._history_snapshot = self._snapshot_pixels()

    def _commit_history_action(self) -> None:
        if self._history_snapshot is None:
            return
        if self._history_snapshot != self.pixels:
            self.undo_stack.append(self._history_snapshot)
            self.redo_stack.clear()
            self._sync_history_buttons()
        self._history_snapshot = None

    def _restore_pixels(self, snapshot: list[list[str]]) -> None:
        for y, row in enumerate(snapshot):
            for x, color in enumerate(row):
                self._set_pixel(x, y, color)

    def _sync_history_buttons(self) -> None:
        self.undo_button.configure(state="normal" if self.undo_stack else "disabled")
        self.redo_button.configure(state="normal" if self.redo_stack else "disabled")

    def _on_press(self, event: tk.Event[tk.Widget]) -> None:
        point = self._canvas_to_grid(event)
        if point is None:
            return
        tool = self.tool.get()
        if tool in {"pencil", "eraser"}:
            self._begin_history_action()
            self._set_pixel(point[0], point[1], self._active_color())
            self.drag_start = point
            return
        if tool == "fill":
            self._begin_history_action()
            self._flood_fill(point[0], point[1], self._active_color())
            self._commit_history_action()
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
        tool = self.tool.get()
        if tool in {"pencil", "eraser"} and self.drag_start is not None:
            self._commit_history_action()
            self.drag_start = None
            return
        point = self._canvas_to_grid(event)
        if point is None or self.drag_start is None:
            self.drag_start = None
            return
        if tool == "line":
            self._begin_history_action()
            self._draw_line(self.drag_start, point, self.current_color.get())
            self._commit_history_action()
        elif tool == "circle":
            self._begin_history_action()
            self._draw_circle(self.drag_start, point, self.current_color.get())
            self._commit_history_action()
        self.drag_start = None

    def _set_pixel(self, x: int, y: int, color: str) -> None:
        if self.pixels[y][x] == color:
            return
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
        self._begin_history_action()
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self._set_pixel(x, y, CUSTOM_ELEMENT_KEY)
        self._commit_history_action()

    def _undo(self, _event: tk.Event[tk.Widget] | None = None) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self._snapshot_pixels())
        snapshot = self.undo_stack.pop()
        self._restore_pixels(snapshot)
        self._sync_history_buttons()

    def _redo(self, _event: tk.Event[tk.Widget] | None = None) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self._snapshot_pixels())
        snapshot = self.redo_stack.pop()
        self._restore_pixels(snapshot)
        self._sync_history_buttons()

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
        self.root.geometry("860x560")
        self.project_root = Path(__file__).resolve().parents[1]
        self.app_state_dir = get_app_state_dir()

        self.audio_path = tk.StringVar()
        self.output_path = tk.StringVar(value="cymatesserae_output.mp4")
        self.width = tk.IntVar(value=1280)
        self.height = tk.IntVar(value=720)
        self.link_dimensions = tk.BooleanVar(value=False)
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
        self.stack_interaction = tk.StringVar(value="none")
        self.chroma_key_color = tk.StringVar()
        self.custom_element_path = tk.StringVar(value=str(self.project_root / "custom_elements" / "custom_element.bmp"))
        self.custom_element_paths: list[Path] = []
        self.graphic_vars = {name: tk.BooleanVar(value=False) for name in GRAPHIC_OPTIONS}
        self._next_channel_id = 1
        self.channels: list[dict[str, object]] = [self._create_layer_state("", "Channel 1", custom_index=1)]
        self.status = tk.StringVar(value="Ready.")
        self.running_process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.log_path: Path | None = None
        self.active_run_mode = "idle"
        self.stop_requested = False
        self.channels_popup: tk.Toplevel | None = None
        self.channels_popup_frame: ttk.Frame | None = None
        self._dimension_link_ratio = self.width.get() / max(self.height.get(), 1)
        self._dimension_link_guard = False
        self.dimension_link_button: tk.Canvas | None = None

        self._build()
        self._bind_dimension_linking()
        self._size_main_window_to_content()

    def _size_window_to_content(
        self,
        window: tk.Toplevel,
        parent: tk.Widget | None = None,
        *,
        min_width: int = 520,
        min_height: int = 320,
        max_width: int = 980,
        max_height: int = 820,
    ) -> None:
        window.update_idletasks()
        width = min(max(window.winfo_reqwidth(), min_width), max_width)
        height = min(max(window.winfo_reqheight(), min_height), max_height)
        window.geometry(f"{width}x{height}")
        window.minsize(min(width, max_width), min(height, max_height))
        if parent is not None:
            _position_toplevel_near_parent(window, parent)

    def _size_main_window_to_content(self) -> None:
        self.root.update_idletasks()
        width = min(max(self.root.winfo_reqwidth(), 760), 980)
        height = min(max(self.root.winfo_reqheight(), 420), 760)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(760, 420)

    def _ttk_background(self, style_name: str = "TFrame") -> str:
        style = ttk.Style(self.root)
        return style.lookup(style_name, "background") or self.root.cget("bg")

    def _build(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)

        self._build_file_section(container)
        self._build_render_section(container)
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

        project_buttons = ttk.Frame(frame)
        project_buttons.grid(row=2, column=1, columnspan=2, sticky="w", pady=(10, 0))
        load_project_button = ttk.Button(project_buttons, text="Load Project", command=self._load_project)
        load_project_button.pack(side="left")
        save_project_button = ttk.Button(project_buttons, text="Save Project", command=self._save_project)
        save_project_button.pack(side="left", padx=(8, 0))
        load_preset_button = ttk.Button(project_buttons, text="Load Preset", command=self._load_preset)
        load_preset_button.pack(side="left", padx=(8, 0))
        save_preset_button = ttk.Button(project_buttons, text="Save Preset", command=self._save_preset)
        save_preset_button.pack(side="left", padx=(8, 0))

        self._tooltip(audio_label, "Path to the source audio file that will drive the animation.")
        self._tooltip(audio_entry, "You can paste a full path here or use Browse.")
        self._tooltip(audio_button, "Choose a WAV, MP3, FLAC, OGG, or M4A file.")
        self._tooltip(output_label, "Where the MP4 export should be written.")
        self._tooltip(output_entry, "Preview mode ignores this, but export mode uses it.")
        self._tooltip(output_button, "Pick the output MP4 filename and destination folder.")
        self._tooltip(load_project_button, "Load a saved JSON project and repopulate the current controls.")
        self._tooltip(save_project_button, "Save the current render and channel settings to a reusable JSON project.")
        self._tooltip(load_preset_button, "Load a saved visual preset and apply it without replacing the current audio or output path.")
        self._tooltip(save_preset_button, "Save the current visual settings and channels as a reusable preset.")

    def _build_render_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Render", padding=12)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for idx in range(6):
            frame.columnconfigure(idx, weight=1)

        size_frame = ttk.LabelFrame(frame, text="Dimensions", padding=10)
        size_frame.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(0, 4))
        size_frame.columnconfigure(0, weight=0)
        size_frame.columnconfigure(1, weight=1)
        size_frame.columnconfigure(2, weight=0)
        size_frame.columnconfigure(3, weight=0)
        size_frame.columnconfigure(4, weight=1)

        width_label = ttk.Label(size_frame, text="Width")
        width_label.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        width_spinbox = ttk.Spinbox(size_frame, textvariable=self.width, from_=320, to=3840, increment=1)
        width_spinbox.grid(row=0, column=1, sticky="ew", pady=4)

        self.dimension_link_button = tk.Canvas(
            size_frame,
            width=26,
            height=26,
            highlightthickness=0,
            bd=0,
            bg=self._ttk_background("TLabelframe"),
            cursor="hand2",
            relief="flat",
        )
        self.dimension_link_button.grid(row=0, column=2, pady=4, padx=14)
        self.dimension_link_button.bind("<Button-1>", lambda _event: self._toggle_dimension_link())

        height_label = ttk.Label(size_frame, text="Height")
        height_label.grid(row=0, column=3, sticky="w", padx=(0, 8), pady=4)
        height_spinbox = ttk.Spinbox(size_frame, textvariable=self.height, from_=180, to=2160, increment=1)
        height_spinbox.grid(row=0, column=4, sticky="ew", pady=4)

        self._spinbox(frame, "FPS", self.fps, 1, 1, 12, 120, tooltip="Frames per second. Higher values look smoother but render slower.")
        self._spinbox(frame, "Points", self.points, 1, 3, 20, 600, tooltip="Base point count for the motion field. More points means denser visuals.")
        self._entry(frame, "Duration", self.duration, 2, 1, tooltip="Optional limit in seconds for quick tests. Leave blank to use the full audio.")
        self._spinbox(frame, "Seed", self.seed, 2, 3, 0, 999999, tooltip="Random seed for repeatable results.")
        self._spinbox(frame, "Plate M", self.plate_m, 3, 1, 1, 16, tooltip="Horizontal cymatic plate mode. Higher values create tighter nodal spacing.")
        self._spinbox(frame, "Plate N", self.plate_n, 3, 3, 1, 16, tooltip="Vertical cymatic plate mode. Pair it with Plate M to change the nodal grid.")
        self._combo(frame, "Stack Interaction", self.stack_interaction, STACK_INTERACTION_OPTIONS, 4, 1, tooltip="How enabled channels interact while layered together. Pulse, duck, and spotlight respond to the audio automatically.")
        self._entry(frame, "Chroma Key", self.chroma_key_color, 4, 3, tooltip="Optional solid key color such as 00ff00 or ff00ff. The exact color will be reserved for the background.")
        cymatic_button = ttk.Checkbutton(frame, text="Cymatic mode", variable=self.cymatic)
        cymatic_button.grid(row=4, column=4, columnspan=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self._tooltip(width_label, "Final video width in pixels.")
        self._tooltip(width_spinbox, "Final video width in pixels.")
        self._tooltip(height_label, "Final video height in pixels.")
        self._tooltip(height_spinbox, "Final video height in pixels.")
        self._tooltip(self.dimension_link_button, "Click to link or unlink width and height scaling.")
        self._tooltip(cymatic_button, "Pull motion toward vibrating-plate nodal lines for more cymatic behavior.")
        self._refresh_dimension_link_button()

    def _bind_dimension_linking(self) -> None:
        self.width.trace_add("write", lambda *_: self._handle_dimension_change("width"))
        self.height.trace_add("write", lambda *_: self._handle_dimension_change("height"))

    def _refresh_dimension_link_ratio(self) -> None:
        self._dimension_link_ratio = self.width.get() / max(self.height.get(), 1)
        self._refresh_dimension_link_button()

    def _toggle_dimension_link(self) -> None:
        self.link_dimensions.set(not self.link_dimensions.get())
        self._refresh_dimension_link_ratio()

    def _refresh_dimension_link_button(self) -> None:
        if self.dimension_link_button is None:
            return
        canvas = self.dimension_link_button
        canvas.delete("all")
        canvas.create_text(13, 13, text="🔗", fill="#1f1f1f", font=("Segoe UI Emoji", 12))
        if self.link_dimensions.get():
            return
        canvas.create_line(6, 6, 20, 20, fill="#b22222", width=2)
        canvas.create_line(20, 6, 6, 20, fill="#b22222", width=2)

    def _handle_dimension_change(self, changed: str) -> None:
        if self._dimension_link_guard:
            return
        if not self.link_dimensions.get():
            self._refresh_dimension_link_ratio()
            return

        self._dimension_link_guard = True
        try:
            ratio = self._dimension_link_ratio if self._dimension_link_ratio > 0 else (self.width.get() / max(self.height.get(), 1))
            if changed == "width":
                new_height = max(2, int(round(self.width.get() / max(ratio, 1e-6))))
                self.height.set(new_height)
            else:
                new_width = max(2, int(round(self.height.get() * ratio)))
                self.width.set(new_width)
        finally:
            self._dimension_link_guard = False

    def _default_custom_element_path(self, custom_index: int | None = None) -> Path:
        candidates = []
        if custom_index is not None:
            candidates.append(self.project_root / "custom_elements" / f"custom_element{custom_index}.bmp")
            candidates.append(self.project_root / "custom_elements" / f"custom_element_{custom_index}.bmp")
        candidates.append(self.project_root / "custom_elements" / "custom_element.bmp")
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _create_layer_state(self, family: str, title: str, custom_index: int | None = None) -> dict[str, object]:
        default_path = self._default_custom_element_path(custom_index)
        cycle_defaults = {name: tk.BooleanVar(value=(name == family)) for name in GRAPHIC_OPTIONS}
        channel_id = self._next_channel_id
        self._next_channel_id += 1
        layer = {
            "channel_id": channel_id,
            "family": family,
            "title": tk.StringVar(value=title),
            "summary": tk.StringVar(),
            "enabled": tk.BooleanVar(value=False),
            "opacity": tk.DoubleVar(value=1.0),
            "transparent_color_path": tk.StringVar(),
            "transparent_colors": [],
            "graphic_cycle_vars": cycle_defaults,
            "beats_per_switch": tk.IntVar(value=4),
            "response_gain": tk.DoubleVar(value=1.0),
            "style_a": tk.StringVar(value="ceramic"),
            "style_b": tk.StringVar(value="neon"),
            "morph_rate": tk.DoubleVar(value=0.18),
            "layer_count": tk.IntVar(value=3),
            "overlap": tk.DoubleVar(value=0.35),
            "reorg_mode": tk.StringVar(value="burst"),
            "pattern_layout": tk.StringVar(value="flow"),
            "grid_strength": tk.DoubleVar(value=0.82),
            "geometry_rigidity": tk.DoubleVar(value=0.75),
            "layer_rigidity": tk.DoubleVar(value=0.55),
            "tile_overlap": tk.DoubleVar(value=0.25),
            "grid_columns": tk.IntVar(value=12),
            "grid_rows": tk.IntVar(value=6),
            "grid_pattern": tk.StringVar(value="rect"),
            "cell_alternation": tk.StringVar(value="none"),
            "custom_element_path": tk.StringVar(value=str(default_path)),
            "custom_element_paths": [],
            "transparent_colors_listbox": None,
            "listbox": None,
            "section": None,
        }
        for var in cycle_defaults.values():
            var.trace_add("write", lambda *_args, current=layer: self._refresh_layer_summary(current))
        self._refresh_layer_summary(layer)
        return layer

    def _selected_graphic_cycle(self, layer: dict[str, object]) -> tuple[str, ...]:
        cycle_vars: dict[str, tk.BooleanVar] = layer["graphic_cycle_vars"]  # type: ignore[assignment]
        cycle = tuple(name for name in GRAPHIC_OPTIONS if cycle_vars[name].get())
        if not cycle:
            return ()
        return cycle

    def _summarize_layer_graphics(self, layer: dict[str, object]) -> str:
        selected = self._selected_graphic_cycle(layer)
        if not selected:
            return "no elements selected"
        cycle_label = " -> ".join(selected)
        if len(selected) > 1:
            return f"cycle | {cycle_label}"
        return cycle_label

    def _refresh_layer_summary(self, layer: dict[str, object]) -> None:
        summary_var = layer.get("summary")
        if isinstance(summary_var, tk.StringVar):
            summary_var.set(self._summarize_layer_graphics(layer))

    def _render_layer_row(self, parent: ttk.Frame, layer: dict[str, object], removable: bool = True) -> None:
        row = ttk.Frame(parent, padding=(0, 0, 0, 8))
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)

        enabled = ttk.Checkbutton(row, text="", variable=layer["enabled"])
        enabled.grid(row=0, column=0, sticky="w", padx=(0, 8))

        title_label = ttk.Label(row, textvariable=layer["title"])
        title_label.grid(row=0, column=1, sticky="w")

        family_label = ttk.Label(row, textvariable=layer["summary"], foreground="#4f4f4f")
        family_label.grid(row=1, column=1, sticky="w", pady=(2, 0))

        ttk.Button(row, text="Edit Settings", command=lambda current=layer: self._open_layer_editor(current)).grid(row=0, column=2, rowspan=2, sticky="e")
        if removable:
            ttk.Button(
                row,
                text="Remove",
                command=lambda current=layer, row_widget=row: self._remove_channel_row(current, row_widget),
            ).grid(row=0, column=3, rowspan=2, sticky="e", padx=(8, 0))

    def _open_layer_editor(self, layer: dict[str, object]) -> None:
        window = tk.Toplevel(self.root)
        window.title(f"{layer['title'].get()} Channel")
        window.transient(self.root)
        window.bind("<Destroy>", lambda _event: self._refresh_channels_popup(), add="+")

        outer = ttk.Frame(window, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0, bd=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        container = ttk.Frame(canvas, padding=16)
        container.columnconfigure(0, weight=1)
        window_id = canvas.create_window((0, 0), window=container, anchor="nw")

        def sync_scroll_region(_event: tk.Event[tk.Widget] | None = None) -> None:
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except tk.TclError:
                return

        def sync_canvas_width(_event: tk.Event[tk.Widget]) -> None:
            try:
                canvas.itemconfigure(window_id, width=_event.width)
            except tk.TclError:
                return

        def on_mousewheel(event: tk.Event[tk.Widget]) -> None:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            canvas.yview_scroll(int(-delta / 120), "units")

        container.bind("<Configure>", sync_scroll_region, add="+")
        canvas.bind("<Configure>", sync_canvas_width, add="+")
        canvas.bind("<MouseWheel>", on_mousewheel, add="+")
        container.bind("<MouseWheel>", on_mousewheel, add="+")

        header = ttk.LabelFrame(container, text="Channel", padding=12)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)
        ttk.Checkbutton(header, text="Enable this channel", variable=layer["enabled"]).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Name").grid(row=0, column=1, sticky="w", padx=(12, 8))
        ttk.Entry(header, textvariable=layer["title"]).grid(row=0, column=2, sticky="ew")
        header.columnconfigure(2, weight=1)
        helper = ttk.Label(
            header,
            text="This channel renders alongside any other enabled channels. Start by naming it, then choose the element(s) it contains below.",
            wraplength=560,
            justify="left",
        )
        helper.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        row_idx = 1
        graphics_frame = ttk.LabelFrame(container, text="Elements", padding=12)
        graphics_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10))
        for idx in range(4):
            graphics_frame.columnconfigure(idx, weight=1)
        graphics_helper = ttk.Label(
            graphics_frame,
            text="Choose the element(s) this channel contains. If you select more than one, this channel will cycle through them automatically.",
            wraplength=560,
            justify="left",
        )
        graphics_helper.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        cycle_vars: dict[str, tk.BooleanVar] = layer["graphic_cycle_vars"]  # type: ignore[assignment]
        preset_families = tuple(family for family in GRAPHIC_FAMILIES if family["name"] != "custom")
        presets_expanded = any(cycle_vars[family["name"]].get() for family in preset_families)
        presets_section = CollapsibleSection(
            graphics_frame,
            "Preset Elements",
            "Built-in circles, lines, scribbles, geometrics, and Voronoi elements.",
            expanded=presets_expanded,
        )
        presets_section.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        for idx in range(4):
            presets_section.body.columnconfigure(idx, weight=1)
        rotation_helper = ttk.Label(
            presets_section.body,
            text="Choose additional elements here. When more than one element is selected, this channel rotates through them on the beat.",
            wraplength=560,
            justify="left",
        )
        rotation_helper.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        for idx, family in enumerate(preset_families):
            ttk.Checkbutton(
                presets_section.body,
                text=family["title"],
                variable=cycle_vars[family["name"]],
            ).grid(row=1 + idx // 2, column=idx % 2, sticky="w", padx=(0, 12), pady=2)
        self._spinbox(
            presets_section.body,
            "Beats Per Switch",
            layer["beats_per_switch"],
            4,
            1,
            1,
            32,
            tooltip="How many beats this channel keeps one graphic before switching to the next selected graphic.",
        )
        self._spinbox(
            presets_section.body,
            "Response Gain",
            layer["response_gain"],
            4,
            3,
            0.1,
            4.0,
            increment=0.1,
            tooltip="How strongly this channel reacts to audio pulses and onset triggers.",
        )
        row_idx += 1

        # Custom elements section - only visible if family is "custom"
        custom_frame = ttk.LabelFrame(container, text="Custom Elements", padding=12)
        custom_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10))
        custom_frame.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            custom_frame,
            text="Include custom elements in this channel",
            variable=cycle_vars["custom"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        custom_content = ttk.Frame(custom_frame)
        custom_content.grid(row=1, column=0, sticky="ew")
        custom_content.columnconfigure(1, weight=1)
        ttk.Label(custom_content, text="New element").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(custom_content, textvariable=layer["custom_element_path"]).grid(row=0, column=1, sticky="ew")
        ttk.Button(custom_content, text="Add File", command=lambda current=layer: self._choose_custom_elements_for_layer(current)).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(custom_content, text="Open Paint Editor", command=lambda current=layer: self._open_paint_editor_for_layer(current)).grid(row=1, column=1, sticky="w", pady=(8, 0))

        list_frame = ttk.Frame(custom_content)
        list_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        list_frame.columnconfigure(0, weight=1)
        ttk.Label(list_frame, text="Custom sprite list").grid(row=0, column=0, sticky="w")
        listbox = tk.Listbox(list_frame, height=5)
        listbox.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        layer["listbox"] = listbox
        buttons = ttk.Frame(list_frame)
        buttons.grid(row=1, column=1, sticky="ns", padx=(8, 0))
        ttk.Button(buttons, text="Add Draft", command=lambda current=layer: self._add_current_custom_path_for_layer(current)).pack(fill="x")
        ttk.Button(buttons, text="Remove", command=lambda current=layer: self._remove_selected_custom_element_for_layer(current)).pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="Clear", command=lambda current=layer: self._clear_custom_elements_for_layer(current)).pack(fill="x", pady=(6, 0))
        self._refresh_custom_layer_listbox(layer)
        
        # Hide/show custom frame based on family selection
        def sync_custom_visibility() -> None:
            if "custom" in self._selected_graphic_cycle(layer):
                custom_content.grid()
                return
            custom_content.grid_remove()
        
        for var in cycle_vars.values():
            var.trace_add("write", lambda *_: (sync_custom_visibility(), self._refresh_channels_popup()))
        layer["title"].trace_add("write", lambda *_: self._refresh_channels_popup())
        sync_custom_visibility()
        row_idx += 1

        style_frame = ttk.LabelFrame(container, text="Style / Motion", padding=12)
        style_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10))
        for idx in range(6):
            style_frame.columnconfigure(idx, weight=1)
        style_helper = ttk.Label(
            style_frame,
            text="These settings affect this channel only. Overlap passes are internal copies drawn within this one channel.",
            wraplength=560,
            justify="left",
        )
        style_helper.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))
        self._combo(style_frame, "Style A", layer["style_a"], STYLE_OPTIONS, 1, 1)
        self._combo(style_frame, "Style B", layer["style_b"], STYLE_OPTIONS, 1, 3)
        self._spinbox(style_frame, "Morph Rate", layer["morph_rate"], 2, 1, 0.0, 2.0, increment=0.05)
        self._spinbox(style_frame, "Overlap Passes", layer["layer_count"], 2, 3, 1, 8)
        self._spinbox(style_frame, "Overlap", layer["overlap"], 3, 1, 0.0, 1.5, increment=0.05)
        self._combo(style_frame, "Reorg", layer["reorg_mode"], REORG_OPTIONS, 3, 3)
        self._spinbox(style_frame, "Opacity", layer["opacity"], 4, 1, 0.0, 1.0, increment=0.05)
        
        # Transparent colors section
        transparent_colors_label = ttk.Label(style_frame, text="Transparent Colors")
        transparent_colors_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 4))
        transparent_color_entry = ttk.Entry(style_frame, textvariable=layer["transparent_color_path"])
        transparent_color_entry.grid(row=5, column=2, columnspan=2, sticky="ew", pady=(8, 4))
        transparent_color_button = ttk.Button(
            style_frame,
            text="Pick",
            command=lambda current=layer, parent=window: self._choose_transparent_color_for_layer(current, parent),
        )
        transparent_color_button.grid(row=5, column=4, sticky="w", padx=(8, 0), pady=(8, 4))
        self._tooltip(transparent_colors_label, "Optional colors to key out inside this channel.")
        self._tooltip(transparent_color_entry, "Hex color value to be added to the transparent list.")
        self._tooltip(transparent_color_button, "Pick a color to add to the transparent colors list.")
        
        # Transparent colors listbox
        tc_list_frame = ttk.Frame(style_frame)
        tc_list_frame.grid(row=6, column=0, columnspan=6, sticky="ew", pady=(4, 0))
        tc_list_frame.columnconfigure(0, weight=1)
        ttk.Label(tc_list_frame, text="Colors to key out").grid(row=0, column=0, sticky="w")
        tc_listbox = tk.Listbox(tc_list_frame, height=3)
        tc_listbox.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        layer["transparent_colors_listbox"] = tc_listbox
        tc_buttons = ttk.Frame(tc_list_frame)
        tc_buttons.grid(row=1, column=1, sticky="ns", padx=(8, 0))
        ttk.Button(tc_buttons, text="Add", command=lambda current=layer: self._add_transparent_color_for_layer(current)).pack(fill="x")
        ttk.Button(tc_buttons, text="Remove", command=lambda current=layer: self._remove_transparent_color_for_layer(current)).pack(fill="x", pady=(6, 0))
        ttk.Button(tc_buttons, text="Clear", command=lambda current=layer: self._clear_transparent_colors_for_layer(current)).pack(fill="x", pady=(6, 0))
        self._refresh_transparent_colors_listbox(layer)
        row_idx += 1

        grid_frame = ttk.LabelFrame(container, text="Grid Layout", padding=12)
        grid_frame.grid(row=row_idx, column=0, sticky="ew")
        for idx in range(6):
            grid_frame.columnconfigure(idx, weight=1)
        grid_helper = ttk.Label(
            grid_frame,
            text="Grid controls also belong to this channel only.",
            wraplength=560,
            justify="left",
        )
        grid_helper.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))
        self._combo(grid_frame, "Pattern Layout", layer["pattern_layout"], PATTERN_LAYOUT_OPTIONS, 1, 1)
        self._combo(grid_frame, "Grid Pattern", layer["grid_pattern"], GRID_PATTERN_OPTIONS, 1, 3)
        self._combo(grid_frame, "Cell Alternation", layer["cell_alternation"], CELL_ALTERNATION_OPTIONS, 2, 1)
        self._spinbox(grid_frame, "Grid Strength", layer["grid_strength"], 2, 3, 0.0, 1.0, increment=0.05)
        self._spinbox(grid_frame, "Geometry Rigidity", layer["geometry_rigidity"], 3, 1, 0.0, 1.0, increment=0.05)
        self._spinbox(grid_frame, "Layer Rigidity", layer["layer_rigidity"], 3, 3, 0.0, 1.0, increment=0.05)
        self._spinbox(grid_frame, "Tile Overlap", layer["tile_overlap"], 4, 1, 0.0, 1.0, increment=0.05)
        self._spinbox(grid_frame, "Grid Columns", layer["grid_columns"], 4, 3, 0, 200)
        self._spinbox(grid_frame, "Grid Rows", layer["grid_rows"], 5, 1, 0, 200)
        self._size_window_to_content(window, self.root, min_width=720, min_height=520, max_width=920, max_height=760)
        sync_scroll_region()

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

    def _build_grid_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Grid Layout", padding=12)
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        for idx in range(6):
            frame.columnconfigure(idx, weight=1)

        helper = ttk.Label(frame, text="These controls matter most when Pattern Layout is set to grid.")
        helper.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))
        self._tooltip(helper, "Use this block to control rigid screen-filling tile layouts and alternating cell behavior.")

        self._combo(frame, "Pattern Layout", self.pattern_layout, PATTERN_LAYOUT_OPTIONS, 1, 1, tooltip="Use flow for freer motion or grid for more stationary placements.")
        self._combo(frame, "Grid Pattern", self.grid_pattern, GRID_PATTERN_OPTIONS, 1, 3, tooltip="Alternative cell arrangements for grid mode.")
        self._combo(frame, "Cell Alternation", self.cell_alternation, CELL_ALTERNATION_OPTIONS, 2, 1, tooltip="Alternate inverse orientation and/or inverse color between neighboring grid cells.")
        self._spinbox(frame, "Grid Strength", self.grid_strength, 2, 3, 0.0, 1.0, increment=0.05, tooltip="How firmly grid layout holds patterns in place.")
        self._spinbox(frame, "Geometry Rigidity", self.geometry_rigidity, 3, 1, 0.0, 1.0, increment=0.05, tooltip="How rigidly tile positions lock to the grid in grid mode.")
        self._spinbox(frame, "Layer Rigidity", self.layer_rigidity, 3, 3, 0.0, 1.0, increment=0.05, tooltip="How much layer shear, offset, and pulse are suppressed in grid mode.")
        self._spinbox(frame, "Tile Overlap", self.tile_overlap, 4, 1, 0.0, 1.0, increment=0.05, tooltip="How tightly stationary tiles and custom sprites overlap.")
        self._spinbox(frame, "Grid Columns", self.grid_columns, 4, 3, 0, 200, tooltip="Explicit grid column count in grid mode. Use 0 to auto-fit.")
        self._spinbox(frame, "Grid Rows", self.grid_rows, 5, 1, 0, 200, tooltip="Explicit grid row count in grid mode. Use 0 to auto-fit.")

    def _build_graphics_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Graphics", padding=12)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        helper = ttk.Label(frame, text="Create channels here, then choose the reusable element(s) each channel uses.")
        helper.grid(row=0, column=0, sticky="w", pady=(0, 10))

        popup_row = ttk.Frame(frame)
        popup_row.grid(row=1, column=0, sticky="w", pady=(0, 10))
        ttk.Button(popup_row, text="Open Channels", command=self._open_channels_popup).pack(side="left")

        cycle_note = ttk.Label(
            frame,
            text="Workflow: create a channel, choose its reusable element(s), and if you select more than one the channel will cycle through them automatically.",
            wraplength=520,
            justify="left",
        )
        cycle_note.grid(row=2, column=0, sticky="w")

    def _populate_channels_popup(self, parent: ttk.Frame) -> None:
        for child in parent.winfo_children():
            child.destroy()
        for layer in self.channels:
            self._render_layer_row(parent, layer, removable=True)

    def _open_channels_popup(self) -> None:
        if self.channels_popup is not None:
            try:
                if self.channels_popup.winfo_exists():
                    self.channels_popup.lift()
                    self.channels_popup.focus_force()
                    return
            except tk.TclError:
                pass
            self.channels_popup = None
            self.channels_popup_frame = None

        window = tk.Toplevel(self.root)
        window.title("Channels")
        window.transient(self.root)
        self.channels_popup = window
        window.bind("<Destroy>", lambda _event: self._on_channels_popup_destroyed(), add="+")

        container = ttk.Frame(window, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)

        actions = ttk.Frame(container)
        actions.pack(anchor="w", pady=(0, 10))
        layers_frame = ttk.Frame(container)
        layers_frame.pack(fill="both", expand=True)
        layers_frame.columnconfigure(0, weight=1)
        self.channels_popup_frame = layers_frame

        helper = ttk.Label(container, text="Channels work like mixer lanes. Each one can reuse built-in and custom elements, and enabled channels layer together with the selected stack interaction.")
        helper.pack(anchor="w", pady=(0, 10))

        ttk.Button(actions, text="Create Channel", command=lambda: self._add_channel_and_refresh(layers_frame)).pack(side="left")
        self._populate_channels_popup(layers_frame)
        self._size_window_to_content(window, self.root, min_width=660, min_height=360, max_width=900, max_height=780)

    def _on_channels_popup_destroyed(self) -> None:
        self.channels_popup = None
        self.channels_popup_frame = None
        self._clear_dead_channel_widgets()

    def _clear_dead_channel_widgets(self) -> None:
        for layer in self.channels:
            listbox = layer.get("listbox")
            if isinstance(listbox, tk.Listbox):
                try:
                    if not listbox.winfo_exists():
                        layer["listbox"] = None
                except tk.TclError:
                    layer["listbox"] = None
            transparent_colors_listbox = layer.get("transparent_colors_listbox")
            if isinstance(transparent_colors_listbox, tk.Listbox):
                try:
                    if not transparent_colors_listbox.winfo_exists():
                        layer["transparent_colors_listbox"] = None
                except tk.TclError:
                    layer["transparent_colors_listbox"] = None

    def _add_channel_and_refresh(self, layers_frame: ttk.Frame) -> None:
        self._add_channel()
        self._populate_channels_popup(layers_frame)

    def _refresh_channels_popup(self) -> None:
        frame = self.channels_popup_frame
        if frame is None:
            return
        try:
            if not frame.winfo_exists():
                self.channels_popup_frame = None
                return
        except tk.TclError:
            self.channels_popup_frame = None
            return
        self._populate_channels_popup(frame)

    def _build_actions(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        frame.grid(row=3, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        status_label = ttk.Label(frame, textvariable=self.status)
        status_label.grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(frame)
        actions.grid(row=0, column=1, sticky="e")
        preview_button = ttk.Button(actions, text="Preview", command=lambda: self._launch(preview=True))
        preview_button.pack(side="left", padx=(0, 8))
        export_button = ttk.Button(actions, text="Export MP4", command=lambda: self._launch(preview=False))
        export_button.pack(side="left", padx=(0, 8))
        stop_button = ttk.Button(actions, text="Stop Render", command=self._stop_render)
        stop_button.pack(side="left")
        self._tooltip(status_label, "Shows whether the renderer is idle, running, or finished.")
        self._tooltip(preview_button, "Open a lighter live-only preview window that skips MP4 export work.")
        self._tooltip(export_button, "Render the current settings to an MP4 file.")
        self._tooltip(stop_button, "Stop the active preview or export process.")

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
            parent=self.root,
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if path:
            self.audio_path.set(path)
            current_output = self.output_path.get().strip()
            output_name = Path(path).with_suffix(".mp4").name
            if current_output:
                output_guess = Path(current_output).with_name(output_name)
            else:
                output_guess = Path(output_name)
            self.output_path.set(str(output_guess))

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save MP4 as",
            parent=self.root,
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if path:
            self.output_path.set(path)

    def _add_channel(self) -> None:
        index = len(self.channels) + 1
        self.channels.append(self._create_layer_state("", f"Channel {index}", custom_index=index))

    def _remove_channel(self, layer: dict[str, object]) -> None:
        channel_id = layer.get("channel_id")
        self.channels = [current for current in self.channels if current.get("channel_id") != channel_id]
        if not self.channels:
            self.channels.append(self._create_layer_state("", "Channel 1", custom_index=1))

    def _remove_channel_and_refresh(self, layer: dict[str, object]) -> None:
        self._remove_channel(layer)
        self._refresh_channels_popup()

    def _remove_channel_row(self, layer: dict[str, object], row_widget: ttk.Frame) -> None:
        previous_count = len(self.channels)
        self._remove_channel(layer)
        if len(self.channels) == previous_count:
            return
        try:
            if row_widget.winfo_exists() and self.channels:
                row_widget.destroy()
        except tk.TclError:
            pass
        if len(self.channels) <= 1:
            self._refresh_channels_popup()

    def _refresh_custom_layer_listbox(self, layer: dict[str, object]) -> None:
        listbox = layer.get("listbox")
        if not isinstance(listbox, tk.Listbox):
            return
        try:
            if not listbox.winfo_exists():
                layer["listbox"] = None
                return
            listbox.delete(0, "end")
            for path in layer["custom_element_paths"]:
                listbox.insert("end", str(path))
        except tk.TclError:
            layer["listbox"] = None

    def _append_custom_element_to_layer(self, layer: dict[str, object], path: Path) -> None:
        resolved = path.resolve()
        paths: list[Path] = layer["custom_element_paths"]  # type: ignore[assignment]
        if resolved not in paths:
            paths.append(resolved)
        cycle_vars: dict[str, tk.BooleanVar] = layer["graphic_cycle_vars"]  # type: ignore[assignment]
        cycle_vars["custom"].set(True)
        self._refresh_custom_layer_listbox(layer)

    def _choose_custom_elements_for_layer(self, layer: dict[str, object]) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose custom elements",
            parent=self.root,
            filetypes=[("Bitmap images", "*.bmp *.png"), ("All files", "*.*")],
        )
        if not paths:
            return
        for path in paths:
            self._append_custom_element_to_layer(layer, Path(path))
        layer["custom_element_path"].set(paths[-1])
        layer["enabled"].set(True)

    def _open_paint_editor_for_layer(self, layer: dict[str, object]) -> None:
        PaintEditor(
            self.root,
            Path(layer["custom_element_path"].get().strip()),
            on_save=lambda path, current=layer: self._handle_custom_element_saved_for_layer(current, path),
        )

    def _choose_transparent_color_for_layer(self, layer: dict[str, object], parent: tk.Widget) -> None:
        initial = None
        current = layer["transparent_color_path"].get().strip()
        if current:
            try:
                initial = f"#{normalize_hex_color_text(current)}"
            except ValueError:
                initial = None
        choice = colorchooser.askcolor(color=initial, parent=parent)
        if choice[1]:
            layer["transparent_color_path"].set(choice[1].lstrip("#").lower())
            self._add_transparent_color_for_layer(layer)

    def _refresh_transparent_colors_listbox(self, layer: dict[str, object]) -> None:
        listbox = layer.get("transparent_colors_listbox")
        if not isinstance(listbox, tk.Listbox):
            return
        try:
            if not listbox.winfo_exists():
                layer["transparent_colors_listbox"] = None
                return
            listbox.delete(0, "end")
            for color in layer["transparent_colors"]:
                listbox.insert("end", color)
        except tk.TclError:
            layer["transparent_colors_listbox"] = None

    def _add_transparent_color_for_layer(self, layer: dict[str, object]) -> None:
        text = layer["transparent_color_path"].get().strip()
        if not text:
            messagebox.showerror("Missing color", "Enter a hex color (RRGGBB) first or use the Pick button.")
            return
        try:
            normalized = normalize_hex_color_text(text)
            colors: list[str] = layer["transparent_colors"]  # type: ignore[assignment]
            if normalized not in colors:
                colors.append(normalized)
            self._refresh_transparent_colors_listbox(layer)
            layer["transparent_color_path"].set("")
        except ValueError as e:
            messagebox.showerror("Invalid color", str(e))

    def _remove_transparent_color_for_layer(self, layer: dict[str, object]) -> None:
        listbox = layer.get("transparent_colors_listbox")
        if not isinstance(listbox, tk.Listbox):
            return
        selection = listbox.curselection()
        if not selection:
            return
        colors: list[str] = layer["transparent_colors"]  # type: ignore[assignment]
        for index in reversed(selection):
            del colors[index]
        self._refresh_transparent_colors_listbox(layer)

    def _clear_transparent_colors_for_layer(self, layer: dict[str, object]) -> None:
        colors: list[str] = layer["transparent_colors"]  # type: ignore[assignment]
        colors.clear()
        self._refresh_transparent_colors_listbox(layer)

    def _handle_custom_element_saved_for_layer(self, layer: dict[str, object], path: Path) -> None:
        layer["custom_element_path"].set(str(path))
        self._append_custom_element_to_layer(layer, path)
        layer["enabled"].set(True)

    def _add_current_custom_path_for_layer(self, layer: dict[str, object]) -> None:
        text = layer["custom_element_path"].get().strip()
        if not text:
            messagebox.showerror("Missing path", "Enter a custom element path first.")
            return
        path = Path(text)
        if not path.exists():
            messagebox.showerror("Missing draft", f"Custom element not found:\n{path}\n\nSave it from the paint editor or choose an existing file first.")
            return
        self._append_custom_element_to_layer(layer, path)
        layer["enabled"].set(True)

    def _remove_selected_custom_element_for_layer(self, layer: dict[str, object]) -> None:
        listbox = layer.get("listbox")
        if not isinstance(listbox, tk.Listbox):
            return
        selection = listbox.curselection()
        if not selection:
            return
        paths: list[Path] = layer["custom_element_paths"]  # type: ignore[assignment]
        for index in reversed(selection):
            del paths[index]
        self._refresh_custom_layer_listbox(layer)

    def _clear_custom_elements_for_layer(self, layer: dict[str, object]) -> None:
        paths: list[Path] = layer["custom_element_paths"]  # type: ignore[assignment]
        paths.clear()
        self._refresh_custom_layer_listbox(layer)

    def _active_layer_payloads(self) -> list[dict[str, object]]:
        layers = list(self.channels)
        payloads: list[dict[str, object]] = []
        for layer in layers:
            if not layer["enabled"].get():
                continue
            graphic_cycle = self._selected_graphic_cycle(layer)
            if not graphic_cycle:
                raise ValueError(f"{layer['title'].get()} needs at least one selected element.")
            stored_paths: list[Path] = layer["custom_element_paths"]  # type: ignore[assignment]
            custom_paths = tuple(path for path in stored_paths if path.exists())
            needs_custom = "custom" in graphic_cycle
            if needs_custom and not custom_paths:
                current = layer["custom_element_path"].get().strip()
                if current:
                    current_path = Path(current).resolve()
                    if current_path.exists():
                        custom_paths = (current_path,)
            if needs_custom and not custom_paths:
                raise ValueError(f"{layer['title'].get()} needs at least one custom element.")
            transparent_colors_list: list[str] = layer["transparent_colors"]  # type: ignore[assignment]
            payloads.append(
                {
                    "name": layer["title"].get().strip() or str(layer["family"]),
                    "family": graphic_cycle[0],
                    "enabled": True,
                    "opacity": float(layer["opacity"].get()),
                    "transparent_colors": transparent_colors_list,
                    "graphic_cycle": list(graphic_cycle),
                    "beats_per_switch": int(layer["beats_per_switch"].get()),
                    "response_gain": float(layer["response_gain"].get()),
                    "style_a": layer["style_a"].get(),
                    "style_b": layer["style_b"].get(),
                    "morph_rate": float(layer["morph_rate"].get()),
                    "layer_count": int(layer["layer_count"].get()),
                    "overlap": float(layer["overlap"].get()),
                    "pattern_layout": layer["pattern_layout"].get(),
                    "grid_strength": float(layer["grid_strength"].get()),
                    "geometry_rigidity": float(layer["geometry_rigidity"].get()),
                    "layer_rigidity": float(layer["layer_rigidity"].get()),
                    "tile_overlap": float(layer["tile_overlap"].get()),
                    "grid_columns": int(layer["grid_columns"].get()),
                    "grid_rows": int(layer["grid_rows"].get()),
                    "grid_pattern": layer["grid_pattern"].get(),
                    "cell_alternation": layer["cell_alternation"].get(),
                    "reorg_mode": layer["reorg_mode"].get(),
                    "custom_element_paths": [str(path) for path in custom_paths],
                }
            )
        return payloads

    def _layer_state_to_config(self, layer: dict[str, object]) -> GraphicLayerConfig:
        graphic_cycle = self._selected_graphic_cycle(layer)
        stored_paths: list[Path] = layer["custom_element_paths"]  # type: ignore[assignment]
        transparent_colors_list: list[str] = layer["transparent_colors"]  # type: ignore[assignment]
        transparent_colors = tuple(parse_hex_to_rgb(text) for text in transparent_colors_list)
        return GraphicLayerConfig(
            name=layer["title"].get().strip() or "Channel",
            family=graphic_cycle[0] if graphic_cycle else "voronoi",
            enabled=bool(layer["enabled"].get()),
            opacity=float(layer["opacity"].get()),
            transparent_colors=transparent_colors,
            graphic_cycle=graphic_cycle,
            beats_per_switch=int(layer["beats_per_switch"].get()),
            response_gain=float(layer["response_gain"].get()),
            style_a=layer["style_a"].get(),
            style_b=layer["style_b"].get(),
            morph_rate=float(layer["morph_rate"].get()),
            layer_count=int(layer["layer_count"].get()),
            overlap=float(layer["overlap"].get()),
            pattern_layout=layer["pattern_layout"].get(),
            grid_strength=float(layer["grid_strength"].get()),
            geometry_rigidity=float(layer["geometry_rigidity"].get()),
            layer_rigidity=float(layer["layer_rigidity"].get()),
            tile_overlap=float(layer["tile_overlap"].get()),
            grid_columns=int(layer["grid_columns"].get()),
            grid_rows=int(layer["grid_rows"].get()),
            grid_pattern=layer["grid_pattern"].get(),
            cell_alternation=layer["cell_alternation"].get(),
            reorg_mode=layer["reorg_mode"].get(),
            custom_element_paths=tuple(path.resolve() for path in stored_paths),
        )

    def _build_project_config(self) -> RenderConfig:
        audio = self.audio_path.get().strip()
        if not audio:
            raise ValueError("Choose an audio file before saving a project.")
        duration_text = self.duration.get().strip()
        chroma_key_text = self.chroma_key_color.get().strip()
        graphic_layers = tuple(self._layer_state_to_config(layer) for layer in self.channels)
        return RenderConfig(
            audio_path=Path(audio).resolve(),
            output_path=Path(normalize_mp4_path_text(self.output_path.get())).resolve(),
            width=normalize_video_dimension(self.width.get()),
            height=normalize_video_dimension(self.height.get()),
            fps=int(self.fps.get()),
            point_count=int(self.points.get()),
            preview=False,
            preview_only=False,
            cymatic_mode=bool(self.cymatic.get()),
            plate_mode=(int(self.plate_m.get()), int(self.plate_n.get())),
            duration_limit=float(duration_text) if duration_text else None,
            seed=int(self.seed.get()),
            style_a=self.style_a.get(),
            style_b=self.style_b.get(),
            morph_rate=float(self.morph_rate.get()),
            layer_count=int(self.layers.get()),
            overlap=float(self.overlap.get()),
            pattern_layout=self.pattern_layout.get(),
            grid_strength=float(self.grid_strength.get()),
            geometry_rigidity=float(self.geometry_rigidity.get()),
            layer_rigidity=float(self.layer_rigidity.get()),
            tile_overlap=float(self.tile_overlap.get()),
            grid_columns=int(self.grid_columns.get()),
            grid_rows=int(self.grid_rows.get()),
            grid_pattern=self.grid_pattern.get(),
            cell_alternation=self.cell_alternation.get(),
            reorg_mode=self.reorg_mode.get(),
            graphic_cycle=("voronoi", "circles", "scribbles", "lines", "geometrics"),
            beats_per_switch=int(self.beats_per_switch.get()),
            stack_interaction=self.stack_interaction.get(),
            chroma_key_color=parse_hex_to_rgb(chroma_key_text) if chroma_key_text else None,
            graphic_layers=graphic_layers,
        )

    def _build_preset_source_config(self) -> RenderConfig:
        duration_text = self.duration.get().strip()
        chroma_key_text = self.chroma_key_color.get().strip()
        graphic_layers = tuple(self._layer_state_to_config(layer) for layer in self.channels)
        audio_text = self.audio_path.get().strip() or "preset-placeholder.wav"
        output_text = normalize_mp4_path_text(self.output_path.get()) if self.output_path.get().strip() else "preset-placeholder.mp4"
        return RenderConfig(
            audio_path=Path(audio_text).resolve(),
            output_path=Path(output_text).resolve(),
            width=normalize_video_dimension(self.width.get()),
            height=normalize_video_dimension(self.height.get()),
            fps=int(self.fps.get()),
            point_count=int(self.points.get()),
            preview=False,
            preview_only=False,
            cymatic_mode=bool(self.cymatic.get()),
            plate_mode=(int(self.plate_m.get()), int(self.plate_n.get())),
            duration_limit=float(duration_text) if duration_text else None,
            seed=int(self.seed.get()),
            style_a=self.style_a.get(),
            style_b=self.style_b.get(),
            morph_rate=float(self.morph_rate.get()),
            layer_count=int(self.layers.get()),
            overlap=float(self.overlap.get()),
            pattern_layout=self.pattern_layout.get(),
            grid_strength=float(self.grid_strength.get()),
            geometry_rigidity=float(self.geometry_rigidity.get()),
            layer_rigidity=float(self.layer_rigidity.get()),
            tile_overlap=float(self.tile_overlap.get()),
            grid_columns=int(self.grid_columns.get()),
            grid_rows=int(self.grid_rows.get()),
            grid_pattern=self.grid_pattern.get(),
            cell_alternation=self.cell_alternation.get(),
            reorg_mode=self.reorg_mode.get(),
            graphic_cycle=("voronoi", "circles", "scribbles", "lines", "geometrics"),
            beats_per_switch=int(self.beats_per_switch.get()),
            stack_interaction=self.stack_interaction.get(),
            chroma_key_color=parse_hex_to_rgb(chroma_key_text) if chroma_key_text else None,
            graphic_layers=graphic_layers,
        )

    def _apply_layer_config_to_state(self, state: dict[str, object], layer_config: GraphicLayerConfig) -> None:
        state["title"].set(layer_config.name)
        state["enabled"].set(layer_config.enabled)
        state["opacity"].set(layer_config.opacity)
        state["transparent_colors"] = [rgb_to_hex(color) for color in layer_config.transparent_colors]
        state["transparent_color_path"].set("")
        state["beats_per_switch"].set(layer_config.beats_per_switch)
        state["response_gain"].set(layer_config.response_gain)
        state["style_a"].set(layer_config.style_a)
        state["style_b"].set(layer_config.style_b)
        state["morph_rate"].set(layer_config.morph_rate)
        state["layer_count"].set(layer_config.layer_count)
        state["overlap"].set(layer_config.overlap)
        state["reorg_mode"].set(layer_config.reorg_mode)
        state["pattern_layout"].set(layer_config.pattern_layout)
        state["grid_strength"].set(layer_config.grid_strength)
        state["geometry_rigidity"].set(layer_config.geometry_rigidity)
        state["layer_rigidity"].set(layer_config.layer_rigidity)
        state["tile_overlap"].set(layer_config.tile_overlap)
        state["grid_columns"].set(layer_config.grid_columns)
        state["grid_rows"].set(layer_config.grid_rows)
        state["grid_pattern"].set(layer_config.grid_pattern)
        state["cell_alternation"].set(layer_config.cell_alternation)
        cycle_vars: dict[str, tk.BooleanVar] = state["graphic_cycle_vars"]  # type: ignore[assignment]
        selected = set(layer_config.graphic_cycle or (layer_config.family,))
        for name, var in cycle_vars.items():
            var.set(name in selected)
        custom_paths = [Path(path) for path in layer_config.custom_element_paths]
        state["custom_element_paths"] = custom_paths
        if custom_paths:
            state["custom_element_path"].set(str(custom_paths[-1]))
        self._refresh_layer_summary(state)

    def _apply_project_config(self, config: RenderConfig) -> None:
        self.audio_path.set(str(config.audio_path))
        self.output_path.set(str(config.output_path))
        self.width.set(config.width)
        self.height.set(config.height)
        self.fps.set(config.fps)
        self.points.set(config.point_count)
        self.duration.set("" if config.duration_limit is None else str(config.duration_limit))
        self.seed.set(config.seed)
        self.cymatic.set(config.cymatic_mode)
        self.plate_m.set(config.plate_mode[0])
        self.plate_n.set(config.plate_mode[1])
        self.style_a.set(config.style_a)
        self.style_b.set(config.style_b)
        self.morph_rate.set(config.morph_rate)
        self.layers.set(config.layer_count)
        self.overlap.set(config.overlap)
        self.pattern_layout.set(config.pattern_layout)
        self.grid_strength.set(config.grid_strength)
        self.geometry_rigidity.set(config.geometry_rigidity)
        self.layer_rigidity.set(config.layer_rigidity)
        self.tile_overlap.set(config.tile_overlap)
        self.grid_columns.set(config.grid_columns)
        self.grid_rows.set(config.grid_rows)
        self.grid_pattern.set(config.grid_pattern)
        self.cell_alternation.set(config.cell_alternation)
        self.reorg_mode.set(config.reorg_mode)
        self.beats_per_switch.set(config.beats_per_switch)
        self.stack_interaction.set(config.stack_interaction)
        self.chroma_key_color.set("" if config.chroma_key_color is None else rgb_to_hex(config.chroma_key_color))

        layer_configs = list(config.graphic_layers)
        if not layer_configs:
            layer_configs = [GraphicLayerConfig(name="Channel 1", family="voronoi", enabled=False)]
        self.channels = []
        self._next_channel_id = 1
        for idx, layer_config in enumerate(layer_configs, start=1):
            state = self._create_layer_state(layer_config.family, layer_config.name or f"Channel {idx}", custom_index=idx)
            self._apply_layer_config_to_state(state, layer_config)
            self.channels.append(state)
        self._refresh_channels_popup()
        self.status.set("Project loaded.")

    def _save_project(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Project",
            defaultextension=".json",
            filetypes=[("Cymatesserae Project", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            config = self._build_project_config()
            save_project_file(Path(path), config)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return
        self.status.set(f"Project saved to {Path(path).name}.")

    def _load_project(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Project",
            filetypes=[("Cymatesserae Project", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            config = load_project_file(Path(path))
            self._apply_project_config(config)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc), parent=self.root)
            return

    def _save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Preset",
            defaultextension=".json",
            filetypes=[("Cymatesserae Preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            config = self._build_preset_source_config()
            save_preset_file(Path(path), config)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return
        self.status.set(f"Preset saved to {Path(path).name}.")

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Preset",
            filetypes=[("Cymatesserae Preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        audio_before = self.audio_path.get()
        output_before = self.output_path.get()
        try:
            base_config = self._build_preset_source_config()
            config = load_preset_file(Path(path), base_config)
            self._apply_project_config(config)
            self.audio_path.set(audio_before)
            self.output_path.set(output_before)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc), parent=self.root)
            return
        self.status.set("Preset applied.")

    def _build_command(self, preview: bool) -> list[str]:
        audio = self.audio_path.get().strip()
        if not audio:
            raise ValueError("Choose an audio file first.")

        layer_payloads = self._active_layer_payloads()
        if not layer_payloads:
            raise ValueError("Enable at least one channel.")
        self.app_state_dir.mkdir(parents=True, exist_ok=True)
        config_dir = Path(tempfile.mkdtemp(prefix="gui-run-", dir=self.app_state_dir))
        graphics_config_path = config_dir / "graphics_config.json"
        graphics_config_path.write_text(json.dumps(layer_payloads, indent=2), encoding="utf-8")

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
            "--stack-interaction",
            self.stack_interaction.get(),
            "--seed",
            str(self.seed.get()),
            "--graphics-config",
            str(graphics_config_path),
        ]
        duration = self.duration.get().strip()
        if duration:
            cmd.extend(["--duration", duration])
        chroma_key = self.chroma_key_color.get().strip()
        if chroma_key:
            cmd.extend(["--chroma-key-color", chroma_key])
        if preview:
            cmd.append("--preview-only")
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
            self.app_state_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.app_state_dir / "cymatesserae_gui_last_run.log"
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

        self.stop_requested = False
        self.active_run_mode = "preview" if preview else "export"
        mode_label = "live preview" if preview else "MP4 export"
        self.status.set(f"Started {mode_label}. Initializing renderer and audio analysis... Log: {self.log_path.name if self.log_path else 'n/a'}")
        self.root.after(300, self._poll_process)

    def _stop_render(self) -> None:
        process = self.running_process
        if process is None or process.poll() is not None:
            self.status.set("No render is currently running.")
            return
        self.stop_requested = True
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            try:
                process.terminate()
            except Exception as exc:
                self.stop_requested = False
                messagebox.showerror("Stop failed", str(exc))
                return
        self.status.set("Stopping render process...")
        self.root.after(300, self._poll_process)

    def _poll_process(self) -> None:
        if not self.running_process:
            return
        code = self.running_process.poll()
        if code is None:
            self.status.set(self._current_runtime_status())
            self.root.after(1000, self._poll_process)
            return
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        if self.stop_requested:
            self.status.set("Render stopped.")
        elif code == 0:
            mode_label = "Live preview" if self.active_run_mode == "preview" else "Render"
            self.status.set(f"{mode_label} finished successfully.")
        else:
            self.status.set(f"Render exited with code {code}.")
            details = self._read_log_tail()
            messagebox.showwarning("Render exited", f"The render process ended with exit code {code}.\n\n{details}")
        self.running_process = None
        self.active_run_mode = "idle"
        self.stop_requested = False

    def _current_runtime_status(self) -> str:
        if self.log_path is None or not self.log_path.exists():
            return "Renderer is running. Waiting for the first log update..."
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            return f"Renderer is running. Could not read progress log: {exc}"

        for line in reversed(lines):
            text = line.strip()
            if not text:
                continue
            if "[cymatesserae]" in text:
                detail = text.split("[cymatesserae]", 1)[1].strip()
                prefix = "Live preview" if self.active_run_mode == "preview" else "Render"
                return f"{prefix}: {detail}"
            if "frame=" in text:
                match = re.search(r"frame=\s*(\d+).*?fps=\s*([0-9.]+).*?speed=\s*([0-9.]+)x", text)
                if match:
                    frame, fps, speed = match.groups()
                    return f"Export: encoded frame {frame} at {fps} fps, about {speed}x realtime."
                return f"Export: {text}"

        return "Renderer is running. Waiting for the first progress marker..."

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


def parse_hex_to_rgb(text: str) -> tuple[int, int, int]:
    normalized = normalize_hex_color_text(text)
    return tuple(int(normalized[idx : idx + 2], 16) for idx in (0, 2, 4))


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    return "".join(f"{channel:02x}" for channel in color)
