from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .renderer import GraphicLayerConfig, RenderConfig


def parse_chroma_key_color(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise argparse.ArgumentTypeError("Chroma key color must be a 6-digit hex value like 00ff00 or #00ff00.")
    try:
        return tuple(int(text[idx : idx + 2], 16) for idx in (0, 2, 4))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Chroma key color must be valid hexadecimal.") from exc


def normalize_output_path(path: Path) -> Path:
    if path.suffix.lower() == ".mp4":
        return path
    if path.suffix:
        return path.with_suffix(".mp4")
    return path.with_name(f"{path.name}.mp4")


def normalize_video_dimension(value: int) -> int:
    number = max(2, int(value))
    if number % 2 == 0:
        return number
    return number + 1


def load_graphic_layers(path: Path | None) -> tuple[GraphicLayerConfig, ...]:
    if path is None:
        return ()
    from .renderer import GraphicLayerConfig

    raw = json.loads(path.read_text(encoding="utf-8"))
    layers: list[GraphicLayerConfig] = []
    for idx, item in enumerate(raw):
        custom_paths = tuple(Path(text) for text in item.get("custom_element_paths", []))
        transparent_colors_raw = item.get("transparent_colors", [])
        transparent_colors = tuple(
            parse_chroma_key_color(str(color)) for color in transparent_colors_raw
        ) if transparent_colors_raw else ()
        layers.append(
            GraphicLayerConfig(
                name=str(item.get("name", f"layer_{idx + 1}")),
                family=str(item.get("family", "voronoi")),
                enabled=bool(item.get("enabled", True)),
                opacity=float(item.get("opacity", 1.0)),
                transparent_colors=transparent_colors,
                graphic_cycle=tuple(str(value) for value in item.get("graphic_cycle", [])),
                beats_per_switch=int(item.get("beats_per_switch", 4)),
                response_gain=float(item.get("response_gain", 1.0)),
                style_a=str(item.get("style_a", "ceramic")),
                style_b=str(item.get("style_b", "neon")),
                morph_rate=float(item.get("morph_rate", 0.18)),
                layer_count=int(item.get("layer_count", 3)),
                overlap=float(item.get("overlap", 0.35)),
                pattern_layout=str(item.get("pattern_layout", "flow")),
                grid_strength=float(item.get("grid_strength", 0.82)),
                geometry_rigidity=float(item.get("geometry_rigidity", 0.75)),
                layer_rigidity=float(item.get("layer_rigidity", 0.55)),
                tile_overlap=float(item.get("tile_overlap", 0.25)),
                grid_columns=int(item.get("grid_columns", 0)),
                grid_rows=int(item.get("grid_rows", 0)),
                grid_pattern=str(item.get("grid_pattern", "rect")),
                cell_alternation=str(item.get("cell_alternation", "none")),
                reorg_mode=str(item.get("reorg_mode", "burst")),
                custom_element_paths=custom_paths,
            )
        )
    return tuple(layers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cymatesserae",
        description="Transform audio into cymatic Voronoi tessellations.",
    )
    parser.add_argument("audio", type=Path, nargs="?", help="Path to the input audio file.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the desktop control panel instead of running directly from the CLI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cymatesserae_output.mp4"),
        help="MP4 output path.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second.")
    parser.add_argument(
        "--points",
        type=int,
        default=180,
        help="Number of Voronoi seed points.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the animation in a window while rendering.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Open a lighter live preview window without exporting frames to FFmpeg.",
    )
    parser.add_argument(
        "--cymatic",
        action="store_true",
        help="Enable cymatic nodal-line attraction mode.",
    )
    parser.add_argument(
        "--plate-mode",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        default=(4, 6),
        help="Plate vibration mode pair for cymatic attraction.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length used for librosa feature extraction.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=2048,
        help="FFT window size used for the STFT.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional limit in seconds for quick renders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic tessellations.",
    )
    parser.add_argument(
        "--style",
        default="ceramic",
        choices=["ceramic", "neon", "lava", "glass", "monolith"],
        help="Primary visual style preset.",
    )
    parser.add_argument(
        "--style-b",
        default="neon",
        choices=["ceramic", "neon", "lava", "glass", "monolith"],
        help="Secondary visual style preset used for morphing.",
    )
    parser.add_argument(
        "--morph-rate",
        type=float,
        default=0.18,
        help="Rate at which the renderer drifts between style presets.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of overlapping tessellation layers.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.35,
        help="Amount of spatial overlap and echo between layers.",
    )
    parser.add_argument(
        "--pattern-layout",
        default="flow",
        choices=["flow", "grid"],
        help="Whether patterns move freely or stay more locked to a grid.",
    )
    parser.add_argument(
        "--grid-strength",
        type=float,
        default=0.82,
        help="How strongly grid layout resists drift and stays stationary.",
    )
    parser.add_argument(
        "--geometry-rigidity",
        type=float,
        default=0.75,
        help="How rigidly points lock to their grid positions in grid layout.",
    )
    parser.add_argument(
        "--layer-rigidity",
        type=float,
        default=0.55,
        help="How much overlapping layers resist shear, offset, and pulse in grid layout.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="How tightly grid and custom elements overlap when using stationary layouts.",
    )
    parser.add_argument(
        "--grid-columns",
        type=int,
        default=0,
        help="Optional explicit number of grid columns when using grid layout. Use 0 to auto-fit.",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=0,
        help="Optional explicit number of grid rows when using grid layout. Use 0 to auto-fit.",
    )
    parser.add_argument(
        "--grid-pattern",
        default="rect",
        choices=["rect", "brick", "hex", "diamond"],
        help="Cell arrangement to use when pattern layout is grid.",
    )
    parser.add_argument(
        "--cell-alternation",
        default="none",
        choices=["none", "orientation", "color", "both"],
        help="Alternate inverse orientation and/or inverse color across grid cells.",
    )
    parser.add_argument(
        "--reorg-mode",
        default="burst",
        choices=["burst", "swirl", "split", "shockwave"],
        help="How percussive hits reorganize tesserae points.",
    )
    parser.add_argument(
        "--graphic-cycle",
        default="voronoi,circles,scribbles,lines,geometrics",
        help="Comma-separated list of graphic families to switch between on the beat.",
    )
    parser.add_argument(
        "--beats-per-switch",
        type=int,
        default=4,
        help="How many beats each graphic family stays active before switching.",
    )
    parser.add_argument(
        "--stack-interaction",
        default="none",
        choices=["none", "crossfade", "shuffle", "pulse", "duck", "spotlight"],
        help="How enabled channels interact while stacked together, including audio-responsive modes.",
    )
    parser.add_argument(
        "--chroma-key-color",
        type=parse_chroma_key_color,
        default=None,
        help="Optional solid background color for chroma keying, as RRGGBB or #RRGGBB.",
    )
    parser.add_argument(
        "--custom-element",
        type=Path,
        nargs="+",
        default=None,
        help="One or more painted sprite assets used by the custom graphic family.",
    )
    parser.add_argument(
        "--graphics-config",
        type=Path,
        default=None,
        help="Optional JSON file describing per-graphic layer settings from the GUI.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.gui:
        from .gui import launch_gui

        launch_gui()
        return 0

    if args.audio is None:
        raise SystemExit("Audio file is required unless you use --gui.")
    if args.audio.name.lower() == "preview" and not args.audio.exists() and not args.preview:
        raise SystemExit(
            "`preview` is a flag, not a subcommand.\n"
            "Use: python -m cymatesserae <audio-file> --preview [options]\n"
            "Or launch the GUI with: python -m cymatesserae --gui"
        )

    from .renderer import RenderConfig, render_project

    config = RenderConfig(
        audio_path=args.audio,
        output_path=normalize_output_path(args.output),
        width=normalize_video_dimension(args.width),
        height=normalize_video_dimension(args.height),
        fps=args.fps,
        point_count=args.points,
        preview=args.preview or args.preview_only,
        preview_only=args.preview_only,
        cymatic_mode=args.cymatic,
        plate_mode=tuple(args.plate_mode),
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        duration_limit=args.duration,
        seed=args.seed,
        style_a=args.style,
        style_b=args.style_b,
        morph_rate=args.morph_rate,
        layer_count=args.layers,
        overlap=args.overlap,
        pattern_layout=args.pattern_layout,
        grid_strength=args.grid_strength,
        geometry_rigidity=args.geometry_rigidity,
        layer_rigidity=args.layer_rigidity,
        tile_overlap=args.tile_overlap,
        grid_columns=args.grid_columns,
        grid_rows=args.grid_rows,
        grid_pattern=args.grid_pattern,
        cell_alternation=args.cell_alternation,
        reorg_mode=args.reorg_mode,
        graphic_cycle=tuple(part.strip() for part in args.graphic_cycle.split(",") if part.strip()),
        beats_per_switch=args.beats_per_switch,
        stack_interaction=args.stack_interaction,
        chroma_key_color=args.chroma_key_color,
        custom_element_paths=tuple(args.custom_element or ()),
        graphic_layers=load_graphic_layers(args.graphics_config),
    )
    render_project(config)
    return 0
