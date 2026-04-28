from __future__ import annotations

import argparse
from pathlib import Path

from .renderer import RenderConfig, render_project


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
        "--chroma-key-color",
        type=parse_chroma_key_color,
        default=None,
        help="Optional solid background color for chroma keying, as RRGGBB or #RRGGBB.",
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

    config = RenderConfig(
        audio_path=args.audio,
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        point_count=args.points,
        preview=args.preview,
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
        reorg_mode=args.reorg_mode,
        graphic_cycle=tuple(part.strip() for part in args.graphic_cycle.split(",") if part.strip()),
        beats_per_switch=args.beats_per_switch,
        chroma_key_color=args.chroma_key_color,
    )
    render_project(config)
    return 0
