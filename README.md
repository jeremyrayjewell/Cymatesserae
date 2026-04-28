# Cymatesserae

Cymatesserae is a Python-based audiovisual engine that converts audio signals into responsive, beat-synchronized visual systems.

It generates cymatic-style mosaics, switches between graphic families based on rhythmic structure, and exports synchronized video output.

Part of the Aggregatron Records experimental media toolkit.

---

## What It Does

- Analyzes audio for bass energy, spectral centroid, contrast, harmonic/percussive content, BPM, and beat positions.
- Maps features to motion, color, tessellation scale, and style transitions.
- Renders multiple visual families: voronoi, circles, scribbles, lines, geometrics.
- Supports cymatic nodal-line attraction (plate simulation behavior).
- Exports synchronized MP4 via FFmpeg.
- Includes a desktop GUI for interactive control.
- Supports chroma-key output for compositing workflows.

---

## Pipeline

audio → feature extraction → parameter mapping → visual system → frame rendering → video export

---

## Install

From the project root:

```powershell
python -m pip install -r requirements.txt
```

Requirements:

- Python 3.13 is supported in this repo's current setup  
- Recommended: Python 3.10–3.12 for best library compatibility  
- `ffmpeg` must be available on your `PATH`

---

## Quick Start

Preview with the GUI:

```powershell
python -m cymatesserae --gui
```

Preview from the CLI:

```powershell
python -m cymatesserae ".\input.wav" --preview --cymatic
```

Export an MP4:

```powershell
python -m cymatesserae ".\input.wav" --output ".\output.mp4" --width 1280 --height 720 --fps 30 --cymatic
```

---

## Core Ideas

The renderer combines several systems:

- Audio analysis  
  STFT-derived features drive motion and color decisions

- BPM and beat detection  
  Used to switch between graphic families

- Cymatic motion  
  Points are attracted toward nodal lines of a vibrating plate

- Layered rendering  
  Multiple overlapping passes create density and motion echo

- Style morphing  
  Two style presets can blend over time

---

## Graphic Families

- `voronoi`  
  Tessellated shards and cell structures

- `circles`  
  Bubble fields, rings, and pulse-driven clusters

- `scribbles`  
  Loose gestural strands and hand-drawn motion

- `lines`  
  Directional streaks and vector-field sweeps

- `geometrics`  
  Angular shards and crystalline bursts

These can be cycled or combined using beat-driven switching.

---

## GUI

Launch the desktop control panel:

```powershell
python -m cymatesserae --gui
```

The GUI includes controls for:

- audio file and output path
- render size, FPS, seed, and point count
- cymatic mode and plate modes
- style presets and morph rate
- layer count and overlap
- reorg mode (burst, swirl, split, shockwave)
- beat-driven graphic switching
- chroma-key background color

If a GUI render fails, the latest run is logged to:

`cymatesserae_gui_last_run.log`

---

## CLI Options

Common options:

- `--preview`
- `--output output.mp4`
- `--width 1280`
- `--height 720`
- `--fps 30`
- `--points 180`
- `--duration 10`
- `--seed 7` (deterministic rendering)

Style and motion:

- `--style ceramic|neon|lava|glass|monolith`
- `--style-b ceramic|neon|lava|glass|monolith`
- `--morph-rate 0.18`
- `--layers 3`
- `--overlap 0.35`
- `--reorg-mode burst|swirl|split|shockwave`

Cymatic controls:

- `--cymatic`
- `--plate-mode 4 6`

Beat-driven switching:

- `--graphic-cycle voronoi,circles,scribbles,lines,geometrics`
- `--beats-per-switch 4`

Chroma key:

- `--chroma-key-color 00ff00`
- `--chroma-key-color ff00ff`

---

## Example Commands

Warm cymatic shard render (layered morphing + shockwave reorg):

```powershell
python -m cymatesserae ".\input.wav" --preview --cymatic --style ceramic --style-b lava --morph-rate 0.3 --layers 3 --overlap 0.45 --reorg-mode shockwave
```

Cold electric multi-family switcher (fast beat-driven transitions):

```powershell
python -m cymatesserae ".\input.wav" --preview --cymatic --style neon --style-b monolith --graphic-cycle voronoi,circles,scribbles,lines,geometrics --beats-per-switch 2
```

Green-screen export for compositing:

```powershell
python -m cymatesserae ".\input.wav" --output ".\greenscreen.mp4" --style glass --style-b neon --graphic-cycle circles,lines,geometrics --beats-per-switch 1 --chroma-key-color 00ff00
```

---

## Chroma Key Notes

When `--chroma-key-color` is set:

- the background becomes a strict solid color  
- that exact RGB value is reserved for the background  
- foreground colors are adjusted if they collide with the key  
- output minimizes artifacts on the key color channel for cleaner compositing

---

## Project Structure

- `cymatesserae/cli.py`  
  command-line entry point

- `cymatesserae/gui.py`  
  desktop control panel

- `cymatesserae/renderer.py`  
  audio analysis, motion system, and rendering/export pipeline

- `requirements.txt`  
  Python dependencies

---

## Performance

- CPU-based rendering
- Real-time preview depends on point count and resolution
- Export time scales roughly linearly with duration and FPS

---

## Limitations

- No GPU acceleration yet
- High-resolution exports can be slow
- Feature extraction accuracy depends on audio quality

---

## Troubleshooting

If preview from the GUI appears to do nothing:

- check the GUI status line
- open `cymatesserae_gui_last_run.log`
- confirm the audio path is valid
- confirm `ffmpeg` is available on `PATH`

To stop a long export in PowerShell:

```powershell
Ctrl+C
```

If that fails:

```powershell
Stop-Process -Name ffmpeg -Force
Stop-Process -Name python -Force
```

---

## Notes

- Video time is mapped directly to audio duration for synchronization  
- Some `librosa` paths may stall under Python 3.13; SciPy/SoundFile fallbacks are used  
- The system retains a tessera-based motion lineage, though current visual families are more divergent  

---

## Credits

Created for the Aggregatron ecosystem.

**Aggregatron Records**  
Experimental electronic releases, visuals, and artist-world systems.

---

## Author

Jeremy Ray Jewell  
GitHub: https://github.com/jeremyrayjewell  
LinkedIn: https://www.linkedin.com/in/jeremyrayjewell