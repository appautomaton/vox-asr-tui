# Agent Guidelines for TNT

## Mission

Keep TNT functional, stable, and usable first. Prioritize runtime reliability and clear user-visible errors over polish work.

## Project summary

TNT is a terminal voice-to-text TUI:
- `Space` starts recording
- `Space` stops recording and triggers transcription
- transcription runs locally with `qwen_asr` (pure C, CPU-only)

## Platform matrix

- Linux/macOS:
  - default backend: `live` (`sounddevice` + PortAudio)
- Android `proot` (Termux + Debian/Ubuntu):
  - default backend: `termux_api` (clip recording + ffmpeg transcode)
- override on any platform:
  - `TNT_CAPTURE_BACKEND=live`
  - `TNT_CAPTURE_BACKEND=termux_api`

## Non-negotiables

- No network calls at runtime.
- No PyTorch, transformers, or CUDA.
- Use `uv` only (`uv sync`, `uv run`, `uv add`).
- Keep runtime dependencies minimal (`textual`, `sounddevice`, `numpy` + stdlib).
- Keep blocking work off the UI path (use async/worker patterns).

## Source layout

```text
src/tnt/
  app.py           # TUI state machine and keybindings
  audio.py         # capture backends (live + termux_api)
  transcriber.py   # qwen_asr subprocess wrapper
  widgets/
    transcript.py
    status.py
bin/
  qwen-asr/
  qwen_asr
  qwen3-asr-0.6b/
```

## Bootstrap and artifacts

- Canonical bootstrap command:
  - `./bootstrap-qwen-asr.sh`
- Expected outputs:
  - binary: `bin/qwen_asr`
  - model files: `bin/qwen3-asr-0.6b/` (including `.safetensors`)
- Vendored source location:
  - `bin/qwen-asr/`
  - bootstrap expects this source tree to already be present and committed.

## Audio contract

- Required format for inference: 16 kHz, mono, 16-bit PCM WAV.
- App state flow:
  - `idle -> recording -> transcribing -> idle`

## Validation commands

```bash
uv sync
uv run ruff check src/
uv run tnt
```
