# Agent Guidelines for TNT

## Mission

Keep TNT functional, stable, and usable first. Prioritize runtime reliability and clear user-visible errors over polish work.

## Project summary

TNT is a terminal voice-to-text TUI:
- tap `Space` to start recording
- tap `Space` again to stop and transcribe
- hold `Space` to record until release
- `Space` during transcription cancels it
- default ASR: Moonshine v2 medium-streaming (`moonshine_asr` C++ binary, CPU-only)
- fallback ASR: Qwen3-ASR-1.7B (`qwen_asr` pure C binary, CPU-only)

## Platform matrix

- macOS arm64 laptops:
  - primary supported platform
  - capture backend: `live` (`sounddevice` + PortAudio)
  - default ASR backend: `moonshine`
- Linux laptops/desktops:
  - secondary support target
  - capture backend: `live` (`sounddevice` + PortAudio)
  - default ASR backend: `moonshine`
- override on supported platforms:
  - `TNT_ASR_BACKEND=moonshine`
  - `TNT_ASR_BACKEND=qwen`
  - `TNT_INPUT_DEVICE=<index-or-name>`
- unsupported:
  - Android / Termux / proot

## Non-negotiables

- No network calls at runtime.
- No PyTorch, transformers, or CUDA.
- Use `uv` only (`uv sync`, `uv run`, `uv add`).
- Keep runtime dependencies minimal (`textual`, `sounddevice`, `numpy` + stdlib).
- Keep blocking work off the UI path (use async/worker patterns).

## Source layout

\`\`\`text
src/tnt/
  app.py           # TUI state machine and keybindings
  audio.py         # live microphone capture
  transcriber.py   # ASR backend wrappers (moonshine + qwen)
  widgets/
    transcript.py
    status.py
csrc/
  moonshine_asr.cpp  # Native C++ helper: stdin WAV -> Moonshine C API
bin/
  qwen-asr/          # Vendored qwen-asr C source (committed)
  qwen_asr           # Compiled qwen helper (gitignored)
  qwen3-asr-1.7b/    # Qwen model files (gitignored)
  moonshine_asr      # Compiled moonshine helper (gitignored)
  moonshine-runtime/ # Moonshine + ONNX runtime libs (gitignored)
  moonshine-models/  # Moonshine model files (gitignored)
  moonshine-src/     # Downloaded Moonshine source (gitignored)
\`\`\`

## Bootstrap and artifacts

- Moonshine bootstrap (primary):
  - `./bootstrap-moonshine.sh`
  - Downloads Moonshine source `v0.0.49`, builds `libmoonshine`, builds `bin/moonshine_asr`
  - Downloads `medium-streaming-en` quantized model to `bin/moonshine-models/medium-streaming-en/`
  - Installs runtime libs to `bin/moonshine-runtime/<platform>/`
- Qwen bootstrap (optional fallback):
  - `./bootstrap-qwen-asr.sh`
  - Builds `bin/qwen_asr` from vendored `bin/qwen-asr/`
  - Downloads Qwen3-ASR-1.7B to `bin/qwen3-asr-1.7b/`

## Audio contract

- Required format for inference: 16 kHz, mono, 16-bit PCM WAV.
- App state flow:
  - `idle -> recording -> transcribing -> idle`

## Validation commands

\`\`\`bash
uv sync
uv run ruff check src/
uv run tnt
\`\`\`
