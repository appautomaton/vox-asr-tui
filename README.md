# TNT 🧨

Terminal voice-to-text with local ASR backends:

- Moonshine v2 medium-streaming (`moonshine-streaming-medium`) via Moonshine C API
- Qwen3-ASR-1.7B via `qwen_asr` C binary

Tap `Space` to start recording, tap it again to transcribe, or hold `Space` to record until release. All local, no runtime network calls.

## Setup

> [!NOTE]
> Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), a C/C++ toolchain, and CMake.
>
> Reference platform: macOS arm64 laptops. Linux remains secondary and Android/Termux is not supported on this branch.
>
> On macOS, if compile tools are missing, run `xcode-select --install`.

### Quick start

```bash
git clone https://github.com/appautomaton/tnt-asr.git
cd tnt-asr
uv sync
./bootstrap-moonshine.sh
# Optional fallback backend
./bootstrap-qwen-asr.sh
uv run tnt
```

### Backend setup scripts

- `./bootstrap-moonshine.sh`
  - Downloads pinned Moonshine source (`v0.0.49`)
  - Builds `bin/moonshine_asr`
  - Installs runtime libs under `bin/moonshine-runtime/<platform>/`
  - Downloads medium-streaming model files to `bin/moonshine-models/medium-streaming-en/`
- `./bootstrap-qwen-asr.sh`
  - Builds `bin/qwen_asr` from vendored `bin/qwen-asr/`
  - Downloads Qwen3-ASR-1.7B files to `bin/qwen3-asr-1.7b/`

### Run

```bash
uv run tnt
```

## ASR backend selection

- Default backend: `moonshine`
- Backend env var: `TNT_ASR_BACKEND=moonshine` or `TNT_ASR_BACKEND=qwen`
- If selected backend is missing, TNT automatically falls back to the other backend and shows a warning.

## Keybindings

| Key | Action |
|-----|--------|
| <kbd>Space</kbd> | Start / stop recording, or hold-to-record until release |
| <kbd>m</kbd> | Switch ASR backend (idle only) |
| <kbd>c</kbd> | Copy last transcript entry to clipboard |
| <kbd>C</kbd> | Copy all transcript entries to clipboard |
| <kbd>x</kbd> | Clear transcript |
| <kbd>q</kbd> | Quit |

## Project structure

```text
src/tnt/
├── app.py             # Textual TUI, state machine, keybindings
├── audio.py           # Live microphone capture
├── transcriber.py     # ASR backend wrappers (moonshine + qwen)
└── widgets/
    ├── transcript.py  # Scrollable transcript log
    └── status.py      # Recording indicator + audio level visualizer
csrc/
└── moonshine_asr.cpp  # Native helper (stdin WAV -> Moonshine C API)
bin/
├── qwen-asr/          # Upstream qwen-asr C source snapshot (committed)
├── qwen_asr           # Compiled qwen helper (gitignored)
├── qwen3-asr-1.7b/    # Qwen model files (gitignored)
├── moonshine_asr      # Compiled moonshine helper (gitignored)
├── moonshine-runtime/ # Moonshine + ONNX runtime libs (gitignored)
└── moonshine-models/  # Moonshine model files (gitignored)
```

## Notes

> [!TIP]
> - Input audio expected by inference path: 16 kHz, mono PCM WAV.
> - Inference is CPU-only and local.
> - Runtime model/binary artifacts are gitignored and fetched/built locally via bootstrap.

## Third-party attribution

- Qwen ASR C implementation: [`antirez/qwen-asr`](https://github.com/antirez/qwen-asr)
- Moonshine Voice: [`moonshine-ai/moonshine`](https://github.com/moonshine-ai/moonshine)
- ONNX Runtime: [`microsoft/onnxruntime`](https://github.com/microsoft/onnxruntime)
- Third-party notice is included in [`LICENSE`](LICENSE).

## License

MIT. See [`LICENSE`](LICENSE).
