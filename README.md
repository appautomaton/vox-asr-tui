# TNT 🧨

Terminal voice-to-text powered by [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via [antirez/qwen-asr](https://github.com/antirez/qwen-asr) (pure C inference, no PyTorch).

Press Space to record, Space again to transcribe. All local, no network calls.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and a C compiler (for the ASR binary).

On macOS (Apple Silicon), BLAS uses Apple Accelerate automatically (`make blas` path).  
If compile tools are missing, run `xcode-select --install`.

### Platform usage

- Linux/macOS: use the standard setup and run steps in this section.
- Android `proot` (Termux + Debian/Ubuntu): use `README_android.md`.

### Quick start (recommended)

```bash
git clone <your-repo-url>
cd <your-repo-dir>
uv sync
./bootstrap-qwen-asr.sh
uv run tnt
```

Everything stays inside this repo (`bin/`), so there is no `/tmp` build step.
`bootstrap-qwen-asr.sh` uses vendored source at `bin/qwen-asr/`, builds `bin/qwen_asr`, and downloads Qwen3-ASR-0.6B files to `bin/qwen3-asr-0.6b/`.
`bin/qwen-asr/` is intended to be retained and committed in this repository.

`model.safetensors` is ~1.7 GB.

### Run

```bash
uv run tnt
```

## Android

Android `proot` (Termux + Debian/Ubuntu) has a different setup path.  
Use `README_android.md` for full instructions.

## Keybindings

| Key     | Action                                          |
|---------|-------------------------------------------------|
| `Space` | Start/stop recording                            |
| `c`     | Copy last transcript entry to system clipboard  |
| `C`     | Copy all transcript entries to system clipboard |
| `x`     | Clear transcript                                |
| `q`     | Quit                                            |

## Project structure

```
src/tnt/
├── app.py           # Textual TUI, state machine, keybindings
├── audio.py         # Mic capture via sounddevice, WAV encoding
├── transcriber.py   # Subprocess wrapper for qwen_asr binary
└── widgets/
    ├── transcript.py  # Scrollable transcript log
    └── status.py      # Recording indicator + audio level visualizer
bin/
├── qwen-asr              # upstream C source snapshot used for local builds
├── qwen_asr              # compiled C binary (gitignored)
└── qwen3-asr-0.6b/       # model weights (gitignored)
```

## Third-party attribution

- ASR binary source: [`antirez/qwen-asr`](https://github.com/antirez/qwen-asr)
- Upstream license: MIT
- Third-party notice is included in `LICENSE`.

## Notes

- Audio format: 16 kHz, mono, 16-bit PCM — what the qwen_asr binary expects.
- Inference is CPU-only via the C binary. No GPU, no PyTorch, no transformers.
- The binary and model weights are gitignored. Each developer downloads them locally per the instructions above.

## License

MIT. See `LICENSE`.
