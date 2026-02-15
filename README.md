# TNT 🧨

Terminal voice-to-text powered by [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via [antirez/qwen-asr](https://github.com/antirez/qwen-asr) (pure C inference, no PyTorch).

Press Space to record, Space again to transcribe. All local, no network calls.

## Setup

> [!NOTE]
> Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and a C compiler (for the ASR binary).
>
> On macOS (Apple Silicon), BLAS uses Apple Accelerate automatically (`make blas` path).
> If compile tools are missing, run `xcode-select --install`.

| Platform | Instructions |
|----------|-------------|
| Linux / macOS | Follow the steps below |
| Android proot (Termux + Debian/Ubuntu) | See [`README_android.md`](README_android.md) |

### Quick start

```bash
git clone https://github.com/appautomaton/tnt-asr.git
cd tnt-asr
uv sync
./bootstrap-qwen-asr.sh
uv run tnt
```

> [!IMPORTANT]
> `model.safetensors` is ~1.7 GB. The bootstrap script will download it on first run.

<details>
<summary>What does bootstrap do?</summary>

- Uses vendored source at `bin/qwen-asr/` to compile `bin/qwen_asr`
- Downloads Qwen3-ASR-0.6B model files to `bin/qwen3-asr-0.6b/`
- Everything stays inside the repo (`bin/`), no `/tmp` build step
- `bin/qwen-asr/` is intended to be retained and committed in this repository

</details>

### Run

```bash
uv run tnt
```

## Keybindings

| Key | Action |
|-----|--------|
| <kbd>Space</kbd> | Start / stop recording |
| <kbd>c</kbd> | Copy last transcript entry to clipboard |
| <kbd>C</kbd> | Copy all transcript entries to clipboard |
| <kbd>x</kbd> | Clear transcript |
| <kbd>q</kbd> | Quit |

## Project structure

```text
src/tnt/
├── app.py             # Textual TUI, state machine, keybindings
├── audio.py           # Capture backends (live + termux_api)
├── transcriber.py     # Subprocess wrapper for qwen_asr binary
└── widgets/
    ├── transcript.py  # Scrollable transcript log
    └── status.py      # Recording indicator + audio level visualizer
bin/
├── qwen-asr/          # Upstream C source snapshot (committed)
├── qwen_asr           # Compiled binary (gitignored)
└── qwen3-asr-0.6b/    # Model weights (gitignored)
```

## Notes

> [!TIP]
> - Audio format: 16 kHz, mono, 16-bit PCM — what the `qwen_asr` binary expects.
> - Inference is CPU-only via the C binary. No GPU, no PyTorch, no transformers.
> - The binary and model weights are gitignored. Each developer downloads them locally via bootstrap.

## Third-party attribution

- ASR binary source: [`antirez/qwen-asr`](https://github.com/antirez/qwen-asr)
- Upstream license: MIT
- Third-party notice is included in [`LICENSE`](LICENSE).

## License

MIT. See [`LICENSE`](LICENSE).
