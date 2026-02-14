# TNT on Android (Termux + proot)

This guide is only for Android + Termux + `proot` distro (Debian/Ubuntu, etc.).

## Important

Do not use `--isolated` when launching your distro. TNT needs Termux-side tools/paths.

Use:

```bash
proot-distro login --user dev debian
```

Avoid:

```bash
proot-distro login --isolated --user dev debian
```

If you use a wrapper like `pd sh`, remove `--isolated` there too.

## 1) Termux host setup (outside proot)

Install from F-Droid:

- `Termux`
- `Termux:API`

Install host package:

```bash
pkg update
pkg install -y termux-api
```

Grant microphone permission when Android prompts.

## 2) proot setup (inside distro)

```bash
sudo apt update
sudo apt install -y build-essential git libopenblas-dev libportaudio2 portaudio19-dev ffmpeg
```

`libopenblas-dev` is important. Without it, bootstrap builds a slower non-BLAS `qwen_asr`.

## 3) Project setup (repo root, inside proot)

```bash
rm -rf .venv
UV_NO_CACHE=1 UV_LINK_MODE=copy uv sync --refresh --reinstall
./bootstrap-qwen-asr.sh
```

Bootstrap behavior is fixed:

- uses vendored `bin/qwen-asr`
- builds `bin/qwen_asr`
- downloads Qwen3-ASR-0.6B files into `bin/qwen3-asr-0.6b`

## 4) Run

```bash
./launch-tnt-proot.sh
```

Launcher behavior:

- sets `PATH` for Termux binaries
- sets `TMPDIR`
- defaults `TNT_CAPTURE_BACKEND=termux_api`
- runs `uv run tnt`

## Quick checks

```bash
command -v termux-microphone-record
command -v ffmpeg
uv run which tnt
```

## Troubleshooting

- No mic capture:
  - verify you are not using `--isolated`
  - ensure `termux-api` is installed on host and `Termux:API` app is installed
- Slow transcription:
  - install `libopenblas-dev` and rerun `./bootstrap-qwen-asr.sh`
- Wheel/import issues:
  - `rm -rf .venv && UV_NO_CACHE=1 UV_LINK_MODE=copy uv sync --refresh --reinstall`
- Still no audio after permission prompt:
  - test manually in proot:
    - `termux-microphone-record -f test.m4a -e aac -r 16000 -c 1 -l 2`
