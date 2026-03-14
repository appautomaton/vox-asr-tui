"""ASR transcription backends via local subprocess binaries."""

import asyncio
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Literal, Protocol

AsrBackend = Literal["moonshine", "qwen"]
DEFAULT_ASR_BACKEND: AsrBackend = "moonshine"


class Transcriber(Protocol):
    """Shared interface used by the TUI for all ASR backends."""

    @property
    def backend(self) -> AsrBackend:
        """Backend identifier."""

    @property
    def model_label(self) -> str:
        """Human-readable model label for UI rendering."""

    async def transcribe_async(self, wav_bytes: bytes, timeout: float = 120) -> str:
        """Run transcription asynchronously with timeout."""

    def kill_process(self) -> None:
        """Kill any in-flight subprocess work."""


def resolve_asr_backend() -> AsrBackend:
    """Resolve ASR backend from env var with a moonshine default."""
    requested = os.environ.get("TNT_ASR_BACKEND", "").strip().lower()
    if requested == "qwen":
        return "qwen"
    if requested == "moonshine":
        return "moonshine"
    return DEFAULT_ASR_BACKEND


def other_asr_backend(backend: AsrBackend) -> AsrBackend:
    """Return the secondary backend used for fallback."""
    return "qwen" if backend == "moonshine" else "moonshine"


def model_label_for_backend(backend: AsrBackend) -> str:
    """Return the display label for a backend."""
    return "moonshine-streaming-medium" if backend == "moonshine" else "qwen3-asr-1.7b"


def hint_label_for_backend(backend: AsrBackend) -> str:
    """Return a compact backend label for narrow UI areas."""
    return "moonshine" if backend == "moonshine" else "qwen"


def create_transcriber(backend: AsrBackend) -> Transcriber:
    """Instantiate a backend transcriber."""
    if backend == "moonshine":
        return MoonshineTranscriber()
    return QwenTranscriber()


def create_transcriber_with_fallback(
    preferred_backend: AsrBackend,
) -> tuple[Transcriber, AsrBackend, str | None]:
    """Create transcriber with automatic fallback to the alternate backend."""
    first_error: Exception | None = None
    try:
        return create_transcriber(preferred_backend), preferred_backend, None
    except Exception as exc:
        first_error = exc

    fallback_backend = other_asr_backend(preferred_backend)
    try:
        transcriber = create_transcriber(fallback_backend)
    except Exception as fallback_exc:
        raise RuntimeError(
            "No usable ASR backend found.\n"
            f"{preferred_backend}: {first_error}\n"
            f"{fallback_backend}: {fallback_exc}"
        ) from fallback_exc

    warning = (
        f"{preferred_backend} unavailable: {first_error}. "
        f"Falling back to {fallback_backend}."
    )
    return transcriber, fallback_backend, warning


class _SubprocessTranscriber:
    """Common subprocess wrapper used by concrete backend adapters."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._killed = False

    @property
    def backend(self) -> AsrBackend:
        raise NotImplementedError

    @property
    def model_label(self) -> str:
        raise NotImplementedError

    def _command(self) -> list[str]:
        raise NotImplementedError

    def _process_env(self) -> dict[str, str] | None:
        return None

    def transcribe(self, wav_bytes: bytes, timeout: float = 120) -> str:
        """Run transcription synchronously. Returns transcribed text."""
        return self._transcribe_sync(wav_bytes, timeout)

    async def transcribe_async(self, wav_bytes: bytes, timeout: float = 120) -> str:
        """Run transcription in a worker thread with cancellation support."""
        self._killed = False
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, self._transcribe_sync, wav_bytes, timeout)
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            self.kill_process()
            # Wait for the worker thread to unblock after the process is killed.
            try:
                await asyncio.wait_for(fut, timeout=5)
            except Exception:
                pass
            raise

    def _transcribe_sync(self, wav_bytes: bytes, timeout: float) -> str:
        """Blocking transcription via Popen.communicate()."""
        if self._killed:
            raise asyncio.CancelledError()
        proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._process_env(),
        )
        self._proc = proc
        # Check if kill_process() fired between Popen and the assignment above.
        if self._killed:
            try:
                proc.kill()
                proc.communicate()
            except Exception:
                pass
            raise asyncio.CancelledError()
        try:
            stdout, stderr = proc.communicate(input=wav_bytes, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise asyncio.TimeoutError() from None
        finally:
            self._proc = None

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"{self.backend} exited with code {proc.returncode}: {stderr_text}"
            )

        return stdout.decode("utf-8", errors="replace").strip()

    def kill_process(self) -> None:
        """Kill any active subprocess and close its stdin to unblock communicate()."""
        self._killed = True
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.kill()
            except ProcessLookupError:
                pass


class QwenTranscriber(_SubprocessTranscriber):
    """Wraps the qwen_asr binary for speech-to-text transcription."""

    REQUIRED_MODEL_FILES = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "vocab.json",
        "merges.txt",
    )

    def __init__(
        self,
        binary_path: str = "bin/qwen_asr",
        model_dir: str = "bin/qwen3-asr-1.7b",
    ) -> None:
        super().__init__()
        self.binary_path = Path(binary_path).resolve()
        self.model_dir = Path(model_dir).resolve()
        self._validate()

    @property
    def backend(self) -> AsrBackend:
        return "qwen"

    @property
    def model_label(self) -> str:
        return "qwen3-asr-1.7b"

    def _validate(self) -> None:
        """Check that qwen binary and model directory are present."""
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"qwen_asr binary not found at {self.binary_path}\n"
                "Run ./bootstrap-qwen-asr.sh from the project root."
            )
        if not os.access(self.binary_path, os.X_OK):
            raise FileNotFoundError(
                f"qwen_asr binary at {self.binary_path} is not executable.\n"
                f"Run: chmod +x {self.binary_path}"
            )
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found at {self.model_dir}\n"
                "Run ./bootstrap-qwen-asr.sh from the project root."
            )
        missing = [
            name for name in self.REQUIRED_MODEL_FILES if not (self.model_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Qwen model is incomplete at {self.model_dir}\n"
                f"Missing files: {', '.join(missing)}\n"
                "Run ./bootstrap-qwen-asr.sh from the project root."
            )

    def _command(self) -> list[str]:
        return [
            str(self.binary_path),
            "-d",
            str(self.model_dir),
            "--stdin",
            "--silent",
        ]


class MoonshineTranscriber(_SubprocessTranscriber):
    """Wraps the moonshine_asr helper binary for Moonshine v2 streaming."""

    REQUIRED_MODEL_FILES = (
        "adapter.ort",
        "cross_kv.ort",
        "decoder_kv.ort",
        "encoder.ort",
        "frontend.ort",
        "streaming_config.json",
        "tokenizer.bin",
    )

    def __init__(
        self,
        binary_path: str = "bin/moonshine_asr",
        model_dir: str = "bin/moonshine-models/medium-streaming-en",
        runtime_root: str = "bin/moonshine-runtime",
    ) -> None:
        super().__init__()
        self.binary_path = Path(binary_path).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.runtime_dir = self.runtime_root / _runtime_platform_id()
        self._validate()

    @property
    def backend(self) -> AsrBackend:
        return "moonshine"

    @property
    def model_label(self) -> str:
        return "moonshine-streaming-medium"

    def _validate(self) -> None:
        """Check that moonshine helper, runtime, and model artifacts are present."""
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"moonshine_asr binary not found at {self.binary_path}\n"
                "Run ./bootstrap-moonshine.sh from the project root."
            )
        if not os.access(self.binary_path, os.X_OK):
            raise FileNotFoundError(
                f"moonshine_asr at {self.binary_path} is not executable.\n"
                f"Run: chmod +x {self.binary_path}"
            )
        if not self.runtime_dir.exists():
            raise FileNotFoundError(
                f"Moonshine runtime directory not found at {self.runtime_dir}\n"
                "Run ./bootstrap-moonshine.sh from the project root."
            )
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Moonshine model directory not found at {self.model_dir}\n"
                "Run ./bootstrap-moonshine.sh from the project root."
            )
        missing = [
            name for name in self.REQUIRED_MODEL_FILES if not (self.model_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Moonshine model is incomplete at {self.model_dir}\n"
                f"Missing files: {', '.join(missing)}\n"
                "Run ./bootstrap-moonshine.sh from the project root."
            )

    def _command(self) -> list[str]:
        return [
            str(self.binary_path),
            "--model-dir",
            str(self.model_dir),
            "--model-arch",
            "medium-streaming",
        ]

    def _process_env(self) -> dict[str, str] | None:
        env = os.environ.copy()
        if sys.platform == "darwin":
            key = "DYLD_LIBRARY_PATH"
        else:
            key = "LD_LIBRARY_PATH"

        existing = env.get(key, "")
        runtime = str(self.runtime_dir)
        env[key] = f"{runtime}:{existing}" if existing else runtime
        return env


def _runtime_platform_id() -> str:
    """Map runtime platform to the local moonshine-runtime artifact directory."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        if machine != "arm64":
            raise RuntimeError(
                f"Unsupported Moonshine platform: macOS {machine}. Expected arm64."
            )
        return "macos-arm64"

    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            return "linux-x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux-aarch64"
        raise RuntimeError(
            f"Unsupported Moonshine platform: Linux {machine}. "
            "Expected x86_64 or aarch64."
        )

    raise RuntimeError(f"Unsupported Moonshine platform: {sys.platform}/{machine}")
