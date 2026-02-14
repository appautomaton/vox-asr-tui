"""Mic capture backends for live desktop and Termux API clip recording."""

import io
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - env-dependent import failure
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR: Exception | None = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

CaptureBackend = Literal["live", "termux_api"]


class Recorder(Protocol):
    """Shared recorder interface used by the app state machine."""

    @property
    def is_recording(self) -> bool:
        """Whether capture is currently active."""

    def start(self) -> None:
        """Begin capture."""

    def stop(self) -> bytes:
        """Stop capture and return WAV bytes."""

    def elapsed(self) -> float:
        """Elapsed seconds since capture start."""

    def get_level(self) -> float:
        """Level meter value in range 0.0..1.0."""


def create_recorder(
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
) -> tuple[Recorder, CaptureBackend]:
    """Build the best available recorder backend for this environment."""
    backend = resolve_capture_backend()
    requested = os.environ.get("TNT_CAPTURE_BACKEND", "").strip().lower()
    explicit_backend = requested in {"live", "termux_api"}
    errors: dict[str, str] = {}

    def build_live() -> MicRecorder:
        return MicRecorder(sample_rate=sample_rate, channels=channels, dtype=dtype)

    def build_termux() -> TermuxMicRecorder:
        return TermuxMicRecorder(
            sample_rate=sample_rate,
            channels=channels,
            dtype=dtype,
        )

    if backend == "termux_api":
        try:
            return (build_termux(), "termux_api")
        except RuntimeError as exc:
            errors["termux_api"] = str(exc)
            if explicit_backend:
                raise

        try:
            return (build_live(), "live")
        except RuntimeError as exc:
            errors["live"] = str(exc)
            raise RuntimeError(_format_backend_errors(errors)) from exc

    try:
        return (build_live(), "live")
    except RuntimeError as exc:
        errors["live"] = str(exc)
        if explicit_backend:
            raise

    try:
        return (build_termux(), "termux_api")
    except RuntimeError as exc:
        errors["termux_api"] = str(exc)
        raise RuntimeError(_format_backend_errors(errors)) from exc


def _format_backend_errors(errors: dict[str, str]) -> str:
    """Build a compact error message when all capture backends are unavailable."""
    lines = ["No usable audio capture backend found."]
    for backend in ("live", "termux_api"):
        detail = errors.get(backend)
        if detail:
            lines.append(f"{backend}: {detail}")
    lines.append("Set TNT_CAPTURE_BACKEND=live or TNT_CAPTURE_BACKEND=termux_api.")
    return "\n".join(lines)


def resolve_capture_backend() -> CaptureBackend:
    """Resolve capture backend from env var or runtime environment."""
    requested = os.environ.get("TNT_CAPTURE_BACKEND", "").strip().lower()
    if requested == "live":
        return "live"
    if requested == "termux_api":
        return "termux_api"
    if _in_proot():
        return "termux_api"
    return "live"


def _in_proot() -> bool:
    """Best-effort detection for proot sessions."""
    return any(
        key in os.environ for key in ("PROOT_TMP_DIR", "PROOT_LOADER", "PROOT_VERSION")
    )


def _termux_command_available() -> bool:
    """Whether termux microphone API wrapper is invokable from this shell."""
    return shutil.which("termux-microphone-record") is not None


class MicRecorder:
    """Records audio from the default microphone and returns WAV bytes."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        device: int | str | None = None,
    ) -> None:
        if sd is None:
            detail = str(_SOUNDDEVICE_IMPORT_ERROR) if _SOUNDDEVICE_IMPORT_ERROR else ""
            raise RuntimeError(
                "Live microphone backend unavailable: sounddevice failed to load.\n"
                f"Reason: {detail}\n"
                "Install PortAudio or set TNT_CAPTURE_BACKEND=termux_api."
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = self._resolve_device(device)

        self._stream: Any | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False
        self._start_time: float = 0.0
        self._current_level: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        """Open an InputStream and begin capturing audio."""
        if self._recording:
            return

        with self._lock:
            self._chunks = []
            self._current_level = 0.0

        self._start_time = time.monotonic()
        self._recording = True

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:
            self._recording = False
            self._stream = None
            raise RuntimeError(self._build_mic_error(str(exc))) from exc

    def stop(self) -> bytes:
        """Stop recording, close the stream, return WAV bytes."""
        if not self._recording:
            return b""

        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._chunks:
                return b""
            audio_data = np.concatenate(self._chunks)
            self._chunks = []

        return self._encode_wav(audio_data)

    def elapsed(self) -> float:
        """Seconds since start() was called."""
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    def get_level(self) -> float:
        """Current RMS amplitude normalized to 0.0-1.0."""
        with self._lock:
            return self._current_level

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Called on the audio thread for each chunk."""
        del frames, time_info, status
        chunk = indata.copy()

        # Compute RMS level on a perceptual dB scale.
        samples = chunk.astype(np.float64)
        rms = np.sqrt(np.mean(samples**2))
        # Map RMS to 0.0-1.0 using dB scale (-60 dB .. 0 dB).
        if rms > 1.0:
            db = 20.0 * math.log10(rms / 32767.0)
            normalized = max(0.0, min(1.0, (db + 60.0) / 60.0))
        else:
            normalized = 0.0

        with self._lock:
            self._chunks.append(chunk)
            self._current_level = normalized

    def _encode_wav(self, audio_data: np.ndarray) -> bytes:
        """Encode numpy int16 audio data as WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()

    def _resolve_device(self, explicit: int | str | None) -> int | str | None:
        """Pick input device from argument or TNT_INPUT_DEVICE env var."""
        if explicit is not None:
            return explicit

        env_value = os.environ.get("TNT_INPUT_DEVICE", "").strip()
        if not env_value:
            return None

        if env_value.isdigit():
            return int(env_value)
        return env_value

    def _build_mic_error(self, base_error: str) -> str:
        """Create a user-facing mic error with actionable setup hints."""
        lines = [base_error]
        lines.append("Set TNT_INPUT_DEVICE to an input index/name if needed.")
        lines.extend(self._list_input_hints(limit=5))

        if _termux_command_available():
            lines.append("Fallback available: set TNT_CAPTURE_BACKEND=termux_api.")
        if _in_proot():
            lines.append(
                "proot note: live mic requires host audio forwarding (PulseAudio/PipeWire)."
            )
        return "\n".join(lines)

    def _list_input_hints(self, limit: int = 5) -> list[str]:
        """Return a short list of discoverable input devices."""
        if sd is None:
            return ["No audio device list available from PortAudio."]

        try:
            devices = sd.query_devices()
        except Exception:
            return ["No audio device list available from PortAudio."]

        hints: list[str] = []
        count = 0
        for idx, dev in enumerate(devices):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in <= 0:
                continue
            name = str(dev.get("name", f"device-{idx}"))
            hints.append(f"Input device {idx}: {name} (max_in={max_in})")
            count += 1
            if count >= limit:
                break

        if not hints:
            return ["No input-capable devices reported by PortAudio."]
        return hints


class TermuxMicRecorder:
    """Record clips with termux-microphone-record, then transcode to WAV."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        del dtype  # Termux API controls encoder format; ffmpeg normalizes to WAV.
        self.sample_rate = sample_rate
        self.channels = channels

        self._recording = False
        self._start_time: float = 0.0
        self._tmp_dir: Path | None = None
        self._raw_path: Path | None = None
        self._wav_path: Path | None = None
        self._validate_tools()

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        """Start Termux API recording to a temporary compressed file."""
        if self._recording:
            return

        self._cleanup_paths()
        tmp_dir = Path(tempfile.mkdtemp(prefix="tnt-termux-"))
        raw_path = tmp_dir / "capture.opus"
        wav_path = tmp_dir / "capture.wav"

        # Clear stale recorder state if an earlier session crashed.
        subprocess.run(
            ["termux-microphone-record", "-q"],
            capture_output=True,
            text=True,
            check=False,
        )

        start_cmd = [
            "termux-microphone-record",
            "-f",
            str(raw_path),
            "-e",
            "opus",
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-l",
            "0",
        ]
        result = subprocess.run(start_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self._cleanup_paths(tmp_dir)
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                "Failed to start termux microphone recording.\n"
                f"{stderr or 'Unknown error from termux-microphone-record.'}"
            )

        self._tmp_dir = tmp_dir
        self._raw_path = raw_path
        self._wav_path = wav_path
        self._start_time = time.monotonic()
        self._recording = True

    def stop(self) -> bytes:
        """Stop recording, convert clip to WAV, and return bytes."""
        if not self._recording:
            return b""

        self._recording = False
        quit_result = subprocess.run(
            ["termux-microphone-record", "-q"],
            capture_output=True,
            text=True,
            check=False,
        )

        raw_path = self._raw_path
        wav_path = self._wav_path
        if raw_path is None or wav_path is None:
            self._cleanup_paths()
            return b""

        for _ in range(20):
            if raw_path.exists() and raw_path.stat().st_size > 0:
                break
            time.sleep(0.05)

        if not raw_path.exists() or raw_path.stat().st_size == 0:
            self._cleanup_paths()
            return b""

        ffmpeg_cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "wav",
            str(wav_path),
        ]
        convert = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if convert.returncode != 0:
            stderr = (convert.stderr or convert.stdout or "").strip()
            self._cleanup_paths()
            raise RuntimeError(
                "Failed to convert Termux recording to WAV with ffmpeg.\n"
                f"{stderr or 'Unknown ffmpeg conversion error.'}"
            )

        if not wav_path.exists():
            self._cleanup_paths()
            raise RuntimeError("WAV conversion completed but output file was not created.")

        wav_bytes = wav_path.read_bytes()
        self._cleanup_paths()

        if not wav_bytes and quit_result.returncode != 0:
            stderr = (quit_result.stderr or quit_result.stdout or "").strip()
            raise RuntimeError(
                "Failed to stop termux microphone recording.\n"
                f"{stderr or 'Unknown error from termux-microphone-record -q.'}"
            )

        return wav_bytes

    def elapsed(self) -> float:
        """Seconds since recording started."""
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    def get_level(self) -> float:
        """Clip backend has no live meter; return a constant active level."""
        return 0.25 if self._recording else 0.0

    def _cleanup_paths(self, tmp_dir: Path | None = None) -> None:
        """Remove temporary recording directory."""
        target = tmp_dir if tmp_dir is not None else self._tmp_dir
        if target is not None:
            shutil.rmtree(target, ignore_errors=True)
        self._tmp_dir = None
        self._raw_path = None
        self._wav_path = None

    def _validate_tools(self) -> None:
        """Validate required shell tools for termux capture backend."""
        missing: list[str] = []
        if shutil.which("termux-microphone-record") is None:
            missing.append("termux-microphone-record not found in PATH")
        if shutil.which("ffmpeg") is None:
            missing.append("ffmpeg not found in PATH")

        if missing:
            details = "\n".join(missing)
            raise RuntimeError(
                "Termux capture backend unavailable.\n"
                f"{details}\n"
                "Install termux-api tools and ffmpeg, or set TNT_CAPTURE_BACKEND=live."
            )
