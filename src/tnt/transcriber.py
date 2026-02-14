"""ASR transcription via the qwen_asr C binary."""

import asyncio
import os
import subprocess
from pathlib import Path


class QwenTranscriber:
    """Wraps the qwen_asr binary for speech-to-text transcription."""

    def __init__(
        self,
        binary_path: str = "bin/qwen_asr",
        model_dir: str = "bin/qwen3-asr-0.6b",
    ) -> None:
        self.binary_path = Path(binary_path).resolve()
        self.model_dir = Path(model_dir).resolve()

        self._validate()

    def _validate(self) -> None:
        """Check that the binary and model directory exist."""
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
        safetensor_files = list(self.model_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {self.model_dir}\n"
                "The model directory must contain the Qwen3-ASR-0.6B "
                "safetensors weight files."
            )

    def transcribe(self, wav_bytes: bytes) -> str:
        """Run transcription synchronously. Returns the transcribed text."""
        result = subprocess.run(
            [
                str(self.binary_path),
                "-d",
                str(self.model_dir),
                "--stdin",
                "--silent",
            ],
            input=wav_bytes,
            capture_output=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"qwen_asr exited with code {result.returncode}: {stderr}")

        return result.stdout.decode("utf-8").strip()

    async def transcribe_async(self, wav_bytes: bytes) -> str:
        """Run transcription asynchronously. Returns the transcribed text."""
        proc = await asyncio.create_subprocess_exec(
            str(self.binary_path),
            "-d",
            str(self.model_dir),
            "--stdin",
            "--silent",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=wav_bytes)

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"qwen_asr exited with code {proc.returncode}: {stderr_text}"
            )

        return stdout.decode("utf-8").strip()
