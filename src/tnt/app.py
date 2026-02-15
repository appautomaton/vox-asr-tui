"""Textual TUI app for voice-to-text transcription."""

import asyncio
import subprocess

from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from tnt.audio import Recorder, create_recorder
from tnt.transcriber import QwenTranscriber
from tnt.widgets.status import StatusPanel
from tnt.widgets.transcript import TranscriptView


class HeaderBar(Static):
    """Custom header: title left, model info + state indicator right."""

    DEFAULT_CSS = """
    HeaderBar {
        dock: top;
        height: 1;
        background: #100025;
        color: #f8f4ff;
        padding: 0 1;
    }
    """

    state: reactive[str] = reactive("idle")

    def render(self) -> Table:
        left = Text()
        left.append("● ", style="bold #39ff14")
        left.append("TNT", style="bold #ff4fd8")
        left.append(" 🧨", style="bold #ffb703")
        left.append(" — voice → text", style="#7afcff")

        right = Text()
        right.append("qwen3-asr-0.6b", style="bold #8be9fd")
        right.append(" │ ", style="#7a6aa5")
        right.append("16kHz", style="bold #f1fa8c")
        right.append(" │ ", style="#7a6aa5")
        match self.state:
            case "idle":
                right.append("▮▮ IDLE", style="bold #7afcff")
            case "recording":
                right.append("● REC", style="bold #ff5ccf")
            case "transcribing":
                right.append("◌ ...", style="bold #ffd166")

        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            expand=True,
            padding=0,
        )
        table.add_column(justify="left", ratio=1, no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        table.add_row(left, right)
        return table


class HintBar(Static):
    """Bottom bar showing keybindings with state-dependent labels."""

    DEFAULT_CSS = """
    HintBar {
        dock: bottom;
        height: 1;
        background: #140a2e;
        color: #f8f4ff;
        padding: 0 1;
    }
    """

    state: reactive[str] = reactive("idle")

    def render(self) -> Text:
        action = "stop" if self.state == "recording" else "record"
        action_color = "#ff8ad8" if self.state == "recording" else "#9bff7a"
        text = Text()
        text.append(" Space ", style="bold #090014 on #39ff14")
        text.append(f" {action}  ", style=f"bold {action_color}")
        text.append(" c ", style="bold #090014 on #00e5ff")
        text.append(" copy last  ", style="#9cf6ff")
        text.append(" C ", style="bold #090014 on #ff47d4")
        text.append(" copy all  ", style="#ff9ce8")
        text.append(" x ", style="bold #090014 on #ffd166")
        text.append(" clear  ", style="#ffe8a3")
        text.append(" q ", style="bold #090014 on #ff6b6b")
        text.append(" quit", style="#ffb3b3")
        return text


class TntApp(App):
    """Voice-to-text TUI powered by Qwen3-ASR."""

    CSS = """
    Screen {
        layout: vertical;
        background: #090014;
        color: #f8f4ff;
    }

    #main-layout {
        height: 1fr;
    }

    #main-layout TranscriptView {
        width: 3fr;
    }

    #main-layout StatusPanel {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("space", "toggle_recording", "Record", show=False),
        Binding("c", "copy_last", "Copy last", show=False),
        Binding("C", "copy_all", "Copy all", show=False),
        Binding("x", "clear_transcript", "Clear", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    state: reactive[str] = reactive("idle")

    def __init__(self) -> None:
        super().__init__()
        self.recorder: Recorder
        self.recorder, self.capture_backend = create_recorder()
        self._transcriber: QwenTranscriber | None = None
        self._recording_timer = None
        self._recording_session_id = 0

    def _init_transcriber(self) -> QwenTranscriber:
        """Lazily initialize the transcriber, raising clear errors."""
        if self._transcriber is None:
            self._transcriber = QwenTranscriber()
        return self._transcriber

    def compose(self) -> ComposeResult:
        yield HeaderBar()
        with Horizontal(id="main-layout"):
            yield TranscriptView()
            yield StatusPanel()
        yield HintBar()

    def watch_state(self, value: str) -> None:
        try:
            self.query_one(HeaderBar).state = value
            self.query_one(StatusPanel).state = value
            self.query_one(HintBar).state = value
        except Exception:
            pass

    def _update_recording_info(self) -> None:
        """Periodic callback during recording to update timer and level."""
        if not self.recorder.is_recording:
            return
        panel = self.query_one(StatusPanel)
        panel.update_elapsed(self.recorder.elapsed())
        panel.push_level(self.recorder.get_level())

    def action_toggle_recording(self) -> None:
        """Space key: toggle between idle/recording states."""
        match self.state:
            case "idle":
                self._start_recording()
            case "recording":
                self._stop_recording()
            case "transcribing":
                self.notify(
                    "Transcription in progress, please wait.", severity="warning"
                )

    def _start_recording(self) -> None:
        """Begin mic capture."""
        try:
            self.recorder.start()
        except Exception as e:
            self.notify(f"Mic error ({self.capture_backend}): {e}", severity="error")
            return

        self._recording_session_id += 1
        self.state = "recording"
        self._recording_timer = self.set_interval(0.1, self._update_recording_info)

    def _stop_recording(self) -> None:
        """Stop mic capture, launch transcription worker."""
        if self._recording_timer is not None:
            self._recording_timer.stop()
            self._recording_timer = None

        self.state = "transcribing"
        session_id = self._recording_session_id
        self.query_one(TranscriptView).show_placeholder()
        self.run_worker(self._stop_and_transcribe(session_id))

    async def _stop_and_transcribe(self, session_id: int) -> None:
        """Async worker: stop capture and transcribe without blocking the UI."""
        tv = self.query_one(TranscriptView)
        try:
            wav_bytes = await asyncio.wait_for(asyncio.to_thread(self.recorder.stop), 30)
        except asyncio.TimeoutError:
            tv.remove_placeholder()
            self.notify(
                f"Stop timed out ({self.capture_backend}); reset and try again.",
                severity="error",
            )
            if session_id == self._recording_session_id:
                self.state = "idle"
            return
        except Exception as e:
            tv.remove_placeholder()
            self.notify(f"Stop error ({self.capture_backend}): {e}", severity="error")
            if session_id == self._recording_session_id:
                self.state = "idle"
            return

        if not wav_bytes:
            tv.remove_placeholder()
            self.notify("No audio captured.", severity="warning")
            if session_id == self._recording_session_id:
                self.state = "idle"
            return

        try:
            transcriber = self._init_transcriber()
            text = await transcriber.transcribe_async(wav_bytes)
            tv.remove_placeholder()
            if text:
                tv.append(text)
            else:
                self.notify("No speech detected.", severity="warning")
        except FileNotFoundError as e:
            tv.remove_placeholder()
            self.notify(str(e), severity="error")
        except RuntimeError as e:
            tv.remove_placeholder()
            self.notify(f"Transcription failed: {e}", severity="error")
        except Exception as e:
            tv.remove_placeholder()
            self.notify(f"Error: {e}", severity="error")
        finally:
            if session_id == self._recording_session_id:
                self.state = "idle"

    def action_copy_last(self) -> None:
        """Copy the last transcript entry to clipboard."""
        text = self.query_one(TranscriptView).get_last()
        if not text:
            self.notify("Nothing to copy.", severity="warning")
            return
        self._copy_to_clipboard(text)

    def action_copy_all(self) -> None:
        """Copy all transcript entries to clipboard."""
        text = self.query_one(TranscriptView).get_all()
        if not text:
            self.notify("Nothing to copy.", severity="warning")
            return
        self._copy_to_clipboard(text)

    def _copy_to_clipboard(self, text: str) -> None:
        """Try to copy text to system clipboard."""
        commands: list[tuple[list[str], bool, str]] = [
            (["termux-clipboard-set", text], False, "termux-clipboard-set"),
            (["pbcopy"], True, "pbcopy"),
            (["wl-copy"], True, "wl-copy"),
            (["xclip", "-selection", "clipboard"], True, "xclip"),
        ]
        for cmd, use_stdin, label in commands:
            try:
                input_bytes = text.encode("utf-8") if use_stdin else None
                proc = subprocess.run(
                    cmd,
                    input=input_bytes,
                    capture_output=True,
                    timeout=2,
                )
                if proc.returncode == 0:
                    self.notify(f"Copied to clipboard ({label}).")
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        self.notify(
            "Clipboard not available; text stored in buffer.", severity="warning"
        )

    def action_clear_transcript(self) -> None:
        """Clear all transcript entries."""
        self.query_one(TranscriptView).clear()
        self.notify("Transcript cleared.")


def main() -> None:
    app = TntApp()
    app.run()


if __name__ == "__main__":
    main()
