"""Textual TUI app for voice-to-text transcription."""

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
        background: $surface-lighten-1;
        padding: 0 1;
    }
    """

    state: reactive[str] = reactive("idle")

    def render(self) -> Table:
        left = Text()
        left.append("● ", style="green")
        left.append("TNT", style="bold")
        left.append(" 🧨")
        left.append(" — voice → text")

        right = Text()
        right.append("qwen3-asr-0.6b │ 16kHz │ ")
        match self.state:
            case "idle":
                right.append("▮▮ IDLE", style="dim")
            case "recording":
                right.append("● REC", style="red")
            case "transcribing":
                right.append("◌ ...", style="yellow")

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
        background: $surface;
        padding: 0 1;
    }
    """

    state: reactive[str] = reactive("idle")

    def render(self) -> Text:
        action = "stop" if self.state == "recording" else "record"
        text = Text()
        text.append(" Space ", style="reverse")
        text.append(f" {action}  ")
        text.append(" c ", style="reverse")
        text.append(" copy last  ")
        text.append(" C ", style="reverse")
        text.append(" copy all  ")
        text.append(" x ", style="reverse")
        text.append(" clear  ")
        text.append(" q ", style="reverse")
        text.append(" quit")
        return text


class TntApp(App):
    """Voice-to-text TUI powered by Qwen3-ASR."""

    CSS = """
    Screen {
        layout: vertical;
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

        self.state = "recording"
        self._recording_timer = self.set_interval(0.1, self._update_recording_info)

    def _stop_recording(self) -> None:
        """Stop mic capture, launch transcription worker."""
        if self._recording_timer is not None:
            self._recording_timer.stop()
            self._recording_timer = None

        wav_bytes = self.recorder.stop()
        if not wav_bytes:
            self.state = "idle"
            self.notify("No audio captured.", severity="warning")
            return

        self.state = "transcribing"
        self.query_one(TranscriptView).show_placeholder()
        self.run_worker(self._do_transcribe(wav_bytes))

    async def _do_transcribe(self, wav_bytes: bytes) -> None:
        """Async worker: runs transcription without blocking the UI."""
        try:
            transcriber = self._init_transcriber()
            text = await transcriber.transcribe_async(wav_bytes)
            tv = self.query_one(TranscriptView)
            tv.remove_placeholder()
            if text:
                tv.append(text)
            else:
                self.notify("No speech detected.", severity="warning")
        except FileNotFoundError as e:
            self.query_one(TranscriptView).remove_placeholder()
            self.notify(str(e), severity="error")
        except RuntimeError as e:
            self.query_one(TranscriptView).remove_placeholder()
            self.notify(f"Transcription failed: {e}", severity="error")
        except Exception as e:
            self.query_one(TranscriptView).remove_placeholder()
            self.notify(f"Error: {e}", severity="error")
        finally:
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
        for cmd in [
            ["pbcopy"],
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
        ]:
            try:
                proc = subprocess.run(
                    cmd,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=2,
                )
                if proc.returncode == 0:
                    self.notify("Copied to clipboard.")
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
