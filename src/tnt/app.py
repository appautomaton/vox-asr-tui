"""Textual TUI app for voice-to-text transcription."""

import asyncio
import signal
import subprocess

from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from tnt.audio import Recorder, create_recorder
from tnt.transcriber import (
    AsrBackend,
    Transcriber,
    create_transcriber_with_fallback,
    hint_label_for_backend,
    model_label_for_backend,
    other_asr_backend,
    resolve_asr_backend,
)
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
    backend_label: reactive[str] = reactive(model_label_for_backend("moonshine"))

    def render(self) -> Table:
        left = Text()
        left.append("● ", style="bold #39ff14")
        left.append("TNT", style="bold #ff4fd8")
        left.append(" 🧨", style="bold #ffb703")
        left.append(" — voice → text", style="#7afcff")

        right = Text()
        right.append(self.backend_label, style="bold #8be9fd")
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
    backend_label: reactive[str] = reactive(hint_label_for_backend("moonshine"))

    def render(self) -> Text:
        match self.state:
            case "recording":
                action, action_color = "stop", "#ff8ad8"
            case "transcribing":
                action, action_color = "cancel", "#ffd166"
            case _:
                action, action_color = "record", "#9bff7a"
        text = Text()
        text.append(" Space ", style="bold #090014 on #39ff14")
        text.append(f" {action}  ", style=f"bold {action_color}")
        text.append(" c ", style="bold #090014 on #00e5ff")
        text.append(" copy last  ", style="#9cf6ff")
        text.append(" C ", style="bold #090014 on #ff47d4")
        text.append(" copy all  ", style="#ff9ce8")
        text.append(" x ", style="bold #090014 on #ffd166")
        text.append(" clear  ", style="#ffe8a3")
        text.append(" m ", style="bold #090014 on #9bf6ff")
        text.append(f" {self.backend_label}  ", style="#cff7ff")
        text.append(" q ", style="bold #090014 on #ff6b6b")
        text.append(" quit", style="#ffb3b3")
        return text


class TntApp(App):
    """Voice-to-text TUI powered by local ASR backends."""

    _SPACE_PENDING_STOP_SECONDS = 0.18
    _SPACE_HOLD_RELEASE_WINDOW_SECONDS = 0.30

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
        Binding("m", "switch_asr", "Switch ASR", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    state: reactive[str] = reactive("idle")

    def __init__(self) -> None:
        super().__init__()
        self.recorder: Recorder
        self.recorder = create_recorder()
        self.asr_backend: AsrBackend = resolve_asr_backend()
        self._transcriber: Transcriber | None = None
        self._transcriber_backend: AsrBackend | None = None
        self._recording_timer = None
        self._recording_session_id = 0
        self._transcribe_worker = None
        self._space_recording_mode = "ready"
        self._space_mode_generation = 0

    def _init_transcriber(self) -> Transcriber:
        """Lazily initialize transcriber with automatic backend fallback."""
        if self._transcriber is not None and self._transcriber_backend == self.asr_backend:
            return self._transcriber

        transcriber, active_backend, warning = create_transcriber_with_fallback(
            self.asr_backend
        )
        self._transcriber = transcriber
        self._transcriber_backend = active_backend

        if active_backend != self.asr_backend:
            self.asr_backend = active_backend
            self._refresh_backend_ui()
            if warning:
                self.notify(warning, severity="warning")
        else:
            self._refresh_backend_ui()

        return transcriber

    def compose(self) -> ComposeResult:
        yield HeaderBar()
        with Horizontal(id="main-layout"):
            yield TranscriptView()
            yield StatusPanel()
        yield HintBar()

    def on_mount(self) -> None:
        self._refresh_backend_ui()

    def watch_state(self, value: str) -> None:
        try:
            self.query_one(HeaderBar).state = value
            self.query_one(StatusPanel).state = value
            self.query_one(HintBar).state = value
        except Exception:
            pass

    def _refresh_backend_ui(self) -> None:
        """Refresh backend labels shown in header and hint bar."""
        label = model_label_for_backend(self.asr_backend)
        hint_label = hint_label_for_backend(self.asr_backend)
        try:
            self.query_one(HeaderBar).backend_label = label
            self.query_one(HintBar).backend_label = hint_label
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
        """Space key: tap toggles, while a held key records until release."""
        match self.state:
            case "idle":
                self._start_recording()
            case "recording":
                self._handle_recording_space()
            case "transcribing":
                self._cancel_transcription()

    def _handle_recording_space(self) -> None:
        """Interpret Space during recording as a tap stop or held-key repeat."""
        match self._space_recording_mode:
            case "hold":
                self._arm_space_hold_release_timer()
            case "pending_stop":
                self._space_recording_mode = "hold"
                self._arm_space_hold_release_timer()
            case _:
                self._space_recording_mode = "pending_stop"
                self._space_mode_generation += 1
                generation = self._space_mode_generation
                self.set_timer(
                    self._SPACE_PENDING_STOP_SECONDS,
                    lambda: self._resolve_pending_space_stop(generation),
                )

    def _arm_space_hold_release_timer(self) -> None:
        """Refresh the inferred release timer while key-repeat is still arriving."""
        self._space_mode_generation += 1
        generation = self._space_mode_generation
        self.set_timer(
            self._SPACE_HOLD_RELEASE_WINDOW_SECONDS,
            lambda: self._finish_space_hold(generation),
        )

    def _resolve_pending_space_stop(self, generation: int) -> None:
        """Commit a stop when a follow-up repeat does not arrive."""
        if generation != self._space_mode_generation:
            return

        if self.state == "recording" and self._space_recording_mode == "pending_stop":
            self._space_recording_mode = "ready"
            self._stop_recording()

    def _finish_space_hold(self, generation: int) -> None:
        """Stop recording once key-repeat stops, which approximates key release."""
        if generation != self._space_mode_generation:
            return

        if self.state == "recording" and self._space_recording_mode == "hold":
            self._space_recording_mode = "ready"
            self._stop_recording()

    def _reset_space_recording_mode(self) -> None:
        """Clear inferred Space press state and invalidate pending timers."""
        self._space_recording_mode = "ready"
        self._space_mode_generation += 1

    def action_switch_asr(self) -> None:
        """Switch ASR backend while idle."""
        if self.state != "idle":
            self.notify("Switch ASR only while idle.", severity="warning")
            return

        if self._transcriber is not None:
            self._transcriber.kill_process()
            self._transcriber = None
            self._transcriber_backend = None

        self.asr_backend = other_asr_backend(self.asr_backend)
        self._refresh_backend_ui()
        self.notify(f"ASR backend: {model_label_for_backend(self.asr_backend)}.")

    def action_quit(self) -> None:
        """Quit the app, killing any in-flight transcription subprocess first."""
        if self._transcriber is not None:
            self._transcriber.kill_process()
        self.exit()

    def _cancel_transcription(self) -> None:
        """Cancel a running transcription and kill the subprocess."""
        self._reset_space_recording_mode()
        if self._transcriber is not None:
            self._transcriber.kill_process()
        if self._transcribe_worker is not None:
            self._transcribe_worker.cancel()

    def _start_recording(self) -> None:
        """Begin mic capture."""
        self._reset_space_recording_mode()
        try:
            self.recorder.start()
        except Exception as e:
            self.notify(f"Mic error: {e}", severity="error")
            return

        self._recording_session_id += 1
        self.state = "recording"
        self._recording_timer = self.set_interval(0.1, self._update_recording_info)

    def _stop_recording(self) -> None:
        """Stop mic capture, launch transcription worker."""
        self._reset_space_recording_mode()
        if self._recording_timer is not None:
            self._recording_timer.stop()
            self._recording_timer = None

        self.state = "transcribing"
        session_id = self._recording_session_id
        duration = self.recorder.elapsed()
        self.query_one(TranscriptView).show_placeholder()
        self._transcribe_worker = self.run_worker(
            self._stop_and_transcribe(session_id, duration)
        )

    async def _stop_and_transcribe(self, session_id: int, duration: float) -> None:
        """Async worker: stop capture and transcribe without blocking the UI."""
        tv = self.query_one(TranscriptView)
        try:
            wav_bytes = await asyncio.wait_for(asyncio.to_thread(self.recorder.stop), 30)
        except asyncio.TimeoutError:
            tv.remove_placeholder()
            self.notify(
                "Stop timed out; reset and try again.",
                severity="error",
            )
            if session_id == self._recording_session_id:
                self.state = "idle"
            return
        except Exception as e:
            tv.remove_placeholder()
            self.notify(f"Stop error: {e}", severity="error")
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
                tv.append(text, duration=duration)
                try:
                    label = await asyncio.wait_for(
                        asyncio.to_thread(self._try_clipboard_copy, text), timeout=5
                    )
                    if label:
                        self.notify(f"Copied to clipboard ({label}).")
                except asyncio.TimeoutError:
                    pass
            else:
                self.notify("No speech detected.", severity="warning")
        except asyncio.TimeoutError:
            tv.remove_placeholder()
            self.notify("Transcription timed out.", severity="error")
        except asyncio.CancelledError:
            if self._transcriber is not None:
                self._transcriber.kill_process()
            tv.remove_placeholder()
            self.notify("Transcription cancelled.", severity="warning")
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
            self._transcribe_worker = None
            if session_id == self._recording_session_id:
                self.state = "idle"

    def action_copy_last(self) -> None:
        """Copy the last transcript entry to clipboard."""
        text = self.query_one(TranscriptView).get_last()
        if not text:
            self.notify("Nothing to copy.", severity="warning")
            return
        label = self._try_clipboard_copy(text)
        if label:
            self.notify(f"Copied to clipboard ({label}).")
        else:
            self.notify("Clipboard not available; text stored in buffer.", severity="warning")

    def action_copy_all(self) -> None:
        """Copy all transcript entries to clipboard."""
        text = self.query_one(TranscriptView).get_all()
        if not text:
            self.notify("Nothing to copy.", severity="warning")
            return
        label = self._try_clipboard_copy(text)
        if label:
            self.notify(f"Copied to clipboard ({label}).")
        else:
            self.notify("Clipboard not available; text stored in buffer.", severity="warning")

    def _try_clipboard_copy(self, text: str) -> str | None:
        """Try to copy text to system clipboard.

        Returns the backend label on success, or None on failure.
        Does NOT call self.notify() — callers handle notification so
        this method is safe to run in a worker thread via asyncio.to_thread.
        """
        commands: list[tuple[list[str], bool, str]] = [
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
                    return label
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    def action_clear_transcript(self) -> None:
        """Clear all transcript entries."""
        self.query_one(TranscriptView).clear()
        self.notify("Transcript cleared.")


def main() -> None:
    app = TntApp()

    # On SIGINT (Ctrl-C), kill any in-flight subprocess so the worker
    # thread unblocks and the thread pool can join cleanly on exit.
    # Do NOT raise KeyboardInterrupt here — let the default handler do that
    # after we've cleaned up, otherwise the signal re-enters the asyncio loop.
    _orig_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig: int, frame: object) -> None:
        if app._transcriber is not None:
            app._transcriber.kill_process()
        signal.signal(signal.SIGINT, _orig_sigint)
        signal.raise_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        app.run()
    finally:
        if app._transcriber is not None:
            app._transcriber.kill_process()


if __name__ == "__main__":
    main()
