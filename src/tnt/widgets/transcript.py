"""Scrollable transcript log widget."""

from textual.containers import VerticalScroll
from textual.widgets import Static

ENTRY_TONES = (
    "#2f2a44",  # dusk lavender
    "#2a3547",  # dusty blue
    "#3a2e44",  # muted plum
    "#2b3b3f",  # dark pastel teal
    "#3d2f37",  # muted rose
    "#2e3a33",  # moss slate
)


class TranscriptEntry(Static):
    """A single transcript entry."""

    DEFAULT_CSS = """
    TranscriptEntry {
        padding: 0 1;
        margin: 0 0 1 0;
        color: #f4efff;
    }
    """


class TranscriptPlaceholder(Static):
    """Placeholder shown during transcription."""

    DEFAULT_CSS = """
    TranscriptPlaceholder {
        padding: 0 1;
        margin: 0 0 1 0;
        color: #ff71ce;
    }
    """


class TranscriptView(VerticalScroll):
    """Scrollable container of transcript entries."""

    DEFAULT_CSS = """
    TranscriptView {
        border: solid #8e7dff;
        border-title-color: #42f5ff;
        background: #0d041f;
        color: #f4efff;
        padding: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries: list[str] = []
        self.border_title = "Transcript"

    def append(self, text: str) -> None:
        """Add a new entry and scroll to bottom."""
        self.remove_placeholder()
        tone = ENTRY_TONES[len(self._entries) % len(ENTRY_TONES)]
        self._entries.append(text)
        entry = TranscriptEntry(text)
        entry.styles.background = tone
        self.mount(entry)
        self.scroll_end(animate=False)

    def show_placeholder(self) -> None:
        """Show a transcription-in-progress cursor."""
        self.remove_placeholder()
        self.mount(TranscriptPlaceholder("[#ff71ce]▊[/]", id="transcript-placeholder"))
        self.scroll_end(animate=False)

    def remove_placeholder(self) -> None:
        """Remove the transcription placeholder if present."""
        try:
            self.query_one("#transcript-placeholder").remove()
        except Exception:
            pass

    def get_last(self) -> str:
        """Return the last transcript entry, or empty string."""
        return self._entries[-1] if self._entries else ""

    def get_all(self) -> str:
        """Return all transcript entries joined by newlines."""
        return "\n".join(self._entries)

    def clear(self) -> None:
        """Remove all transcript entries."""
        self._entries.clear()
        self.query(TranscriptEntry).remove()
        self.remove_placeholder()
