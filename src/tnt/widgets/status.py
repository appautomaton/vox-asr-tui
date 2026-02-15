"""Recording state indicator and audio level visualizer."""

from collections import deque

from rich.text import Text

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

WAVEFORM_COLS = 16
WAVEFORM_ROWS = 5
IDLE_LEVEL = 0.12


class StatusPanel(Widget):
    """Shows recording state, elapsed time, and audio level."""

    DEFAULT_CSS = """
    StatusPanel {
        border: solid #00e5ff;
        border-title-color: #ff9df0;
        background: #12082a;
        color: #f8f4ff;
        layout: vertical;
        align: center middle;
    }

    StatusPanel > Static {
        width: 100%;
        height: auto;
    }

    #waveform {
        height: 5;
    }

    #state-label {
        margin: 1 0 0 0;
    }

    #state-timer {
        color: #7afcff;
    }
    """

    state: reactive[str] = reactive("idle")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = "Status"
        self._levels: deque[float] = deque(
            [IDLE_LEVEL] * WAVEFORM_COLS, maxlen=WAVEFORM_COLS
        )

    def compose(self) -> ComposeResult:
        yield Static(id="waveform")
        yield Static(id="state-label")
        yield Static(id="state-timer")

    def on_mount(self) -> None:
        self._refresh_display()

    def watch_state(self, value: str) -> None:
        match value:
            case "idle":
                self._levels = deque(
                    [IDLE_LEVEL] * WAVEFORM_COLS, maxlen=WAVEFORM_COLS
                )
            case "transcribing":
                # Keep frozen recording levels (don't reset)
                pass
        self._refresh_display()

    def push_level(self, level: float) -> None:
        """Push a new audio level sample and refresh the waveform."""
        self._levels.append(level)
        try:
            self.query_one("#waveform", Static).update(self._render_waveform())
        except Exception:
            pass

    def update_elapsed(self, seconds: float) -> None:
        """Update the timer display."""
        mins = int(seconds) // 60
        secs = seconds - (mins * 60)
        text = Text(f"{mins:02d}:{secs:04.1f}s", style="bold #7afcff", justify="center")
        try:
            self.query_one("#state-timer", Static).update(text)
        except Exception:
            pass

    def _refresh_display(self) -> None:
        try:
            self.query_one("#waveform", Static).update(self._render_waveform())
            self.query_one("#state-label", Static).update(self._render_label())
            if self.state != "recording":
                self.query_one("#state-timer", Static).update("")
        except Exception:
            pass

    def _render_waveform(self) -> Text:
        palettes = {
            "idle": ("#5f6cff", "#5ad8ff", "#8f7dff", "#6ef3ff"),
            "recording": (
                "#ff4fd8",
                "#ff6f91",
                "#ff9f1c",
                "#ffe347",
                "#42f5ff",
                "#7f5dff",
            ),
            "transcribing": ("#ffe347", "#ffb347", "#ff71ce", "#8ef6ff", "#6d7bff"),
        }
        palette = palettes.get(self.state, palettes["idle"])
        text = Text(justify="center")

        for row in range(WAVEFORM_ROWS):
            if row > 0:
                text.append("\n")
            for col in range(WAVEFORM_COLS):
                height = round(self._levels[col] * WAVEFORM_ROWS)
                threshold = WAVEFORM_ROWS - row
                if height >= threshold:
                    text.append("●", style=f"bold {palette[col % len(palette)]}")
                else:
                    text.append(" ")

        return text

    def _render_label(self) -> Text:
        text = Text(justify="center")
        match self.state:
            case "idle":
                text.append("■ READY", style="bold #7afcff")
            case "recording":
                text.append("● RECORDING", style="bold #ff5ccf")
            case "transcribing":
                text.append("◌ TRANSCRIBING", style="bold #ffd166")
        return text
