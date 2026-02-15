"""Recording state indicator and audio level visualizer."""

import math
from collections import deque

from rich.text import Text

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

WAVEFORM_COLS = 16
WAVEFORM_UPPER_ROWS = 3
WAVEFORM_LOWER_ROWS = 3
WAVEFORM_ROWS = WAVEFORM_UPPER_ROWS + WAVEFORM_LOWER_ROWS
IDLE_LEVEL = 0.12

# Upper half: fill from bottom up (8 sub-levels per cell).
_UPPER_BLOCKS = " ▁▂▃▄▅▆▇█"
_UPPER_SUBCELLS = WAVEFORM_UPPER_ROWS * 8

# Lower half: fill from top down (2 sub-levels per cell: ▀ and █).
_LOWER_BLOCKS = " ▀█"
_LOWER_SUBCELLS = WAVEFORM_LOWER_ROWS * 2


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
        height: 6;
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
        self.capture_backend: str = "live"
        self._levels: deque[float] = deque(
            [IDLE_LEVEL] * WAVEFORM_COLS, maxlen=WAVEFORM_COLS
        )
        self._sine_tick: int = 0
        self._transcribe_timer = None

    def compose(self) -> ComposeResult:
        yield Static(id="waveform")
        yield Static(id="state-label")
        yield Static(id="state-timer")

    def on_mount(self) -> None:
        self._refresh_display()

    def watch_state(self, value: str) -> None:
        # Stop transcribing animation when leaving that state.
        if self._transcribe_timer is not None:
            self._transcribe_timer.stop()
            self._transcribe_timer = None

        match value:
            case "idle":
                self._levels = deque(
                    [IDLE_LEVEL] * WAVEFORM_COLS, maxlen=WAVEFORM_COLS
                )
                self._sine_tick = 0
            case "transcribing":
                self._sine_tick = 0
                self._transcribe_timer = self.set_interval(
                    0.1, self._tick_transcribe_animation
                )
        self._refresh_display()

    def push_level(self, level: float) -> None:
        """Push a new audio level sample and refresh the waveform."""
        if self.capture_backend == "termux_api" and self.state == "recording":
            self._sine_tick += 1
            self._apply_sine_levels(amplitude=0.40, baseline=0.15, speed=0.25)
        else:
            self._levels.append(level)

        try:
            self.query_one("#waveform", Static).update(self._render_waveform())
        except Exception:
            pass

    def _tick_transcribe_animation(self) -> None:
        """Periodic callback that animates a sine wave during transcription."""
        self._sine_tick += 1
        self._apply_sine_levels(amplitude=0.25, baseline=0.10, speed=0.15)
        try:
            self.query_one("#waveform", Static).update(self._render_waveform())
        except Exception:
            pass

    def _apply_sine_levels(
        self, amplitude: float, baseline: float, speed: float
    ) -> None:
        """Overwrite all columns with a sine wave pattern."""
        self._levels.clear()
        for i in range(WAVEFORM_COLS):
            phase = (i / WAVEFORM_COLS) * 2.0 * math.pi
            t = self._sine_tick * speed
            self._levels.append(baseline + amplitude * abs(math.sin(phase + t)))

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
                level = self._levels[col]
                color = palette[col % len(palette)]

                if row < WAVEFORM_UPPER_ROWS:
                    # Upper half: bars grow upward from center.
                    upper_sub = round(level * _UPPER_SUBCELLS)
                    row_from_center = WAVEFORM_UPPER_ROWS - 1 - row
                    below = row_from_center * 8
                    fill = max(0, min(8, upper_sub - below))
                    char = _UPPER_BLOCKS[fill]
                else:
                    # Lower half: bars grow downward from center.
                    lower_sub = round(level * _LOWER_SUBCELLS)
                    row_from_center = row - WAVEFORM_UPPER_ROWS
                    above = row_from_center * 2
                    fill = max(0, min(2, lower_sub - above))
                    char = _LOWER_BLOCKS[fill]

                if char != " ":
                    text.append(char, style=f"bold {color}")
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
