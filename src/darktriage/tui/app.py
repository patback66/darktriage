"""Textual TUI dashboard for DarkTriage."""

from __future__ import annotations

from typing import List, Optional

from darktriage.config import Config
from darktriage.core.exif_reader import Assessment
from darktriage.core.xmp_writer import (
    ExifToolNotFoundError,
    SidecarRequest,
    XmpWriter,
)

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Label

from .widgets import MetricsView


class DarkTriageApp(App):
    """Interactive dashboard: image list, metric breakdown, XMP commit."""

    CSS = """
    #file-list Button {
        width: 100%;
    }
    #metrics-view {
        min-height: 7;
        border: round $accent;
        padding: 1 2;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        assessments: Optional[List[Assessment]] = None,
        directory: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        super().__init__()
        self.assessments = list(assessments or [])
        self.directory = directory or ""
        self.config = config
        self.selected: Optional[Assessment] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="file-list"):
            for assessment in self.assessments:
                yield Button(
                    assessment.raw_path.name,
                    id=f"file-{assessment.raw_path.stem}",
                    classes="file-item",
                )
        with Vertical():
            yield MetricsView(id="metrics-view")
            yield Button("Commit XMP sidecars", id="commit", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.selected = self.assessments[0] if self.assessments else None
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        self.query_one(MetricsView).update(self.selected)

    def _assessment_for_button(self, button: Button) -> Optional[Assessment]:
        for assessment in self.assessments:
            if button.id == f"file-{assessment.raw_path.stem}":
                return assessment
        return None

    def on_button_pressed(self, event) -> None:
        button = event.button
        if button.id == "commit":
            self._commit_sidecars()
            return

        assessment = self._assessment_for_button(button)
        if assessment is not None:
            self.selected = assessment
            self._refresh_metrics()

    def _commit_sidecars(self) -> None:
        create_backup = self.config.safety.create_backup if self.config else True
        try:
            writer = XmpWriter(create_backup=create_backup)
        except ExifToolNotFoundError as exc:
            self.notify(str(exc), severity="error")
            return

        requests = [
            SidecarRequest(
                raw_path=a.raw_path,
                rating=a.rating,
                label=a.label,
                ev_offset=a.ev_offset,
            )
            for a in self.assessments
            if a.rating is not None
        ]
        results = writer.write_batch(requests)
        written = sum(1 for r in results if r.success and not r.skipped)
        preserved = sum(1 for r in results if r.skipped)

        message = f"Committed XMP sidecars for {written} file(s)"
        if preserved:
            message += f", preserved {preserved} existing sidecar(s)"
        self.notify(message)