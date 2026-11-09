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
from textual.widgets import Button, Footer, Header, Label, ProgressBar

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
    #status {
        height: 1;
        color: $text-muted;
    }
    #commit-progress {
        height: 1;
        margin: 1 0;
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
        self._commit_in_progress = False

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
            yield Label("Idle", id="status")
            yield ProgressBar(id="commit-progress", total=0, show_eta=False)
            yield Button(
                "Commit XMP sidecars", id="commit", variant="success"
            )
        yield Footer()

    def on_mount(self) -> None:
        self.selected = self.assessments[0] if self.assessments else None
        self._refresh_metrics()
        progress = self.query_one(ProgressBar)
        progress.update(total=0, progress=0)

    def _refresh_metrics(self) -> None:
        self.query_one(MetricsView).update(self.selected)

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Label).update(message)

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
        if self._commit_in_progress:
            return
        self._commit_in_progress = True
        commit_button = self.query_one("#commit", Button)
        commit_button.disabled = True

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

        progress = self.query_one(ProgressBar)
        if requests:
            progress.update(total=len(requests), progress=0)
            self._set_status(f"Committing {len(requests)} XMP sidecar(s)...")
        else:
            progress.update(total=0, progress=0)
            self._set_status("No sidecars to commit")

        try:
            writer = XmpWriter(
                create_backup=self.config.safety.create_backup
                if self.config
                else True
            )
        except ExifToolNotFoundError as exc:
            self._set_status(str(exc))
            self.notify(str(exc), severity="error")
            self._reset_commit_ui()
            return

        if not requests:
            self._reset_commit_ui()
            return

        def _advance() -> None:
            progress.advance(1)

        try:
            results = writer.write_batch(requests, on_complete=_advance)
            progress.update(progress=progress.total)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Commit failed: {exc}")
            self.notify(str(exc), severity="error")
            self._reset_commit_ui()
            return

        written = sum(1 for r in results if r.success and not r.skipped)
        preserved = sum(1 for r in results if r.skipped)
        message = f"Committed XMP sidecars for {written} file(s)"
        if preserved:
            message += f", preserved {preserved} existing sidecar(s)"
        self._set_status(message)
        self.notify(message)
        self._reset_commit_ui()

    def _reset_commit_ui(self) -> None:
        self._commit_in_progress = False
        commit_button = self.query_one("#commit", Button)
        commit_button.disabled = False
        progress = self.query_one(ProgressBar)
        progress.update(progress=0)