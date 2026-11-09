"""Textual TUI dashboard for DarkTriage.

Analysis (pairing, EXIF read, parallel scoring) runs in a background worker so
the UI stays responsive on large directories.  A progress bar shows scoring
progress, and the file list is populated once analysis completes.
"""

from __future__ import annotations

from typing import List, Optional

from darktriage.cli import _analyze_directory
from darktriage.config import Config
from darktriage.core.engine import (
    DEFAULT_CHUNK_SIZE,
    format_label_histogram,
    format_rating_histogram,
    summarize_assessments,
)
from darktriage.core.exif_reader import Assessment, discover_pairs
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
    #analyze-status {
        height: 1;
        color: $text-muted;
    }
    #analyze-progress {
        height: 1;
        margin: 1 0;
    }
    #summary {
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
        *,
        threshold: Optional[float] = None,
        max_dim: int = 1920,
        workers: Optional[int] = None,
        use_embedded_preview: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        preview_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.assessments: List[Assessment] = list(assessments or [])
        self.directory = directory or ""
        self.config = config
        self.threshold = threshold
        self.max_dim = max_dim
        self._scoring_workers = workers
        self.use_embedded_preview = use_embedded_preview
        self.chunk_size = chunk_size
        self.preview_cache_dir = preview_cache_dir
        self.selected: Optional[Assessment] = None
        self._commit_in_progress = False
        self._analysis_ready = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Analyzing images...", id="analyze-status")
        yield ProgressBar(id="analyze-progress", total=0, show_eta=False)
        yield Label("", id="summary")
        with VerticalScroll(id="file-list"):
            pass  # populated after analysis completes
        with Vertical():
            yield MetricsView(id="metrics-view")
            yield Label("Idle", id="status")
            yield ProgressBar(id="commit-progress", total=0, show_eta=False)
            yield Button(
                "Commit XMP sidecars", id="commit", variant="success", disabled=True
            )
        yield Footer()

    def on_mount(self) -> None:
        self.selected = None
        self._refresh_metrics()
        self.run_worker(self._do_analysis, thread=True, name="analyze")

    def _refresh_metrics(self) -> None:
        try:
            self.query_one(MetricsView).update(self.selected)
        except Exception:  # widget not yet mounted (e.g. before compose)
            pass

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Label).update(message)

    def _set_analysis_status(self, message: str) -> None:
        self.query_one("#analyze-status", Label).update(message)

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

    # ------------------------------------------------------------------
    # Background analysis (runs on mount)
    # ------------------------------------------------------------------

    def _do_analysis(self) -> None:
        """Run the full analysis pipeline in a worker thread."""
        try:
            self.call_from_thread(self._set_analysis_status, "Discovering images...")
            pairs = discover_pairs(self.directory)
            if self.use_embedded_preview:
                jpg_count = len(pairs)
            else:
                jpg_count = len([p for p in pairs if p.jpg_path is not None])

            progress = self.query_one("#analyze-progress", ProgressBar)
            self.call_from_thread(
                progress.update, total=jpg_count, progress=0
            )
            self.call_from_thread(self._set_analysis_status, "Scoring images...")

            def advance() -> None:
                self.call_from_thread(progress.advance, 1)

            assessments = _analyze_directory(
                self.directory,
                self.config,
                threshold_seconds=self.threshold,
                max_dim=self.max_dim,
                workers=self._scoring_workers,
                on_complete=advance,
                use_embedded_preview=self.use_embedded_preview,
                chunk_size=self.chunk_size,
                preview_cache_dir=self.preview_cache_dir,
                pairs=pairs,
            )
            self.call_from_thread(self._analysis_done, assessments)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._analysis_failed, str(exc))

    def _analysis_done(self, assessments: List[Assessment]) -> None:
        """Populate the UI once background analysis completes."""
        self.assessments = list(assessments)
        self._analysis_ready = True

        progress = self.query_one("#analyze-progress", ProgressBar)
        progress.update(progress=progress.total)
        self._set_analysis_status(
            f"Analysis complete — {len(self.assessments)} file(s)"
        )

        summary = summarize_assessments(self.assessments)
        if summary.total:
            text = format_rating_histogram(summary, percent=True)
            if summary.by_label:
                text += "  |  " + format_label_histogram(summary)
            if summary.unscoreable:
                text += f"  |  unscoreable {summary.unscoreable}"
            self.query_one("#summary", Label).update(text)

        file_list = self.query_one("#file-list", VerticalScroll)
        file_list.remove_children()
        for assessment in self.assessments:
            file_list.mount(
                Button(
                    assessment.raw_path.name,
                    id=f"file-{assessment.raw_path.stem}",
                    classes="file-item",
                )
            )

        commit_button = self.query_one("#commit", Button)
        commit_button.disabled = not self.assessments

        self.selected = self.assessments[0] if self.assessments else None
        self._refresh_metrics()
        self._set_status("Ready")

    def _analysis_failed(self, message: str) -> None:
        self._set_analysis_status(f"Analysis failed: {message}")
        self._set_status("Analysis failed")
        self.notify(message, severity="error")
        commit_button = self.query_one("#commit", Button)
        commit_button.disabled = True

    # ------------------------------------------------------------------
    # XMP commit (with progress)
    # ------------------------------------------------------------------

    def _commit_sidecars(self) -> None:
        if self._commit_in_progress or not self._analysis_ready:
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

        progress = self.query_one(ProgressBar, id="commit-progress")
        if requests:
            progress.update(total=len(requests), progress=0)
            self._set_status(f"Committing {len(requests)} XMP sidecar(s)...")
        else:
            progress.update(total=0, progress=0)
            self._set_status("No sidecars to commit")
            self._reset_commit_ui()
            return

        try:
            writer = XmpWriter(
                create_backup=self.config.safety.create_backup
                if self.config
                else True,
                label_case=self.config.darktable.label_case
                if self.config
                else "title",
            )
        except ExifToolNotFoundError as exc:
            self._set_status(str(exc))
            self.notify(str(exc), severity="error")
            self._reset_commit_ui()
            return

        def _advance() -> None:
            self.call_from_thread(progress.advance, 1)

        try:
            results = writer.write_batch(requests, on_complete=_advance)
            self.call_from_thread(
                lambda: progress.update(progress=progress.total)
            )
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
        commit_button.disabled = not self._analysis_ready or not self.assessments
        progress = self.query_one(ProgressBar, id="commit-progress")
        progress.update(progress=0)
