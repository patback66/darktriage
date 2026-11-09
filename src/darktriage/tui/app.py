"""Textual TUI Application instance for DarkTriage."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from darktriage.core.metrics import analyze_image

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView

from .widgets import MetricsView


class FileListView(ListView):
    """List of triaged files with persistent file paths (for selection)."""

    def __init__(self, paths: Iterable[Path], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths = list(paths)
        self.selected_path: Optional[Path] = None

    def set_paths(self, paths: Iterable[Path]) -> None:
        self.paths = list(paths)
        self.clear()
        for p in self.paths:
            with ListItem():
                yield Button(str(p.name), id=f"file-{p.name}", value=p)

    def on_button_pressed(self, event) -> None:
        button = event.sender
        if hasattr(button, "value") and button.value is not None:
            self.selected_path = button.value  # type: ignore[attr-defined]
            self.app.update_metrics(self.selected_path)


class DarkTriageApp(App):
    """Terminal dashboard for triaging a RAW folder and viewing metrics."""

    CSS = """
    Button.file-item {
        width: 100%;
    }
    #metrics-view Label {
        padding: 1 2;
    }
    """
    BINDINGS = [("q", "app.quit", "Quit"), ("d", "toggle_debug", "Debug")]

    def __init__(
        self,
        file_paths: Optional[Iterable[Path]] = None,
        metrics_map: Optional[Dict[Path, dict]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.file_paths = list(file_paths or [])
        self.metrics_map = metrics_map or {}
        self.selected_path: Optional[Path] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield FileListView(self.file_paths, id="file-list")
        with Vertical():
            yield Label("Select a file to view its triage metrics")
            yield MetricsView(id="metrics-view")
        yield Footer()

    def on_mount(self) -> None:
        file_list = self.query_one(FileListView)
        file_list.set_paths(self.file_paths)
        if self.file_paths:
            file_list.selected_path = self.file_paths[0]
            self.update_metrics(self.file_paths[0])

    def on_button_pressed(self, event) -> None:
        file_list = self.query_one(FileListView)
        file_list.on_button_pressed(event)

    def update_metrics(self, path: Path) -> None:
        self.selected_path = path
        metrics = self.metrics_map.get(path)
        metrics_view = self.query_one(MetricsView)
        metrics_view.update(metrics)

    def action_toggle_debug(self) -> None:
        self.log.debug("Debug logging toggled")
        self.notify("Debug output toggled")
