"""Custom Textual components for the DarkTriage dashboard."""

from __future__ import annotations

from typing import Optional

from textual.containers import Vertical
from textual.widgets import Label


class MetricsView(Vertical):
    """Displays the metrics of the currently selected image."""

    def update(self, metrics: Optional[dict]) -> None:
        self.clear()
        if metrics is None:
            self.mount(Label("No metrics yet"))
            return

        rating = metrics["rating"]
        rows = [
            ("File", ""),
            ("Rating", f"{rating.rating}/5  [{rating.label}]"),
            ("Sharpness", f"{metrics['sharpness']:.2f}"),
            ("Mean exposure", f"{metrics['exposure']['mean_luminance']:.1f}"),
            ("Highlight clipping", f"{metrics['exposure']['highlight_ratio'] * 100:.1f}%"),
            ("Composition", f"{metrics['composition']['composition_score']:.1f}"),
        ]
        for name, value in rows:
            self.mount(Label(f"[{bold}]{name}[/bold]   {value}"))
