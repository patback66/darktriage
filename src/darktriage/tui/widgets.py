"""Custom Textual widgets for the DarkTriage dashboard."""

from __future__ import annotations

from typing import Optional

from textual.containers import Vertical
from textual.widgets import Label

from darktriage.core.exif_reader import Assessment


class MetricsView(Vertical):
    """Metric breakdown for the currently selected image."""

    def update(self, assessment: Optional[Assessment]) -> None:
        self.clear()

        if assessment is None:
            self.mount(Label("No file selected"))
            return

        self.mount(Label(f"[b]{assessment.raw_path.name}[/b]"))

        if assessment.rating is None:
            self.mount(Label(f"Status: {assessment.error or 'unscoreable'}"))
            return

        rating = f"{assessment.rating}/5"
        if assessment.label:
            rating += f"  [{assessment.label}]"

        burst = "-"
        if assessment.burst_index is not None:
            burst = str(assessment.burst_index)
            if assessment.is_burst_winner:
                burst += " (winner)"

        rows = [
            ("Rating", rating),
            ("Sharpness", f"{assessment.sharpness:.2f}"),
            ("Composition", f"{assessment.composition_score:.2f}"),
            ("Mean luminance", f"{assessment.mean_luminance:.1f}"),
            ("Clipped highlights", f"{assessment.clipped_highlight_pct:.1f}%"),
            ("EV offset", f"{assessment.ev_offset:+.2f}" if assessment.ev_offset is not None else "-"),
            ("Burst", burst),
        ]
        for name, value in rows:
            self.mount(Label(f"[b]{name}:[/b] {value}"))