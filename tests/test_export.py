"""Tests for darktriage.export.darktable_cli."""

from __future__ import annotations

from darktriage.export.darktable_cli import (
    ExportJob,
    batch_export,
    build_command,
    read_xmp_rating,
)


def test_read_xmp_rating_xml_element(tmp_path):
    xmp = tmp_path / "DSC.NEF.xmp"
    xmp.write_text("<xmp:Rating>4</xmp:Rating>")
    assert read_xmp_rating(xmp) == 4


def test_read_xmp_rating_attribute(tmp_path):
    xmp = tmp_path / "DSC.NEF.xmp"
    xmp.write_text('<rdf:Description xmp:Rating="5"/>')
    assert read_xmp_rating(xmp) == 5


def test_batch_export_filters_by_min_rating(tmp_path):
    for name, rating in [("A", 5), ("B", 2), ("C", 4)]:
        (tmp_path / f"{name}.NEF").write_bytes(b"raw")
        (tmp_path / f"{name}.NEF.xmp").write_text(f"<xmp:Rating>{rating}</xmp:Rating>")

    result = batch_export(tmp_path, min_rating=4, dry_run=True)
    exported = sorted(job.nef.stem for job in result.jobs)
    skipped = sorted(path.stem for path in result.skipped)

    assert exported == ["A", "C"]
    assert skipped == ["B"]
    assert result.success is True
    assert result.dry_run is True


def test_batch_export_isolated_output_directory(tmp_path):
    (tmp_path / "A.NEF").write_bytes(b"raw")
    (tmp_path / "A.NEF.xmp").write_text("<xmp:Rating>5</xmp:Rating>")

    result = batch_export(tmp_path, min_rating=4, dry_run=True)
    # Default output directory is <target>/exported, isolated from sources.
    assert result.jobs[0].output.parent.name == "exported"
    assert result.jobs[0].output.suffix == ".jpg"


def test_build_command_chained_styles_and_resolution(tmp_path):
    xmp = tmp_path / "DSC.NEF.xmp"
    xmp.write_text("<xmp:Rating>5</xmp:Rating>")
    job = ExportJob(
        nef=tmp_path / "DSC.NEF",
        xmp=xmp,
        output=tmp_path / "exported" / "DSC.jpg",
        rating=5,
    )

    command = build_command(
        job, styles=["portrait", "sharpen"], width=2048, height=1365
    )
    assert command[0] == "darktable-cli"
    assert command.count("--style") == 2
    assert command[command.index("--style") + 1] == "portrait"
    assert "--width" in command and "2048" in command
    assert "--height" in command and "1365" in command
    assert str(job.output) in command