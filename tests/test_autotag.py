"""Tests for deterministic subject auto-tagging and dc:Subject writing."""

from __future__ import annotations

import numpy as np
from PIL import Image

from darktriage.core.autotag import (
    SceneStats,
    SubjectTagger,
    analyse_scene,
    derive_tags,
    detect_faces,
)
from darktriage.core.xmp_writer import (
    XmpWriter,
    read_sidecar_subjects,
    xmp_sidecar_path,
)


def _sky_image(path):
    arr = np.zeros((120, 160, 3), dtype=np.uint8)
    arr[:60, :] = (0, 0, 255)  # RGB blue sky (top half)
    arr[60:, :] = (128, 128, 128)
    Image.fromarray(arr).save(path, quality=95)
    return path


def _green_image(path):
    arr = np.full((120, 160, 3), (30, 140, 30), dtype=np.uint8)  # RGB green
    Image.fromarray(arr).save(path, quality=95)
    return path


def _grey_image(path, value=128):
    Image.fromarray(np.full((120, 160, 3), value, dtype=np.uint8)).save(path, quality=95)
    return path


def test_derive_tags_portrait_people_group():
    scene = SceneStats(0.0, 0.0, 0.0, 128.0)
    single = derive_tags(face_count=1, largest_face_ratio=0.10, scene=scene)
    assert single == ("people", "portrait")
    group = derive_tags(face_count=3, largest_face_ratio=0.02, scene=scene)
    assert "people" in group and "group" in group and "portrait" not in group


def test_derive_tags_scene_only():
    sky = SceneStats(0.5, 0.0, 0.0, 140.0)
    assert "landscape" in derive_tags(face_count=0, largest_face_ratio=0.0, scene=sky)
    green = SceneStats(0.0, 0.8, 0.0, 120.0)
    assert "nature" in derive_tags(face_count=0, largest_face_ratio=0.0, scene=green)
    dark = SceneStats(0.0, 0.0, 0.0, 20.0)
    assert "night" in derive_tags(face_count=0, largest_face_ratio=0.0, scene=dark)


def test_analyse_scene_sky_and_green():
    sky = np.zeros((100, 100, 3), dtype=np.uint8)
    sky[:50, :] = (255, 0, 0)  # BGR blue
    assert analyse_scene(sky).sky_fraction > 0.3

    green = np.zeros((100, 100, 3), dtype=np.uint8)
    green[:, :] = (0, 140, 0)  # BGR green
    assert analyse_scene(green).green_fraction > 0.6


def test_detect_faces_blank_returns_empty():
    assert detect_faces(np.zeros((200, 200), dtype=np.uint8)) == []


def test_tagger_landscape_and_nature(tmp_path):
    tagger = SubjectTagger(max_dim=320)
    assert "landscape" in tagger.analyze(_sky_image(tmp_path / "sky.jpg")).tags
    assert "nature" in tagger.analyze(_green_image(tmp_path / "green.jpg")).tags


def test_tagger_blank_has_no_tags(tmp_path):
    assert SubjectTagger(max_dim=320).analyze(_grey_image(tmp_path / "grey.jpg")).tags == ()


def test_read_sidecar_subjects(tmp_path):
    xmp = tmp_path / "D.NEF.xmp"
    xmp.write_text(
        "<x:xmpmeta><rdf:RDF><rdf:Description>"
        "<dc:subject><rdf:Bag><rdf:li>portrait</rdf:li><rdf:li>people</rdf:li></rdf:Bag></dc:subject>"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    assert read_sidecar_subjects(xmp) == ("portrait", "people")


def test_writer_adds_subject_tags_to_new_sidecar(tmp_path):
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    result = writer.write(
        tmp_path / "D.NEF", rating=4, label="green", tags=("portrait", "people")
    )
    joined = " ".join(result.command)
    assert "-XMP-dc:Subject=portrait" in joined
    assert "-XMP-dc:Subject+=people" in joined


def test_writer_preserves_existing_subjects(tmp_path):
    xmp = xmp_sidecar_path(tmp_path / "D.NEF")
    xmp.write_text(
        "<x:xmpmeta><rdf:RDF><rdf:Description>"
        "<dc:subject><rdf:Bag><rdf:li>portrait</rdf:li></rdf:Bag></dc:subject>"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    result = writer.write(
        tmp_path / "D.NEF", rating=4, label="green", tags=("portrait", "landscape")
    )
    joined = " ".join(result.command)
    assert "-XMP-dc:Subject+=landscape" in joined
    assert "Subject+=portrait" not in joined
