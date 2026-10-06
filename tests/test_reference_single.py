from pathlib import Path
import zipfile

from taikonation.reference import import_osz


def test_single_diff_set_is_preserved_as_style_exemplar(tmp_path: Path):
    archive = tmp_path / "single.osz"
    osu = """osu file format v14

[General]
AudioFilename: audio.mp3
Mode: 1

[Metadata]
Title:Single
Artist:Artist
Creator:Nifty
Version:Adrenaline
BeatmapID:1
BeatmapSetID:456

[TimingPoints]
0,500,4,2,1,70,1,0

[HitObjects]
256,192,0,1,0,0:0:0:0:
256,192,250,1,8,0:0:0:0:
"""
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("audio.mp3", b"audio")
        zf.writestr("Single (Nifty) [Adrenaline].osu", osu)

    manifest = import_osz(archive, tmp_path / "out")
    assert manifest["reference_role"] == "style_exemplar"
    assert manifest["maps"][0]["difficulty"] is None
    assert manifest["maps"][0]["difficulty_mapper"] == "Nifty"
