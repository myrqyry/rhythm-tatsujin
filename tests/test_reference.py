from pathlib import Path
import zipfile

from taikonation.reference import (
    analyze_taiko_map,
    difficulty_mapper,
    import_osz,
    normalize_taiko_difficulty,
    parse_osu,
    summarize_manifests,
    summarize_mappers,
)


def _osu(*, mode: int, version: str, beatmap_id: int, set_id: int, notes: list[tuple[int, int]]):
    objects = "\n".join(
        f"256,192,{time},1,{hitsound},0:0:0:0:"
        for time, hitsound in notes
    )
    return f"""osu file format v14

[General]
AudioFilename: audio.mp3
Mode: {mode}

[Metadata]
Title:Reference Song
Artist:Reference Artist
Creator:Nifty
Version:{version}
BeatmapID:{beatmap_id}
BeatmapSetID:{set_id}

[Difficulty]
HPDrainRate:5
CircleSize:5
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,2,1,70,1,0

[HitObjects]
{objects}
"""


def test_difficulty_aliases_cover_taiko_names_without_substring_false_positives():
    assert normalize_taiko_difficulty("kantan!") == "easy"
    assert normalize_taiko_difficulty("Futsuu") == "normal"
    assert normalize_taiko_difficulty("ftsu") == "normal"
    assert normalize_taiko_difficulty("Nifty's Muzukashii") == "hard"
    assert normalize_taiko_difficulty("mzkshi") == "hard"
    assert normalize_taiko_difficulty("oni!") == "oni"
    assert normalize_taiko_difficulty("ni") == "oni"
    assert normalize_taiko_difficulty("Timeless Oni") == "oni"
    assert normalize_taiko_difficulty("O.N.I.") == "oni"
    assert normalize_taiko_difficulty("F.U.T.S.U.U.") == "normal"
    assert normalize_taiko_difficulty("M.U.Z.U.K.A.S.H.I.I.") == "hard"
    assert normalize_taiko_difficulty("Inner Oni") == "ura"
    assert normalize_taiko_difficulty("Inneroni") == "ura"
    assert normalize_taiko_difficulty("Expressionism") is None
    assert normalize_taiko_difficulty("Helloni") is None
    assert normalize_taiko_difficulty("Adrenaline") is None


def test_guest_mapper_is_recovered_from_difficulty_name():
    assert difficulty_mapper("Mentai", "Nifty's Oni") == "Nifty"
    assert difficulty_mapper("Nifty", "Oni") == "Nifty"


def test_analysis_tracks_density_and_subdivision_complexity():
    easy = parse_osu(
        _osu(
            mode=1,
            version="Kantan",
            beatmap_id=1,
            set_id=99,
            notes=[(0, 0), (1000, 8), (2000, 0)],
        )
    )
    oni = parse_osu(
        _osu(
            mode=1,
            version="Oni",
            beatmap_id=2,
            set_id=99,
            notes=[(0, 0), (125, 8), (250, 0), (500, 0), (1000, 8)],
        )
    )

    easy_stats = analyze_taiko_map(easy)
    oni_stats = analyze_taiko_map(oni)

    assert easy_stats["difficulty"] == "easy"
    assert oni_stats["difficulty"] == "oni"
    assert oni_stats["notes_per_second"] > easy_stats["notes_per_second"]
    assert oni_stats["quarter_beat_or_faster_share"] > easy_stats["quarter_beat_or_faster_share"]


def test_import_osz_filters_standard_maps_and_keeps_audio_local(tmp_path: Path):
    archive = tmp_path / "reference.osz"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("audio.mp3", b"not-real-audio")
        zf.writestr(
            "Reference (Nifty) [Kantan].osu",
            _osu(
                mode=1,
                version="Kantan",
                beatmap_id=10,
                set_id=123,
                notes=[(0, 0), (1000, 8), (2000, 0)],
            ),
        )
        zf.writestr(
            "Reference (Nifty) [Oni].osu",
            _osu(
                mode=1,
                version="Oni",
                beatmap_id=11,
                set_id=123,
                notes=[(0, 0), (250, 8), (500, 0), (750, 8), (1000, 0)],
            ),
        )
        zf.writestr(
            "Reference [Standard].osu",
            _osu(
                mode=0,
                version="Insane",
                beatmap_id=12,
                set_id=123,
                notes=[(0, 0)],
            ),
        )

    manifest = import_osz(archive, tmp_path / "output")
    set_dir = tmp_path / "output" / "123"

    assert manifest["reference_role"] == "difficulty_ladder"
    assert [item["difficulty"] for item in manifest["maps"]] == ["easy", "oni"]
    assert (set_dir / "audio.mp3").read_bytes() == b"not-real-audio"
    assert len(list((set_dir / "maps").glob("*.osu"))) == 2
    assert manifest["maps"][0]["density_ratio_to_oni"] < 1.0
    assert manifest["maps"][1]["density_ratio_to_oni"] == 1.0

    summary = summarize_manifests([manifest])
    assert summary["easy"]["maps"] == 1
    assert summary["oni"]["maps"] == 1

    mapper_profiles = summarize_mappers([manifest])
    assert mapper_profiles["Nifty"]["maps"] == 2
    assert mapper_profiles["Nifty"]["sets"] == 1
    assert mapper_profiles["Nifty"]["difficulty_profile"]["oni"]["maps"] == 1
