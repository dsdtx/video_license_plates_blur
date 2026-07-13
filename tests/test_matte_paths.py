from blur_plates import resolve_matte_output_path


def test_prores_forces_mov():
    assert resolve_matte_output_path("out.mp4", "prores") == "out.mov"


def test_prores_keeps_mov():
    assert resolve_matte_output_path("clip.mov", "prores") == "clip.mov"


def test_hevc_forces_mp4():
    assert resolve_matte_output_path("out.mkv", "hevc") == "out.mp4"


def test_hevc_keeps_mp4():
    assert resolve_matte_output_path("clip.mp4", "hevc") == "clip.mp4"


def test_preserves_directory_and_stem():
    assert resolve_matte_output_path("/a/b/matte.mxf", "prores") == "/a/b/matte.mov"
