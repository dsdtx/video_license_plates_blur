import numpy as np
from blur_plates import render_matte_frame


def test_hard_edges_white_inside_black_outside():
    m = render_matte_frame(100, 100, [(40, 40, 60, 60)], padding=0, feather=0)
    assert m.shape == (100, 100, 3)
    assert m.dtype == np.uint8
    # Clearly inside the box → white
    assert (m[50, 50] == [255, 255, 255]).all()
    # Clearly outside the box → black
    assert (m[10, 10] == [0, 0, 0]).all()


def test_padding_expands_white_region():
    m = render_matte_frame(100, 100, [(40, 40, 60, 60)], padding=5, feather=0)
    # 37 is inside the 5px-padded region (starts at x=35), still white
    assert (m[50, 37] == 255).all()
    # 30 is outside the padded region, still black
    assert (m[50, 30] == 0).all()


def test_empty_rects_is_all_black():
    m = render_matte_frame(64, 48, [], padding=8, feather=0)
    assert m.shape == (64, 48, 3)
    assert m.max() == 0


def test_feather_introduces_intermediate_values():
    hard = render_matte_frame(100, 100, [(40, 40, 60, 60)], padding=0, feather=0)
    soft = render_matte_frame(100, 100, [(40, 40, 60, 60)], padding=0, feather=5)
    # Hard matte is purely 0 or 255; feathered matte has grey edge pixels.
    assert set(np.unique(hard)).issubset({0, 255})
    assert ((soft > 0) & (soft < 255)).any()


def test_rect_with_extra_elements_ignored():
    # Plates may carry source tags beyond the first 4 coords; only coords matter.
    m = render_matte_frame(100, 100, [(40, 40, 60, 60, "sahi", 0.9)], padding=0)
    assert (m[50, 50] == 255).all()
