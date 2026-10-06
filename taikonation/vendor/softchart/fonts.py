"""Font helpers for generated preview images."""

from __future__ import annotations

import os


def cjk_font_path(bold: bool = False) -> str | None:
    names = (
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Bold.otf",
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        ),
        (
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ),
        (
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        ),
        (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ),
    )
    for bold_path, regular_path in names:
        path = bold_path if bold else regular_path
        if os.path.exists(path):
            return path
    return None
