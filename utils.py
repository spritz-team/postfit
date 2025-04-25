import matplotlib as mpl
import sys

mpl.use("Agg")


def darker_color(color, darker_factor=4 / 5):
    rgb = list(mpl.colors.to_rgba(color)[:-1])
    rgb[0] = rgb[0] * darker_factor
    rgb[1] = rgb[1] * darker_factor
    rgb[2] = rgb[2] * darker_factor
    return tuple(rgb)


cmap_petroff = [
    "#5790fc",
    "#f89c20",
    "#e42536",
    "#964a8b",
    "#9c9ca1",
    "#7a21dd",
]
cmap_pastel = [
    "#A1C9F4",
    "#FFB482",
    "#8DE5A1",
    "#FF9F9B",
    "#D0BBFF",
    "#DEBB9B",
    "#FAB0E4",
    "#CFCFCF",
    "#FFFEA3",
    "#B9F2F0",
]


def get_analysis_dict(path):
    sys.path.insert(0, 'configs')
    exec(f"import {path} as analysis_cfg", globals(), globals())

    return analysis_cfg.__dict__  # type: ignore # noqa: F821
