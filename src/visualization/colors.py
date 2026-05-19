import numpy as np


def hex_to_rgb(hex_str: str) -> list[int]:
    h = hex_str.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]


def normalize_color_map(color_map: dict) -> dict[int, list[int]]:
    out = {}
    for k, v in color_map.items():
        out[int(k)] = hex_to_rgb(v) if isinstance(v, str) else [int(c) for c in v]
    return out


def labels_to_colors(labels: np.ndarray, color_map: dict) -> np.ndarray:
    norm = normalize_color_map(color_map)
    default = [128, 128, 128]
    colors = np.array([norm.get(int(l), default) for l in labels], dtype=np.float32)
    return colors / 255.0
