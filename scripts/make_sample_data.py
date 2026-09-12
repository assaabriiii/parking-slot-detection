"""Write tiny synthetic PNG frames for data/sample (stdlib only)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "frames"
WIDTH, HEIGHT = 640, 360

# Expected labels for the OpenCV heuristic: dark/textured = occupied, light/flat = free
SPOTS = [
    # id, (x0, y0, x1, y1), occupied
    ("spot_01", (20, 180, 140, 330), True),
    ("spot_02", (150, 180, 270, 330), False),
    ("spot_03", (280, 180, 400, 330), True),
    ("spot_04", (410, 180, 530, 330), False),
    ("spot_05", (540, 180, 620, 330), True),
]


def _png(rgb: list[list[tuple[int, int, int]]]) -> bytes:
    h = len(rgb)
    w = len(rgb[0])
    raw = b"".join(b"\x00" + bytes(c for pix in row for c in pix) for row in rgb)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _paint(seed: int) -> list[list[tuple[int, int, int]]]:
    pavement = (70 + seed, 120, 80)
    grid = [[pavement for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if y < 160:
                grid[y][x] = (90, 140 - seed, 95)
    for _, (x0, y0, x1, y1), occupied in SPOTS:
        if occupied:
            body = (30, 30, 70 + seed * 8)
        else:
            body = (200, 200, 205)
        for y in range(y0, y1):
            for x in range(x0, x1):
                # Light stripe on occupied cars so Laplacian variance is high.
                if occupied and ((x + y) % 7 == 0):
                    grid[y][x] = (180, 180, 190)
                else:
                    grid[y][x] = body
        yellow = (240, 210, 40)
        for x in range(x0, x1):
            grid[y0][x] = yellow
            grid[y1 - 1][x] = yellow
        for y in range(y0, y1):
            grid[y][x0] = yellow
            grid[y][x1 - 1] = yellow
    return grid


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        path = OUT / f"frame_{i:03d}.png"
        path.write_bytes(_png(_paint(i)))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
