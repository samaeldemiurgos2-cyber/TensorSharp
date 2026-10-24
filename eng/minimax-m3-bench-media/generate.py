#!/usr/bin/env python3
"""Regenerate the image fixtures of eng/minimax-m3-bench.py's `image` scenario.

  python3 eng/minimax-m3-bench-media/generate.py

Every picture is drawn into a pixel buffer by this script (explicit 5x7 glyphs,
filled circles and squares), so the content does not depend on installed fonts.
PNGs are written with the standard library; JPEGs need Pillow (only for the
encoder). manifest.json lists each file with its question and the regular
expression an answer must match; the benchmark reads only the manifest and the
files, never this script.
"""
from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

GLYPHS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
}


class Canvas:
    def __init__(self, width, height, color):
        self.width, self.height = width, height
        self.pixels = bytearray(bytes(color) * (width * height))

    def rect(self, x, y, w, h, color):
        x0, x1 = max(0, x), min(self.width, x + w)
        for row in range(max(0, y), min(self.height, y + h)):
            offset = (row * self.width + x0) * 3
            self.pixels[offset:offset + (x1 - x0) * 3] = bytes(color) * (x1 - x0)

    def circle(self, cx, cy, r, color):
        for row in range(cy - r, cy + r + 1):
            dy = row - cy
            half = int((r * r - dy * dy) ** 0.5)
            self.rect(cx - half, row, 2 * half + 1, 1, color)

    def text(self, text, x, y, scale, color):
        for char in text:
            for row, bits in enumerate(GLYPHS[char]):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self.rect(x + column * scale, y + row * scale, scale, scale, color)
            x += 6 * scale

    def text_centered(self, text, scale, color):
        width = (6 * len(text) - 1) * scale
        self.text(text, (self.width - width) // 2, (self.height - 7 * scale) // 2, scale, color)

    def png(self):
        rows = b"".join(b"\0" + self.pixels[r * self.width * 3:(r + 1) * self.width * 3] for r in range(self.height))

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))

    def jpeg(self):
        try:
            from PIL import Image
        except ImportError:
            sys.exit("generate.py: Pillow is needed to encode the JPEG fixtures (pip install pillow)")
        import io
        out = io.BytesIO()
        Image.frombytes("RGB", (self.width, self.height), bytes(self.pixels)).save(out, "JPEG", quality=90)
        return out.getvalue()


WHITE, BLACK, GREY = (255, 255, 255), (20, 20, 20), (225, 225, 225)
RED, BLUE, GREEN, NAVY = (220, 30, 30), (30, 60, 220), (30, 160, 60), (20, 40, 110)


def number_png():
    c = Canvas(448, 224, WHITE)
    c.text_centered("7319", 12, BLACK)
    return c.png()


def shapes_jpeg():
    c = Canvas(448, 448, GREY)
    c.circle(125, 224, 85, RED)
    c.rect(270, 150, 145, 145, BLUE)
    return c.jpeg()


def word_jpeg():
    c = Canvas(560, 196, WHITE)
    c.text_centered("OCEAN", 14, NAVY)
    return c.jpeg()


def count_png():
    c = Canvas(448, 448, WHITE)
    for x, y in ((40, 50), (250, 90), (130, 290)):
        c.rect(x, y, 110, 110, GREEN)
    c.circle(350, 340, 50, RED)
    return c.png()


FIXTURES = [
    ("number-7319.png", number_png, "image/png",
     "What number is written in this image? Reply with the digits only.", r"7\s*,?\s*3\s*1\s*9"),
    ("shapes-red-circle.jpg", shapes_jpeg, "image/jpeg",
     "This image shows a circle and a square. What color is the circle? Reply with one word.", r"\bred\b"),
    ("word-ocean.jpg", word_jpeg, "image/jpeg",
     "What word is written in this image? Reply with the word only.", r"\bocean\b"),
    ("count-green-squares.png", count_png, "image/png",
     "How many green squares are in this image? Reply with a number.", r"\b(3|three)\b"),
]


def main():
    manifest = []
    for name, draw, mime, question, expect in FIXTURES:
        data = draw()
        (HERE / name).write_bytes(data)
        manifest.append({"file": name, "mime": mime, "question": question, "expect_regex": expect})
        print(f"{name}: {len(data)} bytes")
    (HERE / "manifest.json").write_text(json.dumps({"images": manifest}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
