import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

CELL = 14          # size of each character cell in px
FONT_SIZE = 14
CHARSET_MATRIX = "アイウエオ0123456789ABCDEF"   # matrix-style glyphs
CHARSET_SPARKLE = "o+*.:"                        # later "sparkle" glyphs

font_matrix = ImageFont.truetype("DejaVuSansMono.ttf", FONT_SIZE)
font_sparkle = ImageFont.truetype("DejaVuSansMono.ttf", FONT_SIZE)

cap = cv2.VideoCapture("vid1.mp4")
frame_idx = 0
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Get a mask from the source frame (edges or luminance)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)              # or just threshold `gray`
    mask = cv2.dilate(edges, None, iterations=2)

    h, w = mask.shape
    out = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(out)

    # 2. Blend between "matrix" style early and "sparkle" style late
    t = frame_idx / n_frames
    charset = CHARSET_MATRIX if t < 0.7 else CHARSET_SPARKLE
    font = font_matrix if t < 0.7 else font_sparkle
    color = (0, 255, 70) if t < 0.7 else (255, 255, 255)

    # 3. Walk a grid of cells; only draw a glyph where the mask says "on"
    #    and randomly skip some for a sparse/noisy look
    for y in range(0, h, CELL):
        for x in range(0, w, CELL):
            if mask[y:y+CELL, x:x+CELL].mean() > 20 and random.random() < 0.6:
                ch = random.choice(charset)
                draw.text((x, y), ch, font=font, fill=color)

    out.save(f"frames_out/frame_{frame_idx:03d}.png")
    frame_idx += 1

cap.release()