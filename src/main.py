import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CONFIG = {
    "input_path":  "../input/vid1.mp4",
    "output_path": "../output/Video-995_matrix.mp4",

    "font_matrix_path":  "../fonts/matrix_glyphs.ttf",
    "font_sparkle_path": "../fonts/sparkle_glyphs.ttf",

    "charset_matrix":  "アイウエオカキクケコ0123456789",
    "charset_sparkle": "o+*.:",

    "cell_size": 9,           # px per glyph cell (bigger = fewer, larger glyphs)
    "font_size": 7,           # kept smaller than cell_size so stamps can jitter a bit within a cell

    # --- edge detection ---
    "canny_low": 40,
    "canny_high": 120,

    "empty_thresh": 0.04,      # below this: pure black. Raise to suppress more background noise.
    "fill_thresh": 0.22,       # above this: solid texture fill (e.g. the tablecloth)
    "sparse_glyph_prob": 0.55, # chance of drawing in the sparse/outline band
    "fill_stamps_per_cell": 2, # how many overlapping glyphs to stamp in the dense band (higher = more solid)

    "density_ema": 0.5,        # 0 = no smoothing (recompute cold each frame), closer to 1 = smoother/slower to react

    "original_dim": 0.0,       # how much of the original frame shows through (0=pure black bg, like reference)
    "glyph_alpha": 0.95,       # opacity of glyphs over the composite

    "color_matrix":  (60, 255, 90),   # BGR green
    "color_sparkle": (255, 255, 255), # BGR white

    "style_switch_at": 0.65,   # fraction of video where style shifts matrix -> sparkle

    "interpolate_factor": 1,   # 1 = no interpolation, 2 = double the frame rate, etc.
    "output_fps": None,        # None = keep source fps * interpolate_factor

    "context_cycle_sec": 2.5,     # length of one (effect + raw) cycle, in seconds
    "context_raw_sec": 1.2,       # how much of each cycle is raw original video (1-2s per your ask)
    "context_crossfade_sec": 0.15, # soft cross-fade in/out of the raw burst, avoids a jarring hard cut
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_font(path, size):
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    print(f"[matrix_fx] WARNING: font not found at {path}, using PIL default.")
    return ImageFont.load_default()


def interpolate_frames(frame_a, frame_b, n_between):
    """Optical-flow based in-between frames (frame_a -> frame_b)."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = gray_a.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    out = []
    for i in range(1, n_between + 1):
        t = i / (n_between + 1)
        map_x = (grid_x + flow[..., 0] * t).astype(np.float32)
        map_y = (grid_y + flow[..., 1] * t).astype(np.float32)
        warped = cv2.remap(frame_a, map_x, map_y, cv2.INTER_LINEAR)
        out.append(warped)
    return out


class GlyphRenderer:
    """Caches pre-rendered glyph bitmaps so we're not calling PIL per-cell per-frame."""

    def __init__(self, font, charset, color, cell_size):
        self.cell_size = cell_size
        self.glyphs = []
        for ch in charset:
            img = Image.new("RGBA", (cell_size, cell_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), ch, font=font, fill=(color[2], color[1], color[0], 255))
            self.glyphs.append(np.array(img))  # RGBA numpy

    def random_glyph(self):
        return random.choice(self.glyphs)


def compute_cell_density(edges, cell_size):
    """
    Downsample the edge map into a per-cell density grid (0..1), via box
    filtering. Each output cell value = fraction of edge pixels within it.
    """
    h, w = edges.shape
    kernel = np.ones((cell_size, cell_size), np.float32) / (cell_size * cell_size)
    density_full = cv2.filter2D((edges > 0).astype(np.float32), -1, kernel)
    
    return density_full[cell_size // 2::cell_size, cell_size // 2::cell_size]


def stamp_glyphs_by_density(canvas_bgr, density_grid, renderer, cfg):
    """
    Walk the density grid; for each cell decide black / sparse-outline /
    dense-fill based on the thresholds in cfg, and stamp glyph bitmaps
    accordingly onto canvas_bgr.
    """
    cs = renderer.cell_size
    alpha = cfg["glyph_alpha"]
    rows, cols = density_grid.shape

    for r in range(rows):
        for c in range(cols):
            d = density_grid[r, c]
            if d <= cfg["empty_thresh"]:
                continue  # stays black

            y, x = r * cs, c * cs
            if d <= cfg["fill_thresh"]:
                # sparse band: outline/contour glyphs
                if random.random() > cfg["sparse_glyph_prob"]:
                    continue
                stamps = 1
            else:
                # dense band: solid texture fill
                stamps = cfg["fill_stamps_per_cell"]

            for _ in range(stamps):
                glyph = renderer.random_glyph()
                gh, gw = glyph.shape[:2]
                # small random jitter within the cell for the multi-stamp fill look
                jx = x + random.randint(0, max(cs - gw, 0))
                jy = y + random.randint(0, max(cs - gh, 0))
                roi = canvas_bgr[jy:jy + gh, jx:jx + gw]
                if roi.shape[:2] != (gh, gw):
                    continue
                a = (glyph[..., 3:4].astype(np.float32) / 255.0) * alpha
                glyph_bgr = glyph[..., :3][..., ::-1]  # RGBA->BGR
                canvas_bgr[jy:jy + gh, jx:jx + gw] = (
                    roi.astype(np.float32) * (1 - a) + glyph_bgr.astype(np.float32) * a
                ).astype(np.uint8)


def raw_weight(t, cfg):
    """
    Returns 0..1: how much of the RAW original frame should show at time t
    (seconds into the output). 1 = fully raw, 0 = fully effect. Ramps up/down
    over context_crossfade_sec at the edges of each raw burst window so the
    cut isn't jarring.
    """
    cycle = cfg["context_cycle_sec"]
    raw_dur = cfg["context_raw_sec"]
    fade = max(cfg["context_crossfade_sec"], 1e-6)

    if cycle <= 0 or raw_dur <= 0:
        return 0.0

    pos = t % cycle
    if pos >= raw_dur:
        return 0.0
    if pos < fade:
        return pos / fade
    if pos > raw_dur - fade:
        return max((raw_dur - pos) / fade, 0.0)
    return 1.0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(config=CONFIG):
    here = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(here, config["input_path"]))
    output_path = os.path.normpath(os.path.join(here, config["output_path"]))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = config["output_fps"] or (src_fps * config["interpolate_factor"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

    font_matrix = load_font(os.path.join(here, config["font_matrix_path"]), config["font_size"])
    font_sparkle = load_font(os.path.join(here, config["font_sparkle_path"]), config["font_size"])
    renderer_matrix = GlyphRenderer(font_matrix, config["charset_matrix"], config["color_matrix"], config["cell_size"])
    renderer_sparkle = GlyphRenderer(font_sparkle, config["charset_sparkle"], config["color_sparkle"], config["cell_size"])

    cs = config["cell_size"]
    grid_h = h // cs
    grid_w = w // cs
    density_ema = np.zeros((grid_h, grid_w), dtype=np.float32)

    prev_frame = None
    frame_idx = 0
    output_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_to_process = [frame]
        if prev_frame is not None and config["interpolate_factor"] > 1:
            in_between = interpolate_frames(prev_frame, frame, config["interpolate_factor"] - 1)
            frames_to_process = in_between + [frame]

        for f in frames_to_process:
            t = output_frame_idx / out_fps
            w_raw = raw_weight(t, config)

            if w_raw >= 1.0:
                writer.write(f)
                output_frame_idx += 1
                continue

            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, config["canny_low"], config["canny_high"])

            density = compute_cell_density(edges, cs)
            density = density[:grid_h, :grid_w]

            k = config["density_ema"]
            density_ema = density_ema * k + density * (1 - k)

            if config["original_dim"] > 0:
                base = (f.astype(np.float32) * config["original_dim"]).astype(np.uint8)
            else:
                base = np.zeros_like(f)

            progress = frame_idx / max(n_frames, 1)
            renderer = renderer_matrix if progress < config["style_switch_at"] else renderer_sparkle

            stamp_glyphs_by_density(base, density_ema, renderer, config)

            if w_raw > 0.0:
                base = (
                    f.astype(np.float32) * w_raw + base.astype(np.float32) * (1 - w_raw)
                ).astype(np.uint8)

            writer.write(base)
            output_frame_idx += 1

        prev_frame = frame
        frame_idx += 1
        # if frame_idx % 20 == 0:
        #     print(f"[matrix_fx] processed {frame_idx}/{n_frames} source frames")

    cap.release()
    writer.release()
    print(f"[matrix_fx] done -> {output_path}")


if __name__ == "__main__":
    run()
