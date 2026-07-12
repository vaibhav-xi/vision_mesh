import os
import numpy as np
import cv2
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CONFIG = {
    "input_path":  "../input/vid2.mp4",
    "output_path": "../output/vid2_rgbfx.mp4",

    # (still-image mode only - ignored if input_path is a video, since the
    # video's own length/fps is used instead)
    "duration_sec": 18.0,
    "fps": 30,

    # --- persistent canvas / feedback ---
    # each frame, canvas = (1-keyframe_alpha)*effect_output + keyframe_alpha*original.
    # Higher keyframe_alpha = stays closer to the source image (calmer, more
    # recognizable). Lower = melts further into abstraction over time.
    "keyframe_alpha": 0.05,

    # --- effect schedule ---
    "segment_sec": 1.15,       # how long each effect gets before crossfading to the next
    "crossfade_sec": 0.35,     # overlap/blend time between consecutive effects
    "effect_order": ["wave_rgb", "chromatic_shift", "pixel_sort", "liquid_warp", "feedback_zoom"],

    # --- wave_rgb ---
    "wave_cycle_seconds": 3.0,
    "wave_propagation_mode": "radial",   # horizontal / vertical / diagonal / radial / none
    "wave_radial_origin": (0.5, 0.4),
    "wave_sweep_seconds": 6.0,

    # --- chromatic_shift ---
    "chroma_amp_px": 14.0,       # how far channels pull apart, in pixels
    "chroma_freq_hz": 0.35,      # how fast the pulsing is

    # --- pixel_sort ---
    "sort_band_height_px": 50,
    "sort_scan_period_sec": 2.5,  # how long for the scan band to travel top->bottom

    # --- liquid_warp ---
    "liquid_amp_px": 10.0,
    "liquid_freq": 4.0,           # spatial frequency of the ripples
    "liquid_speed_hz": 0.25,

    # --- feedback_zoom ---
    "zoom_amp": 0.02,             # +/- fractional scale pulsing
    "zoom_period_sec": 4.0,
    "rotate_amp_deg": 1.2,
    "rotate_period_sec": 5.0,
}


# --------------------------------------------------------------------------
# Effect implementations
# --------------------------------------------------------------------------
def effect_wave_rgb(canvas, t, cfg, grids):
    delay_sec = grids["wave_delay_sec"]
    local_t = t - delay_sec
    phase = local_t / cfg["wave_cycle_seconds"]
    R, G, B = canvas[..., 0], canvas[..., 1], canvas[..., 2]
    newR = 128 + 127 * np.sin(2*np.pi * (phase + (R*5 + G*3 - B*2) / 256.0))
    newG = 128 + 127 * np.sin(2*np.pi * (phase*1.3 + (G*5 + B*3 - R*2) / 256.0))
    newB = 128 + 127 * np.sin(2*np.pi * (phase*1.7 + (B*5 + R*3 - G*2) / 256.0))
    return np.stack([newR, newG, newB], axis=-1)


def effect_chromatic_shift(canvas, t, cfg, grids):
    h, w = canvas.shape[:2]
    amp = cfg["chroma_amp_px"]
    freq = cfg["chroma_freq_hz"]
    dxr = amp * np.sin(2*np.pi*freq*t)
    dxg = amp * np.sin(2*np.pi*freq*t + 2.094)  # +120deg phase
    dxb = amp * np.sin(2*np.pi*freq*t + 4.188)  # +240deg phase

    def shift_channel(ch, dx, dy):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(ch, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    R = shift_channel(canvas[..., 0], dxr, dxr * 0.3)
    G = shift_channel(canvas[..., 1], dxg, -dxg * 0.2)
    B = shift_channel(canvas[..., 2], dxb, dxb * 0.4)
    return np.stack([R, G, B], axis=-1)


def effect_pixel_sort(canvas, t, cfg, grids):
    h, w = canvas.shape[:2]
    band_h = cfg["sort_band_height_px"]
    period = cfg["sort_scan_period_sec"]
    band_y = int(((t % period) / period) * h)
    y0, y1 = band_y, min(band_y + band_h, h)
    out = canvas.copy()
    if y1 > y0:
        band = out[y0:y1]
        brightness = band.mean(axis=2)
        order = np.argsort(brightness, axis=1)
        for c in range(3):
            band[..., c] = np.take_along_axis(band[..., c], order, axis=1)
        out[y0:y1] = band
    return out


def effect_liquid_warp(canvas, t, cfg, grids):
    h, w = canvas.shape[:2]
    amp = cfg["liquid_amp_px"]
    freq = cfg["liquid_freq"]
    speed = cfg["liquid_speed_hz"]
    dx = amp * np.sin(2*np.pi * (grids["ys"]/h*freq + t*speed))
    dy = amp * np.cos(2*np.pi * (grids["xs"]/w*freq + t*speed*0.8))
    map_x = (grids["xs"] + dx).astype(np.float32)
    map_y = (grids["ys"] + dy).astype(np.float32)
    return cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def effect_feedback_zoom(canvas, t, cfg, grids):
    h, w = canvas.shape[:2]
    scale = 1.0 + cfg["zoom_amp"] * np.sin(2*np.pi * t / cfg["zoom_period_sec"])
    angle = cfg["rotate_amp_deg"] * np.sin(2*np.pi * t / cfg["rotate_period_sec"])
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    return cv2.warpAffine(canvas, M, (w, h), borderMode=cv2.BORDER_REFLECT)


EFFECTS = {
    "wave_rgb": effect_wave_rgb,
    "chromatic_shift": effect_chromatic_shift,
    "pixel_sort": effect_pixel_sort,
    "liquid_warp": effect_liquid_warp,
    "feedback_zoom": effect_feedback_zoom,
}


# --------------------------------------------------------------------------
# Scheduler: which effect(s) are active at time t, with crossfade blend
# --------------------------------------------------------------------------
def get_active_effects(t, cfg):
    order = cfg["effect_order"]
    seg = cfg["segment_sec"]
    fade = cfg["crossfade_sec"]
    n = len(order)
    cycle_len = seg * n

    t_in_cycle = t % cycle_len
    idx = int(t_in_cycle // seg)
    t_in_seg = t_in_cycle - idx * seg

    name_a = order[idx % n]
    name_b = order[(idx + 1) % n]

    if t_in_seg >= seg - fade:
        blend = (t_in_seg - (seg - fade)) / fade
        return name_a, name_b, blend
    return name_a, name_a, 0.0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(config=CONFIG):
    here = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(here, config["input_path"]))
    output_path = os.path.normpath(os.path.join(here, config["output_path"]))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext not in IMAGE_EXTENSIONS

    video_cap = None
    if is_video:
        video_cap = cv2.VideoCapture(input_path)
        if not video_cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {input_path}")
        fps = video_cap.get(cv2.CAP_PROP_FPS) or config["fps"]
        w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first = video_cap.read()
        if not ret:
            raise RuntimeError("Input video has no frames")
        canvas = first[..., ::-1].astype(np.float32)  # BGR -> RGB
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind so the loop below re-reads frame 0
    else:
        img = Image.open(input_path).convert("RGB")
        static_source = np.array(img).astype(np.float32)  # HxWx3 RGB
        h, w = static_source.shape[:2]
        fps = config["fps"]
        n_frames = int(config["duration_sec"] * fps)
        canvas = static_source.copy()

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    grids = {"ys": ys, "xs": xs}

    mode = config["wave_propagation_mode"]
    if mode == "none":
        wave_delay_sec = np.zeros((h, w), dtype=np.float32)
    else:
        if mode == "horizontal":
            metric = xs / w
        elif mode == "vertical":
            metric = ys / h
        elif mode == "diagonal":
            metric = (xs / w + ys / h) / 2.0
        elif mode == "radial":
            ox, oy = config["wave_radial_origin"]
            cx, cy = ox * w, oy * h
            metric = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            metric = metric / metric.max()
        else:
            raise ValueError(f"Unknown wave_propagation_mode: {mode}")
        wave_delay_sec = metric * config["wave_sweep_seconds"]
    grids["wave_delay_sec"] = wave_delay_sec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    keyframe_alpha = config["keyframe_alpha"]

    print(f"[rgb_fx] mode={'video' if is_video else 'image'}  input={input_path}")
    print(f"[rgb_fx] effect order: {config['effect_order']}  "
          f"segment_sec={config['segment_sec']}  frames={n_frames}  fps={fps:.2f}")

    frame_idx = 0
    while frame_idx < n_frames:
        if is_video:
            ret, src_frame = video_cap.read()
            if not ret:
                break
            keyframe = src_frame[..., ::-1].astype(np.float32)
        else:
            keyframe = static_source

        t = frame_idx / fps
        name_a, name_b, blend = get_active_effects(t, config)

        out = EFFECTS[name_a](canvas, t, config, grids)
        if blend > 0.0:
            out_b = EFFECTS[name_b](canvas, t, config, grids)
            out = out * (1 - blend) + out_b * blend

        canvas = np.clip(out * (1 - keyframe_alpha) + keyframe * keyframe_alpha, 0, 255).astype(np.float32)

        bgr = canvas[..., ::-1].astype(np.uint8)
        writer.write(bgr)

        if frame_idx % 30 == 0:
            active = name_a if blend == 0 else f"{name_a}->{name_b} ({blend:.2f})"
            print(f"[rgb_fx] rendered {frame_idx}/{n_frames}  active={active}")

        frame_idx += 1

    if video_cap is not None:
        video_cap.release()
    writer.release()
    print(f"[rgb_fx] done -> {output_path}")


if __name__ == "__main__":
    run()
