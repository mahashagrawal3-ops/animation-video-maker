# animation-video-maker
coding to video maker
#!/usr/bin/env python3
"""
make_animation_from_audio.py

- Transcribes audio to subtitles using whisper (optional).
- Generates a vertical 1080x1920 animated video (cartoon-horror style) using MoviePy + Pillow.
- Renders subtitles burned into the video and uses the original audio.

Usage:
    python make_animation_from_audio.py

Outputs:
    output_video.mp4
    subtitles.srt   (if transcription was run)
"""

import os
import math
from pathlib import Path
from typing import List, Tuple
import subprocess
import sys

# -- External libs --
from moviepy.editor import (
    VideoClip,
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from tqdm import tqdm

# Optional transcription (Whisper)
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# ---------------------------
# USER CONFIG
# ---------------------------
AUDIO_PATH = "/mnt/data/Headfone _ Danny, based on Dracula (Hindi), Ep 01 _ Hindi Audio Series _ Horror Vampire Story.mp3"
OUTPUT_VIDEO = "output_video.mp4"
SRT_PATH = "subtitles.srt"  # created if transcription runs
MODEL_NAME = "small"  # whisper model: tiny, base, small, medium, large (choose based on your machine)
FPS = 24
W, H = 1080, 1920  # vertical 9:16 1080p vertical
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # change if necessary
SUB_FONT_SIZE = 48
TITLE = "Dracula (Horror — Animated Audio Adaptation)"

# animation parameters
MOON_X, MOON_Y = W * 0.8, H * 0.2
MOON_RADIUS = 140

# ---------------------------
# UTILS: Time formatting for SRT
# ---------------------------
def sec_to_srt_time(t: float) -> str:
    millis = int((t - math.floor(t)) * 1000)
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

# ---------------------------
# TRANSCRIPTION (whisper)
# ---------------------------
def transcribe_audio_to_srt(audio_path: str, srt_out: str, model_name: str = "small") -> None:
    if not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper is not installed/available in this environment.")
    print("Loading Whisper model:", model_name)
    model = whisper.load_model(model_name)
    print("Transcribing audio (this may take time)...")
    # Note: Using word_timestamps requires whisperx or modifications. We'll use segment-level timestamps.
    result = model.transcribe(audio_path, language="hi", fp16=False)
    # result['segments'] is a list with start/end/text
    segments = result.get("segments", [])
    # write srt
    with open(srt_out, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = sec_to_srt_time(seg["start"])
            end = sec_to_srt_time(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print("Wrote SRT to", srt_out)

# ---------------------------
# SRT PARSER (simple)
# ---------------------------
def parse_srt(srt_path: str) -> List[Tuple[float, float, str]]:
    """
    Returns list of (start_sec, end_sec, text)
    """
    items = []
    if not os.path.exists(srt_path):
        return items
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    for b in blocks:
        lines = b.splitlines()
        if len(lines) >= 3:
            # lines[0] index, lines[1] times, rest text
            times = lines[1]
            try:
                start_s, end_s = times.split(" --> ")
                def parse_time(s):
                    hh, mm, ss_ms = s.split(":")
                    ss, ms = ss_ms.split(",")
                    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
                start = parse_time(start_s.strip())
                end = parse_time(end_s.strip())
                text = " ".join(lines[2:])
                items.append((start, end, text))
            except Exception:
                continue
    return items

# ---------------------------
# ANIMATION FRAME GENERATOR (PIL)
# ---------------------------
def make_frame_image(t: float, duration: float) -> Image.Image:
    """
    Create a stylized horror background frame with a vampire silhouette and motion effects.
    t: time in seconds
    """
    img = Image.new("RGBA", (W, H), (10, 8, 20, 255))
    draw = ImageDraw.Draw(img)

    # Background: vertical gradient (dark purple -> near black)
    for y in range(H):
        ratio = y / H
        r = int(12 + (30 - 12) * (1 - ratio))
        g = int(8 + (10 - 8) * (1 - ratio))
        b = int(20 + (40 - 20) * (1 - ratio))
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Moon with subtle flicker
    flicker = 8 * math.sin(t * 2.1)
    moon_r = int(MOON_RADIUS + flicker)
    moon_xy = (int(MOON_X - moon_r), int(MOON_Y - moon_r), int(MOON_X + moon_r), int(MOON_Y + moon_r))
    draw.ellipse(moon_xy, fill=(240, 230, 200, 255))
    # Moon halo
    halo = Image.new("RGBA", (W, H))
    hd = ImageDraw.Draw(halo)
    for i in range(1, 8):
        alpha = int(30 / i)
        r = moon_r + 20 + i * 6
        hd.ellipse((MOON_X - r, MOON_Y - r, MOON_X + r, MOON_Y + r), outline=(240,230,200,alpha))
    img = Image.alpha_composite(img, halo)

    # Distant castle silhouette (parallax)
    castle = Image.new("RGBA", (W, H))
    cd = ImageDraw.Draw(castle)
    base_y = int(H * 0.62)
    # draw simple towers
    towers_x = [W * 0.08, W * 0.20, W * 0.36, W * 0.55, W * 0.72]
    for i, tx in enumerate(towers_x):
        w_t = int(80 + 40 * (i % 2))
        h_t = int(180 + 40 * ((i + 1) % 2))
        tx_int = int(tx)
        cd.rectangle((tx_int - w_t//2, base_y - h_t, tx_int + w_t//2, base_y), fill=(12, 10, 15, 255))
        # little battlements
        for b in range(3):
            cd.rectangle((tx_int - w_t//2 + 8 + b*20, base_y - h_t - 10, tx_int - w_t//2 + 18 + b*20, base_y - h_t), fill=(10,8,12,255))
    # castle base
    cd.rectangle((0, base_y, W, base_y + 300), fill=(8,6,10,255))
    # slight blur for depth
    castle = castle.filter(ImageFilter.GaussianBlur(radius=2))
    img = Image.alpha_composite(img, castle)

    # Foreground: vampire silhouette (simple)
    vamp = Image.new("RGBA", (W, H))
    vd = ImageDraw.Draw(vamp)
    # vampire position bobbing
    bob = 12 * math.sin(t * 2.5)
    vamp_x = int(W * 0.37 + 30 * math.sin(t * 0.9))
    vamp_y = int(base_y - 120 + bob)
    # body (cape)
    cape_coords = [(vamp_x - 160, vamp_y + 200), (vamp_x + 160, vamp_y - 160)]
    vd.polygon([(vamp_x-100, vamp_y+180),
                (vamp_x-160, vamp_y+200),
                (vamp_x-30, vamp_y-90),
                (vamp_x, vamp_y-140),
                (vamp_x+30, vamp_y-90),
                (vamp_x+160, vamp_y+200),
                (vamp_x+100, vamp_y+180)], fill=(10, 10, 10, 255))
    # head and ears
    vd.ellipse((vamp_x-30, vamp_y-170, vamp_x+30, vamp_y-110), fill=(18,18,18,255))
    # collar spikes
    vd.polygon([(vamp_x-50, vamp_y-110),(vamp_x-20, vamp_y-80),(vamp_x-5, vamp_y-100)], fill=(30,30,30,255))
    vd.polygon([(vamp_x+50, vamp_y-110),(vamp_x+20, vamp_y-80),(vamp_x+5, vamp_y-100)], fill=(30,30,30,255))
    # little reflective eyes
    eye_offset = 20
    vd.ellipse((vamp_x - eye_offset - 5, vamp_y - 150, vamp_x - eye_offset + 5, vamp_y - 140), fill=(180, 50, 50, 255))
    vd.ellipse((vamp_x + eye_offset - 5, vamp_y - 150, vamp_x + eye_offset + 5, vamp_y - 140), fill=(180, 50, 50, 255))
    vamp = vamp.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = Image.alpha_composite(img, vamp)

    # Motion graphics - floating mist layers (animated with sine-based alpha)
    mist = Image.new("RGBA", (W, H))
    md = ImageDraw.Draw(mist)
    # draw a few wavy shapes
    for i in range(4):
        offset = int(90 * math.sin(t * (0.2 + i*0.1) + i))
        y = int(H * (0.45 + i * 0.06))
        md.ellipse((-200 + offset, y - 80, W + 200 + offset, y + 80), fill=(220, 220, 240, int(18 + 8 * math.sin(t * 0.6 + i))))
    img = Image.alpha_composite(img, mist)

    # Vignette
    vign = Image.new("L", (W, H), 0)
    vd2 = ImageDraw.Draw(vign)
    maxr = math.hypot(W/2, H/2)
    for i in range(10):
        alpha = int(20 + i*12)
        bbox = [i*30, i*30, W - i*30, H - i*30]
        vd2.rectangle(bbox, outline=alpha, fill=None)
    img = img.convert("RGB")
    # final mild contrast
    return img

# ---------------------------
# SUBTITLE RENDERING (moviepy TextClip per segment)
# ---------------------------
def create_subtitle_clips(subs: List[Tuple[float,float,str]], duration: float) -> List[ImageClip]:
    clips = []
    # Ensure font available
    font = FONT_PATH if os.path.exists(FONT_PATH) else None
    for start, end, text in subs:
        txt = TextClip(
            txt=text,
            fontsize=SUB_FONT_SIZE,
            font=font,
            method="caption",
            size=(int(W*0.9), None),
            align="center"
        ).set_position(("center", H - int(H*0.12))).set_start(start).set_duration(max(0.5, end - start))
        # Add semi-transparent background box
        # moviepy doesn't support background color for TextClip easily, so create a black rectangle clip
        box = (TextClip(" ", fontsize=SUB_FONT_SIZE, font=font, size=(int(W*0.9), int(SUB_FONT_SIZE*1.8)), color='white')
               .on_color(size=(int(W*0.9), int(SUB_FONT_SIZE*1.8)), color=(0,0,0), col_opacity=0.45)
               .set_position(("center", H - int(H*0.12))).set_start(start).set_duration(max(0.5, end-start)))
        clips.append(box)
        clips.append(txt)
    return clips

# ---------------------------
# MAIN: Build video
# ---------------------------
def build_video(audio_path: str, srt_path: str = None, output_path: str = OUTPUT_VIDEO):
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    print(f"Audio duration: {duration:.1f}s")

    # make a VideoClip from PIL frames
    def make_frame(t):
        pil_img = make_frame_image(t, duration)
        return np.asarray(pil_img)

    base_clip = VideoClip(make_frame, duration=duration).set_fps(FPS)

    # Title overlay (fade in/out first 3s)
    font = FONT_PATH if os.path.exists(FONT_PATH) else None
    title_clip = TextClip(TITLE, fontsize=64, font=font, method="label").set_position(("center", int(H*0.08))).set_duration(4).fadeout(1.0)

    # If we have subs, parse and render them
    subtitle_clips = []
    if srt_path and os.path.exists(srt_path):
        subs = parse_srt(srt_path)
        subtitle_clips = create_subtitle_clips(subs, duration)

    # Compose everything
    layers = [base_clip, title_clip]
    layers.extend(subtitle_clips)
    final = CompositeVideoClip(layers, size=(W,H)).set_audio(audio)

    # render
    print("Rendering video... this may take some minutes depending on CPU/GPU.")
    final.write_videofile(output_path, fps=FPS, codec="libx264", audio_codec="aac", bitrate="6M")
    print("Done. Output:", output_path)

# ---------------------------
# ENTRY POINT
# ---------------------------
def main():
    # check audio path
    if not os.path.exists(AUDIO_PATH):
        print("Audio file not found at:", AUDIO_PATH)
        sys.exit(1)

    # If SRT exists already, skip transcription
    if os.path.exists(SRT_PATH):
        print("Found existing SRT:", SRT_PATH)
    else:
        # try to transcribe if whisper is available
        if WHISPER_AVAILABLE:
            try:
                transcribe_audio_to_srt(AUDIO_PATH, SRT_PATH, model_name=MODEL_NAME)
            except Exception as e:
                print("Transcription failed:", e)
                print("Proceeding without subtitles. You can provide an SRT file named", SRT_PATH)
        else:
            print("Whisper not available. If you want subtitles, install whisper or provide an SRT file named", SRT_PATH)

    build_video(AUDIO_PATH, SRT_PATH, OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
