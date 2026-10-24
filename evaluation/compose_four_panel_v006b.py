#!/usr/bin/env python3
"""Compose the nine existing V006-B rollouts into an 81-second four-panel MP4.

No MuJoCo simulation, checkpoint loading, or mutation of recorded data.
Requires numpy, Pillow, and an ffmpeg executable with libx264.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'evaluation' / 'results'
TASKS = (
    ('Task 1', (0, 0), 'video_test_act_only', 'video_test_assisted15', 'video_task1_teacher'),
    ('Task 2', (10, 10), 'video_task2_act', 'video_task2_assisted', 'video_task2_teacher'),
    ('Task 3', (-10, -10), 'video_task3_act', 'video_task3_assisted', 'video_task3_teacher'),
)
W, H = 1920, 1080
PW, PH = W // 2, H // 2
FPS, SECONDS = 25, 27
NFRAMES = FPS * SECONDS
BG = '#0b1220'
PANEL = '#142036'
FG = '#edf2fb'
MUTED = '#9dacc3'
BLUE = '#56adff'
CYAN = '#30d6c8'
ORANGE = '#ffbf69'
GRID = '#34445d'
RED = '#ff7373'


def font(size: int, bold: bool = False):
    options = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for path in options:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

F18 = font(18)
F20 = font(20)
F24 = font(24)
F29 = font(29, True)
F33 = font(33, True)


def data(folder: Path):
    with np.load(folder / 'rollout.npz', allow_pickle=False) as archive:
        needed = ('time', 'state', 'command', 'control_mode')
        for k in needed:
            if k not in archive:
                raise ValueError(f'Missing {k} in {folder}')
        result = {k: archive[k].copy() for k in needed}
    return result


def validate(folder: Path, expect_mode: str):
    meta_path = folder / 'result.json'
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('mode') != expect_mode:
        raise ValueError(f'{folder}: mode expected {expect_mode}, got {meta.get("mode")}')
    if int(meta.get('frame_count', -1)) != NFRAMES or abs(meta.get('duration_seconds', -1) - SECONDS) > 1e-6:
        raise ValueError(f'{folder}: expected {NFRAMES} frames over {SECONDS}s')
    frames = sorted((folder / 'frames').glob('*.png'))
    if len(frames) != NFRAMES:
        raise ValueError(f'{folder}: expected {NFRAMES} PNGs, found {len(frames)}')
    arrays = data(folder)
    if len(arrays['time']) != 1350 or arrays['state'].shape != (1350, 7) or arrays['command'].shape != (1350, 7):
        raise ValueError(f'{folder}: unexpected arrays')
    if np.any(np.diff(arrays['time']) <= 0) or not np.all(np.isfinite(arrays['state'])):
        raise ValueError(f'{folder}: invalid time/state')
    if not np.allclose(arrays['time'], np.arange(1350) * .02, atol=1e-5):
        raise ValueError(f'{folder}: unexpected time axis')
    return meta, arrays, frames


def sync(t_ref: np.ndarray, series: np.ndarray, t_src: np.ndarray):
    if series.ndim == 1:
        return np.interp(t_ref, t_src, series)
    return np.column_stack([np.interp(t_ref, t_src, series[:, j]) for j in range(series.shape[1])])


def plot_base(title: str, ylabel: str, lines: list[tuple[str, np.ndarray, str]], t: np.ndarray,
              ymin: float, ymax: float, show_switch=True):
    image = Image.new('RGB', (PW, PH), PANEL)
    d = ImageDraw.Draw(image)
    d.text((40, 22), title, font=F29, fill=FG)
    d.text((40, 65), ylabel, font=F18, fill=MUTED)
    left, top, right, bottom = 95, 124, 915, 447

    def xx(v):
        return round(left + (v / SECONDS) * (right - left))

    def yy(v):
        return round(bottom - (float(v) - ymin) / (ymax - ymin) * (bottom - top))

    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        y = yy(v)
        d.line((left, y, right, y), fill=GRID, width=1)
        label = f'{v:.1f}' if ymax > 1 else f'{v:.2f}'
        d.text((22, y - 11), label, font=F18, fill=MUTED)
    for sec in (0, 5, 10, 15, 20, 25, 27):
        x = xx(sec)
        d.line((x, top, x, bottom), fill=GRID, width=1)
        d.text((x - 12, bottom + 12), str(sec), font=F18, fill=MUTED)
    d.text((right - 50, bottom + 47), 'time (s)', font=F18, fill=MUTED)
    if show_switch:
        x = xx(15)
        for y in range(top, bottom, 13):
            d.line((x, y, x, min(y + 7, bottom)), fill=ORANGE, width=2)
        d.text((x + 9, top + 7), 'Switch 15s', font=F18, fill=ORANGE)
    for label, series, color in lines:
        if len(series) != len(t):
            raise ValueError(f'Plot length mismatch for {label}')
        xy = [(xx(float(tt)), max(top, min(bottom, yy(v)))) for tt, v in zip(t, series)]
        if len(xy) >= 2:
            d.line(xy, fill=color, width=3, joint='curve')
    # Legend, bottom band
    xpos = left
    for label, _, color in lines:
        d.line((xpos, 501, xpos + 24, 501), fill=color, width=4)
        d.text((xpos + 31, 490), label, font=F18, fill=FG)
        xpos += 190 if len(label) < 12 else 225
    return image, (xx, top, bottom)


def robot_panel(image_path: Path, heading: str, status: str, time_s: float,
                task_name: str, outcome: bool, switch=None):
    panel = Image.new('RGB', (PW, PH), BG)
    with Image.open(image_path) as original:
        shot = ImageOps.contain(original.convert('RGB'), (PW - 24, PH - 135), Image.Resampling.LANCZOS)
    x = (PW - shot.width) // 2
    y = 75 + ((PH - 140) - shot.height) // 2
    panel.paste(shot, (x, y))
    d = ImageDraw.Draw(panel)
    d.rectangle((0, 0, PW, 66), fill=PANEL)
    d.text((24, 14), heading, font=F33, fill=FG)
    d.text((PW - 180, 22), f'{time_s:05.2f}s', font=F24, fill=MUTED)
    d.rectangle((0, PH - 66, PW, PH), fill=PANEL)
    d.text((24, PH - 48), task_name, font=F24, fill=FG)
    color = ORANGE if status == 'TEACHER' else CYAN
    d.text((305, PH - 48), status, font=F24, fill=color)
    if switch is not None:
        d.text((507, PH - 48), 'switch @ 15s', font=F18, fill=MUTED)
    d.text((PW - 132, PH - 48), 'PASS' if outcome else 'FAIL',
           font=F24, fill=CYAN if outcome else RED)
    return panel


def place_plot(base: Image.Image, mapping, t_sec: float):
    img = base.copy()
    d = ImageDraw.Draw(img)
    xx, top, bottom = mapping
    x = xx(t_sec)
    d.line((x, top, x, bottom), fill='#ffffff', width=3)
    d.ellipse((x - 5, top - 5, x + 5, top + 5), fill='#ffffff')
    return img


def build_task(task, root=RESULTS):
    title, offset, act_name, assisted_name, teacher_name = task
    dirs = (root / act_name, root / assisted_name, root / teacher_name)
    records = [validate(folder, mode) for folder, mode in zip(dirs, ('act', 'assisted', 'teacher'))]
    (meta_a, a, fa), (meta_b, b, fb), (meta_t, teacher, _) = records
    if abs(meta_b.get('switch_seconds', -1) - 15) > 1e-6:
        raise ValueError(f'{title}: assisted switch must be 15 seconds')
    if any(not np.allclose(meta.get('cube_offset_mm', []), offset) for meta, _, _ in records):
        raise ValueError(f'{title}: cube offsets mismatch')
    if not (np.all(a['control_mode'] == 0)
            and np.all(teacher['control_mode'] == 1)
            and np.all(b['control_mode'][:750] == 1)
            and np.all(b['control_mode'][750:] == 0)):
        raise ValueError(f'{title}: unexpected controller transition')
    t = teacher['time']
    qa = sync(t, a['state'][:, :6], a['time'])
    qb = sync(t, b['state'][:, :6], b['time'])
    qt = teacher['state'][:, :6]
    rmse_a = np.sqrt(np.mean((qa - qt)**2, axis=1))
    rmse_b = np.sqrt(np.mean((qb - qt)**2, axis=1))
    cmd_t = teacher['command'][:, 6]
    cmd_a = sync(t, a['command'][:, 6], a['time'])
    cmd_b = sync(t, b['command'][:, 6], b['time'])
    maximum_rmse = max(float(rmse_a.max()), float(rmse_b.max()), .05)
    maximum_rmse = np.ceil(maximum_rmse / .05) * .05
    err_base, err_map = plot_base(
        'JOINT TRACKING ERROR', 'RMSE vs teacher  |  6 arm joints  |  rad',
        [('ACT Only', rmse_a, BLUE), ('Teacher -> ACT', rmse_b, CYAN)], t,
        0., maximum_rmse)
    grip_base, grip_map = plot_base(
        'GRIPPER COMMAND', 'Applied actuator command  |  0 = open, 255 = close',
        [('Teacher', cmd_t, ORANGE), ('ACT Only', cmd_a, BLUE), ('Teacher -> ACT', cmd_b, CYAN)], t,
        0., 255.)
    return title, offset, (meta_a, fa), (meta_b, fb), err_base, err_map, grip_base, grip_map


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=RESULTS)
    p.add_argument('--output', type=Path, default=ROOT / 'evaluation/results/final_demo_v006b.mp4')
    p.add_argument('--preset', default='veryfast', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium'])
    p.add_argument('--crf', type=int, default=22)
    p.add_argument('--limit-frames', type=int, default=0, help='Smoke test: first N frames of each task; 0 = all')
    args = p.parse_args()
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found: install with sudo apt install ffmpeg')
    if args.output.exists():
        raise FileExistsError(f'Output already exists (not overwriting): {args.output}')
    items = [build_task(task, args.results_dir) for task in TASKS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-an', '-c:v', 'libx264', '-preset', args.preset,
           '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    print('Output:', args.output, flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for task_id, item in enumerate(items, 1):
            title, offset, act, assisted, err_base, err_map, grip_base, grip_map = item
            print(f'{title}: offset XY = {offset} mm', flush=True)
            frame_count = NFRAMES if args.limit_frames == 0 else min(args.limit_frames, NFRAMES)
            for frame in range(frame_count):
                second = frame / FPS
                mode = 'TEACHER' if second < 15 else 'ACT'
                left = robot_panel(act[1][frame], 'ACT ONLY', 'ACT', second,
                                   title, act[0]['provisional_geometric_success'])
                right = robot_panel(assisted[1][frame], 'TEACHER -> ACT', mode, second,
                                    title, assisted[0]['provisional_geometric_success'], switch=15)
                bottom_left = place_plot(err_base, err_map, second)
                bottom_right = place_plot(grip_base, grip_map, second)
                canvas = Image.new('RGB', (W, H), BG)
                canvas.paste(left, (0, 0))
                canvas.paste(right, (PW, 0))
                canvas.paste(bottom_left, (0, PH))
                canvas.paste(bottom_right, (PW, PH))
                if proc.stdin is None:
                    raise RuntimeError('ffmpeg stdin closed')
                proc.stdin.write(canvas.tobytes())
                if (frame + 1) % 225 == 0:
                    print(f'  {frame + 1}/{frame_count} frames', flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f'ffmpeg returned {proc.returncode}')
    except BaseException:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        proc.kill()
        proc.wait()
        if args.output.exists():
            args.output.unlink()
        raise
    print('Saved:', args.output, flush=True)
    print('Duration:', len(TASKS) * (NFRAMES if args.limit_frames == 0 else min(args.limit_frames, NFRAMES)) / FPS, 'seconds', flush=True)


if __name__ == '__main__':
    main()
