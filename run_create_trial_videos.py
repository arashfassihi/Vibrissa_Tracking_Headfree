"""Create original trial videos from numbered image folders before head DLC.

MP4-only output avoids large AVI intermediates. Optional H.264 uses FFmpeg.
Frame rate must match acquisition; the old DeepWhiskerCuts code used 40 Hz.
"""
import argparse
import math
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


def image_files(folder):
    extensions = {'.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = [p for p in Path(folder).iterdir()
             if p.is_file() and p.suffix.lower() in extensions]
    if not files:
        raise ValueError(f'No supported images in {folder}')
    if any(not p.stem.isdigit() for p in files):
        raise ValueError(f'Expected numeric image filenames in {folder}')
    files.sort(key=lambda p: int(p.stem))
    numbers = [int(p.stem) for p in files]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f'Duplicate image frame numbers in {folder}')
    if any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
        raise ValueError(f'Missing image frame numbers in {folder}; inspect before making videos')
    return files


def verify_video(path, expected_count, size, fps):
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot decode existing/output video: {path}')
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isclose(actual_fps, fps, rel_tol=1e-4, abs_tol=1e-3):
            raise ValueError(f'{path}: frame rate {actual_fps} differs from requested {fps}')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                raise ValueError(f'{path}: unexpected frame dimensions')
            count += 1
    finally:
        capture.release()
    if count != expected_count:
        raise ValueError(f'{path}: decoded {count} frames, expected {expected_count}')


def resolve_ffmpeg(binary=None):
    resolved = shutil.which(str(binary or 'ffmpeg'))
    if resolved is None:
        raise FileNotFoundError('H.264 requires FFmpeg on PATH or --ffmpeg "path/to/ffmpeg.exe"')
    return resolved


def write_h264(images, first, path, fps, binary, crf, preset):
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise ValueError('H.264 yuv420p requires even image dimensions; no resizing is applied')
    command = [binary, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', f'{width}x{height}',
               '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', preset, '-crf', str(crf), '-threads', '2',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    # A file for stderr avoids deadlocks if FFmpeg reports more than a pipe buffer.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors,
                                   stdout=subprocess.DEVNULL)
        try:
            for n, image in enumerate(images):
                frame = first if n == 0 else cv2.imread(str(image), cv2.IMREAD_COLOR)
                if frame is None or frame.shape != first.shape:
                    raise ValueError(f'Unreadable image or inconsistent dimensions: {image}')
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            status = process.wait()
            if status != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace'))
        except BaseException as error:
            if process.poll() is None:
                process.terminate()
            process.wait()
            process.stdin.close()
            if isinstance(error, BrokenPipeError):
                errors.seek(0)
                raise RuntimeError('FFmpeg failed: ' + errors.read().decode(errors='replace')) from error
            raise


def create_trial_video(trial_folder, fps, output_format='both', encoder='mp4v',
                       skip_existing=False, output_dir=None, ffmpeg=None,
                       crf=18, preset='veryfast'):
    folder = Path(trial_folder)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Frame rate must be positive and finite')
    images = image_files(folder)
    if output_format not in ('mp4', 'both') or encoder not in ('mp4v', 'h264'):
        raise ValueError('Unsupported video format or encoder')
    if not 0 <= crf <= 51:
        raise ValueError('CRF must be in 0..51')
    destination = folder.parent if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    avi = destination / (folder.name + '.avi')
    mp4 = destination / (folder.name + 'video.mp4')
    outputs = [avi, mp4] if output_format == 'both' else [mp4]
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f'Cannot decode image: {images[0]}')
    height, width = first.shape[:2]
    pending = []
    for path in outputs:
        if path.exists():
            if not skip_existing:
                raise FileExistsError(f'Refusing to overwrite existing video: {path}')
            verify_video(path, len(images), (width, height), fps)
            print(f'Skipped verified video: {path.name}', flush=True)
        else:
            pending.append(path)
    if not pending:
        return mp4
    binary = resolve_ffmpeg(ffmpeg) if encoder == 'h264' and mp4 in pending else None
    # Publish only validated outputs. Failed attempts leave no final-name partials.
    with tempfile.TemporaryDirectory(prefix=f'.trial{folder.name}_', dir=destination) as tmp:
        staged = {path: Path(tmp) / path.name for path in pending}
        writers = []
        try:
            for path in pending:
                if path == mp4 and encoder == 'h264':
                    continue
                codec = 'FFV1' if path == avi else 'mp4v'
                writer = cv2.VideoWriter(str(staged[path]), cv2.VideoWriter_fourcc(*codec),
                                         fps, (width, height))
                writers.append(writer)
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create {path} with codec {codec}')
            if writers:
                for n, image in enumerate(images):
                    frame = first if n == 0 else cv2.imread(str(image), cv2.IMREAD_COLOR)
                    if frame is None or frame.shape != first.shape:
                        raise ValueError(f'Unreadable image or inconsistent dimensions: {image}')
                    for writer in writers:
                        writer.write(frame)
        finally:
            for writer in writers:
                writer.release()
        if binary is not None:
            write_h264(images, first, staged[mp4], fps, binary, crf, preset)
        for path in pending:
            verify_video(staged[path], len(images), (width, height), fps)
        for path in pending:
            staged[path].rename(path)
    print(f'Created {", ".join(path.name for path in pending)}: {len(images)} frames at {fps:g} fps', flush=True)
    return mp4


def create_session_videos(session_dir, fps, trial=None, workers=1, **options):
    session = Path(session_dir)
    if not session.is_dir():
        raise NotADirectoryError(session)
    folders = sorted((p for p in session.iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda p: int(p.name))
    if trial is not None:
        folders = [p for p in folders if int(p.name) == trial]
    if not folders:
        raise ValueError('No matching numbered trial folders')
    if workers < 1:
        raise ValueError('workers must be positive')
    def convert(folder):
        print(f'Trial {folder.name}: starting', flush=True)
        return create_trial_video(folder, fps, **options)
    if workers == 1:
        return [convert(folder) for folder in folders]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(convert, folders))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', required=True)
    parser.add_argument('--fps', type=float, required=True, help='Actual acquisition frame rate')
    parser.add_argument('--trial', type=int, help='One trial ID; omit to create all trials')
    parser.add_argument('--format', choices=('mp4', 'both'), default='mp4', dest='output_format',
                        help='Default: MP4 only. Use both for an additional lossless AVI.')
    parser.add_argument('--encoder', choices=('mp4v', 'h264'), default='mp4v')
    parser.add_argument('--skip-existing', action='store_true', help='Validate and skip existing requested outputs')
    parser.add_argument('--output-dir', help='Separate destination; default is the session folder')
    parser.add_argument('--ffmpeg', help='FFmpeg executable; otherwise search PATH')
    parser.add_argument('--crf', type=int, default=18, help='H.264 quality: lower is higher quality; default 18')
    parser.add_argument('--preset', choices=('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow'),
                        default='veryfast', help='H.264 encoding speed/compression tradeoff')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent trials; start with 2 on an SSD')
    args = parser.parse_args()
    create_session_videos(args.session_dir, args.fps, args.trial, workers=args.workers,
                          output_format=args.output_format, encoder=args.encoder,
                          skip_existing=args.skip_existing, output_dir=args.output_dir,
                          ffmpeg=args.ffmpeg, crf=args.crf, preset=args.preset)


if __name__ == '__main__':
    main()
