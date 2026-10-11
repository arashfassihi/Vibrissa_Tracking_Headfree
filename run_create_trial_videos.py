"""Create original trial videos from numbered image folders before head DLC.

Uses OpenCV directly, without the legacy machine-specific ffmpeg settings.
Frame rate must match acquisition; the old DeepWhiskerCuts code used 40 Hz.
"""
import argparse
import math
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


def create_trial_video(trial_folder, fps):
    folder = Path(trial_folder)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Frame rate must be positive and finite')
    images = image_files(folder)
    avi = folder.parent / (folder.name + '.avi')
    mp4 = folder.parent / (folder.name + 'video.mp4')
    if avi.exists() or mp4.exists():
        raise FileExistsError(f'Refusing to overwrite existing videos: {avi}, {mp4}')
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f'Cannot decode image: {images[0]}')
    height, width = first.shape[:2]
    writers = []
    try:
        for path, codec in ((avi, 'FFV1'), (mp4, 'mp4v')):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec),
                                     fps, (width, height))
            writers.append(writer)
            if not writer.isOpened():
                raise RuntimeError(f'Cannot create {path} with codec {codec}')
        for n, image in enumerate(images):
            frame = first if n == 0 else cv2.imread(str(image), cv2.IMREAD_COLOR)
            if frame is None or frame.shape != first.shape:
                raise ValueError(f'Unreadable image or inconsistent dimensions: {image}')
            for writer in writers:
                writer.write(frame)
    finally:
        for writer in writers:
            writer.release()
    # Decode every output frame to catch failed writes before passing to DLC.
    for path in (avi, mp4):
        capture = cv2.VideoCapture(str(path))
        count = 0
        try:
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
        finally:
            capture.release()
        if count != len(images):
            raise RuntimeError(f'{path}: decoded {count} frames, expected {len(images)}')
    print(f'Created {avi.name} and {mp4.name}: {len(images)} frames at {fps:g} fps', flush=True)
    return mp4


def create_session_videos(session_dir, fps, trial=None):
    session = Path(session_dir)
    if not session.is_dir():
        raise NotADirectoryError(session)
    folders = sorted((p for p in session.iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda p: int(p.name))
    if trial is not None:
        folders = [p for p in folders if int(p.name) == trial]
    if not folders:
        raise ValueError('No matching numbered trial folders')
    return [create_trial_video(folder, fps) for folder in folders]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', required=True)
    parser.add_argument('--fps', type=float, required=True, help='Actual acquisition frame rate')
    parser.add_argument('--trial', type=int, help='One trial ID; omit to create all trials')
    args = parser.parse_args()
    create_session_videos(args.session_dir, args.fps, args.trial)


if __name__ == '__main__':
    main()
