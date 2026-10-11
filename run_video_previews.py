"""Save a numbered frame from each MP4 as a labeled JPEG for Explorer browsing."""
import argparse
from pathlib import Path
import re

import cv2


def create_previews(video_dir, frame=400):
    """frame is one-based: frame 400 has zero-based video index 399."""
    folder = Path(video_dir)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if frame < 1:
        raise ValueError('Frame must be at least 1 (one-based)')
    videos = sorted((p for p in folder.iterdir() if p.suffix.lower() == '.mp4'),
                    key=lambda p: [int(s) if s.isdigit() else s.lower()
                                   for s in re.split(r'(\d+)', p.name)])
    if not videos:
        raise ValueError('No MP4 videos found')
    output = folder / f'previews_frame{frame}'
    output.mkdir(exist_ok=True)
    results = []
    for video in videos:
        target = output / f'{video.stem}_frame{frame}.jpg'
        if target.exists():
            raise FileExistsError(f'Preview already exists: {target}')
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {video}')
            # Sequential reads avoid codec-dependent seek errors and verify that
            # exactly the requested source frame was reached.
            image = None
            for _ in range(frame):
                ok, image = capture.read()
                if not ok:
                    print(f'Skipped {video.name}: fewer than {frame} decodable frames', flush=True)
                    image = None
                    break
        finally:
            capture.release()
        if image is None:
            continue
        # Put the label below the image so it never hides the head or whiskers.
        preview = cv2.copyMakeBorder(image, 0, 42, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(preview, f'{video.name} | frame {frame}', (10, image.shape[0] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(target), preview, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f'Cannot save {target}')
        results.append(target)
        print(f'Saved {target.name}', flush=True)
    print(f'{len(results)} previews saved in {output}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir', required=True)
    parser.add_argument('--frame', type=int, default=400,
                        help='One-based frame number (default 400 = video index 399)')
    args = parser.parse_args()
    create_previews(args.video_dir, args.frame)


if __name__ == '__main__':
    main()
