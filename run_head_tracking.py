"""Run the original DeepWhiskerCuts head DLC analysis and filtering step.

Run in an existing DeepLabCut environment with the trained top-view model.
No video splitting or side-whisker model is run by this entry point.
"""
import argparse
from pathlib import Path


def find_videos(video=None, session_dir=None):
    if video is not None:
        path = Path(video)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        return [path]
    folder = Path(session_dir)
    if not folder.is_dir():
        raise NotADirectoryError(f"Session folder not found: {folder}")
    # Match the original get_top_videos convention, excluding side movies.
    videos = sorted(folder.glob('*video.mp4'))
    if not videos:
        raise FileNotFoundError(f"No *video.mp4 originals found in {folder}")
    return videos


def run_head_tracking(config, videos, shuffle):
    config = Path(config)
    if not config.is_file():
        raise FileNotFoundError(f"Model config not found: {config}")
    import deeplabcut

    for video in videos:
        print(f'Head DLC: {video}', flush=True)
        deeplabcut.analyze_videos(str(config), [str(video)],
                                 shuffle=shuffle, save_as_csv=True)
        deeplabcut.filterpredictions(str(config), [str(video)],
                                    shuffle=shuffle, save_as_csv=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Trained head-model config.yaml')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--video', help='One original video')
    inputs.add_argument('--session-dir', help='Folder containing original *video.mp4 files')
    parser.add_argument('--shuffle', type=int, required=True,
                        help='Trained head-model shuffle number; verify the model used previously')
    args = parser.parse_args()
    if args.shuffle < 1:
        parser.error('--shuffle must be positive')
    videos = find_videos(args.video, args.session_dir)
    run_head_tracking(args.config, videos, args.shuffle)


if __name__ == '__main__':
    main()
