"""Select visually diverse original frames across numbered trial folders.

Creates an UNLABELED image set. Appearance diversity is a proxy for pose coverage,
not a verified head-position measurement. Human labels are required for training.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import shutil
from datetime import datetime

import cv2
import numpy as np


def collect_candidates(session, per_trial):
    if per_trial < 1:
        raise ValueError('Candidates per trial must be positive')
    folders = sorted((p for p in Path(session).iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda p: int(p.name))
    records, thumbnails = [], []
    for folder in folders:
        images = [p for p in folder.iterdir() if p.is_file() and p.stem.isdigit()
                  and p.suffix.lower() in {'.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'}]
        images.sort(key=lambda p: int(p.stem))
        if not images:
            print(f'Skipped trial {folder.name}: no numbered images', flush=True)
            continue
        if len({int(p.stem) for p in images}) != len(images):
            raise ValueError(f'Duplicate source frame numbers in {folder}')
        indices = np.unique(np.linspace(0, len(images)-1, min(per_trial, len(images))).round().astype(int))
        for index in indices:
            image = cv2.imread(str(images[index]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode source image: {images[index]}')
            height, width = image.shape[:2]
            thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (48, 24),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
            # Near-uniform exposures carry little useful pose information.
            if thumb.std() < 2:
                print(f'Skipped near-uniform image: {images[index]}', flush=True)
                continue
            records.append({'trial': int(folder.name), 'source_frame_number': int(images[index].stem),
                            'source_index0': int(index), 'source_path': str(images[index].resolve()),
                            'width': width, 'height': height})
            thumbnails.append(thumb)
        print(f'Trial {folder.name}: sampled candidates', flush=True)
    if not records:
        raise ValueError('No usable original-image candidates found')
    return records, np.stack(thumbnails)


def appearance_features(thumbnails):
    # Mean centering reduces exposure differences. Background-relative differences
    # emphasize changes in animal appearance/location in a fixed camera scene.
    centered = thumbnails - thumbnails.mean(axis=(1, 2), keepdims=True)
    background = np.median(centered, axis=0)
    difference = centered - background
    gradients = np.diff(centered, axis=2, prepend=centered[:, :, :1])
    return np.concatenate([difference.reshape(len(thumbnails), -1),
                           .5 * gradients.reshape(len(thumbnails), -1)], axis=1) / 255.


def select_diverse(records, features, count, max_per_trial, min_frame_gap):
    if count < 1 or max_per_trial < 1 or min_frame_gap < 0:
        raise ValueError('Count and per-trial limit must be positive; gap must be nonnegative')
    # Deterministic greedy farthest-point sampling, constrained by trial coverage
    # and within-trial temporal spacing. No random training labels are generated.
    distances = np.sum((features - features.mean(axis=0))**2, axis=1)
    first = int(np.argmin(distances))
    nearest = np.full(len(records), np.inf)
    selected, scores, used = [], [], Counter()
    selected_positions = {}
    while len(selected) < min(count, len(records)):
        eligible = np.array([used[r['trial']] < max_per_trial and
            all(abs(r['source_index0']-position) >= min_frame_gap
                for position in selected_positions.get(r['trial'], []))
            for r in records])
        eligible[selected] = False
        if not eligible.any():
            break
        # Prefer unseen trials until they have all contributed, if count allows.
        unseen = eligible & np.array([used[r['trial']] == 0 for r in records])
        if unseen.any():
            eligible = unseen
        if not selected:
            chosen = first
            score = 0.
        else:
            chosen = int(np.argmax(np.where(eligible, nearest, -np.inf)))
            score = float(nearest[chosen])
        selected.append(chosen)
        scores.append(score)
        trial = records[chosen]['trial']
        used[trial] += 1
        selected_positions.setdefault(trial, []).append(records[chosen]['source_index0'])
        distance = np.sum((features-features[chosen])**2, axis=1)
        nearest = np.minimum(nearest, distance)
    return selected, scores


def write_manifest(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheets(selected, output, frame_dir):
    for start in range(0, len(selected), 20):
        page = selected[start:start+20]
        canvas = np.full((math.ceil(len(page)/4)*180, 4*320, 3), 245, np.uint8)
        for i, row in enumerate(page):
            image = cv2.imread(str(frame_dir / row['selected_file']))
            height, width = image.shape[:2]
            factor = min(316/width, 146/height)
            resized = cv2.resize(image, (max(1, round(width*factor)), max(1, round(height*factor))))
            y, x = (i//4)*180, (i%4)*320
            canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
            label = f"Trial {row['trial']} | source frame {row['source_frame_number']}"
            cv2.putText(canvas, label, (x+4, y+169), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
        path = output / f'contact_sheet_{start//20+1:02d}.jpg'
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError(f'Cannot save {path}')


def install_for_labeling(config, name, frame_dir, reference_video):
    """Register one mixed-trial image folder using one real reference video.

    The reference video is only DLC's folder-registration anchor; images retain
    their true trial/frame identities in filenames and the accompanying manifest.
    Do not extract frames again from this reference or it would mix sample sets.
    """
    from deeplabcut.utils import auxiliaryfunctions
    config = Path(config).resolve()
    cfg = auxiliaryfunctions.read_config(str(config))
    project = Path(cfg['project_path']).resolve()
    if project != config.parent:
        raise ValueError('Copied config project_path must point to its local parent folder')
    if cfg.get('multianimalproject', False):
        raise ValueError('This installer supports the single-animal head project only')
    reference = Path(reference_video).resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    labels = project / 'labeled-data' / name
    video = project / 'videos' / (name+reference.suffix.lower())
    registered = [Path(p.replace('\\', '/')).stem for p in cfg.get('video_sets', {})]
    if labels.exists() or video.exists() or name in registered:
        raise FileExistsError(f'Dataset name already exists in project: {name}')
    capture = cv2.VideoCapture(str(reference))
    try:
        ok, image = capture.read()
        if not ok:
            raise ValueError('Cannot decode the reference video')
        height, width = image.shape[:2]
    finally:
        capture.release()
    shutil.copytree(frame_dir, labels)
    video.parent.mkdir(exist_ok=True)
    shutil.copy2(reference, video)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    shutil.copy2(config, str(config)+f'.before_sampling_{stamp}.bak')
    cfg.setdefault('video_sets', {})[str(video)] = {'crop': f'0, {width}, 0, {height}'}
    auxiliaryfunctions.write_config(str(config), cfg)
    print(f'UNLABELED images installed for Label Frames: {labels}', flush=True)
    print('Do not run Extract Frames for this folder. Human annotation is required.', flush=True)
    return labels


def sample_training_frames(session_dir, output_dir, count=80, candidates_per_trial=16,
                           max_per_trial=4, min_frame_gap=50, name='auto_training_samples',
                           exclude_trials=()):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Dataset name must use only letters, digits, underscores or hyphens')
    session, output = Path(session_dir), Path(output_dir)
    if not session.is_dir():
        raise NotADirectoryError(session)
    if output.exists():
        raise FileExistsError(f'Choose a fresh output folder: {output}')
    records, thumbnails = collect_candidates(session, candidates_per_trial)
    keep = np.array([r['trial'] not in set(exclude_trials) for r in records])
    records = [r for r, good in zip(records, keep) if good]
    if not records:
        raise ValueError('No candidates after excluding trials')
    thumbnails = thumbnails[keep]
    features = appearance_features(thumbnails)
    indices, scores = select_diverse(records, features, count, max_per_trial, min_frame_gap)
    output.mkdir(parents=True, exist_ok=False)
    frame_dir = output / 'selected_frames'
    frame_dir.mkdir()
    selected = []
    for rank, (index, score) in enumerate(zip(indices, scores), 1):
        row = records[index].copy()
        row.update(selection_rank=rank, appearance_novelty_score=score,
                   selected_file=f"img_trial{row['trial']:05d}_frame{row['source_frame_number']:07d}.png")
        image = cv2.imread(row['source_path'], cv2.IMREAD_COLOR)
        if image is None or not cv2.imwrite(str(frame_dir / row['selected_file']), image):
            raise RuntimeError(f"Cannot export {row['source_path']}")
        selected.append(row)
    write_manifest(output / 'selected_manifest.csv', selected)
    write_manifest(output / 'candidate_manifest.csv', records)
    write_contact_sheets(selected, output, frame_dir)
    settings = dict(session=str(session.resolve()), requested_count=count, selected_count=len(selected),
                    candidate_count=len(records), selected_trial_count=len({r['trial'] for r in selected}),
                    candidates_per_trial=candidates_per_trial, max_per_trial=max_per_trial,
                    min_frame_gap=min_frame_gap, name=name, exclude_trials=list(exclude_trials),
                    method='mean-centered low-resolution appearance plus gradients; greedy diversity with trial constraints',
                    status='UNLABELED; not a finished training dataset or verified pose coverage',
                    source_frame_number='numeric original image filename; source_index0 is sorted file position',
                    near_uniform_threshold_std=2, descriptor_size=[48, 24])
    (output / 'sampling_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print(json.dumps(settings, indent=2), flush=True)
    if len(selected) < count:
        print('Fewer frames available under the spacing/per-trial constraints; no constraints were silently relaxed.', flush=True)
    return selected, frame_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--name', default='auto_training_samples')
    parser.add_argument('--count', type=int, default=80)
    parser.add_argument('--candidates-per-trial', type=int, default=16)
    parser.add_argument('--max-per-trial', type=int, default=4)
    parser.add_argument('--min-frame-gap', type=int, default=50)
    parser.add_argument('--exclude-trials', type=int, nargs='*', default=[])
    parser.add_argument('--dlc-config', help='Optional: install into the copied DLC project for manual labeling')
    parser.add_argument('--reference-video', help='One readable video to register the combined labeling folder')
    args = parser.parse_args()
    if bool(args.dlc_config) != bool(args.reference_video):
        parser.error('--dlc-config and --reference-video must be supplied together')
    _, frames = sample_training_frames(args.session_dir, args.output_dir, args.count,
        args.candidates_per_trial, args.max_per_trial, args.min_frame_gap, args.name, args.exclude_trials)
    if args.dlc_config:
        install_for_labeling(args.dlc_config, args.name, frames, args.reference_video)


if __name__ == '__main__':
    main()
