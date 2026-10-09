"""Head-stabilized side videos with explicit, invertible pixel coordinates.

New outputs only. All geometry uses zero-based (x,y) pixel centres. The AVI
frame index always equals the original source frame index, including bad poses.
Requires numpy and OpenCV; no pandas, Pillow or DLC installation is needed.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 315, 700
NOSE_TARGET = np.array([385., 350.])
SIDE_START = {"L": 80., "R": 375.}


def map_points(points, matrix, input_base=0, output_base=0):
    points = np.asarray(points, dtype=float).reshape(-1, 2) - input_base
    hom = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return hom[:, :2] / hom[:, 2:] + output_base


def frame_transforms(nose, snout):
    """Original -> sides, preserving the historical layout and head direction.

    Nose-to-snout points up in the stabilized image. Matrices map POINTS
    forward; warpAffine receives those same forward matrices without
    WARP_INVERSE_MAP. No expanded-rotation canvas needs reconstructing.
    """
    nose, snout = np.asarray(nose, float), np.asarray(snout, float)
    if not np.isfinite([nose, snout]).all() or np.linalg.norm(snout-nose) < 1:
        raise ValueError("Invalid head landmarks")
    angle = np.degrees(np.arctan2(*(snout-nose)[::-1])) + 90.
    rotation = np.eye(3)
    rotation[:2] = cv2.getRotationMatrix2D(tuple(nose), float(angle), 1.)
    rotation[:2, 2] += NOSE_TARGET - nose
    result = {}
    for side in ("L", "R"):
        crop = np.eye(3); crop[0, 2] = -SIDE_START[side]
        matrix = crop @ rotation
        if side == "R":
            # A 315-pixel image spans x=0..314, not x=0..315.
            reflection = np.array([[-1., 0., WIDTH-1.], [0., 1., 0.], [0., 0., 1.]])
            matrix = reflection @ matrix
        result[side] = matrix
    return result, angle


def read_dlc(path, nose_name="Nose", snout_name="Snout"):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader); bodyparts = next(reader); coordinates = next(reader)
        columns = {(b.lower(), c.lower()): i for i, (b, c) in enumerate(zip(bodyparts, coordinates))}
        wanted = [(name.lower(), coordinate) for name in (nose_name, snout_name)
                  for coordinate in ("x", "y", "likelihood")]
        missing = [key for key in wanted if key not in columns]
        if missing:
            raise ValueError(f"Missing DLC columns: {missing}")
        rows = []
        for row in reader:
            if not row: continue
            frame = int(row[0])
            if frame != len(rows):
                raise ValueError("DLC rows must have consecutive zero-based source frame IDs")
            rows.append([float(row[columns[key]]) for key in wanted])
    return np.asarray(rows, float).reshape(-1, 6)


def prepare_poses(raw, min_likelihood=.7, max_gap=5, smoothing_radius=5):
    xy = raw[:, [0, 1, 3, 4]].copy()
    distances = np.linalg.norm(xy[:, :2]-xy[:, 2:], axis=1)
    good = (np.isfinite(raw).all(axis=1) & (raw[:, 2] >= min_likelihood)
            & (raw[:, 5] >= min_likelihood) & (distances >= 5) & (distances <= 200))
    xy[~good] = np.nan
    interpolated = np.zeros(len(raw), bool)
    # Fill only short INTERIOR gaps; long gaps and missing ends stay invalid.
    i = 0
    while i < len(raw):
        if good[i]: i += 1; continue
        end = i
        while end < len(raw) and not good[end]: end += 1
        if i > 0 and end < len(raw) and end-i <= max_gap:
            weights = np.arange(1, end-i+1)/(end-i+1)
            xy[i:end] = xy[i-1] + weights[:, None]*(xy[end]-xy[i-1])
            interpolated[i:end] = True
        i = end
    valid = good | interpolated
    # Smooth only within contiguous valid runs; never bridge a long bad gap.
    smoothed = xy.copy(); i = 0
    while i < len(raw):
        if not valid[i]: i += 1; continue
        end = i
        while end < len(raw) and valid[end]: end += 1
        run = xy[i:end]
        cumulative = np.vstack([np.zeros((1, 4)), np.cumsum(run, axis=0)])
        indices = np.arange(len(run));lo = np.maximum(0, indices-smoothing_radius)
        hi = np.minimum(len(run), indices+smoothing_radius+1)
        smoothed[i:end] = (cumulative[hi]-cumulative[lo])/(hi-lo)[:, None]
        i = end
    return smoothed, good, interpolated, valid


def side_mask(sigma=60.):
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    return (1.-np.exp(-4*np.log(2)*((xx-WIDTH)**2+(yy-HEIGHT/2)**2)/sigma**2)).astype(np.float32)


def transform_image(image, matrix, mask=None, contrast=1.05):
    output = cv2.warpAffine(image, matrix[:2], (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if mask is not None: output = np.clip(output.astype(float)*mask[..., None], 0, 255).astype(np.uint8)
    if contrast != 1:
        mean = float(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).mean())
        output = np.clip(mean + contrast*(output.astype(float)-mean), 0, 255).astype(np.uint8)
    return output


def preprocess(video_path, dlc_path, output_dir, stem=None, max_frames=None,
               min_likelihood=.7, max_gap=5, smoothing_radius=5, contrast=1.05,
               nose_name="Nose", snout_name="Snout"):
    video_path, dlc_path, output_dir = Path(video_path), Path(dlc_path), Path(output_dir)
    # Require a fresh directory: do not mix regenerated geometry with old traces.
    output_dir.mkdir(parents=True, exist_ok=False)
    raw = read_dlc(dlc_path, nose_name, snout_name)
    poses, good, interpolated, valid = prepare_poses(raw, min_likelihood, max_gap, smoothing_radius)
    stem = video_path.stem if stem is None else stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0: raise ValueError("Video has no valid frame rate")
    writers = {}; records = []; count = 0; mask = side_mask()
    paths = {"L": output_dir/f"Mask{stem}L.avi", "R": output_dir/f"Mirror{stem}R.avi"}
    try:
        for side in ("L", "R"):
            # Lossless output keeps faint whisker intensities intact.
            writers[side] = cv2.VideoWriter(str(paths[side]), cv2.VideoWriter_fourcc(*"FFV1"), fps, (WIDTH, HEIGHT))
            if not writers[side].isOpened(): raise RuntimeError("FFV1 AVI writer is unavailable")
        while max_frames is None or count < max_frames:
            ok, frame = cap.read()  # Always advance the ORIGINAL video.
            if not ok: break
            if count >= len(raw): raise ValueError("Video has more frames than DLC; alignment is unsafe")
            matrices = {side: np.full((3, 3), np.nan) for side in ("L", "R")}
            angle = np.nan
            if valid[count]: matrices, angle = frame_transforms(poses[count, :2], poses[count, 2:])
            record = {"source_frame0": count, "processed_frame0": count, "source_time_s": count/fps,
                      "pose_valid": int(valid[count]), "raw_pose_good": int(good[count]),
                      "pose_interpolated": int(interpolated[count]), "rotation_deg": angle}
            for name, value in zip(("nose_x0", "nose_y0", "snout_x0", "snout_y0"), poses[count]): record[name] = value
            for side in ("L", "R"):
                matrix = matrices[side]
                inverse = np.linalg.inv(matrix) if valid[count] else matrix.copy()
                for prefix, transform in ((side, matrix), (side+"inv", inverse)):
                    for row in range(3):
                        for column in range(3): record[f"{prefix}{row}{column}"] = transform[row, column]
                image = transform_image(frame, matrix, mask, contrast) if valid[count] else np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                writers[side].write(image)
            records.append(record);count += 1
        if not records: raise ValueError("No source frames decoded")
        if max_frames is None and count != len(raw):
            raise ValueError(f"Video/DLC frame count mismatch: {count} versus {len(raw)}")
    finally:
        cap.release()
        for writer in writers.values(): writer.release()
    metadata_path = output_dir/f"{stem}_frame_transforms.csv"
    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    config = {"source_video": str(video_path.resolve()), "source_dlc": str(dlc_path.resolve()),
              "frames_written": count, "fps": fps, "pixel_coordinates": "zero-based (x,y) pixel centres",
              "frame_mapping": "processed_frame0 == source_frame0; invalid poses yield black frames and NaN matrices",
              "width": WIDTH, "height": HEIGHT, "nose_target_before_side_crop": NOSE_TARGET.tolist(),
              "side_crop_x0": SIDE_START, "right_reflection": "x_out = width-1-x_in",
              "min_likelihood": min_likelihood, "max_interpolated_gap": max_gap,
              "smoothing_radius": smoothing_radius, "contrast": contrast,
              "valid_pose_frames": int(valid[:count].sum()), "raw_good_frames": int(good[:count].sum()),
              "interpolated_frames": int(interpolated[:count].sum()),
              "compatibility": "NEW geometry: rerun DLC/Janelia on these outputs; do not reuse legacy side traces"}
    (output_dir/f"{stem}_preprocessing.json").write_text(json.dumps(config, indent=2))
    return config, metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True);parser.add_argument("--dlc", required=True)
    parser.add_argument("--output-dir", required=True);parser.add_argument("--stem")
    parser.add_argument("--max-frames", type=int);parser.add_argument("--min-likelihood", type=float, default=.7)
    parser.add_argument("--max-gap", type=int, default=5);parser.add_argument("--smoothing-radius", type=int, default=5)
    parser.add_argument("--contrast", type=float, default=1.05)
    parser.add_argument("--nose-name", default="Nose");parser.add_argument("--snout-name", default="Snout")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1: parser.error("max-frames must be positive")
    if args.max_gap < 0 or args.smoothing_radius < 0: parser.error("gap and smoothing radius must be nonnegative")
    config, metadata = preprocess(args.video, args.dlc, args.output_dir, args.stem, args.max_frames,
                                  args.min_likelihood, args.max_gap, args.smoothing_radius,
                                  args.contrast, args.nose_name, args.snout_name)
    print(json.dumps(config, indent=2));print(f"Transform table: {metadata}")


if __name__ == "__main__": main()
