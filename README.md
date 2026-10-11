# Vibrissa Tracking Headfree

Head-stabilized whisker tracking and touch analysis, based on William's
[DeepWhiskerCuts](https://github.com/madwilliam/DeepWhiskerCuts), with updated
video preprocessing and invertible coordinate mapping.

## Updated top-view preprocessing

The new command generates head-stabilized left/right videos and saves the exact
forward/inverse coordinate transforms for every frame. It requires NumPy and
OpenCV and runs independently of the DeepLabCut environment.

From the repository root:

```powershell
python run_preprocess_top_view.py --video "path\2video.mp4" --dlc "path\2videoDLC_resnet50_Topview3435Jul28shuffle3_500000_filtered.csv" --output-dir "path\new_trial2" --stem 2 --max-frames 100
```

Remove `--max-frames 100` for the full trial. The output directory must be new.
Outputs are `Mask2L.avi`, `Mirror2R.avi`, `2_frame_transforms.csv`, and
`2_preprocessing.json`. Videos use the source frame rate and lossless FFV1 AVI;
check that your DLC/Janelia environment can decode this codec before a full run.

This command replaces the legacy splitting step for new processing. The old
`run_top_view.py` and pipeline entry points still call the legacy splitter; they
do not automatically select this command. After preprocessing, run your whisker
DLC/Janelia tracking on the new side videos in the new output directory.

### Geometry and frame mapping

- Read each original frame once and make one affine warp per side.
- Combine rotation, cropping, and right-side reflection in the same matrix used
  for point mapping. For a 315-pixel-wide image, reflection is `x = 314 - x`.
- Use the same smoothed landmarks for orientation and translation.
- Preserve all source frames. Invalid head poses produce black frames and NaN
  transforms, and the MATLAB helper rejects them. Short interior pose gaps are
  interpolated and flagged. Never treat a black frame as a usable observation.
- Frame IDs and native point coordinates are zero-based. No frame compaction is
  performed: `processed_frame0 == source_frame0`.
- Metadata is real CSV, rather than CSV with an `.xlsx` extension.

The legacy splitter reads an image before its frame loop without advancing
inside the loop, and its NaN quality flags are unsafe as a truth-value selector.
The new command avoids both issues. Rotation/cropping and contrast are not
pixel-identical to legacy outputs. **Rerun tracking on these new videos. Never
use the new matrices to map curves detected on old side videos.**

Landmarks default to `Nose`/`Snout`. See `--help` for smoothing, likelihood, gap,
contrast, and landmark-name options. JSON records settings and input paths.

### Map a detected whisker back in MATLAB

```matlab
addpath('path/to/DeepWhiskerCuts/matlab');
T = readtable('path/to/new_trial2/2_frame_transforms.csv');
[xy, sourceFrame0] = map_whisker_coordinates( ...
    [whiskerX(:), whiskerY(:)], T, processedFrame0, 'R', 'to_original', 0, 1);
% Read original sourceFrame0 with your zero-based video reader.
imshow(originalFrame); hold on; plot(xy(:,1), xy(:,2), 'r-');
```

The last two arguments specify the input and output pixel base. Use output
base 1 for MATLAB `imshow` overlays, or 0 for native OpenCV coordinates.
Use `'L'` for left-side traces. Pass a loaded table when mapping many curves.

### Validation

```powershell
python -m unittest discover -s tests -v
```

Tests cover known landmark positions, reflection, image/coordinate agreement
across rotations, inverse mapping, pose gaps, advancing video frames, and frame
preservation. The isolated implementation was also previewed on the first 100
frames of session 15, trial 2, and its MATLAB mapping helper was checked with
forward/inverse round trips. Legacy overlay reconstruction still requires the
actual historical preprocessing and inverse-transform script.

### Update a computation computer

After publication, clone this repository on a computation computer:

```powershell
git clone https://github.com/arashfassihi/Vibrissa_Tracking_Headfree.git
cd Vibrissa_Tracking_Headfree
```

For later updates from inside that folder:

```powershell
git pull --ff-only
```

Check `git status` and preserve any local configuration or work before updating.
Select the Python environment available on each computer; local VS Code settings
and validation videos are excluded from new commits.
