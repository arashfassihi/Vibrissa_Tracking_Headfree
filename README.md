# Vibrissa Tracking Headfree

Head-stabilized whisker tracking and touch analysis, based on William's
[DeepWhiskerCuts](https://github.com/madwilliam/DeepWhiskerCuts), with updated
video preprocessing and invertible coordinate mapping.

## First: create videos if the session contains only trial images

For numbered trial folders containing numerically named PNG/BMP/JPEG/TIFF images:

```powershell
python run_create_trial_videos.py --session-dir "path\session" --fps 40 --trial 0
```

The default now creates only `0video.mp4` (mp4v) next to folder `0`, using numeric
image order. Add `--format both` to also create a lossless FFV1 AVI. Set `--fps`
to the acquisition rate; 40 matches the legacy code. Omit `--trial` for all trials.

For compact, high-quality H.264 videos of the whole session:

```powershell
python run_create_trial_videos.py --session-dir "path\session" --fps 40 --encoder h264 --crf 18 --preset veryfast --workers 2 --skip-existing
```

FFmpeg with `libx264` must be on PATH, or pass `--ffmpeg "C:\ffmpeg\bin\ffmpeg.exe"`.
The mp4v option needs only OpenCV; H.264 offers explicit quality control. Both
keep all frames at the original resolution and requested frame rate. H.264 uses
yuv420p for even dimensions and yuv444p when either dimension is odd, preserving
the exact image size without padding, cropping, or resizing. Some Windows players
may not play 4:4:4 H.264; the command validates decoding through OpenCV, as used
by the tracking pipeline. CRF 18 is a starting point for good
quality; lower values such as 15 preserve more detail at the cost of larger files.
Inspect fine whisker detail before adopting lossy compression for analysis.
The original images remain unchanged. `veryfast` controls encoding speed and
compression efficiency; it does not change playback speed. Start with two workers
on an SSD; extra concurrency can slow down a busy or mechanical disk.

`--skip-existing` checks decodable frame counts, dimensions, and frame rate before
leaving each existing requested output unchanged. It does not check the source
image contents or re-encode an old mp4v file to H.264. To encode all trials with a
consistent new quality setting, add `--output-dir "path\session\training_videos_h264"`;
use this folder when adding videos to DLC. No existing AVI files are deleted.

The command rejects missing/duplicate frame numbers and inconsistent image sizes.
Outputs are staged in a temporary directory and published only after all output
frames decode correctly. Normal exceptions and Ctrl+C clean up staging files;
a forcibly killed process may leave a staging directory. Non-image files and
nonnumeric subfolders are ignored. Conversion waits for completion without the
old machine-specific ffmpeg settings. Full decode validation adds a verification
pass, including for skipped videos.

To run the optional real-H.264 integration test, set `VIBRISSA_TEST_FFMPEG` to your
FFmpeg executable before running the test suite.

## Browse frame previews

```powershell
python run_video_previews.py --video-dir "path\session\training_videos_h264" --frame 400
```

This saves a JPEG per MP4 into `previews_frame400`, with the filename and frame
number underneath the image. `--frame` is one-based: 400 means video index 399.
Videos with fewer frames are reported and skipped. Videos are never modified,
and existing preview JPEGs are not overwritten. Browse the JPEG folder using
Explorer's large icons; labels make it possible to select the corresponding
video for DLC. Images are decoded sequentially up to the requested frame.

## Next: run head tracking with DeepLabCut

The original head analysis/filtering step is available as an independent entry
point. Activate your existing working DeepLabCut environment first; NumPy and
OpenCV alone do not provide DLC or the trained model weights.

```powershell
python run_head_tracking.py --config "path\Topview3435-Arash-2021-07-28\config.yaml" --video "path\2video.mp4" --shuffle 3
```

Replace the paths and verify the shuffle number. The original repository settings
specify shuffle 1, whereas the previously inspected session-15 CSV filenames use
shuffle 3. Use the intended trained head model, rather than assuming either is
correct for every computer. Model files and weights must be accessible there.
For a full session replace `--video ...` with `--session-dir "path\session"`;
this selects original `*video.mp4` files only. Outputs follow DLC's usual naming
and destination behavior. Use the generated `*_filtered.csv` for preprocessing.

This command reuses the original `analyze_videos` then `filterpredictions` calls,
without importing the legacy pipeline's side-model settings or running splitting.
Automated tests mock DLC to check dispatch; actual inference requires validation
in the computation computer's DLC environment.

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
