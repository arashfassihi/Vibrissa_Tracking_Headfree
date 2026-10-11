import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from run_create_trial_videos import create_trial_video, image_files


class TrialVideoTests(unittest.TestCase):
    def test_numeric_order_and_decodable_videos_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / '0'
            trial.mkdir()
            for n in (10, 8, 9):
                cv2.imwrite(str(trial / f'{n}.png'), np.full((64, 80, 3), n * 10, np.uint8))
            (trial / 'notes.txt').write_text('ignore this file')
            self.assertEqual([p.stem for p in image_files(trial)], ['8', '9', '10'])
            output = create_trial_video(trial, 40.)
            capture = cv2.VideoCapture(str(output))
            levels = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    levels.append(frame.mean())
            finally:
                capture.release()
            self.assertEqual(len(levels), 3)
            self.assertTrue(levels[0] < levels[1] < levels[2])
            with self.assertRaises(FileExistsError):
                create_trial_video(trial, 40.)

    def test_missing_frames_are_not_silently_compacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in (0, 2):
                (Path(tmp) / f'{n}.png').touch()
            with self.assertRaisesRegex(ValueError, 'Missing image frame'):
                image_files(tmp)
