import tempfile
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from run_create_trial_videos import create_trial_video, create_session_videos, image_files


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

    def test_mp4_only_batch_resume_and_reject_incomplete_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for trial_id in ('10', '2'):
                trial = root / trial_id
                trial.mkdir()
                for n in range(3):
                    cv2.imwrite(str(trial / f'{n}.png'), np.full((64, 80, 3), n * 40, np.uint8))
            (root / 'preview_trial0_v2').mkdir()
            videos = create_session_videos(root, 40., workers=2, output_format='mp4')
            self.assertEqual([p.name for p in videos], ['2video.mp4', '10video.mp4'])
            self.assertEqual(list(root.glob('*.avi')), [])
            before = [p.read_bytes() for p in videos]
            create_session_videos(root, 40., output_format='mp4', skip_existing=True)
            self.assertEqual([p.read_bytes() for p in videos], before)
            videos[0].write_bytes(b'incomplete')
            with self.assertRaises(ValueError):
                create_trial_video(root / '2', 40., output_format='mp4', skip_existing=True)

    @unittest.skipUnless(os.environ.get('VIBRISSA_TEST_FFMPEG'), 'Set VIBRISSA_TEST_FFMPEG for H.264 integration')
    def test_h264_odd_dimensions_are_preserved(self):
        for height, width in ((65, 80), (64, 81), (65, 81)):
            with self.subTest(height=height, width=width), tempfile.TemporaryDirectory() as tmp:
                trial = Path(tmp) / '0'
                trial.mkdir()
                for n in range(2):
                    frame = np.full((height, width, 3), 80 + n * 50, np.uint8)
                    cv2.imwrite(str(trial / f'{n}.png'), frame)
                video = create_trial_video(trial, 40., output_format='mp4', encoder='h264',
                                           ffmpeg=os.environ['VIBRISSA_TEST_FFMPEG'])
                capture = cv2.VideoCapture(str(video))
                try:
                    for n in range(2):
                        ok, frame = capture.read()
                        self.assertTrue(ok)
                        self.assertEqual(frame.shape, (height, width, 3))
                    self.assertFalse(capture.read()[0])
                finally:
                    capture.release()

    @unittest.skipUnless(os.environ.get('VIBRISSA_TEST_FFMPEG'), 'Set VIBRISSA_TEST_FFMPEG for H.264 integration')
    def test_real_h264_preserves_frames_and_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / '0'
            trial.mkdir()
            for n in range(4):
                frame = np.full((64, 80, 3), 40 + n * 30, np.uint8)
                cv2.line(frame, (5, 10), (70, 50), (255, 255, 255), 1)
                cv2.imwrite(str(trial / f'{n}.png'), frame)
            video = create_trial_video(trial, 40., output_format='mp4', encoder='h264',
                                       ffmpeg=os.environ['VIBRISSA_TEST_FFMPEG'])
            capture = cv2.VideoCapture(str(video))
            levels = []
            try:
                for n in range(4):
                    ok, frame = capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape, (64, 80, 3))
                    levels.append(frame.mean())
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()
            self.assertTrue(all(b > a for a, b in zip(levels, levels[1:])))
            self.assertFalse((root / '0.avi').exists())
            self.assertEqual(list(root.glob('.trial0_*')), [])
