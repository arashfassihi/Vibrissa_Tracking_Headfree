import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from run_video_previews import create_previews


class PreviewTests(unittest.TestCase):
    def test_exact_one_based_frame_and_short_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, levels in (('0video', [30, 80, 160]), ('1video', [20])):
                writer = cv2.VideoWriter(str(root / f'{name}.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), 40., (80, 64))
                self.assertTrue(writer.isOpened())
                for level in levels:
                    writer.write(np.full((64, 80, 3), level, np.uint8))
                writer.release()
            before = (root / '0video.mp4').read_bytes()
            results = create_previews(root, frame=3)
            self.assertEqual(len(results), 1)
            image = cv2.imread(str(results[0]))
            self.assertEqual(image.shape[:2], (106, 80))
            self.assertLess(abs(image[:60].mean() - 160), 10)
            self.assertEqual((root / '0video.mp4').read_bytes(), before)
            with self.assertRaises(FileExistsError):
                create_previews(root, frame=3)
