import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from run_head_tracking import find_videos, run_head_tracking


class HeadTrackingTests(unittest.TestCase):
    def test_session_discovery_excludes_side_movies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('2video.mp4', '1video.mp4', 'Mask2L.avi', 'Mirror2R.avi'):
                (root / name).touch()
            self.assertEqual([p.name for p in find_videos(session_dir=root)],
                             ['1video.mp4', '2video.mp4'])

    def test_original_dlc_analysis_then_filter_with_explicit_shuffle(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.yaml'
            config.touch()
            video = Path(tmp) / '2video.mp4'
            dlc = Mock()
            with patch.dict(sys.modules, {'deeplabcut': dlc}):
                run_head_tracking(config, [video], 3)
            self.assertEqual([c[0] for c in dlc.mock_calls],
                             ['analyze_videos', 'filterpredictions'])
            dlc.analyze_videos.assert_called_once_with(
                str(config), [str(video)], shuffle=3, save_as_csv=True)
            dlc.filterpredictions.assert_called_once_with(
                str(config), [str(video)], shuffle=3, save_as_csv=True)
