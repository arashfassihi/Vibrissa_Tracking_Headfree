import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from run_sample_training_frames import (
    appearance_features, install_for_labeling, sample_training_frames, select_diverse,
)


class TrainingSamplingTests(unittest.TestCase):
    def test_diversity_coverage_spacing_and_repeatability(self):
        records = [dict(trial=t, source_index0=f) for t in range(3) for f in (0, 10, 100)]
        features = np.array([[t, f/100.] for t in range(3) for f in (0, 10, 100)])
        chosen, _ = select_diverse(records, features, 6, 2, 50)
        repeated, _ = select_diverse(records, features, 6, 2, 50)
        self.assertEqual(chosen, repeated)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len({records[i]['trial'] for i in chosen[:3]}), 3)
        for trial in range(3):
            positions = [records[i]['source_index0'] for i in chosen if records[i]['trial'] == trial]
            self.assertEqual(len(positions), 2)
            self.assertGreaterEqual(abs(positions[0]-positions[1]), 50)

    def test_original_images_manifest_and_holdout_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / 'session'
            session.mkdir()
            for trial_id in (0, 2, 10):
                trial = session / str(trial_id)
                trial.mkdir()
                for frame_id in (1, 2, 10, 20):
                    image = np.full((65, 81, 3), 160, np.uint8)
                    cv2.circle(image, (10+frame_id, 10+trial_id), 7, (20, 20, 20), -1)
                    cv2.imwrite(str(trial / f'{frame_id}.png'), image)
            (session / 'training_videos_h264').mkdir()
            selected, frames = sample_training_frames(session, root/'samples', count=4,
                candidates_per_trial=4, max_per_trial=2, min_frame_gap=1, exclude_trials=[10])
            self.assertEqual(len(selected), 4)
            self.assertEqual({row['trial'] for row in selected}, {0, 2})
            for row in selected:
                source = cv2.imread(row['source_path'])
                exported = cv2.imread(str(frames / row['selected_file']))
                np.testing.assert_array_equal(exported, source)
                self.assertEqual(exported.shape[:2], (65, 81))
            with (root/'samples'/'selected_manifest.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            self.assertTrue((root/'samples'/'contact_sheet_01.jpg').is_file())
            settings = json.loads((root/'samples'/'sampling_settings.json').read_text())
            self.assertIn('UNLABELED', settings['status'])
            with self.assertRaises(FileExistsError):
                sample_training_frames(session, root/'samples')

    def test_installer_registers_images_and_preserves_existing_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root/'project'
            project.mkdir()
            (project/'labeled-data'/'old').mkdir(parents=True)
            old_labels = project/'labeled-data'/'old'/'CollectedData_Arash.csv'
            old_labels.write_text('existing labels')
            config = project/'config.yaml'
            config.write_text('original config')
            frames = root/'frames'
            frames.mkdir()
            cv2.imwrite(str(frames/'img_trial0_frame10.png'), np.zeros((64, 80, 3), np.uint8))
            reference = root/'reference.mp4'
            writer = cv2.VideoWriter(str(reference), cv2.VideoWriter_fourcc(*'mp4v'), 40., (80, 64))
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((64, 80, 3), np.uint8))
            writer.release()
            cfg = dict(project_path=str(project), scorer='Arash',
                       video_sets={'old.avi': {'crop': '0, 80, 0, 64'}}, iteration=0)
            auxiliary = Mock()
            auxiliary.read_config.return_value = cfg
            utils = Mock(auxiliaryfunctions=auxiliary)
            with patch.dict(sys.modules, {'deeplabcut': Mock(), 'deeplabcut.utils': utils}):
                label_dir = install_for_labeling(config, 'new_samples', frames, reference)
                self.assertEqual(old_labels.read_text(), 'existing labels')
                self.assertTrue((label_dir/'img_trial0_frame10.png').exists())
                self.assertTrue((project/'videos'/'new_samples.mp4').exists())
                self.assertIn('old.avi', cfg['video_sets'])
                self.assertIn(str(project/'videos'/'new_samples.mp4'), cfg['video_sets'])
                self.assertEqual(cfg['iteration'], 0)
                self.assertEqual(len(list(project.glob('config.yaml.before_sampling_*.bak'))), 1)
                auxiliary.write_config.assert_called_once()
                with self.assertRaises(FileExistsError):
                    install_for_labeling(config, 'new_samples', frames, reference)

    def test_features_discount_uniform_brightness_changes(self):
        base = np.arange(48*24, dtype=np.float32).reshape(24, 48) % 100
        features = appearance_features(np.stack([base, base+30]))
        np.testing.assert_allclose(features[0], features[1], atol=1e-6)
