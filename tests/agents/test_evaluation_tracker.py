"""Lightweight tracking contracts without Node, SSH or optional ONNX packages."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.track_training import atomic_json, tracker_lock
from evaluation.export_model import export
from scripts.train_job import last_frames


class TestEvaluationTracker(unittest.TestCase):
    def test_atomic_metrics_and_exclusive_tracker(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metrics.json'
            atomic_json(path,{'message':'训练评估','frames':3_000_000_000})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['frames'],3_000_000_000)
            with tracker_lock(Path(folder)/'tracker.lock'):
                with self.assertRaises((RuntimeError,OSError)):
                    with tracker_lock(Path(folder)/'tracker.lock'):pass
            with tracker_lock(Path(folder)/'tracker.lock'):pass

    def test_unchanged_export_does_not_import_or_load_torch(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'model.tar';checkpoint.write_bytes(b'checkpoint fixture')
            model=Path(folder)/'model.onnx';model.write_bytes(b'onnx fixture')
            stat=checkpoint.stat()
            atomic_json(model.with_suffix('.json'),dict(frames=42,checkpoint_identity=[stat.st_mtime_ns,stat.st_size]))
            with patch('torch.load',side_effect=AssertionError('Unchanged checkpoint was loaded')):
                self.assertFalse(export(checkpoint,model,True)['changed'])

    def test_training_completion_uses_recorded_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'logs.csv'
            path.write_text('# _tick,frames,loss\n0,640,1.0\n1,3000000000,0.1\n')
            self.assertEqual(last_frames(path),3_000_000_000)
