import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from predict import build_arg_parser, resolve_model_name

class ModelSelectionTests(unittest.TestCase):
    def test_recorded_model_must_be_a_supported_name(self):
        with tempfile.TemporaryDirectory() as directory, patch('predict.MODELS_DIR', directory):
            marker = Path(directory) / 'best_model.txt'
            for value in ('../outside', '/tmp/arbitrary', '', 'unknown'):
                marker.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    resolve_model_name(build_arg_parser().parse_args([]))

    def test_supported_recorded_and_explicit_models_work(self):
        with tempfile.TemporaryDirectory() as directory, patch('predict.MODELS_DIR', directory):
            self.assertEqual(resolve_model_name(build_arg_parser().parse_args([])), 'random_forest')
            (Path(directory) / 'best_model.txt').write_text(' logistic_regression\n')
            self.assertEqual(resolve_model_name(build_arg_parser().parse_args([])), 'logistic_regression')
            self.assertEqual(resolve_model_name(build_arg_parser().parse_args(['--model', 'random_forest'])), 'random_forest')
