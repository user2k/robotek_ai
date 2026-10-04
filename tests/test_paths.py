"""Ścieżki konfiguracji i izolacja modeli między gałęziami Git."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from config import config
from curriculum import PLAN_PATH, load_plan, save_plan, default_plan
from sampling_gui import SETTINGS_PATH
from visual_agent import DEFAULT_MODEL
from gui import MODEL_PATH
from train import latest_path


class PathTests(unittest.TestCase):
    def test_absolute_defaults(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(PLAN_PATH, root/'config'/'levels.json')
        self.assertEqual(SETTINGS_PATH, root/'config'/'bptt_settings.json')
        self.assertEqual(DEFAULT_MODEL, MODEL_PATH)
        self.assertEqual(MODEL_PATH.parent, root/'models'/config.MODEL_VERSION)
        self.assertEqual(latest_path(MODEL_PATH).parent, MODEL_PATH.parent)

    def test_branch_names_are_separate(self):
        versions = []
        for branch in ('main', 'plain-ground', 'feature/plain-ground', 'feature%2Fplain-ground'):
            with patch('config.config.subprocess.check_output', return_value=branch) as git:
                versions.append(config.model_version())
                self.assertEqual(git.call_args.kwargs['cwd'], config.PROJECT_ROOT)
        self.assertEqual(versions[0], 'main')
        self.assertEqual(len(set(versions)), 4)
        self.assertTrue(all('/' not in version and '\\' not in version for version in versions))

    def test_detached_head_and_missing_git(self):
        with patch('config.config.subprocess.check_output', side_effect=['', '0123456789ab']):
            self.assertEqual(config.model_version(), 'detached-0123456789ab')
        with patch('config.config.subprocess.check_output', side_effect=FileNotFoundError):
            with self.assertRaises(RuntimeError):
                config.model_version()

    def test_save_plan_creates_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config'/'levels.json'
            save_plan(default_plan(), path)
            self.assertEqual(load_plan(path), default_plan())

    def test_launch_from_another_directory(self):
        code = ('import sys; sys.path.insert(0, sys.argv[1]); '
                'from config.config import MODEL_VERSION; '
                'from curriculum import load_plan; from sampling_gui import load_settings; '
                'assert load_plan(); assert "enabled" in load_settings(); print(MODEL_VERSION)')
        with tempfile.TemporaryDirectory() as folder:
            version = subprocess.check_output([sys.executable, '-c', code, str(config.PROJECT_ROOT)],
                                               cwd=folder, text=True).strip()
        self.assertEqual(version, config.MODEL_VERSION)
