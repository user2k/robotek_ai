"""Ścieżki projektu i osobny katalog modeli dla każdej gałęzi Git."""
from pathlib import Path
import subprocess
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'config'
PLAN_PATH = CONFIG_DIR / 'levels.json'
SETTINGS_PATH = CONFIG_DIR / 'bptt_settings.json'


def model_version():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=PROJECT_ROOT,
                                       text=True, stderr=subprocess.PIPE).strip()
    try:
        branch = git('branch', '--show-current')
        if not branch:
            return 'detached-' + git('rev-parse', '--short=12', 'HEAD')
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('Nie można ustalić gałęzi Git dla katalogu modeli.') from error
    # Slash w nazwie brancha nie tworzy podkatalogu innej gałęzi.
    return quote(branch, safe='')


MODEL_VERSION = model_version()
MODEL_DIR = PROJECT_ROOT / 'models' / MODEL_VERSION
MODEL_PATH = MODEL_DIR / 'agent_continuous.pt'
