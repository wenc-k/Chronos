"""Locate the pinned Wan runtime installed by the environment setup."""
import os
import sys
from pathlib import Path

WAN_ROOT = Path(os.environ.get('WAN22_ROOT', Path(__file__).resolve().parents[1] / '.deps' / 'Wan2.2')).expanduser().resolve()
if not (WAN_ROOT / 'wan' / 'image2video.py').is_file():
    raise RuntimeError('Install the pinned Wan2.2 source using inference/environment, or set WAN22_ROOT')
sys.path.insert(0, str(WAN_ROOT))
