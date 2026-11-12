"""Apply project-local paths before importing plotting or DeepXDE."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def configure():
    for key, suffix in {
        'MPLCONFIGDIR': '.cache/matplotlib', 'XDG_CACHE_HOME': '.cache',
        'TORCH_HOME': '.cache/torch', 'TEMP': '.cache/tmp', 'TMP': '.cache/tmp',
        'CUDA_CACHE_PATH': '.cache/cuda', 'TRITON_CACHE_DIR': '.cache/triton',
        'TORCH_EXTENSIONS_DIR': '.cache/torch_extensions',
    }.items():
        path = ROOT / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['DDE_BACKEND'] = 'pytorch'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

configure()
