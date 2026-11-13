"""Load independently generated validation pairs without changing training data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]


def load_validation(path,device,check=False):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Validation file must be inside project')
    manifest=json.loads(path.with_name('manifest.json').read_text())
    with path.open('rb') as handle: actual=hashlib.file_digest(handle,'sha256').hexdigest()
    if actual!=manifest['validation_sha256']: raise ValueError('Validation data hash mismatch')
    with np.load(path) as archive:
        arrays=[archive[k].astype(np.float32) for k in ['branch','trunk','reference']]
    n=manifest['count']
    assert [a.shape for a in arrays]==[(n,100),(n,1),(n,1)]
    assert all(np.isfinite(a).all() for a in arrays)
    assert ((arrays[1]>=0)&(arrays[1]<=3)).all()
    return tuple(torch.tensor(a[:8] if check else a,device=device) for a in arrays)
