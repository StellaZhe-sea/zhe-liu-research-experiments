"""Load existing Caputo data exactly; never generate new functions or labels."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RUNS = dict(legendre="caputo_legendre_20261017_191641_544492",
            polyfractonomial="caputo_polyfractonomial_20261020_125251_304947",
            grf="caputo_grf_20261020_135640_070892")


def load_data(case):
    run = ROOT/"results"/RUNS[case]
    if case == "legendre":
        result = []
        sources = [run/"data"/(name+".npz") for name in ("train","test","test0")]
        for path in sources:
            suffix = "train" if path.stem=="train" else "test"
            with np.load(str(path)) as data:
                result.append(tuple(data[key+"_"+suffix].astype(np.float32,copy=False)
                                    for key in ("X_u","X_y","Y")))
    else:
        path = (ROOT/"data"/"caputo_polyfractonomial_disjoint"/"compact_data.mat"
                if case=="polyfractonomial" else run/"compact_data.npz")
        sources = [path]
        if path.suffix==".mat":
            d = loadmat(str(path))
            raw = {"query":d["query"]}
            for name in ("train","test","example"):
                raw[name+"_u"] = d["u_"+name] if name!="example" else d["example_u"]
                raw[name+"_ref"] = d["ref_"+name] if name!="example" else d["example_ref"]
        else:
            with np.load(str(path)) as archive:
                raw = {k:archive[k] for k in archive.files}
        result=[]
        for name in ("train","test","example"):
            u = raw[name+"_u"].astype(np.float32)
            reference = raw[name+"_ref"].astype(np.float32)
            query = raw["query"].astype(np.float32)
            result.append((np.repeat(u,100,axis=0),np.tile(query,(len(u),1)),reference.reshape(-1,1)))
    for branch,trunk,reference in result:
        assert branch.shape==(len(reference),15) and trunk.shape==(len(reference),2)
        assert reference.shape==(len(reference),1)
        assert all(np.isfinite(a).all() for a in (branch,trunk,reference))
    assert len(result[0][0])==len(result[1][0])==1000000
    return result, sources, run
