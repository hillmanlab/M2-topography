"""Read a pre-downloaded AllenSDK MouseConnectivityCache without installing allensdk.

allensdk (2.16.2, the only version that installs on Python 3.10) force-downgrades
numpy to 1.23.5 and pandas to 2.3.3 -> 1.5.3 -- see `pip install --dry-run allensdk`.
That would silently unwind the pandas-2 fixes applied across ~60 other notebooks in
this repo if installed into the shared `ncomms` env.

But allensdk's own caching behavior is simple: if the manifest-specified file already
exists on disk, it just reads it -- no network, no recomputation. Once every file a
notebook touches has already been downloaded (as here), that read is exactly:

    injection/projection density  -> nrrd.read(<cache>/experiment_<id>/..._10.nrrd)
    structure mask                -> nrrd.read(<cache>/annotation/ccf_2017/
                                                 structure_masks/resolution_10/
                                                 structure_<id>.nrrd)
    experiment metadata table     -> json.load(<cache>/experiments.json)

This class reproduces just the four MouseConnectivityCache methods
figure-4-allen-connectivity.ipynb calls, with the same signatures, so the notebook
needs no changes beyond swapping the import and the `mcc = ...` construction line.
Only requires `pynrrd`, which -- unlike allensdk -- installs with zero side effects
(`pip install --dry-run pynrrd` touches nothing already in the env).

Falls back to raising FileNotFoundError with the expected path if a file is missing,
rather than silently trying to download it -- this is a read-only cache reader, not a
replacement for allensdk when data hasn't been fetched yet.
"""
import json
from pathlib import Path

import nrrd
import pandas as pd


class LocalMouseConnectivityCache:
    """Drop-in replacement for the 4 MouseConnectivityCache methods this repo uses,
    backed entirely by files already downloaded into `manifest_dir`."""

    def __init__(self, manifest_file, resolution=10):
        # Matches allensdk's own constructor signature: manifest_file is the path to
        # manifest.json itself, not its containing directory.
        self.root = Path(manifest_file).parent
        self.resolution = resolution

    def _nrrd(self, rel_path):
        p = self.root / rel_path
        if not p.exists():
            raise FileNotFoundError(
                f'{p} not cached locally -- this reader does not download. '
                'Use the real allensdk (in a separate env) to fetch it first.')
        data, header = nrrd.read(str(p))
        return data, header

    def get_injection_density(self, experiment_id):
        return self._nrrd(f'experiment_{experiment_id}/'
                          f'injection_density_{self.resolution}.nrrd')

    def get_projection_density(self, experiment_id):
        return self._nrrd(f'experiment_{experiment_id}/'
                          f'projection_density_{self.resolution}.nrrd')

    def get_structure_mask(self, structure_id):
        return self._nrrd(f'annotation/ccf_2017/structure_masks/'
                          f'resolution_{self.resolution}/structure_{structure_id}.nrrd')

    def get_experiments(self, dataframe=False):
        """Matches mcc.get_experiments(): cached data_set_id is renamed to id, as
        allensdk does.

        Two call conventions in this repo, both preserved:
          mcc.get_experiments(dataframe=True)   -> DataFrame indexed BY id, so
                                                    `.loc[experiment_id]` works
          mcc.get_experiments()                 -> plain list of dicts; the caller
                                                    wraps it in pd.DataFrame(...)
                                                    itself, giving a default range
                                                    index with 'id' as a column
        """
        p = self.root / 'experiments.json'
        if not p.exists():
            raise FileNotFoundError(f'{p} not cached locally.')
        with open(p) as f:
            records = json.load(f)
        for r in records:
            r['id'] = r.pop('data_set_id')
        if dataframe:
            return pd.DataFrame(records).set_index('id')
        return records
