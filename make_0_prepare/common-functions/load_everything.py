"""Partial reconstruction of the server-side `load_everything` helper module.

**This module did not survive the migration off the Linux server.** 16 notebooks do
`from load_everything import *`, including
`make_1_figures/figure_5/figure-5-ncomms-stats.ipynb`, which produces main Fig 5i
and Supplementary Fig 6b -- so all of them currently die at import with
`ModuleNotFoundError: No module named 'load_everything'`. The original is not in
this repo, not in its git history, not anywhere on the data volume, and not even in
`common-functions/__pycache__` (so it was never once imported on this machine).

Rather than guess at the whole module, this reconstructs only the names the
surviving notebooks actually call, and makes every *other* name fail loudly with an
explanation instead of silently resolving to something plausible-but-wrong.

Reconstructed:

    load_model(path)   Every call site passes a path to a plain pickle -- e.g.
                       load_model(fetch_path + mouse + '_data_joint.pickle') and
                       load_model(fetch_path_sensory + 'brightest_ind_' + mouse).
                       All such files on disk are protocol-4 pickles that
                       harness.load_pickle reads correctly (verified against
                       index-S1/brightest_ind_cm229,
                       index-M2/brightest_frontline_ind-cm229 and
                       modeling/cm219_data_joint.pickle). So this is an alias for
                       harness.load_pickle, which additionally installs the
                       pandas-1.x unpickling shim these files need under pandas 2.

Deliberately NOT reconstructed:

    plot_concatenated_timecourse(...)   Called once, in figure-5-ncomms-stats.ipynb
                       cell 8, which has no savefig and produces no published
                       panel. Its exact figure layout is unrecoverable, and
                       inventing one would put a fabricated figure in a notebook
                       that ships with the paper. Accessing it raises with this
                       explanation.

If the original turns up (e.g. on a server backup), prefer it over this file.
"""
from harness import load_pickle as _load_pickle

__all__ = ['load_model']

_UNRECONSTRUCTED = {
    'plot_concatenated_timecourse':
        'Called only in figure-5-ncomms-stats.ipynb cell 8, which saves nothing and '
        'produces no published panel. Its original layout is unrecoverable, so it is '
        'deliberately not reimplemented -- a guessed figure has no place in a '
        'notebook published with the paper. Skip that cell.',
}


def load_model(pkl_filename):
    """Load a pickle. Alias for harness.load_pickle -- see this module's docstring
    for why that equivalence is safe."""
    return _load_pickle(pkl_filename)


def __getattr__(name):
    if name in _UNRECONSTRUCTED:
        raise AttributeError(
            f'load_everything.{name} was lost with the server-side original and is '
            f'not reconstructed.\n\n{_UNRECONSTRUCTED[name]}')
    raise AttributeError(
        f'load_everything.{name} does not exist in this partial reconstruction. '
        f'The original module was lost in the migration off the server; only '
        f'{sorted(__all__)} were recovered. See this module\'s docstring.')
