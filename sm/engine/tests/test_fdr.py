import pandas as pd
from sm.engine.db import DB
from unittest.mock import MagicMock, patch
from pandas.util.testing import assert_frame_equal

from sm.engine.fdr import FDR
from sm.engine import MolecularDB

def test_estimate_fdr_returns_correct_df():
    sf_df = pd.DataFrame([[1, '+Cu', True, False],
                          [1, '+Co', True, False],
                          [2, '+Ag', True, False],
                          [2, '+Ar', True, False]],
                         columns=['sf_id', 'adduct', '+H', 'is_target'])
    fdr = FDR(sf_df, ['+H'])
    fdr.fdr_levels = [0.2, 0.8]

    msm_df = pd.DataFrame([[1, '+H', 0.85],
                          [2, '+H', 0.5],
                          [1, '+Cu', 0.5],
                          [1, '+Co', 0.5],
                          [2, '+Ag', 0.75],
                          [2, '+Ar', 0.0]],
                          columns=['sf_id', 'adduct', 'msm']).set_index(['sf_id', 'adduct']).sort_index()
    exp_sf_df = pd.DataFrame([[1, '+H', 0.2], [2, '+H', 0.8]],
                             columns=['sf_id', 'adduct', 'fdr']).set_index(['sf_id', 'adduct'])

    assert_frame_equal(fdr.estimate_fdr(msm_df), exp_sf_df)


def test_estimate_fdr_digitize_works():
    sf_df = pd.DataFrame([[1, '+Cu', True, False],
                          [2, '+Ag', True, False],
                          [3, '+Cl', True, False],
                          [4, '+Co', True, False]],
                         columns=['sf_id', 'adduct', '+H', 'is_target'])
    fdr = FDR(sf_df, ['+H'])
    fdr.fdr_levels = [0.4, 0.8]

    msm_df = pd.DataFrame([[1, '+H', 1.0],
                          [2, '+H', 0.75],
                          [3, '+H', 0.5],
                          [4, '+H', 0.25],
                          [1, '+Cu', 0.75],
                          [2, '+Ag', 0.3],
                          [3, '+Cl', 0.25],
                          [4, '+Co', 0.1]],
                          columns=['sf_id', 'adduct', 'msm']).set_index(['sf_id', 'adduct']).sort_index()
    exp_sf_df = pd.DataFrame([[1, '+H', 0.4],
                              [2, '+H', 0.4],
                              [3, '+H', 0.4],
                              [4, '+H', 0.8]],
                             columns=['sf_id', 'adduct', 'fdr']).set_index(['sf_id', 'adduct'])

    assert_frame_equal(fdr.estimate_fdr(msm_df), exp_sf_df)
