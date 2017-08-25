from io import StringIO
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger('sm-engine')

DECOY_ADDUCTS = ['+He', '+Li', '+Be', '+B', '+C', '+N', '+O', '+F', '+Ne', '+Mg', '+Al', '+Si', '+P', '+S', '+Cl', '+Ar', '+Ca', '+Sc', '+Ti', '+V', '+Cr', '+Mn', '+Fe', '+Co', '+Ni', '+Cu', '+Zn', '+Ga', '+Ge', '+As', '+Se', '+Br', '+Kr', '+Rb', '+Sr', '+Y', '+Zr', '+Nb', '+Mo', '+Ru', '+Rh', '+Pd', '+Ag', '+Cd', '+In', '+Sn', '+Sb', '+Te', '+I', '+Xe', '+Cs', '+Ba', '+La', '+Ce', '+Pr', '+Nd', '+Sm', '+Eu', '+Gd', '+Tb', '+Dy', '+Ho', '+Ir', '+Th', '+Pt', '+Os', '+Yb', '+Lu', '+Bi', '+Pb', '+Re', '+Tl', '+Tm', '+U', '+W', '+Au', '+Er', '+Hf', '+Hg', '+Ta']


class FDR(object):

    def __init__(self, plan, target_adducts):
        """
        Plan is a Pandas dataframe (df) structured as
        sf_id | adduct | ... | <target_adduct_1> | ... | <target_adduct_N> | is_target
        such that df[df[target_adduct]] gives all (molecular formula, decoy adduct) combinations
        for that particular target adduct, grouped by molecular formula
        """
        self._plan = plan
        n_mf = len(plan['sf_id'].unique())
        self.decoy_sample_size = plan[target_adducts[0]].sum() // n_mf
        self.target_adducts = target_adducts
        self.td_df = None
        self.fdr_levels = [0.05, 0.1, 0.2, 0.5]

    @staticmethod
    def _msm_fdr_map(target_msm, decoy_msm):
        target_msm_hits = pd.Series(target_msm.msm.value_counts(), name='target')
        decoy_msm_hits = pd.Series(decoy_msm.msm.value_counts(), name='decoy')
        msm_df = pd.concat([target_msm_hits, decoy_msm_hits], axis=1).fillna(0).sort_index(ascending=False)
        msm_df['target_cum'] = msm_df.target.cumsum()
        msm_df['decoy_cum'] = msm_df.decoy.cumsum()
        msm_df['fdr'] = msm_df.decoy_cum / msm_df.target_cum
        return msm_df.fdr

    def _digitize_fdr(self, fdr_df):
        df = fdr_df.copy().sort_values(by='msm', ascending=False)
        msm_levels = [df[df.fdr < fdr_thr].msm.min() for fdr_thr in self.fdr_levels]
        df['fdr_d'] = 1.
        for msm_thr, fdr_thr in zip(msm_levels, self.fdr_levels):
            row_mask = np.isclose(df.fdr_d, 1.) & np.greater_equal(df.msm, msm_thr)
            df.loc[row_mask, 'fdr_d'] = fdr_thr
        df['fdr'] = df.fdr_d
        return df.drop('fdr_d', axis=1)

    def estimate_fdr(self, msm_df):
        logger.info('Estimating FDR...')

        target_fdr_df_list = []
        for ta in self.target_adducts:
            target_msm = msm_df.loc(axis=0)[:,ta]

            msm_fdr_list = []
            full_decoy_df = self._plan[self._plan[ta]][['sf_id', 'adduct']]
            for i in range(self.decoy_sample_size):
                decoy_subset_df = full_decoy_df[i::self.decoy_sample_size]
                sf_da_list = [tuple(row) for row in decoy_subset_df.values]
                decoy_msm = msm_df.loc[sf_da_list]
                msm_fdr = self._msm_fdr_map(target_msm, decoy_msm)
                msm_fdr_list.append(msm_fdr)

            msm_fdr_avg = pd.Series(pd.concat(msm_fdr_list, axis=1).median(axis=1), name='fdr')
            target_fdr = self._digitize_fdr(target_msm.join(msm_fdr_avg, on='msm'))
            target_fdr_df_list.append(target_fdr.drop('msm', axis=1))

        return pd.concat(target_fdr_df_list, axis=0)

def create_fdr_subsample(df, target_adducts, decoy_adducts, decoys_per_target=20, random_seed=42):
    if random_seed:
        np.random.seed(random_seed)

    df['is_target'] = df['adduct'].isin(target_adducts)
    decoy_df, target_df = [x[1] for x in df.groupby('is_target')]  # False < True
    n_decoy_adducts = len(decoy_df['adduct'].unique())
    n_mf = int(len(decoy_df) / n_decoy_adducts)
    assert decoys_per_target <= n_decoy_adducts

    # assert n_mf == len(decoy_df['mf'].unique())  # slows things down, left here for debugging

    # built-in np.random.choice is relatively slow for drawing samples without replacement,
    # generating a list of M random floats and taking .argsort()[:N] is much faster, as it turns out
    # http://numpy-discussion.10968.n7.nabble.com/Generating-random-samples-without-repeats-tp25666p25707.html
    def _select_decoys(adduct):
        selection = np.zeros_like(decoy_df['adduct'], dtype=bool)
        for i in range(n_mf):
            sub_selection = np.random.rand(n_decoy_adducts).argsort()[:decoys_per_target]
            selection[i * n_decoy_adducts : (i+1) * n_decoy_adducts][sub_selection] = True
        return selection

    for adduct in target_adducts:
        target_df[adduct] = False
        decoy_df[adduct] = _select_decoys(adduct)

    # keep only decoy isotope patterns selected at least for one target adduct
    decoy_df = decoy_df[decoy_df[target_adducts].sum(axis=1) > 0]
    return pd.concat([target_df, decoy_df])
