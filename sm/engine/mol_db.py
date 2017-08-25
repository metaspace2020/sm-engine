from collections import OrderedDict
import pandas as pd
import logging
import requests
import pyarrow.ipc

from sm.engine.db import DB
from sm.engine.util import SMConfig
from sm.engine.fdr import DECOY_ADDUCTS, create_fdr_subsample

logger = logging.getLogger('sm-engine')

SF_INS = 'INSERT INTO sum_formula (db_id, sf) values (%s, %s)'
SF_COUNT = 'SELECT count(*) FROM sum_formula WHERE db_id = %s'
SF_SELECT = 'SELECT id, sf FROM sum_formula WHERE db_id = %s'

class MolDBServiceWrapper(object):
    def __init__(self, service_url):
        self._service_url = service_url
        self._session = requests.Session()

    def _fetch(self, url):
        r = self._session.get(url)
        r.raise_for_status()
        return r.json()['data']

    def find_db_by_id(self, id):
        url = '{}/databases/{}'.format(self._service_url, id)
        return self._fetch(url)

    def find_db_by_name_version(self, name, version=None):
        url = '{}/databases?name={}'.format(self._service_url, name)
        if version:
            url += '&version={}'.format(version)
        return self._fetch(url)

    def fetch_db_sfs(self, db_id):
        return self._fetch('{}/databases/{}/sfs'.format(self._service_url, db_id))

    def fetch_molecules(self, db_id, sf=None):
        if sf:
            url = '{}/databases/{}/molecules?sf={}&fields=mol_id,mol_name'
            return self._fetch(url.format(self._service_url, db_id, sf))
        else:
            # TODO: replace one large request with several smaller ones
            url = '{}/databases/{}/molecules?fields=sf,mol_id,mol_name&limit=10000000'
            return self._fetch(url.format(self._service_url, db_id))

    def fetch_patterns(self, db_id, charge, adducts, pts_per_mz):
        url = '{}/isotopic_patterns/{}/{}/{}'.format(self._service_url, db_id, charge, pts_per_mz)
        r = self._session.post(url, data={'adducts': ','.join(adducts)})
        r.raise_for_status()
        df = pyarrow.ipc.deserialize_pandas(r.content)
        df.rename(columns={
            'sf_id': 'mf',
            'adduct': 'adduct',
            'mzs': 'mzs',
            'intensities': 'centr_ints'
        }, inplace=True)
        return df


class MolecularDB(object):
    """ A class representing a molecule database to search through.
        Provides several data structures used in the engine to speed up computation

        Args
        ----------
        name: str
        version: str
            If None the latest version will be used
        iso_gen_config : dict
            Isotope generator configuration
        mol_db_service : sm.engine.MolDBServiceWrapper
            Molecular database ID/name resolver
        db : DB
            Database connector
        """

    def __init__(self, id=None, name=None, version=None, iso_gen_config=None,
                 mol_db_service=None, db=None):
        self._iso_gen_config = iso_gen_config
        sm_config = SMConfig.get_conf()
        self.mol_db_service = mol_db_service or MolDBServiceWrapper(sm_config['services']['mol_db'])

        if id is not None:
            data = self.mol_db_service.find_db_by_id(id)
        elif name is not None:
            data = self.mol_db_service.find_db_by_name_version(name, version)[0]
        else:
            raise Exception('MolDB id or name should be provided')

        self._id, self._name, self._version = data['id'], data['name'], data['version']
        self._sf_df = None
        self._job_id = None
        self._sfs = None
        self._db = db or DB(sm_config['db'])

    def __str__(self):
        return '{} {}'.format(self.name, self.version)

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def set_job_id(self, job_id):
        self._job_id = job_id

    def get_molecules(self, sf=None):
        """ Returns a dataframe with (mol_id, mol_name) or (sf, mol_id, mol_name) rows

        Args
        ----------
        sf: str
        Returns
        ----------
            pd.DataFrame
        """
        return pd.DataFrame(self.mol_db_service.fetch_molecules(self.id, sf=sf))

    @property
    def sfs(self):
        if not self._sfs:
            sfs = self.mol_db_service.fetch_db_sfs(self.id)
            if self._db.select_one(SF_COUNT, self._id)[0] == 0:
                rows = [(self._id, sf) for sf in sfs]
                self._db.insert(SF_INS, rows)
            self._sfs = OrderedDict(self._db.select(SF_SELECT, self._id))
        return self._sfs

    @property
    def sf_df(self):
        if self._sf_df is None:
            sfs = pd.DataFrame.from_records(list(self.sfs.items()), columns=['sf_id', 'mf'])
            iso_gen_conf = self._iso_gen_config
            charge = '{}{}'.format(iso_gen_conf['charge']['polarity'], iso_gen_conf['charge']['n_charges'])

            target_adducts = iso_gen_conf['adducts']
            full_df = self.mol_db_service.fetch_patterns(self.id, charge,
                                                         target_adducts + DECOY_ADDUCTS,
                                                         iso_gen_conf['isocalc_pts_per_mz'])
            self._sf_df = create_fdr_subsample(full_df, target_adducts, DECOY_ADDUCTS)
            self._check_formula_uniqueness(self._sf_df)

            self._sf_df = pd.merge(self._sf_df, sfs)
            del self._sf_df['mf']

            has_target_entries = self._sf_df['is_target'].sum() > 0
            has_decoy_entries = (~self._sf_df['is_target']).sum() > 0
            assert has_target_entries, 'No formulas matching the criteria were found in theor_peaks! (target)'
            assert has_decoy_entries, 'No formulas matching the criteria were found in theor_peaks! (decoy)'

            logger.info('Loaded %s sum formula, adduct combinations from the DB', self._sf_df.shape[0])
        return self._sf_df

    @staticmethod
    def _check_formula_uniqueness(sf_df):
        uniq_sf_adducts = len({(r.mf, r.adduct) for r in sf_df.itertuples()})
        assert uniq_sf_adducts == sf_df.shape[0], \
            'Not unique formula-adduct combinations {} != {}'.format(uniq_sf_adducts, sf_df.shape[0])

    @staticmethod
    def sf_peak_gen(sf_df):
        for r in sf_df.itertuples():
            for pi, mz in enumerate(r.mzs):
                yield r.sf_id, r.adduct, pi, mz

    def get_ion_peak_df(self):
        return pd.DataFrame(self.sf_peak_gen(self.sf_df),
                            columns=['sf_id', 'adduct', 'peak_i', 'mz']).sort_values(by='mz')

    def get_ion_sorted_df(self):
        return self.sf_df[['sf_id', 'adduct']].copy().set_index(['sf_id', 'adduct']).sort_index()

    def get_sf_peak_ints(self):
        return {(r.sf_id, r.adduct): r.centr_ints for r in self.sf_df.itertuples()}
