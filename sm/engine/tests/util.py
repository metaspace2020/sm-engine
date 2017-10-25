from os.path import join
import pytest
from unittest.mock import patch
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from fabric.api import local
#from pyspark import SparkContext, SparkConf
# lots of patch calls rely on SparkContext name
from pysparkling import Context as SparkContext
import pandas as pd
import numpy as np
import scipy.sparse as sp
from logging.config import dictConfig
from unittest.mock import MagicMock

from sm.engine.db import DB
from sm.engine.dataset_reader import DatasetReader
from sm.engine.search_job import SearchJob
from sm.engine.mol_db import MolecularDB
from sm.engine.theor_peaks_gen import TheorPeaksGenerator
from sm.engine.util import proj_root, sm_log_config, SMConfig, init_logger, logger
from sm.engine import ESExporter, ESIndexManager
from os.path import join

log_config = sm_log_config
log_config['loggers']['sm-engine']['handlers'] = ['console_debug']
init_logger(log_config)


@pytest.fixture(scope='session')
def sm_config():
    SMConfig.set_path(join(proj_root(), 'conf', 'test_config.json'))
    return SMConfig.get_conf(update=True)


@pytest.fixture(scope='module')
def spark_context(request):
    return SparkContext()


@pytest.fixture()
def test_db(sm_config, request):
    db_config = dict(**sm_config['db'])
    db_config['database'] = 'postgres'

    db = DB(db_config, autocommit=True)
    db.alter('DROP DATABASE IF EXISTS sm_test')
    db.alter('CREATE DATABASE sm_test')
    db.close()

    local('psql -h {} -U {} sm_test < {}'.format(
        sm_config['db']['host'], sm_config['db']['user'],
        join(proj_root(), 'scripts/create_schema.sql')))

    def fin():
        db = DB(db_config, autocommit=True)
        try:
            db.alter('DROP DATABASE IF EXISTS sm_test')
        except Exception as e:
            logger.warning('Drop sm_test database failed: %s', e)
        finally:
            db.close()
    request.addfinalizer(fin)


@pytest.fixture()
def ds_config():
    return {
        "databases": [{
            "name": "HMDB",
            "version": "2016"
        }],
        "isotope_generation": {
            "adducts": ["+H", "+Na"],
            "charge": {
                "polarity": "+",
                "n_charges": 1
            },
            "isocalc_sigma": 0.01,
            "isocalc_pts_per_mz": 10000
        },
        "image_generation": {
            "ppm": 1.0,
            "nlevels": 30,
            "q": 99,
            "do_preprocessing": False
        }
    }


@pytest.fixture()
def es(sm_config):
    return Elasticsearch(hosts=["{}:{}".format(sm_config['elasticsearch']['host'],
                                               sm_config['elasticsearch']['port'])])


@pytest.fixture()
def es_dsl_search(es, sm_config):
    return Search(using=es, index=sm_config['elasticsearch']['index'])


@pytest.fixture()
def sm_index(sm_config, request):
    es_config = sm_config['elasticsearch']
    with patch('sm.engine.es_export.DB') as DBMock:
        es_man = ESIndexManager(es_config)
        es_man.delete_index(es_config['index'])
        es_man.create_index(es_config['index'])

    def fin():
        es_man = ESIndexManager(es_config)
        es_man.delete_index(sm_config['elasticsearch']['index'])
    request.addfinalizer(fin)


@pytest.fixture()
def mol_db(sm_config, ds_config):
    data = {'id': 1, 'name': 'HMDB', 'version': '2016'}
    service = MagicMock()
    db = MagicMock()
    service.find_db_by_id.return_value = data
    service.find_db_by_name_version.return_value = data
    SMConfig._config_dict = sm_config

    mol_db = MolecularDB(1, None, None, ds_config['isotope_generation'],
                         mol_db_service=service, db=db)
    mol_db._sf_df = pd.DataFrame(dict(
        sf_id=[1, 2, 3],
        adduct=['+H', '+Na', '+H'],
        mzs=[[100, 101, 102], [200], [150, 151]],
        centr_ints=[[1, 0.1, 0.05], [1], [1, 0.3]]
    ), columns=['sf_id', 'adduct', 'mzs', 'centr_ints'])
    return mol_db

class MockDatasetReader(DatasetReader):
    def __init__(self):
        self.max_y = 80
        self.min_y = 0
        self.max_x = 120
        self.min_x = 0
        self._norm_img_pixel_inds = np.arange(80 * 120)

    def copy_convert_input_data(self):
        pass

class MockSearchJob(SearchJob):
    def _configure_spark(self):
        # avoid using Spark to minimize test running time
        self._sc = None

    def _open_dataset_reader(self, ds):
        self._ds_reader = MockDatasetReader()

    def _search(self, mol_db):
        N = np.random.randint(15, 25)
        df = pd.DataFrame(dict(
            chaos=np.random.uniform(0.9, 1.0, N),
            spatial=np.random.uniform(0.5, 1.0, N),
            spectral=np.random.uniform(0.7, 1.0, N)
        ))
        df['fdr'] = 0.05
        df['msm'] = df.chaos * df.spatial * df.spectral
        df['total_iso_ints'] = [[100000, 30000, 10000, 2000] for _ in range(N)]
        df['min_iso_ints'] = [[0, 0, 0, 0] for _ in range(N)]
        df['max_iso_ints'] = [[1000, 800, 500, 100] for _ in range(N)]

        target_adducts = self._ds.config['isotope_generation']['adducts']
        df['sf_id'] = np.random.choice(list(mol_db.sfs.keys()), N)
        df['adduct'] = np.random.choice(target_adducts, N)

        # generate the peaks so that ElasticSearch export can work
        theor_peaks_gen = TheorPeaksGenerator(self._sc, mol_db, self._ds.config, db=self._db)
        theor_peaks_input = list(zip((mol_db.sfs[id] for id in df.sf_id), df.adduct))
        theor_peaks_gen.generate_theor_peaks(theor_peaks_input)

        # generate ion images to be saved (with 1 on the main diagonal and 0 elsewhere)
        shape = self._ds_reader.get_dims()
        image_keys = list(zip(df.sf_id, df.adduct))
        image_values = [[sp.eye(shape[0], shape[1], format='csr') for i in range(4)]
                        for _ in image_keys]
        images = list(zip(image_keys, image_values))

        # use pysparkling.Context get an RDD-like object
        return df.set_index(['sf_id', 'adduct']), SparkContext().parallelize(images)
