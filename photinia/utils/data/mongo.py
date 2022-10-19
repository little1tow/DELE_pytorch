#!/usr/bin/env python3


"""
@author: xi
"""

import gzip
from typing import Union, List

import numpy as np
import pymongo
from torch.utils.data import Dataset
from tqdm import tqdm


class MongoServer(object):

    def __init__(self,
                 host: Union[str, List[str]],
                 auth_db: str = None,
                 username: str = None,
                 password: str = None,
                 read_preference=None):
        self.host = host
        self.auth_db = auth_db
        self.username = username
        self.password = password
        self.read_preference = read_preference

    def connect(self):
        kwargs = {}
        if self.read_preference:
            kwargs['readPreference'] = self.read_preference
        if self.auth_db:
            kwargs['authSource'] = self.auth_db
        if self.username:
            kwargs['username'] = self.username
        if self.password:
            kwargs['password'] = self.password
        conn = pymongo.MongoClient(self.host, **kwargs)
        return conn


class MongoDataset(Dataset):

    def __init__(self,
                 server: MongoServer,
                 db: str,
                 coll: str,
                 match: Union[dict, None] = None,
                 project: Union[dict, None] = None,
                 fn=None):
        self._server = server
        self._db = db
        self._coll = coll
        self._match = match if match is not None else {}
        self._project = project
        self._fn = fn

        self._conn: Union[pymongo.MongoClient, None] = None
        self._coll_obj: Union[pymongo.collection.Collection, None] = None
        self._id_list = []

        self.connect()
        if self._match:
            count = self._coll_obj.count_documents(self._match)
        else:
            count = self._coll_obj.estimated_document_count()
        self._match['_id'] = {'$ne': None}
        cur = tqdm(
            self._coll_obj.find(self._match, {'_id': 1}).batch_size(40960),
            total=count,
            desc=f'Scanning "{self._coll}"',
            dynamic_ncols=True,
            leave=False,
            unit_scale=True
        )
        for doc in cur:
            self._id_list.append(doc['_id'])
        self.close()

    def connect(self):
        if self._conn is None:
            self._conn = self._server.connect()
            self._coll_obj = self._conn[self._db][self._coll]

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._coll_obj = None

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self._id_list)

    def __getitem__(self, index: int):
        self.connect()
        _id = self._id_list[index]
        doc = self._coll_obj.find_one({'_id': _id}, self._project)
        if callable(self._fn):
            doc = self._fn(doc)
        return doc


def encode_numpy(a: np.ndarray, compress=False, compress_level=1):
    buffer = a.tobytes('C')
    if compress:
        buffer = gzip.compress(buffer, compress_level)
    shape_str = ','.join(str(size) for size in a.shape)
    return buffer, str(a.dtype), shape_str


def decode_numpy(data: tuple, copy=True, compress=False):
    assert isinstance(data, (tuple, list)) and len(data) == 3
    buffer, dtype, shape = data
    if compress:
        buffer = gzip.decompress(buffer)
    if isinstance(shape, str):
        shape = tuple(int(size) for size in shape.split(','))
    a = np.ndarray(buffer=buffer, dtype=dtype, shape=shape, order='C')
    if copy:
        a = np.array(a, copy=True)
    return a
