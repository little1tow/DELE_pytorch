#!/usr/bin/env python3


import hashlib
import io
import struct
import sys
import tarfile
import time

import numpy as np
from bson import BSON
from bson.binary import Binary, USER_DEFINED_SUBTYPE
from bson.codec_options import TypeCodec, CodecOptions, TypeRegistry
from torch.utils.data import Dataset
from tqdm import tqdm

UINT64 = struct.Struct('<Q')


class TarWriter(object):

    def __init__(self, path: str):
        self._path = path
        self._index = []

        self._tar = tarfile.open(path, 'w')
        self._info = tarfile.TarInfo()
        self._info.uname = 'user'
        self._info.gname = 'group'

    def close(self):
        if hasattr(self, '_tar'):
            self._tar.close()
            self._write_index()
            delattr(self, '_tar')

    def write(self, name: str, data: bytes):
        start = self._tar.fileobj.tell()
        self._index.append(start)
        self._info.name = name
        self._info.size = len(data)
        self._info.mtime = time.time()
        self._tar.addfile(self._info, io.BytesIO(data))

    def _write_index(self):
        with io.open(self._path, 'ab') as f:
            index_start = f.tell()
            index_count = len(self._index)
            checksum = hashlib.md5()
            for pos in self._index:
                pos_bin = UINT64.pack(pos)
                checksum.update(pos_bin)
                f.write(pos_bin)
            f.write(b'TARINDEX')
            f.write(UINT64.pack(index_start))
            f.write(UINT64.pack(index_count))
            f.write(checksum.digest())

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self._index)


class TarReader(object):

    def __init__(self, path: str):
        self._path = path
        self._index = []

        self._read_index()
        self._tar = None

    def close(self):
        if hasattr(self, '_tar'):
            if self._tar is not None:
                self._tar.close()
            delattr(self, '_tar')

    def read(self, i):
        if self._tar is None:
            self._tar = tarfile.open(self._path, 'r')
        info = self._index[i]
        if isinstance(info, int):
            self._tar.fileobj.seek(info, io.SEEK_SET)
            info = tarfile.TarInfo.fromtarfile(self._tar)
            self._index[i] = info
        return self._tar.extractfile(info).read()

    def _read_index(self):
        success = False
        with io.open(self._path, 'rb') as f:
            f.seek(-(8 + UINT64.size * 2 + 16), io.SEEK_END)
            if f.read(8) == b'TARINDEX':
                index_start = UINT64.unpack(f.read(UINT64.size))[0]
                index_count = UINT64.unpack(f.read(UINT64.size))[0]
                checksum_bin = f.read(16)
                f.seek(index_start, io.SEEK_SET)
                checksum = hashlib.md5()
                for _ in range(index_count):
                    start_bin = f.read(UINT64.size)
                    checksum.update(start_bin)
                    start = UINT64.unpack(start_bin)[0]
                    self._index.append(start)
                if checksum.digest() == checksum_bin:
                    success = True
                else:
                    self._index = []
                    print('Corrupted tar index.', file=sys.stderr)
            else:
                print('No tar index found.', file=sys.stderr)
        if not success:
            print('Failed to load tar index. It will take some time to build from scratch.', file=sys.stderr)
            with tarfile.open(self._path, 'r') as tar:
                for info in tqdm(tar, leave=False):
                    self._index.append(info)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self._index)


class TarFile(object):

    def __init__(self, path: str, mode: str):
        if mode == 'r':
            self._impl = self._impl_reader = TarReader(path)
        elif mode == 'w':
            self._impl = self._impl_writer = TarWriter(path)
        else:
            raise RuntimeError('Argument "mode" should be one of {"r", "w"}')

    def close(self):
        self._impl.close()

    def write(self, name, data):
        self._impl_writer.write(name, data)

    def read(self, i):
        return self._impl_reader.read(i)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._impl.__exit__(exc_type, exc_val, exc_tb)

    def __len__(self):
        return self._impl.__len__()


class NumpyCodec(TypeCodec):
    sub_type = USER_DEFINED_SUBTYPE + 1
    python_type = np.ndarray
    bson_type = Binary

    def transform_python(self, a: np.ndarray):
        data = NumpyCodec.encode_ndarray(a)
        return Binary(data, NumpyCodec.sub_type)

    def transform_bson(self, data: Binary):
        if data.subtype == NumpyCodec.sub_type:
            return NumpyCodec.decode_ndarray(data)
        return data

    @staticmethod
    def encode_ndarray(a: np.ndarray) -> bytes:
        buf = io.BytesIO()
        buf.write(a.dtype.str.encode())
        buf.write(str(a.shape).encode())
        buf.write(a.tobytes('C'))
        return buf.getvalue()

    @staticmethod
    def decode_ndarray(data: bytes) -> np.ndarray:
        dtype_end = data.find(b'(')
        shape_start = dtype_end + 1
        shape_end = data.find(b')', shape_start)
        dtype = data[:dtype_end]
        shape = tuple(int(size) for size in data[shape_start:shape_end].split(b','))
        buffer = data[shape_end + 1:]
        a = np.ndarray(dtype=dtype, shape=shape, buffer=buffer)
        return np.array(a)


CODEC_OPTIONS = CodecOptions(type_registry=TypeRegistry([NumpyCodec()]))


class BSONTar(object):

    def __init__(self, path: str, mode: str):
        self._path = path
        self._mode = mode

        self._init()

    def _init(self):
        if self._mode == 'r':
            self._impl = self._impl_reader = TarReader(self._path)
        elif self._mode == 'w':
            self._impl = self._impl_writer = TarWriter(self._path)
        else:
            raise RuntimeError()

    def close(self):
        self._impl.close()

    def write(self, doc):
        name = f'{len(self._impl)}.bson'
        data = BSON.encode(doc, codec_options=CODEC_OPTIONS)
        self._impl_writer.write(name, data)

    def read(self, i: int):
        data = self._impl_reader.read(i)
        return BSON(data).decode(codec_options=CODEC_OPTIONS)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._impl.__exit__(exc_type, exc_val, exc_tb)

    def __getitem__(self, i: int):
        return self.read(i)

    def __len__(self):
        return self._impl.__len__()


class TarDataset(Dataset):

    def __init__(self, path, fn=None):
        self._path = path
        self._fn = fn
        self._impl = BSONTar(path, 'r')

    def close(self):
        self._impl.close()

    def __len__(self):
        return self._impl.__len__()

    def __getitem__(self, i: int):
        doc = self._impl.read(i)
        if callable(self._fn):
            doc = self._fn(doc)
        return doc
