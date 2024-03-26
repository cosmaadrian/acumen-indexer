import numpy as np
import pandas as pd
import gzip
import os

from collections import defaultdict
import multiprocessing
from multiprocessing import Pool

# use a user-provided iterator to output the next training instance
# it should output the binary thing and some metadata

def read_from_index(index, dtype = np.float16, use_gzip = False, chunk_paths = './chunks', in_memory = False):
    if in_memory:
        # TODO mmap the chunks in memory and read from them
        pass

    # TODO
    def __read(idx):
        row = index[idx]
        chunk_name = os.path.join(chunk_paths, row['chunk_name'])

        with open(chunk_name, 'rb') as chunk_file:
            chunk_file.seek(row['offset_bytes'])
            data = chunk_file.read(row['size_bytes'])

        if use_gzip:
            data = gzip.decompress(data)

        data = np.frombuffer(data, dtype = dtype).reshape(row['shape'])

        # TODO add metadata

        return data

    return __read

def load_index(index_path):
    return pd.read_csv(index_path)

def split_into_chunks(
        iterator,
        output_path,
        chunk_size_bytes=1024*1024*10,
        verbose = False,
        n_jobs = 1,
        dtype = np.float32,
        use_gzip = False,
    ):

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if verbose:
        print(f'Using {n_jobs} jobs')

    if os.path.exists(output_path):
        raise ValueError(f'{output_path} already exists')

    os.makedirs(output_path, exist_ok=True)

    # TODO use some locks on the iterator to make sure we don't read the same thing twice
    # TODO use a pool of workers to write to the chunks
    # TODO use a lock on the chunk file to make sure we don't write to the same chunk at the same time
    # keep track of offsets and positions

    pass

