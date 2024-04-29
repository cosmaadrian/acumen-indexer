import numpy as np
import pandas as pd
import gzip
import os
import mmap
import ast

from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Manager

def read_from_index(index, dtype = np.float16, use_gzip = False, chunk_paths = './chunks', in_memory = False):
    mmapped_chunk_files = dict()

    if in_memory:
        for chunk_name in index['ai:chunk_name'].unique():
            if chunk_paths is not None:
                chunk_name = os.path.join(chunk_paths, chunk_name)

            with open(chunk_name, 'rb') as f:
                mmapped_chunk_files[chunk_name] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __read(idx):
        row = index.iloc[idx]

        if chunk_paths is not None:
            chunk_name = os.path.join(chunk_paths, row['ai:chunk_name'])
        else:
            chunk_name = row['ai:chunk_name']

        if not in_memory:
            chunk_file = open(chunk_name, 'rb')
        else:
            chunk_file = mmapped_chunk_files[chunk_name]

        chunk_file.seek(row['ai:offset_bytes'])
        data = chunk_file.read(row['ai:size_bytes'])

        if not in_memory:
            chunk_file.close()

        if use_gzip:
            data = gzip.decompress(data)

        data = np.frombuffer(data, dtype = dtype).reshape(ast.literal_eval(row['ai:shape']))

        metadata_keys = [k for k in row.keys() if k not in ['ai:chunk_name', 'ai:offset_bytes', 'ai:size_bytes', 'ai:shape']]
        metadata = {k: row[k] for k in metadata_keys}

        return data, metadata

    return __read

def load_index(index_path):
    return pd.read_csv(index_path)

class Chunk(object):
    def __init__(self, output_path, lock, max_chunk_size = 1024*1024*10, dtype = np.float32, use_gzip = False):
        self.max_chunk_size = max_chunk_size
        self.dtype = dtype
        self.use_gzip = use_gzip
        self.lock = lock

        self.bytes_offset = 0
        self.current_size = 0
        self.current_idx = 0

        self.chunk_path = output_path
        self.current_path_name = f'chunks/chunk_{self.current_idx}.chnk'

        os.makedirs(f'{self.chunk_path}/chunks', exist_ok=True)

    def write(self, data):
        binary_buffer = data.astype(self.dtype).tobytes()

        if self.use_gzip:
            binary_buffer = gzip.compress(binary_buffer)

        if self.current_size + len(binary_buffer) > self.max_chunk_size:
            self.current_idx += 1

            self.current_size = 0
            self.bytes_offset = 0
            self.current_path_name = f'chunks/chunk_{self.current_idx}.chnk'

        with self.lock:
            with open(f'{self.chunk_path}/{self.current_path_name}', 'ab') as f:
                f.write(binary_buffer)

        self.bytes_offset += len(binary_buffer)
        self.current_size += len(binary_buffer)

        return self.current_path_name, self.bytes_offset - len(binary_buffer), len(binary_buffer)

def parallel_split_into_chunks(data_list, read_fn, output_path, chunk):
    process = multiprocessing.current_process()
    print(f"[AcumenIndexer] Started process {process.name}.")

    metadata_csvs = defaultdict(list)

    for d in data_list:
        if read_fn is None:
            data, metadata = d
        else:
            data, metadata = read_fn(d)

        chunk_path, offset, size = chunk.write(data)

        metadata_csvs['ai:chunk_name'].append((chunk_path))
        metadata_csvs['ai:offset_bytes'].append(offset)
        metadata_csvs['ai:size_bytes'].append(size)
        metadata_csvs['ai:shape'].append(data.shape)

        for k, v in metadata.items():
            metadata_csvs[k].append(v)

    csv_output_path = f'{output_path}/{process.pid}.csv'
    metadata_df = pd.DataFrame(metadata_csvs)
    metadata_df.to_csv(csv_output_path, index = False)

    return csv_output_path


def split_into_chunks(
        data_list,
        output_path,
        read_fn = None,
        chunk_size_bytes=1024 * 1024 * 10,
        verbose = False,
        n_jobs = 15,
        dtype = np.float16,
        use_gzip = False,
    ):

    if use_gzip:
        raise Exception('use_gzip = True dosent yet work.')

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if verbose:
        print(f'[AcumenIndexer] Using {n_jobs} jobs')

    if os.path.exists(output_path):
        raise ValueError(f'[AcumenIndexer] {output_path} already exists')

    # if output_path is a file name, throw error
    if os.path.isfile(output_path):
        raise ValueError(f'[AcumenIndexer] {output_path} is a file, not a directory')

    os.makedirs(output_path, exist_ok=True)

    with Manager() as manager:
        chunk_lock = manager.Lock()

        chunk = Chunk(
            output_path = output_path,
            lock = chunk_lock,
            max_chunk_size = chunk_size_bytes,
            dtype = dtype,
            use_gzip = use_gzip,
        )

        with Pool(n_jobs) as pool:
            # split file_names into 4 sublists
            args = [(data_list, read_fn, output_path, chunk) for process_idx in range(n_jobs)]

            output_csvs = pool.starmap(parallel_split_into_chunks, args)

    # merge all the csvs into one
    metadata_df = pd.concat([pd.read_csv(csv) for csv in output_csvs])
    metadata_df.to_csv(os.path.join(output_path, 'index.csv'), index = False)

    # remove the temporary csvs
    for csv in output_csvs:
        try:
            os.remove(csv)
        except Exception as e:
            if verbose:
                print(f'[AcumenIndexer] Could not remove {csv}: {e}')


