<h1 align="center"><span style="font-weight:normal"> <img src="https://github.com/cosmaadrian/acumen-template/blob/master/assets/icon.png" alt="drawing" style="width:30px;"/> Acumen 👉🏻 Indexer 👈🏻</h1>

Coded with love and coffee ☕ by [Adrian Cosma](https://scholar.google.com/citations?user=cdYk_RUAAAAJ&hl=en). But I need more coffee!

<a href="https://www.buymeacoffee.com/cosmadrian" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

# Description

**`AcumenIndexer`** is designed to help with organizing various ***big datasets with many small instances*** into a common, highly efficient format, enabling random accessing, using either RAM or HDD for storing binary data chunks.

## But why?
Currently, the way storing and accessing data is performed is inefficient, especially for begginer data scientists, each practitioner having its own way of doing things. It is not always possible to store the whole dataset in RAM memory, so a usual approach is resorting to splitting each training instance in a separate file. Datasets comprised of many images or small files are very difficult to handle in practice (i.e., transferring the dataset through ssh, zipping takes a long time). Many files in a single folder can lead to performance issues on certain filesystems and lead to crashes.

## But how?
A simple way to overcome the issue of big dataset with many small instances is to store in RAM only the metadata and the index, and use a random access mechanism for big binary chunks of data on disk.

## Say what?
We make use of the native Python I/O operations of `f.seek()`, `f.read()` to read and write from large binary chunk files. We build a custom index based on byte offsets to access any training instance in O(1). Chunks can be `mmap()`-ed into RAM if memory is available to speed up I/O operations.

# Installation

Install the pypi package via pip:

```bash
pip install -U acumenindexer
```

Alternatively, install directly via git:
```bash
pip install -U git+https://github.com/cosmaadrian/acumen-indexer
```

# Usage

## Building an index

TBD

## Reading from index

```python
import numpy as np
import acumenindexer as ai

the_index = ai.load_index('index.csv') # just a pd.DataFrame
read_fn = ai.read_from_index(the_index, use_gzip = False, dtype = np.float16, in_memory = False)

for i in range(10):
    data = read_fn(i)
    print(data) # contains both metadata and actual binary data
```

## Use with PyTorch Datasets

```python
from torch.utils.data import Dataset
import acumenindexer as ai

class CustomDataset(Dataset):
    def __init__(self, index_path):
        self.index = ai.load_index(index_path)
        self.read_fn = ai.read_from_index(self.index, use_gzip = False, dtype = np.float16, in_memory = False)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        data = self.read_fn(idx)
        return data

```

# Benchmarks

TBD

# License
This repository uses [MIT License](LICENSE).
