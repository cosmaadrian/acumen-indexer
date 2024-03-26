from setuptools import setup, find_packages

from acumenindexer import __version__

setup(
    name='acumenindexer',
    version=__version__,

    url='https://github.com/cosmaadrian/acumen-indexer',
    author='Adrian Cosma',
    author_email='cosma.i.adrian@gmail.com',

    py_modules = find_packages(),
)
