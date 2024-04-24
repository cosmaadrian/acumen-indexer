from setuptools import setup, find_packages

from acumenindexer import __version__

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='acumenindexer',
    version=__version__,
    license="MIT",
    url='https://github.com/cosmaadrian/acumen-indexer',
    author='Adrian Cosma',
    author_email='cosma.i.adrian@gmail.com',
    packages = ['acumenindexer'],
    install_requires = requirements,
)
