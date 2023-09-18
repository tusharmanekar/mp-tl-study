from setuptools import setup, find_packages

from utils import __version__

setup(
    name='MP utilities and functions',
    version=__version__,

    url='https://github.com/tusharmanekar/mp-tl-study',
    author='Tushar, David and Arnisa',
    author_email='email@gmail.com',

    install_requires=[
        'returns-decorator',
    ],
    packages=find_packages()
)