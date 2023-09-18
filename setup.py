from setuptools import setup, find_packages

from utils import *
from data_utils import *
from plots import *
from metrics import *

setup(
    name='MP_functions',
    version='1.1',
    url='https://github.com/tusharmanekar/mp-tl-study',
    author='Tushar, David and Arnisa',
    author_email='email@gmail.com',

    packages=find_packages(),
)