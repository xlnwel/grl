import os

from utility.file import load_files


def load_nn():
    load_files(os.path.dirname(os.path.realpath(__file__)))
    
load_nn()
