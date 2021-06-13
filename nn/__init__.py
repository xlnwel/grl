import os, glob
import types
import importlib


def source_file(_file_path):
    """
    Dynamically "sources" a provided file
    """
    basename = os.path.basename(_file_path)
    filename = basename.replace(".py", "")
    # Load the module
    loader = importlib.machinery.SourceFileLoader(filename, _file_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)


def load_files(path="."):
    """
    This function takes a path to a local directory
    and imports all the available files in there.
    """
    # for _file_path in glob.glob(os.path.join(local_dir, "*.py")):
    #     source_file(_file_path)
    for f in glob.glob(f'{path}/*'):
        if os.path.isdir(f):
            load_files(f)
        elif f.endswith('.py') and f != os.path.realpath(__file__):
            source_file(f)


def load_nn():
    load_files(os.path.dirname(os.path.realpath(__file__)))
    
load_nn()
