import os, glob
import types
import importlib


def source_file(file_path):
    """
    Dynamically "sources" a provided file
    """
    basename = os.path.basename(file_path)
    filename = basename.replace(".py", "")
    # Load the module
    loader = importlib.machinery.SourceFileLoader(filename, file_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)


def load_files(path='.', recursively_load=True):
    """
    This function takes a path to a local directory
    and imports all the available files in there.
    """
    # for _file_path in glob.glob(os.path.join(local_dir, "*.py")):
    #     source_file(_file_path)
    for f in glob.glob(f'{path}/*'):
        if recursively_load and os.path.isdir(f):
            load_files(f)
        elif f.endswith('.py') and not f.endswith('__init__.py'):
            source_file(f)


def retrieve_pyfiles(path='.'):
    return [f for f in glob.glob(f'{path}/*') 
        if f.endswith('.py') and not f.endswith('__init__.py')]
