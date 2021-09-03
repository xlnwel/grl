import os
import importlib

from utility.file import retrieve_pyfiles


def retrieve_all_make_env():
    env_dict = {}
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    for i in range(1, 10):
        pkg = 'env' if i == 1 else f'env{i}'
        if importlib.util.find_spec(pkg) is not None:
            env_dir = os.path.join(root_dir, pkg)
            files = retrieve_pyfiles(env_dir)
            filenames = [f.rsplit('/', maxsplit=1)[-1][:-3] for f in files]
            for f in filenames:
                m = importlib.import_module(f'{pkg}.{f}')
                for attr in dir(m):
                    if attr.startswith('make'):
                        env_dict[attr.split('_', maxsplit=1)[1]] = getattr(m, attr)
    
    return env_dict


def make_env(config):
    config = config.copy()
    env_name = config['name'].lower()

    env_dict = retrieve_all_make_env()
    make_built_in_gym = env_dict.pop('built_in_gym')
    env_func = env_dict.get(env_name.split('_', 1)[0], make_built_in_gym)
    env = env_func(config)
    
    return env
