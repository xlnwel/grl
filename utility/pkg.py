import importlib


def pkg_str(root_dir, separator, algo):
    return f'{root_dir}{separator}{algo}'

def get_package(algo, place=0, separator='.', root_dir=None):
    algo = algo.split('-', 1)[place]
    if root_dir is None:
        root_dir = 'algo'
        for i in range(1, 4):
            indexed_root_dir = f'{root_dir}' if i == 1 else f'{root_dir}{i}'
            pkg = pkg_str(indexed_root_dir, '.', algo)
            if importlib.util.find_spec(pkg) is not None:
                pkg = pkg_str(indexed_root_dir, separator, algo)
                break
    else:
        pkg = f'{root_dir}{separator}{algo}'

    return pkg

def import_module(name=None, pkg=None, algo=None, *, config=None, place=0):
    """ import <name> module according to algo or algorithm in config """
    if pkg is None:
        algo = algo or config['algorithm']
        assert isinstance(algo, str), algo
        pkg = get_package(algo=algo, place=place)
    m = importlib.import_module(f'{pkg}.{name}')

    return m

def import_agent(algo=None, *, config=None):
    nn = import_module(name='nn', algo=algo, config=config, place=-1)
    agent = import_module(name='agent', algo=algo, config=config, place=-1)

    return nn.create_model, agent.Agent

def import_main(module, algo=None, *, config=None):
    algo = algo or config['algorithm']
    assert isinstance(algo, str), algo
    pkg = get_package(algo, place={'train': 0, 'eval': -1}[module])
    m = importlib.import_module(f'{pkg}.{module}')

    return m.main

if __name__ == '__main__':
    print(get_package('asap'))
