import os, sys
import logging
import time
import itertools
from multiprocessing import Process
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.utils import deep_update, eval_str
from utility.yaml_op import load_config
from utility.display import pwc
from utility import pkg
from run.grid_search import GridSearch
from run.args import parse_train_args


logger = logging.getLogger(__name__)

def get_algo_name(algo):
    algo_mapping = {
        'r2d2': 'apex-mrdqn',
    }
    if algo in algo_mapping:
        return algo_mapping[algo]
    return algo

def get_config(algo, env):
    def search_add(word, files, filename):
        if [f for f in files if word in f]:
            # if suffix meets any config in the dir, we add it to filename
            filename = f'{word}_{filename}'
        return filename
    algo_dir = pkg.get_package(algo, 0, '/')
    if env == '' and '-' in algo:
        pwc('Config Warning: set Procgen as the default env, otherwise specify env explicitly', color='green')
        env = 'procgen_'
    files = [f for f in os.listdir(algo_dir) if 'config.yaml' in f]
    filename = 'config.yaml'
    if '_' in env:
        prefix = env.split('_')[0]
        filename = search_add(prefix, files, filename)
    if '-' in algo:
        suffix = algo.split('-')[-1]
        if [f for f in files if suffix in f]:
            filename = search_add(suffix, files, filename)
        elif suffix[-1].isdigit():
            suffix = suffix[:-1]
            filename = search_add(suffix, files, filename)
    path = f'{algo_dir}/{filename}'
    pwc(f'Config path: {path}', color='green')
    return load_config(path)

def change_config(kw, model_name, env_config, model_config, agent_config, replay_config):
    if kw:
        for s in kw:
            key, value = s.split('=')
            value = eval_str(value)
            if model_name != '':
                model_name += '-'
            model_name += s

            # change kwargs in config
            configs = []
            config_keys = ['env', 'model', 'agent', 'replay']
            config_values = [env_config, model_config, agent_config, replay_config]

            for k, v in model_config.items():
                if isinstance(v, dict):
                    config_keys.append(k)
                    config_values.append(v)
            for name, config in zip(config_keys, config_values):
                if key in config:
                    configs.append((name, config))
            assert configs, f'"{s}" does not appear in any config!'
            if len(configs) > 1:
                pwc(f'All {key} appeared in the following configs will be changed: '
                        + f'{list([n for n, _ in configs])}.', color='cyan')
                
            for _, c in configs:
                c[key]  = value
            
    return model_name

def load_and_run(directory):
    # load model and log path
    config_file = None
    for root, _, files in os.walk(directory):
        for f in files:
            if 'src' in root:
                break
            if f == 'config.yaml' and config_file is None:
                config_file = os.path.join(root, f)
                break
            elif f =='config.yaml' and config_file is not None:
                pwc(f'Get multiple "config.yaml": "{config_file}" and "{os.path.join(root, f)}"')
                sys.exit()

    config = load_config(config_file)
    env_config = config['env']
    model_config = config['model']
    agent_config = config['agent']
    replay_config = config.get('buffer') or config.get('replay')
    main = pkg.import_main('train', config=agent_config)

    main(env_config, model_config, agent_config, replay_config)

if __name__ == '__main__':
    cmd_args = parse_train_args()
    verbose = getattr(logging, cmd_args.verbose.upper())
    logging.basicConfig(level=verbose)
    
    processes = []
    if cmd_args.directory != '':
        load_and_run(cmd_args.directory)
    else:
        algorithm = list(cmd_args.algorithm)
        environment = list(cmd_args.environment)
        algo_env = list(itertools.product(algorithm, environment))

        logdir = cmd_args.logdir
        prefix = cmd_args.prefix
        model_name = cmd_args.model_name

        for algo, env in algo_env:
            algo = get_algo_name(algo)
            if '-' in algo:
                config = get_config(algo.split('-')[-1], env)
                dist_config = get_config(algo, env)
                assert config, config
                assert dist_config, dist_config
                config = deep_update(config, dist_config)
            else:
                config = get_config(algo, env)
            main = pkg.import_main('train', algo)

            env_config = config['env']
            model_config = config['model']
            agent_config = config['agent']
            replay_config = config.get('buffer') or config.get('replay')
            agent_config['algorithm'] = algo
            if env:
                env_config['name'] = env
            model_name = change_config(
                cmd_args.kwargs, model_name, env_config, 
                model_config, agent_config, replay_config)
            agent_config['model_name'] = model_name
            if cmd_args.grid_search or cmd_args.trials > 1:
                gs = GridSearch(
                    env_config, model_config, agent_config, replay_config, 
                    main, n_trials=cmd_args.trials, logdir=logdir, dir_prefix=prefix,
                    separate_process=len(algo_env) > 1, delay=cmd_args.delay)

                if cmd_args.grid_search:
                    if 'sac' in algo:
                        processes += gs(tsallis_q=[1, 1.2, 1.5])
                    else:
                        processes += gs()
                else:
                    processes += gs()
            else:
                dir_prefix = prefix + '-' if prefix else prefix
                agent_config['root_dir'] = f'{logdir}/{dir_prefix}{env_config["name"]}/{agent_config["algorithm"]}'
                replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
                if len(algo_env) > 1:
                    p = Process(target=main,
                                args=(env_config, 
                                    model_config,
                                    agent_config, 
                                    replay_config))
                    p.start()
                    time.sleep(1)
                    processes.append(p)
                else:
                    main(env_config, model_config, agent_config, replay_config)
    [p.join() for p in processes]
