import time
import logging
from copy import deepcopy
from multiprocessing import Process

from utility.utils import product_flatten_dict
logger = logging.getLogger(__name__)

class GridSearch:
    def __init__(self, env_config, model_config, agent_config, replay_config, 
                train_func, n_trials=1, logdir='logs', dir_prefix='', 
                separate_process=False, delay=1):
        self.env_config = env_config
        self.model_config = model_config
        self.agent_config = agent_config
        self.replay_config = replay_config
        self.train_func = train_func
        self.n_trials = n_trials
        self.logdir = logdir
        self.dir_prefix = dir_prefix
        self.separate_process = separate_process
        self.delay=delay

        self.processes = []

    def __call__(self, **kwargs):
        self._setup_root_dir()
        if kwargs == {} and self.n_trials == 1 and not self.separate_process:
            # if no argument is passed in, run the default setting
            self.train_func(self.env_config, self.model_config, self.agent_config, self.replay_config)        
        else:
            # do grid search
            self.agent_config['model_name'] = ''
            self._change_config(**kwargs)

        return self.processes

    def _setup_root_dir(self):
        if self.dir_prefix:
            self.dir_prefix += '-'
        self.agent_config['root_dir'] = (f'{self.logdir}/'
                                        f'{self.env_config["name"]}/'
                                        f'{self.agent_config["algorithm"]}')

    def _change_config(self, **kwargs):
        kw_list = product_flatten_dict(**kwargs)
        for d in kw_list:
            # deepcopy to avoid unintended conflicts
            env_config = deepcopy(self.env_config)
            model_config = deepcopy(self.model_config)
            agent_config = deepcopy(self.agent_config)
            replay_config = deepcopy(self.replay_config)

            for k, v in d.items():
                # search k in configs
                configs = {}
                for name, config in zip(['env', 'model', 'agent', 'replay'], 
                        [env_config, model_config, agent_config, replay_config]):
                    if k in config:
                        configs[name] = config
                assert configs != [], f'{k} does not appear in any of configs'
                logger.info(f'{k} appears in the following configs: '
                            f'{list([n for n, _ in configs.items()])}.\n')
                # change value in config
                for config in configs.values():
                    if isinstance(config[k], dict):
                        config[k].update(v)
                    else:
                        config[k] = v
                
                if agent_config['model_name']:
                    agent_config['model_name'] += '-'
                # add "key=value" to model name
                agent_config['model_name'] += f'{k}={v}'

            for i in range(1, self.n_trials+1):
                ec = deepcopy(env_config)
                mc = deepcopy(model_config)
                ac = deepcopy(agent_config)
                rc = deepcopy(replay_config)

                if self.n_trials > 1:
                    ac['model_name'] += f'-trial{i}' if ac['model_name'] else f'trial{i}'
                if 'seed' in ec:
                    ec['seed'] = 1000 * i
                if 'video_path' in env_config:
                    ec['video_path'] = (f'{ac["root_dir"]}/'
                                        f'{ac["model_name"]}/'
                                        f'{ec["video_path"]}')
                p = Process(target=self.train_func, args=(ec, mc, ac, rc))
                p.start()
                self.processes.append(p)
                time.sleep(self.delay)   # ensure sub-processs starts in order