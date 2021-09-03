import os, sys, signal
import psutil
import ray

from utility.display import pwc
from core.tf_config import *


def sigint_shutdown_ray():
    """ Shutdown ray when the process is terminated by ctrl+C """
    def handler(sig, frame):
        if ray.is_initialized():
            ray.shutdown()
            pwc('ray has been shutdown by sigint', color='cyan')
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

def cpu_affinity(name=None):
    resources = ray.get_resource_ids()
    if 'CPU' in resources:
        cpus = [v[0] for v in resources['CPU']]
        psutil.Process().cpu_affinity(cpus)
    else:
        cpus = []
        # raise ValueError(f'No cpu is available')
    if name:
        pwc(f'CPUs corresponding to {name}: {cpus}', color='cyan')

def gpu_affinity(name=None):
    gpus = ray.get_gpu_ids()
    if gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    if name:
        pwc(f'GPUs corresponding to {name}: {gpus}', color='cyan')

def get_num_cpus():
    return len(ray.get_resource_ids()['CPU'])

def config_actor(name, config, gpu_idx=0):
    cpu_affinity(name)
    gpu_affinity(name)
    silence_tf_logs()
    num_cpus = get_num_cpus()
    configure_threads(num_cpus, num_cpus)
    use_gpu = configure_gpu(gpu_idx)
    if not use_gpu and 'precision' in config:
        config['precision'] = 32
    configure_precision(config.get('precision', 32))
