import functools
import signal
import sys
import numpy as np

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from core.dataset import create_dataset
from utility.utils import Every, TempStore
from utility.graph import video_summary
from utility.run import Runner, evaluate
from utility.timer import Timer
from utility import pkg
from env.func import create_env


def train(agent, env, eval_env, buffer):
    collect_fn = pkg.import_module('agent', algo=agent.name).collect
    collect = functools.partial(collect_fn, buffer)

    step = agent.env_step
    runner = Runner(env, agent, step=step, nsteps=agent.N_STEPS)
    
    if step == 0 and agent.is_obs_normalized:
        print('Start to initialize running stats...')
        for _ in range(10):
            runner.run(action_selector=env.random_action, step_fn=collect)
            agent.update_obs_rms(np.concatenate(buffer['obs']))
            agent.update_reward_rms(buffer['reward'], buffer['discount'])
            buffer.reset()
        buffer.clear()
        agent.env_step = runner.step
        agent.save(print_terminal_info=True)

    runner.step = step
    # print("Initial running stats:", *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_log = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    rt = Timer('run')
    tt = Timer('train')
    et = Timer('eval')
    lt = Timer('log')
    print('Training starts...')
    while step < agent.MAX_STEPS:
        start_env_step = agent.env_step
        agent.before_run(env)
        with rt:
            step = runner.run(step_fn=collect)
        agent.store(fps=(step-start_env_step)/rt.last())
        # NOTE: normalizing rewards here may introduce some inconsistency 
        # if normalized rewards is fed as an input to the network.
        # One can reconcile this by moving normalization to collect 
        # or feeding the network with unnormalized rewards.
        # The latter is adopted in our implementation. 
        # However, the following line currently doesn't store
        # a copy of unnormalized rewards
        agent.update_reward_rms(buffer['reward'], buffer['discount'])
        buffer.update('reward', agent.normalize_reward(buffer['reward']), field='all')
        agent.record_last_env_output(runner.env_output)
        value = agent.compute_value()
        buffer.finish(value)

        start_train_step = agent.train_step
        with tt:
            agent.learn_log(step)
        agent.store(tps=(agent.train_step-start_train_step)/tt.last())
        buffer.reset()

        if to_eval(agent.train_step) or step > agent.MAX_STEPS:
            with TempStore(agent.get_states, agent.reset_states):
                with et:
                    eval_score, eval_epslen, video = evaluate(
                        eval_env, agent, n=agent.N_EVAL_EPISODES, 
                        record=agent.RECORD, size=(64, 64))
                if agent.RECORD:
                    video_summary(f'{agent.name}/sim', video, step=step)
                agent.store(
                    eval_score=eval_score, 
                    eval_epslen=eval_epslen)

        if to_log(agent.train_step) and agent.contains_stats('score'):
            with lt:
                agent.store(**{
                    'stats/train_step': agent.train_step,
                    'time/run': rt.total(), 
                    'time/train': tt.total(),
                    'time/eval': et.total(),
                    'time/log': lt.total(),
                    'time/run_mean': rt.average(), 
                    'time/train_mean': tt.average(),
                    'time/eval_mean': et.average(),
                    'time/log_mean': lt.average(),
                })
                agent.log(step)
                agent.save()

def main(env_config, model_config, agent_config, buffer_config, train=train):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config['precision'])

    create_model, Agent = pkg.import_agent(config=agent_config)
    Buffer = pkg.import_module('buffer', config=agent_config).Buffer

    use_ray = env_config.get('n_workers', 1) > 1
    if use_ray:
        import ray
        from utility.ray_setup import sigint_shutdown_ray
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config, force_envvec=True)
    eval_env_config = env_config.copy()
    if 'num_levels' in eval_env_config:
        eval_env_config['num_levels'] = 0
    if 'seed' in eval_env_config:
        eval_env_config['seed'] += 1000
    eval_env_config['n_workers'] = 1
    for k in list(eval_env_config.keys()):
        # pop reward hacks
        if 'reward' in k:
            eval_env_config.pop(k)
    eval_env = create_env(eval_env_config, force_envvec=True)

    def sigint_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        env.close()
        eval_env.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    models = create_model(model_config, env)

    buffer_config['n_envs'] = env.n_envs
    buffer_config['state_keys'] = models.state_keys
    buffer_config['use_dataset'] = buffer_config.get('use_dataset', False)
    buffer = Buffer(buffer_config)
    
    if buffer_config['use_dataset']:
        am = pkg.import_module('agent', config=agent_config)
        data_format = am.get_data_format(
            env=env, batch_size=buffer.batch_size,
            sample_size=agent_config.get('sample_size'),
            store_state=agent_config.get('store_state'),
            state_size=models.state_size)
        dataset = create_dataset(buffer, env, 
            data_format=data_format, one_hot_action=False)
    else:
        dataset = buffer

    agent = Agent(
        config=agent_config, 
        models=models, 
        dataset=dataset,
        env=env)

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        buffer=buffer_config
    ))

    train(agent, env, eval_env, buffer)

    if use_ray:
        env.close()
        eval_env.close()
        ray.shutdown()
