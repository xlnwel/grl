from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.utils import Every, TempStore
from utility.graph import video_summary
from utility.run import Runner, evaluate
from utility.timer import Timer
from utility import pkg
from env.func import create_env


def train(agent, env, eval_env, buffer):
    def collect(env, step, reset, next_obs, **kwargs):
        buffer.add(**kwargs)

    step = agent.env_step
    runner = Runner(env, agent, step=step, nsteps=agent.N_STEPS)
    # if step == 0 and agent.is_obs_or_reward_normalized:
    #     print('Start to initialize running stats...')
    #     for _ in range(50):
    #         runner.run(action_selector=env.random_action, step_fn=collect)
    #         agent.update_obs_rms(buffer['obs'])
    #         agent.update_reward_rms(buffer['reward'], buffer['discount'])
    #         buffer.reset()
    # buffer.clear()
    runner.step = step
    # print("Initial running stats:", *[f'{k:.4g}' for k in agent.get_running_stats() if k])
    to_log = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < agent.MAX_STEPS:
        agent.before_run(env)
        for _ in range(agent.N_PI):
            start_env_step = agent.env_step
            with Timer('env') as et:
                step = runner.run(step_fn=collect)
            agent.store(fps=(step-start_env_step)/et.last())
            # NOTE: normalizing rewards here may introduce some inconsistency 
            # if normalized rewards is fed as an input to the network.
            # One can reconcile this by moving normalization to collect 
            # or feeding the network with unnormalized rewards.
            # The latter is adopted in our implementation, 
            # but the following line currently doesn't store a copy of 
            # unnormalized rewards
            agent.update_reward_rms(buffer['reward'], buffer['discount'])
            buffer.update('reward', agent.normalize_reward(buffer['reward']), field='all')
            agent.record_last_env_output(runner.env_output)
            value = agent.compute_value()
            buffer.finish(value)

            start_train_step = agent.train_step
            with Timer('train') as tt:
                agent.learn_log(step)
            agent.store(tps=(agent.train_step-start_train_step)/tt.last())
            agent.update_obs_rms(buffer['obs'])
            buffer.reset()
            if to_log(agent.train_step) and 'score' in agent._logger:
                agent.log(step)
                agent.save()
            if to_eval(agent.train_step) or step > agent.MAX_STEPS:
                with TempStore(agent.get_states, agent.reset_states):
                    with Timer('eval') as eval_time:
                        scores, epslens, video = evaluate(
                            eval_env, agent, record=agent.RECORD, size=(128, 128))
                    if agent.RECORD:
                        video_summary(f'{agent.name}/sim', video, step=step)
                    agent.store(eval_score=scores, eval_epslen=epslens, eval_time=eval_time.total())
            agent.store(env_time=et.total(), train_time=tt.total())

        # auxiliary phase
        buffer.compute_aux_data_with_func(agent.compute_aux_data)
        if agent.AUX_RECOMPUTE:
            value = agent.compute_value()
            buffer.aux_finish(value)
        else:
            buffer.reshape_to_sample()
            buffer.set_ready()

        with Timer('aux_time') as at:
            agent.aux_learn_log(step)
        agent.store(atps=(agent.N_AUX_EPOCHS * agent.N_AUX_MBS)/at.last())
        buffer.aux_reset()

def main(env_config, model_config, agent_config, buffer_config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config['precision'])

    create_model, Agent = pkg.import_agent(config=agent_config)
    Buffer = pkg.import_module('buffer', config=agent_config).Replay

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

    models = create_model(model_config, env)

    buffer_config['n_envs'] = env.n_envs
    buffer_config['state_keys'] = models.state_keys
    buffer = Buffer(buffer_config)
    
    agent = Agent( 
        config=agent_config, 
        models=models, 
        dataset=buffer,
        env=env)

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        buffer=buffer_config
    ))

    train(agent, env, eval_env, buffer)

    if use_ray:
        ray.shutdown()
