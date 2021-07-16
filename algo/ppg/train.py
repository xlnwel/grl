import functools
import numpy as np

from utility.utils import Every, TempStore
from utility.graph import video_summary
from utility.run import Runner, evaluate
from utility.timer import Timer
from utility import pkg
from algo.ppo.train import main


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
    at = Timer('aux')
    print('Training starts...')
    while step < agent.MAX_STEPS:
        agent.before_run(env)
        for _ in range(agent.N_PI):
            start_env_step = agent.env_step
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
            if step > agent.MAX_STEPS or (to_log(agent.train_step) and 'score' in agent._logger):
                with lt:
                    agent.store(**{
                        'train_step': agent.train_step,
                        'time/run': rt.total(), 
                        'time/train': tt.total(),
                        'time/eval': et.total(),
                        'time/log': lt.total(),
                        'time/aux': at.total(),
                        'time/run_mean': rt.average(), 
                        'time/train_mean': tt.average(),
                        'time/eval_mean': et.average(),
                        'time/log_mean': lt.average(),
                        'time/aux_mean': at.average(),
                    })
                    agent.log(step)
                    agent.save()
            if to_eval(agent.train_step) or step > agent.MAX_STEPS:
                with TempStore(agent.get_states, agent.reset_states):
                    with et:
                        scores, epslens, video = evaluate(
                            eval_env, agent, record=agent.RECORD, size=(128, 128))
                    if agent.RECORD:
                        video_summary(f'{agent.name}/sim', video, step=step)
                    agent.store(eval_score=scores, eval_epslen=epslens, eval_time=et.total())
            
            if step >= agent.MAX_STEPS:
                break
        if step >= agent.MAX_STEPS:
            break

        # auxiliary phase
        buffer.compute_aux_data_with_func(agent.compute_aux_data)
        if agent.AUX_RECOMPUTE:
            value = agent.compute_value()
            buffer.aux_finish(value)
        else:
            buffer.reshape_to_sample()
            buffer.set_ready()

        with at:
            agent.aux_learn_log(step)
        agent.store(atps=(agent.N_AUX_EPOCHS * agent.N_AUX_MBS)/at.last())
        buffer.aux_reset()

main = functools.partial(main, train=train)
