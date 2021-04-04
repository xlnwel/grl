from algo.dqn.train import *
# import time

# from core.tf_config import *
# from utility.utils import Every
# from utility.graph import video_summary
# from utility.run import Runner, evaluate
# from utility import pkg
# from env.func import create_env
# from replay.func import create_replay
# from core.dataset import create_dataset


# def train(agent, env, eval_env, replay):
#     def collect_and_learn(env, step, reset, **kwargs):
#         if reset:
#             kwargs['next_obs'] = env.prev_obs()
#         replay.add(**kwargs)
#         agent.learn_log(step)

#     step = agent.env_step
#     collect = lambda *args, **kwargs: replay.add(**kwargs)
#     runner = Runner(env, agent, step=step, nsteps=agent.LOG_PERIOD)
#     while not replay.good_to_learn():
#         step = runner.run(
#             action_selector=env.random_action,
#             step_fn=collect)

#     to_eval = Every(agent.EVAL_PERIOD)
#     print('Training starts...')
#     while step <= int(agent.MAX_STEPS):
#         start_step = step
#         start = time.time()
#         step = runner.run(step_fn=collect_and_learn)
#         agent.store(
#             env_step=agent.env_step,
#             train_step=agent.train_step,
#             fps=(step - start_step) / (time.time() - start))

#         eval_score, eval_epslen, video = evaluate(
#             eval_env, agent, record=agent.RECORD, size=(64, 64))
#         if agent.RECORD:
#             video_summary(f'{agent.name}/sim', video, step=step)
#         agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
#         agent.log(step)
#         agent.save()

# def main(env_config, model_config, agent_config, replay_config):
#     silence_tf_logs()
#     configure_gpu()

#     env = create_env(env_config)
#     assert env.n_envs == 1, \
#         f'n_envs({env.n_envs}) > 1 is not supported here as it messes with n-step'
#     eval_env_config = env_config.copy()
#     eval_env = create_env(eval_env_config)

#     replay = create_replay(replay_config)

#     create_model, Agent = pkg.import_agent(config=agent_config)
#     models = create_model(model_config, env)
#     am = pkg.import_module('agent', config=agent_config)
#     data_format = am.get_data_format(
#         env=env, replay_config=replay_config, 
#         agent_config=agent_config, model=models)
#     dataset = create_dataset(replay, env, data_format=data_format)

#     agent = Agent(
#         name=env.name,
#         config=agent_config, 
#         models=models, 
#         dataset=dataset, 
#         env=env)
    
#     agent.save_config(dict(
#         env=env_config,
#         model=model_config,
#         agent=agent_config,
#         replay=replay_config
#     ))
    
#     # train(agent, env, eval_env, replay)

#     # This training process is used for Mujoco tasks, following the same process as OpenAI's spinningup
#     # out = env.reset()
#     # obs, _, _, _ = out
#     # epslen = 0
#     # from utility.utils import Every
#     # to_log = Every(agent.LOG_PERIOD)
#     # for t in range(int(agent.MAX_STEPS)):
#     #     if t > 1e4:
#     #         action = agent(out)
#     #     else:
#     #         action = env.random_action()

#     #     out = env.step(action)
#     #     next_obs, reward, discount, reset = out
#     #     epslen += 1
#     #     replay.add(obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs)
#     #     obs = next_obs

#     #     if not discount or epslen == env.max_episode_steps:
#     #         agent.store(score=env.score(), epslen=env.epslen())
#     #         assert epslen == env.epslen(), f'{epslen} vs {env.epslen()}'
#     #         obs, _, _, _ = env.reset()
#     #         epslen = 0

#     #     if replay.good_to_learn() and t % agent.TRAIN_PERIOD == 0:
#     #         agent.learn_log(t)
#     #     if to_log(t):
#     #         eval_score, eval_epslen, _ = evaluate(eval_env, agent)

#     #         agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
#     #         agent.log(step=t)
#     #         agent.save()
