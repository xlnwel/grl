## Algorithms implemented in each folder

- airl: AIRL
- apex: Ape-X for distributed training
- daac: Decoupled advantage actor-critic
- dqn: Raibow(except c51)
- dreamer: Dreamer for DMC
- fqf: FQF
- gail: GAIL
- iqn: IQN
- mrdqn: masked recurrent DQN with retrace(𝝀) and 𝛽-LOO(from Reactor)
- mriqn: masked recurrent IQN with retrace(𝝀)
- ppg: PPG with FNN
- ppo: PPO with FNN
- ppo2: PPO with masked lstm
- rnd: Random network distillation
- sac: SAC, TAC
- sacd: SAC for discrete action space and CNN
- sacdiqn: SAC + IQN for discrete action space and CNN
- seed: SEED for distributed training

**Note:** Do not run value-based algorithms on the 'procgen' suite. Instead, run the Ape-X version and fill in the corresponding *config.yaml as you need.