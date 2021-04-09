## Distributed Deep Reinforcement Learning (D2RL)

A modulated and versatile library for deep reinforcement learning, implemented in *Tensorflow2.4*.

## Instructions

If you want to know how an algorithm works, simply study `agent.py` and `nn.py`.

If you want to run some algorithm, see [Get Start](#example).

## Current Implemented Algorithms/Networks

Many algorithms are simply improvements/minor modifications of their predecessors. Therefore, instead of implementing them as separate algorithms, we provide options to turn these on/off in `config.yaml`. Refer to [algo/readme.md](https://github.com/xlnwel/d2rl/blob/master/algo/readme.md) for more details.

### On Policy RL

All implementation details from OpenAI's baselines are implemented for PPO families 

- [x] DAAC
- [x] PPG
- [x] PPO (with FNN)
- [x] PPO2 (with masked LSTM)
- [x] RND

### Off Policy RL

- [x] FQF
- [x] IQN
- [x] M-DQN
- [x] M-IQN
- [x] Rainbow
- [x] Reactor
- [x] Retrace(𝝀)
- [x] RDQN (Recurrent DQN with masked LSTM)
- [x] SAC (w/ or w/o adaptive temperature)
- [x] SACD (SAC for discrete action space)
- [x] TAC
- [x] TBO (Transformed Bellman Operator)

### Distributed RL

*ray1.1.0* is used for distributed training. 

- [x] Ape-X
- [x] R2D2
- [x] SEED

### Model-Based RL

- [x] Dreamer<sup>[1](dreamer)</sup>

### Imitation Learning

- [x] AIRL
- [x] GAIL

### Networks

- [x] CBAM
- [x] Convolutional Attention
- [x] DNC (Differentiable Neural Computer)
- [x] Mask LSTM
- [x] MobileNet Block
- [x] Multi-Head Attention
- [x] Randomized Network (for Generalization)
- [x] ResNet
- [x] SENet
- [x] SN (Spectral Norm)

<a name="example"></a>## Get Started

```shell
python run/train.py algo -e env
```

For available `algo`, please refer to the folder names in `/algo`. To run distributed algorithms, `algo` should be of form `{distributed_architecture}-{algorithm}`. For example, if you want to run Ape-X with DQN, replace 'algo' with `apex-dqn`. Configures are set in `*config.yaml` in each fold following convention `{algo}_{env_suite}_config.yaml`, where `algo` is omitted when there is no ambiguous and `env_suite` is omitted when there is no corresponding suite name. `env` follows convention `{suite}_{name}`, current available `suite` includes `[atari, procgen, dmc]`.  

Examples

```shell
python run/train.py ppo -e LunarLander-v2   # no suite specified
python run/train.py ppo -e procgen_coinrun  # procgen suite
python run/train.py iqn -e procgen_coinrun
python run/train.py apex-iqn -e procgen_coinrun
```

By default, all the checkpoints and loggings are saved to `./logs/{env}/{algo}/{model_name}/`.

You can also make some simple changes to `*config.yaml` from command line

```
# change learning rate to 0.0001, `lr` must appear in `*config.yaml`
python run/train.py ppo -e procgen_coinrun -kw lr=0.0001
```
See more examples in file `start`

## Acknowledge

I'd like to especially thank @danijar for his great help with Dreamer.

## Reference Papers

Machado, Marlos C., Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and Michael Bowling. 2018. “Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.” IJCAI International Joint Conference on Artificial Intelligence 2018-July (2013): 5573–77.

Espeholt, Lasse, Raphaël Marinier, Piotr Stanczyk, Ke Wang, and Marcin Michalski. 2019. “SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference,” 1–19. http://arxiv.org/abs/1910.06591.

Badia, Adrià Puigdomènech, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, and Charles Blundell. 2020. “Agent57: Outperforming the Atari Human Benchmark.” http://arxiv.org/abs/2003.13350.

Burda, Yuri, Harrison Edwards, Amos Storkey, and Oleg Klimov. 2018. “Exploration by Random Network Distillation,” 1–17. http://arxiv.org/abs/1810.12894.

Pardo, Fabio, Arash Tavakoli, Vitaly Levdik, and Petar Kormushev. 2018. “Time Limits in Reinforcement Learning.” 35th International Conference on Machine Learning, ICML 2018 9: 6443–52.

Jaderberg, Max, Wojciech M. Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio Garcia Castañeda, Charles Beattie, et al. 2019. “Human-Level Performance in 3D Multiplayer Games with Population-Based Reinforcement Learning.” Science 364 (6443): 859–65. https://doi.org/10.1126/science.aau6249.

Hafner, Danijar, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. 2019. “Learning Latent Dynamics for Planning from Pixels.” 36th International Conference on Machine Learning, ICML 2019 2019-June: 4528–47.

Graves, Alex, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, et al. 2016. “Hybrid Computing Using a Neural Network with Dynamic External Memory.” Nature 538 (7626): 471–76. https://doi.org/10.1038/nature20101.

Kapturowski, Steven, Georg Ostrovski, John Quan, and Will Dabney. 2019. “Recurrent Experience Replay in Distributed Reinforcement Learning.” In ICLR, 1–19.

Horgan, Dan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado van Hasselt, and David Silver. 2018. “Distributed Prioritized Experience Replay.” In ICLR, 1–19. http://arxiv.org/abs/1803.00933.

Haarnoja, Tuomas, Aurick Zhou, Pieter Abbeel, and Sergey Levine. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” 35th International Conference on Machine Learning, ICML 2018 5: 2976–89.

Munos, Rémi, Thomas Stepleton, Anna Harutyunyan, and Marc G. Bellemare. 2016. “Safe and Efficient Off-Policy Reinforcement Learning.” Advances in Neural Information Processing Systems, no. Nips: 1054–62.

Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. “Proximal Policy Optimization Algorithms.” ArXiv, 1–12.

Cobbe, Karl, Jacob Hilton, Oleg Klimov, and John Schulman. 2020. “Phasic Policy Gradient.” http://arxiv.org/abs/2009.04416.

Haarnoja, Tuomas, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, et al. 2018. “Soft Actor-Critic Algorithms and Applications.” http://arxiv.org/abs/1812.05905.

Christodoulou, Petros. 2019. “Soft Actor-Critic for Discrete Action Settings,” 1–7. http://arxiv.org/abs/1910.07207.

Haarnoja, Tuomas, Haoran Tang, Pieter Abbeel, and Sergey Levine. 2017. “Reinforcement Learning with Deep Energy-Based Policies.” 34th International Conference on Machine 
Learning, ICML 2017 3: 2171–86.

Vieillard, Nino, Olivier Pietquin, and Matthieu Geist. 2020. “Munchausen Reinforcement Learning,” no. NeurIPS. http://arxiv.org/abs/2007.14430.

Howard, Andrew G., Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. 2017. “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.” http://arxiv.org/abs/1704.04861.

Sandler, Mark, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang Chieh Chen. 2018. “MobileNetV2: Inverted Residuals and Linear Bottlenecks.” Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 4510–20. https://doi.org/10.1109/CVPR.2018.00474.

Howard, Andrew, Mark Sandler, Bo Chen, Weijun Wang, Liang Chieh Chen, Mingxing Tan, Grace Chu, et al. 2019. “Searching for MobileNetV3.” Proceedings of the IEEE International Conference on Computer Vision 2019-October: 1314–24. https://doi.org/10.1109/ICCV.2019.00140.

He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. “Deep Residual Learning for Image Recognition.” Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition 2016-December: 770–78. https://doi.org/10.1109/CVPR.2016.90.

He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. “Identity Mappings in Deep Residual Networks.” Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 9908 LNCS: 630–45. https://doi.org/10.1007/978-3-319-46493-0_38.

Tan, Mingxing, and Quoc V. Le. 2019. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” 36th International Conference on Machine Learning, ICML 2019 2019-June: 10691–700.

Graves, Alex, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, et al. 2016. “Hybrid Computing Using a Neural Network with Dynamic External Memory.” Nature 538 (7626): 471–76. https://doi.org/10.1038/nature20101.

Hsin, Carol. 2016. “Implementation and Optimization of Differentiable Neural Computers.” https://web.stanford.edu/class/cs224n/reports/2753780.pdf.

Dzmitry Bahdanau, KyungHyun Cho Yoshua Bengio. 2015. “Neural Machine Translation by Jointly Learning to Align and Translate.” Microbes and Infection 11 (3): 367–73. https://doi.org/10.1016/j.micinf.2008.12.015.

Luong, Minh Thang, Hieu Pham, and Christopher D. Manning. 2015. “Effective Approaches to Attention-Based Neural Machine Translation.” Conference Proceedings - EMNLP 2015: Conference on Empirical Methods in Natural Language Processing, 1412–21. https://doi.org/10.18653/v1/d15-1166.

Xu, Kelvin, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S Zemel, and Yoshua Bengio. 2014. “Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.” https://doi.org/10.1109/72.279181.

Woo, Sanghyun, Jongchan Park, Joon Young Lee, and In So Kweon. 2018. “CBAM: Convolutional Block Attention Module.” Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 11211 LNCS: 3–19. https://doi.org/10.1007/978-3-030-01234-2_1.

Hu, Jie, Li Shen, Samuel Albanie, Gang Sun, and Enhua Wu. 2020. “Squeeze-and-Excitation Networks.” IEEE Transactions on Pattern Analysis and Machine Intelligence 42 (8): 2011–23. https://doi.org/10.1109/TPAMI.2019.2913372.

Espeholt, Lasse, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Boron Yotam, et al. 2018. “IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.” 35th International Conference on Machine Learning, ICML 2018 4: 2263–84.

Hafner, Danijar, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. 2020. “Dream to Control: Learning Behaviors by Latent Imagination.” ICLR, 1–20. http://arxiv.org/abs/1912.01603.

Haarnoja, Tuomas, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, et al. 2018. “Soft Actor-Critic Algorithms and Applications.” http://arxiv.org/abs/1812.05905.

Engstrom, Logan, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. 2019. “Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO.” ICLR, no. January.

Lee, Kimin, Kibok Lee, Jinwoo Shin, and Honglak Lee. 2020. “Network Randomization: A Simple Technique for Generalization in Deep Reinforcement Learning.” Iclr 2020, 1–22. http://arxiv.org/abs/1910.05396.

**Please let me know if I miss any references.**

## Reference Repositories

https://github.com/openai/baselines

https://github.com/google/dopamine

https://github.com/deepmind/dnc

https://github.com/deepmind/trfl

https://github.com/google-research/seed_rl

https://github.com/danijar/dreamer

https://github.com/microsoft/FQF

https://github.com/rwightman/pytorch-image-models

https://github.com/juntang-zhuang/Adabelief-Optimizer

<a name="dreamer">1</a>: dreamer was tested before, but some dependent code such as `Runner` has been changed since then. Due to the expiration of my mojuco liscence, I can no longer test it and I decide to leave it as it is.