# Episodic Multi-agent Reinforcement Learning with Curiosity-driven Exploration

## Note
This codebase accompanies paper [**Episodic Multi-agent Reinforcement Learning with Curiosity-driven Exploration(EMC)**](https://proceedings.neurips.cc/paper/2021/file/1e8ca836c962598551882e689265c1c5-Paper.pdf), 
 and is based on  [PyMARL](https://github.com/oxwhirl/pymarl), [SMAC](https://github.com/oxwhirl/smac), and [QPLEX](https://github.com/wjh720/QPLEX) codebases which are open-sourced. We use the modified SMAC of QPLEX, which is illustrated in the folder `SMAC_ENV`.

The implementation of the following methods can also be found in this codebase, which are finished by the authors of following papers:

- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/pdf/2008.01062)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement learning](https://arxiv.org/abs/1905.05408)
- [**Qatten**: Qatten: A General Framework for Cooperative Multiagent Reinforcement Learning](https://arxiv.org/abs/2002.03939)
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over. We use a modified version of SMAC since we test 17 maps, which is illustrated in the folder of `EMC_smac_env`.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 
We evaluate our method in three environments: gridworld("gridworld_reversed"), Predator and Prey("pred_prey_punish"),17 maps in SMAC("sc2"). We use the default settings in SMAC, and the **results in our paper use Version SC2.4.6.2.69232**.
|  Task config  | Algorithm config|
|  ----  | ----  |
| gridworld_reversed | EMC_toygame |
| pred_prey_punish  | EMC_toygame |
|sc2|EMC_sc2|

Map names for SMAC:
{2s3z,3s5z,1c3s5z,5m_vs_6m,10m_vs_11m,27m_vs_30m,3s5z_vs_3s6z,MMM2,2s_vs_1sc,3s_vs_5z,6h_vs_8z,bane_vs_bane,2c_vs_64zg,corridor,5s10z,7s7z,1c3s8z_vs_1c3s9z.}

For three super hard maps: 3s5z_vs_3s6z,6h_vs_8z,corridor in SMAC, we use fine tuned parameters since they need different scales of exploration.

We provide two ways to run experiments.

#### 1.  Use bash file
We recommand to run experiments by this way. Just replace the default "tasks" and "algos" with your desired ones in EMC/pymarl/src/run_curiosity_test.sh, and then: 
```shell
    cd pymalr/src
    bash run_curiosity_test.sh
```

#### 2. Use command line.
For example, to train EMC on didactic task `gridworld `, run the following command:

```shell
python3 src/main.py --config=EMC_toygame --env-config=gridworld_reversed \
with env_args.map_name=reversed 
```
To train EMC on Predator and Prey, run the following command:
```shell
python3 src/main.py --config=EMC_toygame --env-config=pred_prey_punish \
with env_args.map_name=origin 
```

To train EMC on SC2 setting tasks, run the following command:
```shell
python3 src/main.py --config=EMC_sc2 --env-config=sc2 \
with env_args.map_name=2s3z 
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`





