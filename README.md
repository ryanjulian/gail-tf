# gail-tf
Tensorflow implementation of Generative Adversarial Imitation Learning (and 
behavior cloning)

**disclaimers**: some code is borrowed from @openai/baselines

# TL;DR - What's new in this branch?
The discriminator in the original GAIL compares the expert policy to the generator policy via trajectory rollouts
that contain both the actions and the states (observations) at every time step.

In this setting, however, the discriminator only compares rollouts based on the 5 endeffector features, i.e.
3D vectors pointing from the root (torso) to the head, hands and feet.

## Getting started
First, you can optionally train an expert policy via the following command, or use a pretrained policy.
```bash
# Issue this command in the project home folder
export MPI_NCPU=4  # number of CPU cores to use in parallel
PYTHONPATH=.:$PYTHONPATH mpirun -np $MPI_NCPU python3 gailtf/baselines/trpo_mpi/run_mujoco.py
```
Using this command, a folder named `checkpoint` will be created in which the TensorFlow checkpoints are stored that
contain the learned weights of the policy network. By default, every 100 iterations, a checkpoint will be created
so that the training script can be stopped before all 1e8 episodes are samples (default setting).

Next, sample trajectories (rollouts) by using the expert policy (we provided one in `baselines/expert` for the featurized humanoid environment):
```bash
# Issue this command in baselines/trpo_mpi
python3 run_mujoco.py --env_id HumanoidFeaturized-v1 --hidden --task sample_trajectory --sample_stochastic False --load_model_path ../../expert/trpo.HumanoidFeaturized.0.00-15200
```

This will create a Pickle file inside your current directory. Now copy this pkl file to `rollout` (project home folder).
Then issue the following command to run GAIL using this expert:
```bash
# Issue this command in the project home folder
python3 main.py
```

## Instructions for Running Stuff on Google Cloud Platform
### All the magic happens in the Docker and therefore in .deep-rl-docker (which is the Docker's home folder)
#### (This can be accessed from outside, obviously :) )
Train a TRPO expert in the ryanjulian/gail-tf repo:
```
PYTHONPATH=.:$PYTHONPATH mpirun -np 24 python3 gailtf/baselines/trpo_mpi/run_mujoco.py --num_cpu 24
```
To visualize the policy at a given checkpoint, issue:
```
# copy the policy (which is often stored in checkpoint-folders, e.g. ~/gail-tf/checkpoint/trpo_gail.HumanoidFeaturized.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001)
CUDA_VISIBLE_DEVICES="" python3 run_rl.py --env HumanoidFeaturized-v1 --load examples/trpo.HumanoidFeaturized.0.00-390 --hidden-size 64 --hidden-layers 2
```
To create a roll-out pickle file from a TRPO expert (given by a checkpoint file):
```
python3 gailtf/baselines/trpo_mpi/run_mujoco.py --env_id HumanoidFeaturized-v1 --task sample_trajectory --sample_stochastic False --load_model_path ../../expert/trpo.HumanoidFeaturized.0.00-3570
```
To create a roll-out from mocap data:
```
# Issue this command in uscresl/humanoid-gail:
python3 mocap/generate_rollouts.py
```
To run GAIL on a roll-out PKL file:
```
mpirun -np 24 python3 main.py --expert_path rollout/stochastic.trpo.HumanoidFeaturized.0.00.pkl --num_cpu 24 | tee training.log

# bad practice
sudo mpirun --allow-run-as-root -np 24 python3 main.py --expert_path rollout/stochastic.trpo.HumanoidFeaturized.0.00.pkl --num_cpu 24
```


## What's GAIL?
- model free imtation learning -> low sample efficiency in training time
  - model-based GAIL: End-to-End Differentiable Adversarial Imitation Learning
- Directly extract policy from demonstrations
- Remove the RL optimization from the inner loop od inverse RL
- Some work based on GAIL:
  - Inferring The Latent Structure of Human Decision-Making from Raw Visual 
    Inputs
  - Multi-Modal Imitation Learning from Unstructured Demonstrations using 
  Generative Adversarial Nets
  - Robust Imitation of Diverse Behaviors
  
## Requirements
- python==3.5.2
- mujoco-py==0.5.7
- tensorflow==1.1.0
- gym==0.9.3

## Run the code
I separate the code into two parts: (1) Sampling expert data, (2) Imitation 
learning with GAIL/BC

### Step 1: Generate expert data

#### Train the expert policy using PPO/TRPO, from openai/baselines
Ensure that `$GAILTF` is set to the path to your gail-tf repository, and 
`$ENV_ID` is any valid OpenAI gym environment (e.g. Hopper-v1, HalfCheetah-v1, 
etc.)

##### Configuration
``` bash
export GAILTF=/path/to/your/gail-tf
export ENV_ID="Hopper-v1"
export BASELINES_PATH=$GAILTF/gailtf/baselines/ppo1 # use gailtf/baselines/trpo_mpi for TRPO
export SAMPLE_STOCHASTIC="False"            # use True for stochastic sampling
export STOCHASTIC_POLICY="False"            # use True for a stochastic policy
export PYTHONPATH=$GAILTF:$PYTHONPATH       # as mentioned below
cd $GAILTF
```

##### Train the expert
```bash
python3 $BASELINES_PATH/run_mujoco.py --env_id $ENV_ID
```

The trained model will save in ```./checkpoint```, and its precise name will
vary based on your optimization method and environment ID. Choose the last 
checkpoint in the series.

```bash
export PATH_TO_CKPT=./checkpoint/trpo.Hopper.0.00/trpo.Hopper.00-900
```

#### Use an existing running humanoid policy
A provided expert policy for the `Humanoid-v1` environment is given in the `expert` folder.
```bash
export PATH_TO_CKPT=./expert/trpo.Humanoid.0.00-10100
export ENV_ID="HumanoidFeaturized-v1"
```
By selecting the "featurized" `Humanoid-v1` environment, the latter command will ensure that during sampling,
the 5 endeffector (head, hands, feet) feature vectors are computed and included in the trajectory Pickle file.

##### Sample from the generated expert policy
```bash
python3 $BASELINES_PATH/run_mujoco.py --env_id $ENV_ID --task sample_trajectory --sample_stochastic $SAMPLE_STOCHASTIC --load_model_path $PATH_TO_CKPT
```

This will generate a pickle file that store the expert trajectories in 
```./XXX.pkl``` (e.g. deterministic.ppo.Hopper.0.00.pkl)

```bash
export PICKLE_PATH=./stochastic.trpo.Hopper.0.00.pkl
```

### Step 2: Imitation learning

#### Imitation learning via GAIL

```bash
python3 main.py --env_id $ENV_ID --expert_path $PICKLE_PATH
```

Usage:
```bash
--env_id:          The environment id
--num_cpu:         Number of CPU available during sampling
--expert_path:     The path to the pickle file generated in the [previous section]()
--traj_limitation: Limitation of the exerpt trajectories
--g_step:          Number of policy optimization steps in each iteration
--d_step:          Number of discriminator optimization steps in each iteration
--num_timesteps:   Number of timesteps to train (limit the number of timesteps to interact with environment)
```

To view the summary plots in TensorBoard, issue
```bash
tensorboard --logdir $GAILTF/log
```

##### Evaluate your GAIL agent
```bash
python3 main.py --env_id $ENV_ID --task evaluate --stochastic_policy $STOCHASTIC_POLICY --load_model_path $PATH_TO_CKPT --expert_path $PICKLE_PATH
```

#### Imitation learning via Behavioral Cloning
```bash
python3 main.py --env_id $ENV_ID --algo bc --expert_path $PICKLE_PATH
```

##### Evaluate your BC agent
```bash
python3 main.py --env_id $ENV_ID --algo bc --task evalaute --stochastic_policy $STOCHASTIC_POLICY --load_model_path $PATH_TO_CKPT --expert_path $PICKLE_PATH
```

## Results

Note: The following hyper-parameter setting is the best that I've tested (simple 
grid search on setting with 1500 trajectories), just for your information.

The different curves below correspond to different expert size (1000,100,10,5).

- Hopper-v1 (Average total return of expert policy: 3589)

```bash
python3 main.py --env_id Hopper-v1 --expert_path baselines/ppo1/deterministic.ppo.Hopper.0.00.pkl --g_step 3 --adversary_entcoeff 0
```

![](misc/Hopper-true-reward.png)

- Walker-v1 (Average total return of expert policy: 4392)

```bash
python3 main.py --env_id Walker2d-v1 --expert_path baselines/ppo1/deterministic.ppo.Walker2d.0.00.pkl --g_step 3 --adversary_entcoeff 1e-3
```

![](misc/Walker2d-true-reward.png)

- HalfCheetah-v1 (Average total return of expert policy: 2110)

For HalfCheetah-v1 and Ant-v1, using behavior cloning is needed:
```bash
python3 main.py --env_id HalfCheetah-v1 --expert_path baselines/ppo1/deterministic.ppo.HalfCheetah.0.00.pkl --pretrained True --BC_max_iter 10000 --g_step 3 --adversary_entcoeff 1e-3
```

![](misc/HalfCheetah-true-reward.png)

**You can find more details [here](https://github.com/andrewliao11/gail-tf/blob/master/misc/exp.md), 
GAIL policy [here](https://drive.google.com/drive/folders/0B3fKFm-j0RqeRnZMTUJHSmdIdlU?usp=sharing), 
and BC policy [here](https://drive.google.com/drive/folders/0B3fKFm-j0RqeVFFmMWpHMk85cUk?usp=sharing)**

## Hacking
We don't have a pip package yet, so you'll need to add this repo to your 
PYTHONPATH manually.
```bash
export PYTHONPATH=/path/to/your/repo/with/gailtf:$PYTHONPATH
```

## TODO
* Create pip package/setup.py
* Make style PEP8 compliant
* Create requirements.txt
* Depend on openai/baselines directly and modularize modifications
* openai/robotschool support


## Reference
- Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning, [[arxiv](https://arxiv.org/abs/1606.03476)]
- @openai/imitation
- @openai/baselines
