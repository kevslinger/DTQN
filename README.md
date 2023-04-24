# Deep Transformer Q-Networks for Partially Observable Reinforcement Learning

Deep Transformer Q-Network (DTQN) is an extension of [DQN](https://www.nature.com/articles/nature14236) and [DRQN](https://arxiv.org/abs/1507.06527) designed to encode an agent's history effectively for solving partially observable reinforcement learning tasks.
Our architecture is built from a Transformer Decoder (like [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)).
DTQN is a sequence-to-sequence model; that is, given a history of the agent's interactions (either observation or action-observation) with the environment, DTQN outputs a sequence of Q-values.
Each element in the output sequence represents the Q-values for each action, given the agent's history up to that point.
For instance, the 3rd vector of Q-values was generated based on only the first three interactions between the agent and the environment.
This method allows us to train much more efficiently than a Sequence-to-One method, since we get `Sequence` more data points to use for training.
Our results providence strong evidence indicating a transformer can solve partially observable domains faster than previous recurrent approaches.
Our paper is now publicly available on arXiv! 
You can read it [here](https://arxiv.org/abs/2206.01078).

Please note that we are continuing to work on this repository to extend and improve DTQN.
As such, the code in this branch may not reflect the code submitted with the original paper.
We will keep the [paper](https://github.com/kevslinger/DTQN/tree/paper) branch frozen with the code from the original paper, and only update it as needed to fix bugs.

## Table of Contents
- [Deep Transformer Q-Networks for Partially Observable Reinforcement Learning](#deep-transformer-q-networks-for-partially-observable-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Creating Environment](#creating-environment)
    - [Installing gym-gridverse](#installing-gym-gridverse)
    - [Installing rl-parsers](#installing-rl-parsers)
    - [Installing gym-pomdps](#installing-gym-pomdps)
  - [Running Experiments](#running-experiments)
    - [Experiment Argument Details](#experiment-argument-details)
    - [Ablations](#ablations)
      - [Transformer Decoder Structure](#transformer-decoder-structure)
      - [Positional Encodings](#positional-encodings)
      - [Intermediate Q-value prediction](#intermediate-q-value-prediction)
  - [Citing DTQN](#citing-dtqn)
  - [Contributing](#contributing)

## Installation

To run our code, you must first set up your environment.
We recommend using `virtualenv` with `pip` to set up a virtual environment and dependency manager.
For our experiments, we use `python3.8`; while other versions of python will probably work, they are untested and we cannot guarantee the same performance.

### Creating Environment

First, create a virtual environment with `python3.8`, and then install the dependencies. This can be done with:

```bash
virtualenv venv -p 3.8
source venv/bin/activate
pip install -r requirements.txt
```

This basic setup will allow you to run the `Car Flag` and `Memory Cards` environments from our paper.
If you only have interest in running those domains, skip to the [Running Experiments](#running-experiments) section.
Otherwise, you need to install the `gym-gridverse` repo to run the gridverse experiments, and the `gym-pomdps` and `rl-parsers` repos to run the classic POMDPs experiments.

### Installing gym-gridverse

To install gym-gridverse, you need to clone the `gym-gridverse` [github repo](https://github.com/abaisero/gym-gridverse.git).
Finally, once you have the source code, pip install it into the virtual enviroment.
This can be done as follows:

```bash
git clone git@github.com:abaisero/gym-gridverse.git
cd gym-gridverse
pip install .
```

### Installing rl-parsers

If you also wish to run the Hallway and HeavenHell experiments, you will need to install `rl-parsers` and `gym-pomdps`.
First, install `rl-parsers` by cloning the [github repo](https://github.com/abaisero/rl-parsers.git) and installing it into your python virtual environment.

```bash
git clone git@github.com:abaisero/rl-parsers.git
cd rl-parsers
pip install -e .
```

### Installing gym-pomdps

Next, install `gym-pomdps` by cloning the [github repo](https://github.com/abaisero/gym-pomdps.git) and installing it into your python virtual environment.

```bash
git clone git@github.com:abaisero/gym-pomdps.git
cd gym-pomdps
pip install .
```

Tada! Now, you should be ready to run experiments.

## Running Experiments

Our experiment script is `run.py`. 
The arguments are explained at the bottom of the file.
Using the command
```shell
python run.py
```

You will run the default settings, which will run DTQN on the Car Flag domain as tested in the paper.

Here we provide a mapping from domain name as used in the paper to domain name used in the code:

- Hallway: `POMDP-hallway-episodic-v0`
- HeavenHell: `POMDP-heavenhell_3-episodic-v0`
- Gridverse memory 5x5: `gv_memory.5x5.yaml`
- Gridverse memory 7x7: `gv_memory.7x7.yaml`
- Gridverse Memory 9x9: `gv_memory.9x9.yaml`
- Gridverse four rooms 7x7: `gv_memory_four_rooms.7x7.yaml`
- Car Flag: `DiscreteCarFlag-v0`
- Memory Cards: `Memory-5-v0`

If you want to change the environment, use the `--envs` flag.
You can add multiple `envs` as long as they share the same observation and action space. 
In that case, the agent will collect data with policy rollouts by randomly sampling one of the provided envs.
During evaluation, it will evaluate the policy separately on each environment, and record the results for each domain separately.
To reproduce our results, you may need to change the `inembed` flag as well.
For `POMDP-hallway-episodic-v0`, `POMDP-heavenhell_3-episodic-v0`, and `DiscreteCarFlag-v0` domains, we used `--inembed 64`.
For all others tasks, we use `--inembed 128`.
For instance, to reproduce our Gridverse memory 7x7 experiment, you can use the command:

```shell
python run.py --envs gv_memory.7x7.yaml --inembed 128
```

Which will run for 2,000,000 timesteps.
In our paper, we run our experiments with random seeds 1, 2, 3, 4, 5.

### Experiment Argument Details

We use [weights and biases](https://wandb.ai) to log our results.
If you do not have a weights and biases account, we recommend you get one!
However, if you still do not want to use weights and biases, you can use the `--disable-wandb` flag.
Then your results will be stored to a CSV file in `policies/<project_name>/<env>/<config>.csv`.

If you do not have access to a gpu, set `--device cpu` to train on CPU.

If you want command line outputs to view training results, use `--verbose`.

| Argument | Description |
| -------- | ----------- |
| `--project-name` | The project name (for wandb) or directory name (for local logging) to store the results. |
| `--disable-wandb` | Use `--disable-wandb` to log locally. |
| `--time-limit` | Time limit allowed for job. Useful for some cluster jobs such as slurm. |
| `--model` | Network model to use. |
| `--envs` | Domain to use. You can supply multiple domains, but they must have the same observation and action space. With multiple environments, the agent will sample a new one on each episode reset for conducting policy rollouts and collection experience. During evaluation, it will perform the same evaluation for each domain (Note: this may significantly slow down your run! Consider increasing the eval-frequency or reducing the eval-episodes). |
| `--num-steps` | Number of steps to train the agent. |
| `--tuf` | How many steps between each (hard) target network update. |
| `--lr` | Learning rate for the optimizer. |
| `--batch` | Batch size. |
| `--buf-size` | Number of timesteps to store in replay buffer. Note that we store the max length episodes given by the environment, so episodes that take longer will be padded at the end. This does not affect training but may affect the number of real observations in the buffer. |
| `--eval-frequency` | How many training timesteps between agent evaluations. |
| `--eval-episodes` | Number of episodes for each evaluation period. |
| `--device` | Pytorch device to use. |
| `--context` | For DRQN and DTQN, the context length to use to train the network. |
| `--obs-embed` | For discrete observation domains only. The number of features to give each observation. |
| `--a-embed` | The number of features to give each action. A value of 0 will prevent the policy from using the previous action. |
| `--in-embed` | The dimensionality of the network. In the transformer, this is referred to as `d_model`. |
| `--max-episode-steps` | The maximum number of steps allowed in the environment. If `env` has a `max_episode_steps`, this will be inferred. Otherwise, this argument must be supplied. |
| `--seed` | The random seed to use. |
| `--save-policy` | Use this to save the policy so you can load it later for rendering. |
| `--verbose` | Print out evaluation results as they come in to the console. |
| `--render` | Enjoy mode (NOTE: must have a trained policy saved). |
| `--history` | This is how many (intermediate) Q-values we use to train for each context. To turn off intermediate Q-value prediction, set `--history 1`. To use the entire context, set history equal to the context length. |
| `--heads` | Number of heads to use for the transformer. |
| `--layers` | Number of transformer blocks to use for the transformer. |
| `--dropout` | Dropout probability. |
| `--discount` | Discount factor. |
| `--gate` | Combine step to use. |
| `--identity` | Whether or not to use identity map reordering. |
| `--pos` | The type of positional encodings to use. |
| `--bag-size` | The size of the persistent memory bag. |
| `--slurm-job-id` | The `$SLURM_JOB_ID` assigned to this job. |
| |


### Ablations

#### Transformer Decoder Structure 
To run DTQN with the GRU-like gating mechanism, use `--gate gru`.

To run DTQN with identity map reordering, use `--identity`.

To use both GRU-like gating as well as identity map reordering, use `--gate gru --identity`.

#### Positional Encodings
To run DTQN with the sinusoidal positional encodings, use `--pos sin`.

To run DTQN without positional encodings, use `--pos none`.

#### Intermediate Q-value prediction
To run DTQN without intermediate Q-value prediction, use `--history`.

## Citing DTQN

To cite this paper/code in publications, please use the following bibtex:

```bibtex
@article{esslinger2022dtqn,
  title = {Deep Transformer Q-Networks for Partially Observable Reinforcement Learning},
  author = {Esslinger, Kevin and Platt, Robert and Amato, Christopher},
  journal= {arXiv preprint arXiv:2206.01078},
  year = {2022},
}

```

## Contributing

Feel free to open a pull request with updates, contributions, and modifications.