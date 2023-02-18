import os
import argparse
from time import time, sleep
from typing import Optional

import torch
import wandb
from gym import Env

from utils import env_processing, epsilon_anneal
from utils.agent_utils import MODEL_MAP, get_agent
from utils.random import set_global_seed
from utils.logging_utils import RunningAverage, get_logger, timestamp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-name",
        type=str,
        default="DTQN-test",
        help="The project name (for wandb) or directory name (for local logging) to store the results.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Use `--disable-wandb` to log locally.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time limit allowed for job. Useful for some cluster jobs such as slurm.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DTQN",
        choices=list(MODEL_MAP.keys()),
        help="Network model to use.",
    )
    parser.add_argument(
        "--env", type=str, default="DiscreteCarFlag-v0", help="Domain to use."
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2_000_000,
        help="Number of steps to train the agent.",
    )
    parser.add_argument(
        "--tuf",
        type=int,
        default=10_000,
        help="How many steps between each (hard) target network update.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--buf-size",
        type=int,
        default=500_000,
        help="Number of timesteps to store in replay buffer.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=1_000,
        help="How many timesteps between agent evaluations.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for each evaluation.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Pytorch device to use."
    )
    parser.add_argument(
        "--context",
        type=int,
        default=50,
        help="For DRQN and DTQN, the context length to use to train the network.",
    )
    parser.add_argument(
        "--obsembed",
        type=int,
        default=8,
        help="For discrete observation domains only. The number of features to give each observation.",
    )
    parser.add_argument(
        "--inembed",
        type=int,
        default=128,
        help="The dimensionality of the network. In the transformer, this is referred to as `d_model`.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=-1,
        help="The maximum number of steps allowed in the environment. If `env` has a `max_episode_steps`, this will be inferred. Otherwise, this argument must be supplied.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed to use.")
    parser.add_argument(
        "--save-policy",
        action="store_true",
        help="Use this to save the policy so you can load it later for rendering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out evaluation results as they come in to the console.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enjoy mode (NOTE: must have a trained policy saved).",
    )
    parser.add_argument(
        "--history",
        action="store_false",
        help="Supplying this argument turns off intermediate q-value prediction.",
    )
    # DTQN-Specific
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of heads to use for the transformer.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of transformer blocks to use for the transformer.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability."
    )
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--gate",
        type=str,
        default="res",
        choices=["res", "gru"],
        help="Combine step to use.",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Whether or not to use identity map reordering.",
    )
    parser.add_argument(
        "--pos",
        default=1,
        choices=[1, "1", 0, "0", "sin"],
        help="The type of positional encodings to use.",
    )
    parser.add_argument(
        "--bag-size", type=int, default=0, help="The size of the persistent memory bag."
    )
    # For slurm
    parser.add_argument(
        "--slurm-job-id",
        default=0,
        type=str,
        help="The `$SLURM_JOB_ID` assigned to this job",
    )

    return parser.parse_args()


def evaluate(agent, eval_env: Env, eval_episodes: int, render: Optional[bool] = None):
    """Evaluate the network for n_episodes"""
    # Set networks to eval mode (turns off dropout, etc.)
    agent.eval_on()

    total_reward = 0
    num_successes = 0
    total_steps = 0

    for _ in range(eval_episodes):
        agent.context_reset(eval_env.reset())
        done = False
        ep_reward = 0
        if render:
            eval_env.render()
            sleep(0.5)
        while not done:
            action = agent.get_action(epsilon=0.0)
            obs_next, reward, done, info = eval_env.step(action)
            agent.observe(obs_next, action, reward, done)
            ep_reward += reward
            if render:
                eval_env.render()
                if done:
                    print(f"Episode terminated. Episode reward: {ep_reward}")
                sleep(0.5)
        total_reward += ep_reward
        total_steps += agent.context.timestep
        if info.get("is_success", False) or ep_reward > 0:
            num_successes += 1

    # Set networks back to train mode
    agent.eval_off()
    return (
        num_successes / eval_episodes,
        total_reward / eval_episodes,
        total_steps / eval_episodes,
    )


def train(
    agent,
    env: Env,
    eval_env: Env,
    total_steps: int,
    eps,
    eval_frequency: int,
    eval_episodes: int,
    policy_path: str,
    save_policy: bool,
    logger,
    mean_reward: RunningAverage,
    mean_success_rate: RunningAverage,
    mean_episode_length: RunningAverage,
    time_remaining: Optional[int],
    verbose: bool = False,
):
    start_time = time()
    # Turn on train mode
    agent.eval_off()
    agent.context_reset(env.reset())

    for timestep in range(agent.num_train_steps, total_steps):
        step(agent, env, eps)

        agent.train()
        eps.anneal()

        if timestep % eval_frequency == 0:
            sr, ret, length = evaluate(agent, eval_env, eval_episodes)
            mean_success_rate.add(sr)
            mean_reward.add(ret)
            mean_episode_length.add(length)

            logger.log(
                {
                    "losses/TD_Error": agent.td_errors.mean(),
                    "losses/Grad_Norm": agent.grad_norms.mean(),
                    "losses/Max_Q_Value": agent.qvalue_max.mean(),
                    "losses/Mean_Q_Value": agent.qvalue_mean.mean(),
                    "losses/Min_Q_Value": agent.qvalue_min.mean(),
                    "losses/Max_Target_Value": agent.target_max.mean(),
                    "losses/Mean_Target_Value": agent.target_mean.mean(),
                    "losses/Min_Target_Value": agent.target_min.mean(),
                    "results/Return": ret,
                    "results/Mean_Return": mean_reward.mean(),
                    "results/Success_Rate": sr,
                    "results/Mean_Success_Rate": mean_success_rate.mean(),
                    "results/Episode_Length": length,
                    "results/Mean_Episode_Length": mean_episode_length.mean(),
                    "results/Hours": (time() - start_time) / 3600,
                },
                step=timestep,
            )

            if verbose:
                print(
                    f"[ {timestamp()} ] Training Steps: {timestep}, Success Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}, Hours: {((time() - start_time) / 3600):.2f}"
                )

        if save_policy and timestep % 50_000 == 0:
            torch.save(agent.policy_network.state_dict(), policy_path)

        if time_remaining and time() - start_time >= time_remaining:
            print(
                f"Reached time limit. Saving checkpoint with {agent.num_train_steps} steps completed."
            )

            agent.save_checkpoint(
                policy_path,
                wandb.run.id if logger == wandb else None,
                mean_reward,
                mean_success_rate,
                mean_episode_length,
                eps,
            )
            return


def step(agent, env, eps):
    action = agent.get_action(epsilon=eps.val)
    next_obs, reward, done, info = env.step(action)

    # OpenAI Gym TimeLimit truncation: don't store it in the buffer as done
    if info.get("TimeLimit.truncated", False):
        buffer_done = False
    else:
        buffer_done = done

    agent.observe(next_obs, action, reward, buffer_done)

    if done:
        agent.replay_buffer.flush()
        agent.context_reset(env.reset())


def prepopulate(agent, prepop_steps: int, env: Env):
    timestep = 0
    while timestep < prepop_steps:
        agent.context_reset(env.reset())
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)

            # OpenAI Gym TimeLimit truncation: don't store it in the buffer as done
            if info.get("TimeLimit.truncated", False):
                buffer_done = False
            else:
                buffer_done = done

            agent.observe(next_obs, action, reward, buffer_done, bag=False)
            timestep += 1
        agent.replay_buffer.flush()


def run_experiment(args):
    start_time = time()
    # Create envs, set seed, create RL agent
    env = env_processing.make_env(args.env)
    eval_env = env_processing.make_env(args.env)
    device = torch.device(args.device)
    set_global_seed(args.seed, env, eval_env)

    eps = epsilon_anneal.LinearAnneal(1.0, 0.1, args.num_steps // 10)

    agent = get_agent(
        args.model,
        env,
        args.obsembed,
        args.inembed,
        args.buf_size,
        device,
        args.lr,
        args.batch,
        args.context,
        args.max_episode_steps,
        args.history,
        args.tuf,
        args.discount,
        # DTQN specific
        args.heads,
        args.layers,
        args.dropout,
        args.identity,
        args.gate,
        args.pos,
        args.bag_size,
    )

    print(
        f"[ {timestamp()} ] Creating {args.model} with {sum(p.numel() for p in agent.policy_network.parameters())} parameters"
    )

    # Create logging dir
    policy_save_dir = os.path.join(os.getcwd(), "policies", args.project_name, args.env)
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(
        policy_save_dir,
        f"model={args.model}_env={args.env}_obsembed={args.obsembed}_inembed={args.inembed}_context={args.context}_heads={args.heads}_layers={args.layers}_"
        f"batch={args.batch}_gate={args.gate}_identity={args.identity}_history={args.history}_pos={args.pos}_bag={args.bag_size}_seed={args.seed}",
    )

    # Enjoy mode
    if args.render:
        agent.policy_network.load_state_dict(
            torch.load(policy_path, map_location="cpu")
        )
        evaluate(agent, env, 1_000_000, render=True)

    # If there is already a saved checkpoint, load it and resume training if more steps are needed
    # Or exit early if we have already finished training.
    if os.path.exists(policy_path + "_mini_checkpoint.pt"):
        steps_completed = agent.load_mini_checkpoint(policy_path)["step"]
        print(
            f"Found a mini checkpoint that completed {steps_completed} training steps."
        )
        if steps_completed >= args.num_steps:
            print(f"Removing checkpoint and exiting...")
            if os.path.exists(policy_path + "_checkpoint.pt"):
                os.remove(policy_path + "_checkpoint.pt")
            exit(0)
        else:
            (
                wandb_id,
                mean_reward,
                mean_success_rate,
                mean_episode_length,
                eps_val,
            ) = agent.load_checkpoint(policy_path)
            eps.val = eps_val
            wandb_kwargs = {"resume": "must", "id": wandb_id}
    # Begin training from scratch
    else:
        wandb_kwargs = {"resume": None}
        # Prepopulate the replay buffer
        prepopulate(agent, 50_000, env)
        mean_reward = RunningAverage(10)
        mean_success_rate = RunningAverage(10)
        mean_episode_length = RunningAverage(10)

    # Logging setup
    logger = get_logger(policy_path, args, wandb_kwargs)

    time_remaining = (
        args.time_limit * 3600 - (time() - start_time) if args.time_limit else None
    )

    train(
        agent,
        env,
        eval_env,
        args.num_steps,
        eps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path,
        args.save_policy,
        logger,
        mean_reward,
        mean_success_rate,
        mean_episode_length,
        time_remaining,
        args.verbose,
    )

    # Save a small checkpoint if we finish training to let following runs know we are finished
    agent.save_mini_checkpoint(
        checkpoint_dir=policy_path, wandb_id=wandb.run.id if logger == wandb else None
    )


if __name__ == "__main__":
    run_experiment(get_args())
