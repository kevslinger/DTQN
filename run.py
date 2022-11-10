import os
import time
import argparse
import torch
import wandb
from utils import random, agent_utils, env_processing, logging_utils

# envs
import gym, envs

try:
    import gym_pomdps
except ImportError:
    print(
        "WARNING: ``gym_pomdps`` is not installed. This means you cannot run an experiment with the HeavenHell or Hallway domain."
    )


def run_experiment(args):
    start_time = time.time()
    env = env_processing.make_env(args.env)
    eval_env = env_processing.make_env(args.env)
    device = torch.device(args.device)
    random.set_global_seed(args.seed, env, eval_env)

    save_dir = os.path.join(os.getcwd(), "policies", args.project_name, args.env)
    os.makedirs(save_dir, exist_ok=True)
    policy_path = os.path.join(
        save_dir,
        f"model={args.model}_env={args.env}_obsembed={args.obsembed}_inembed={args.inembed}_context={args.context}_heads={args.heads}_layers={args.layers}_"
        f"batch={args.batch}_gate={args.gate}_identity={args.identity}_history={args.history}_pos={args.pos}_seed={args.seed}",
    )

    agent = agent_utils.get_agent(
        args.model,
        env,
        eval_env,
        args.obsembed,
        args.inembed,
        args.buf_size,
        device,
        args.lr,
        args.batch,
        args.context,
        args.history,
        args.num_steps,
        # DTQN specific
        args.heads,
        args.layers,
        args.dropout,
        args.identity,
        args.gate,
        args.pos,
    )

    print(
        f"[ {logging_utils.timestamp()} ] Creating {args.model} with {sum(p.numel() for p in agent.policy_network.parameters())} parameters"
    )

    # Enjoy mode
    if args.render:
        agent.policy_network.load_state_dict(
            torch.load(policy_path, map_location="cpu")
        )
        agent.policy_network.eval()
        agent.exp_coef.val = 0
        while True:
            agent.evaluate(n_episode=1, render=True)
        exit(0)

    # If we have a checkpoint, load the checkpoint and resume training if more steps are needed.
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
            wandb_kwargs = {"resume": "must", "id": agent.load_checkpoint(policy_path)}
    # Begin training from scratch
    else:
        wandb_kwargs = {"resume": None}
        agent.prepopulate(50_000)

    if args.disable_wandb:
        logger = logging_utils.CSVLogger(policy_path)
    else:
        logging_utils.wandb_init(
            vars(args),
            [
                "model",
                "obsembed",
                "inembed",
                "context",
                "heads",
                "layers",
                "batch",
                "gate",
                "identity",
                "history",
                "pos",
            ],
            **wandb_kwargs,
        )
        logger = wandb

    # Pick up from where we left off in the checkpoint (or 0 if doesn't exist) until max steps
    for timestep in range(agent.num_steps, args.num_steps):
        agent.step()
        agent.train()

        if timestep % args.tuf == 0:
            agent.target_update()

        if timestep % args.eval_frequency == 0:
            sr, ret, length = agent.evaluate(args.eval_episodes)
            if agent.num_steps < len(agent.td_errors):
                td_error = agent.td_errors.sum() / agent.num_steps
                grad_norm = agent.grad_norms.sum() / agent.num_steps
                qmax = agent.qvalue_max.sum() / agent.num_steps
                qmean = agent.qvalue_mean.sum() / agent.num_steps
                qmin = agent.qvalue_min.sum() / agent.num_steps
                tmax = agent.target_max.sum() / agent.num_steps
                tmean = agent.target_mean.sum() / agent.num_steps
                tmin = agent.target_min.sum() / agent.num_steps
            else:
                td_error = agent.td_errors.mean()
                grad_norm = agent.grad_norms.mean()
                qmax = agent.qvalue_max.mean()
                qmean = agent.qvalue_mean.mean()
                qmin = agent.qvalue_min.mean()
                tmax = agent.target_max.mean()
                tmean = agent.target_mean.mean()
                tmin = agent.target_min.mean()

            if agent.num_evals < len(agent.episode_rewards):
                mean_reward = agent.episode_rewards.sum() / agent.num_evals
                mean_success_rate = agent.episode_successes.sum() / agent.num_evals
                mean_episode_length = agent.episode_lengths.sum() / agent.num_evals
            else:
                mean_reward = agent.episode_rewards.mean()
                mean_success_rate = agent.episode_successes.mean()
                mean_episode_length = agent.episode_lengths.mean()

            logger.log(
                {
                    "losses/TD_Error": td_error,
                    "losses/Grad_Norm": grad_norm,
                    "losses/Max_Q_Value": qmax,
                    "losses/Mean_Q_Value": qmean,
                    "losses/Min_Q_Value": qmin,
                    "losses/Max_Target_Value": tmax,
                    "losses/Mean_Target_Value": tmean,
                    "losses/Min_Target_Value": tmin,
                    "results/Return": ret,
                    "results/Mean_Return": mean_reward,
                    "results/Success_Rate": sr,
                    "results/Mean_Success_Rate": mean_success_rate,
                    "results/Episode_Length": length,
                    "results/Mean_Episode_Length": mean_episode_length,
                    "results/Hours": (time.time() - start_time) / 3600,
                },
                step=agent.num_steps,
            )

            if args.verbose:
                curtime = logging_utils.timestamp()
                print(
                    f"[ {curtime} ] Eval #{agent.num_evals} Success Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}, Hours: {((time.time() - start_time) / 3600):.2f}"
                )

        if args.save_policy and not timestep % 50_000:
            torch.save(agent.policy_network.state_dict(), policy_path)

        if (
            args.time_limit is not None
            and ((time.time() - start_time) / 3600) > args.time_limit
        ):
            print(
                f"Reached time limit. Saving checkpoint with {agent.num_steps} steps completed."
            )
            agent.save_checkpoint(wandb.run.id, policy_path)
            exit(0)
    # In case we finish before time limit, we need to output a mini checkpoint so as not to repeat ourselves
    agent.save_mini_checkpoint(wandb_id=wandb.run.id, checkpoint_dir=policy_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-name",
        type=str,
        default="DTQN-Neurips2022",
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
        choices=["DTQN", "DQN", "DRQN", "ADRQN", "DARQN", "ATTN"],
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
        help="Number of contexts to store in replay buffer.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5_000,
        help="How many timesteps between agent evaluations.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for each evluation.",
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
        default=64,
        help="The dimensionality of the network. In the transformer, this is referred to as `d_model`.",
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
        choices=[1, 0, "sin"],
        help="The type of positional encodings to use.",
    )

    run_experiment(parser.parse_args())
