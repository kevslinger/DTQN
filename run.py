import argparse
import torch
import wandb
from random import random
from time import time
from utils import agent_utils, env_processing, logging_utils, epsilon_anneal
from utils.agent_utils import MODEL_MAP
from utils.random import set_global_seed

try:
    import gym_pomdps
except ImportError:
    print(
        "WARNING: ``gym_pomdps`` is not installed. This means you cannot run an experiment with the HeavenHell or "
        "Hallway domain. "
    )

START_TIME = time()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-name",
        type=str,
        default="DTQN-test",
        help="The project name (for wandb) or directory name (for local logging) to store the results.",
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
        help="Number of contexts to store in replay buffer.",
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
        "--device", type=str, default="mps", help="Pytorch device to use."
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
        "--discount", type=float, default=0.99, help="Discount factor."
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

    return parser.parse_args()


def evaluate(agent, timestep, env, eval_frequency, eval_episode):
    if timestep % eval_frequency: return
    agent.eval_on()
    for _ in range(eval_episode):
        ep_reward, episode_length = 0, 0
        obs, done = env.reset(), False
        agent.context_reset()
        while not done:
            episode_length += 1
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        agent.episode_rewards.add(ep_reward)
        agent.episode_lengths.add(episode_length)

        wandb.log(
            {
                "losses/TD_Error": agent.td_errors.mean(),
                "losses/Grad_Norm": agent.grad_norms.mean(),
                "losses/Max_Q_Value": agent.qvalue_max.mean(),
                "losses/Mean_Q_Value": agent.qvalue_mean.mean(),
                "losses/Min_Q_Value": agent.qvalue_min.mean(),
                "losses/Max_Target_Value": agent.target_max.mean(),
                "losses/Mean_Target_Value": agent.target_mean.mean(),
                "losses/Min_Target_Value": agent.target_min.mean(),
                "results/Eval_Return": agent.episode_rewards.mean(),
                "results/Eval_Episode_Length": agent.episode_lengths.mean(),
            },
            timestep,
        )
    agent.eval_off()
    if not timestep: return
    t = time() - START_TIME
    print('{} hours, {} minutes, {} seconds elapsed'.format(t // 3600, t % 3600 // 60, t % 60))
    t_est = t * 1e6 / timestep
    print('{} hours, {} minutes, {} seconds per M step'.format(t_est // 3600, t_est % 3600 // 60, t_est % 60))


def rollout(agent, env, eval_env, steps, eps, train=True, eval_frequency=5000, eval_episodes=10):
    done, cur_obs = True, None
    for i in range(steps):
        if done:
            cur_obs = env.reset()
            agent.context_reset()
        if random() < eps.val:
            action = env.action_space.sample()
        else:
            action = agent.get_action(cur_obs)
        obs, reward, done, _ = env.step(action)
        agent.store_transition(cur_obs, obs, action, reward, done, i)
        cur_obs = obs
        if eps: eps.anneal()
        if train:
            agent.train()
            evaluate(agent, i, eval_env, eval_frequency, eval_episodes)


def run_experiment(args):
    env = env_processing.make_env(args.env)
    eval_env = env_processing.make_env(args.env)
    device = torch.device(args.device)
    set_global_seed(args.seed, env, eval_env)

    eps = epsilon_anneal.LinearAnneal(1.0, 0.1, args.num_steps // 10)

    agent = agent_utils.get_agent(
        args.model,
        env,
        args.obsembed,
        args.inembed,
        args.buf_size,
        device,
        args.lr,
        args.batch,
        args.context,
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
    )

    if not agent:
        print('method name', args.model, 'not found.')
        exit(0)

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
    )

    # prepopulate
    rollout(agent, env, eval_env, 50000, epsilon_anneal.Constant(1.0), train=False)

    # train and eval
    rollout(agent, env, eval_env, args.num_steps, eps, train=True, eval_frequency=args.eval_frequency,
            eval_episodes=args.eval_episodes)


if __name__ == "__main__": run_experiment(get_args())
