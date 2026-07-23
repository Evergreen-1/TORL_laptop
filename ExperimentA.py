"""
Experiment A - Joshua

Dataset:  mujoco/walker2d/medium-v0  (Minari)
Env:      Walker2d-v5                (gymnasium + mujoco)
Algos:    DT, CQL, CDT
Noise:    0%, 25%, 50%, 75% Gaussian injection
Seeds:    [0, 1, 2, 3, 4]

Usage:
  python ExperimentA.py --algo dt --device cuda --noise 0.0 --seed 0 --dt_steps 100000
  python ExperimentA.py --full
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

#general algo + logging
import numpy as np
import torch
import wandb
import torch_directml

#dataset
import minari
import gymnasium as gym

#run dt
import torch.nn as nn
from torch.nn import functional as F
from dt import DecisionTransformer, pad_along_axis
from tqdm import trange

#cql
from cql import TanhGaussianPolicy, FullyConnectedQFunction, ContinuousCQL, ReplayBuffer

#cdt
from cdt import CDT, CDTTrainer, WalkerCDTTrainer

#video of training
from gymnasium.wrappers import RecordVideo

#checkpoints
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# DEVICE DETECTION

def get_device(requested: str = "auto"):
    """
    Returns a device object (not always a string for DirectML).
    All .to(device) calls in the pipeline accept both strings and device objects.
    """
    if requested == "directml":
        
        if torch_directml.is_available():
            print(f"[Device] DirectML: {torch_directml.device()}")
            return torch_directml.device()
        else:
            print("[Device] DirectML not available — falling back to CPU.")
            return "cpu"

    if requested == "auto":
        # Try DirectML first on Windows, then CUDA, then CPU
        try:
            if torch_directml.is_available():
                print(f"[Device] Auto-selected DirectML")
                return torch_directml.device()
        except ImportError:
            pass

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA: {name}")
            return "cuda"

        print("[Device] Using CPU.")
        return "cpu"

    return requested  # explicit "cpu" or "cuda" passed by user


# ARGUMENT PARSING

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",      choices=["dt", "cql", "cdt"], default="dt")
    parser.add_argument("--noise",     type=float, default=0.0)
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--dataset",   type=str,   default="mujoco/walker2d/medium-v0") #mujoco/walker2d/medium-v0
    parser.add_argument("--dt_steps",  type=int,   default=100_000)
    parser.add_argument("--cql_steps", type=int,   default=1_000_000)
    parser.add_argument("--steps", type=int,   default=100_000)
    parser.add_argument("--full",      action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)     #loading checkpoint filepath: checkpoints/...
    parser.add_argument("--resume", action="store_true")            #resuming training 
    return parser.parse_args()



# SEED MANAGEMENT (similar to dt set seed)

def set_seed(seed: int, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.reset(seed=seed)   # gymnasium API uses reset(seed=) not env.seed()
        env.action_space.seed(seed)

# DATASET LOADING (Minari) _______________________________________________________________________________________________

# D4RL reference scores for walker2d normalisation (from d4rl/infos.py)
# These are fixed constants
WALKER2D_REF_MIN = 1.629      # average return of random policy
WALKER2D_REF_MAX = 4592.3     # average return of expert policy #6992.717

def get_normalized_score(raw_return: float) -> float:
    """
    D4RL-equivalent normalised score,
    Returns a value in roughly [0, 100], where 100 = expert level.
    """
    return 100.0 * (raw_return - WALKER2D_REF_MIN) / (WALKER2D_REF_MAX - WALKER2D_REF_MIN)


def load_minari_dataset(dataset_id: str):
    """
    Downloads (first run) and loads a Minari dataset.
    Returns:
      dataset_dict   -flat dict with keys matching D4RL format:
                      observations, actions, rewards, next_observations, terminals
      env            -a live gymnasium env recovered from the dataset
      traj_list      -list of per-episode dicts (for DT/CDT sequence models)
    """


    print(f"[Minari] Loading '{dataset_id}' ...")
    try:
        ds = minari.load_dataset(dataset_id)
    except Exception:
        print(f"[Minari] Dataset not cached locally — downloading ...")
        minari.download_dataset(dataset_id)
        ds = minari.load_dataset(dataset_id)

    env = ds.recover_environment()

    obs_list, nobs_list, act_list, rew_list, term_list = [], [], [], [], []
    traj_list = []

    for ep in ds.iterate_episodes():
        T = len(ep.actions)
        o   = np.array(ep.observations[:-1], dtype=np.float32)   # s_0 .. s_{T-1}
        no  = np.array(ep.observations[1:],  dtype=np.float32)   # s_1 .. s_T
        a   = np.array(ep.actions,           dtype=np.float32)
        r   = np.array(ep.rewards,           dtype=np.float32)

        # terminals: True only if the episode ended by reaching a terminal state
        # (not a timeout). Minari stores both terminations and truncations.
        terms = np.array(ep.terminations, dtype=np.float32)

        obs_list.append(o);  nobs_list.append(no)
        act_list.append(a);  rew_list.append(r);  term_list.append(terms)

        # Build return-to-go for sequence models
        rtg = np.cumsum(r[::-1])[::-1].astype(np.float32)
        traj_list.append({
            "observations": o,
            "actions":      a,
            "rewards":      r,
            "returns":      rtg,
        })

    flat = {
        "observations":      np.concatenate(obs_list),
        "actions":           np.concatenate(act_list),
        "rewards":           np.concatenate(rew_list),
        "next_observations": np.concatenate(nobs_list),
        "terminals":         np.concatenate(term_list),
    }

    n = flat["observations"].shape[0]
    print(f"[Minari] {n:,} transitions | {len(traj_list)} episodes loaded.")
    return flat, env, traj_list



# NOISE INJECTION _______________________________________________________________________________________________

def generate_noise_dict(dataset: dict, noise_fraction: float, seed: int, noise_obs: bool = True, noise_rew: bool = True):
    if noise_fraction == 0.0:
        return None

    rng = np.random.default_rng(seed)
    n = dataset["observations"].shape[0]
    n_corrupt = int(n * noise_fraction)
    idx = rng.choice(n, size=n_corrupt, replace=False)

    obs_std = dataset["observations"].std(axis=0)
    rew_std = float(dataset["rewards"].std())

    obs_noise = (rng.normal(0, 0.1 * obs_std, (n_corrupt, dataset["observations"].shape[1])).astype(np.float32) if noise_obs else np.zeros((n_corrupt, dataset["observations"].shape[1]), dtype=np.float32))
    rew_noise = (rng.normal(0, 0.1 * rew_std, n_corrupt).astype(np.float32) if noise_rew else np.zeros(n_corrupt, dtype=np.float32))

    print(f"[Noise] {noise_fraction*100:.0f}% corrupted with seed {seed}")

    return {
        "idx_to_obs_noise": dict(zip(idx.tolist(), obs_noise)),
        "idx_to_rew_noise": dict(zip(idx.tolist(), rew_noise)),
        "idx_set": set(idx.tolist()),
    }

#def inject_gaussian_noise(dataset: dict, noise_fraction: float, seed: int) -> dict:
    """
    Adds Gaussian noise to observations and rewards of a random subset of
    transitions. Uses an isolated RNG so training seeds are unaffected.
    Noise scale = 0.1 * per-feature std (proportional to data scale).
    """
    if noise_fraction == 0.0:
        print("[Noise] 0% — clean dataset.")
        return dataset

    rng = np.random.default_rng(seed)
    dataset = {k: v.copy() for k, v in dataset.items()}

    n = dataset["observations"].shape[0]
    n_corrupt = int(n * noise_fraction)
    idx = rng.choice(n, size=n_corrupt, replace=False)

    obs_std = dataset["observations"].std(axis=0)
    rew_std  = float(dataset["rewards"].std())

    dataset["observations"][idx] += rng.normal(
        0, 0.1 * obs_std, (n_corrupt, dataset["observations"].shape[1])
    ).astype(np.float32)
    dataset["rewards"][idx] += rng.normal(
        0, 0.1 * rew_std, n_corrupt
    ).astype(np.float32)

    print(f"[Noise] {noise_fraction*100:.0f}% — {n_corrupt:,}/{n:,} transitions corrupted.")
    return dataset

def inject_gaussian_noise(dataset: dict, noise_dict) -> dict:
    if noise_dict is None:
        print("[Noise] 0% — clean dataset.")
        return dataset

    dataset = {k: v.copy() for k, v in dataset.items()}
    idx = np.array(sorted(noise_dict["idx_set"]))

    obs_noise = np.stack([noise_dict["idx_to_obs_noise"][i] for i in idx])
    rew_noise = np.array([noise_dict["idx_to_rew_noise"][i] for i in idx])

    dataset["observations"][idx] += obs_noise
    dataset["rewards"][idx] += rew_noise

    #Adding noise to next observation (s_t+1) by shifting observations backwards within an episode
    prev_idx = idx - 1
    valid = prev_idx >= 0
    same_episode = np.zeros_like(valid)
    same_episode[valid] = dataset["terminals"][prev_idx[valid]] == 0

    valid = valid & same_episode
    dataset["next_observations"][prev_idx[valid]] += obs_noise[valid]

    return dataset

def inject_noise_into_trajs(traj_list: list, noise_dict) -> list:
    """
    Same noise injection but applied to a trajectory list (for DT/CDT).
    Operates on the same random indices as inject_gaussian_noise for consistency.
    """
    if noise_dict is None:
        return traj_list

    new_trajs = []
    flat_cursor = 0
    for traj in traj_list:
        traj = {k: v.copy() for k, v in traj.items()}
        T = len(traj["rewards"])
        for local_i in range(T):
            flat_i = flat_cursor + local_i
            if flat_i in noise_dict["idx_set"]:
                traj["observations"][local_i] += noise_dict["idx_to_obs_noise"][flat_i]
                traj["rewards"][local_i] += noise_dict["idx_to_rew_noise"][flat_i]
        traj["returns"] = np.cumsum(traj["rewards"][::-1])[::-1].astype(np.float32)
        flat_cursor += T
        new_trajs.append(traj)

    return new_trajs


# EVALUATION HELPER (gymnasium API)

def eval_gymnasium(actor_fn, env, n_episodes: int, seed: int, device: str) -> float:
    """
    Run n_episodes rollouts using actor_fn(state) -> action.
    Returns mean raw return (use get_normalized_score() to normalise).
    Uses gymnasium's reset(seed=) API — no env.seed() call.
    """
    returns = []
    for ep_i in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep_i)
        done = False
        ep_ret = 0.0
        while not done:
            action = actor_fn(obs, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


def compute_mean_std(obs: np.ndarray, eps: float = 1e-3):
    return obs.mean(0), obs.std(0) + eps

def normalize_states(obs: np.ndarray, mean, std):
    return (obs - mean) / std

#LOADING CHECKPOINTS

def run_checkpoint_evaluation(checkpoint_path: str, device: str):
    """Loads a saved checkpoint and runs a standalone evaluation rollout loop."""
    
    if not os.path.exists(checkpoint_path):
        print(f"CRITICAL: Checkpoint file not found at {checkpoint_path}")
        return
    
    print(f"\nLOADING CHECKPOINT; Path: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    algo = ckpt["algo"]
    state_mean = ckpt["state_mean"]
    state_std = ckpt["state_std"]

    print(f"Algorithm: {algo.upper()} with best score {ckpt['best_score']:.2f}")
    
    base_env = gym.make("Walker2d-v5")

    class NormWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return (obs - state_mean) / state_std

    eval_env = NormWrapper(base_env)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])

    if algo == "dt":
        class NormWrapper(gym.ObservationWrapper):
            def observation(self, obs):
                return ((obs - state_mean.squeeze()) / state_std.squeeze()).astype(np.float32)

        eval_env = NormWrapper(base_env)
        model = DecisionTransformer(
            state_dim=state_dim, action_dim=action_dim, seq_len=ckpt["seq_len"],
            episode_len=1000, embedding_dim=128, num_layers=3, num_heads=1, max_action=max_action
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        target_return = 3000.0 * ckpt["reward_scale"]
        rets = []
        for ep_i in range(10):
            obs, _ = eval_env.reset(seed=ckpt["seed"] + ep_i)
            states = torch.zeros(1, 1001, state_dim, device=device)
            actions = torch.zeros(1, 1000, action_dim, device=device)
            returns = torch.zeros(1, 1001, device=device)
            timesteps = torch.arange(1000, device=device).unsqueeze(0)
            states[0, 0] = torch.as_tensor(obs, device=device)
            returns[0, 0] = target_return

            ep_ret, done = 0.0, False
            for step in range(1000):
                pred = model(states=states[:, :step+1][:, -ckpt["seq_len"]:],
                             actions=actions[:, :step+1][:, -ckpt["seq_len"]:],
                             returns_to_go=returns[:, :step+1][:, -ckpt["seq_len"]:],
                             time_steps=timesteps[:, :step+1][:, -ckpt["seq_len"]:])
                act = pred[0, -1].clamp(-max_action, max_action).cpu().detach().numpy()
                obs, reward, terminated, truncated, _ = eval_env.step(act)
                actions[0, step] = torch.as_tensor(act, device=device)
                states[0, step+1] = torch.as_tensor(obs, device=device)
                returns[0, step+1] = returns[0, step] - (reward * ckpt["reward_scale"])
                ep_ret += reward
                if terminated or truncated: break
            rets.append(ep_ret)
        raw_score = float(np.mean(rets))

    elif algo == "cql":
        class NormWrapper(gym.ObservationWrapper):
            def observation(self, obs):
                return (obs - state_mean) / state_std

        eval_env = NormWrapper(base_env)
        actor = TanhGaussianPolicy(state_dim, action_dim, max_action, orthogonal_init=True).to(device)
        actor.load_state_dict(ckpt["actor_state"])
        raw_score = eval_gymnasium(lambda o, d: actor.act(o, d), eval_env, 10, ckpt["seed"], device)

    norm = get_normalized_score(raw_score)
    print(f"Checkpoint Evaluation Complete | Algorithm: {algo.upper()}")
    print(f"  → D4RL Normalized Score: {norm:.2f} (Saved Best Was: {ckpt.get('best_score', 0.0):.2f})")

# RUN CQL

def run_cql(flat_dataset: dict, env, seed: int, device: str, max_steps: int, 
            dataset_id: str, noise: float, checkpoint_path: str = None) -> float:

    set_seed(seed, env)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Normalise states
    state_mean, state_std = compute_mean_std(flat_dataset["observations"])
    ds = dict(flat_dataset)
    ds["observations"]      = normalize_states(ds["observations"],      state_mean, state_std)
    ds["next_observations"] = normalize_states(ds["next_observations"], state_mean, state_std)

    # Eval env with same normalisation applied via wrapper

    class NormWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return (obs - state_mean) / state_std

    eval_env = NormWrapper(env)

    # Replay buffer
    buf = ReplayBuffer(state_dim, action_dim,
                       buffer_size=ds["observations"].shape[0] + 100, device=device)
    buf.load_d4rl_dataset(ds)   # same dict format — works unchanged

    actor    = TanhGaussianPolicy(state_dim, action_dim, max_action, orthogonal_init=True).to(device)
    critic_1 = FullyConnectedQFunction(state_dim, action_dim, orthogonal_init=True, n_hidden_layers=3).to(device)
    critic_2 = FullyConnectedQFunction(state_dim, action_dim, orthogonal_init=True, n_hidden_layers=3).to(device)

    trainer = ContinuousCQL(
        critic_1=critic_1, critic_2=critic_2,
        critic_1_optimizer=torch.optim.Adam(critic_1.parameters(), 3e-4),
        critic_2_optimizer=torch.optim.Adam(critic_2.parameters(), 3e-4),
        actor=actor,
        actor_optimizer=torch.optim.Adam(actor.parameters(), 3e-5),
        target_entropy=-action_dim, discount=0.99,
        alpha_multiplier=1.0, use_automatic_entropy_tuning=True,
        backup_entropy=False, policy_lr=3e-5, qf_lr=3e-4,
        soft_target_update_rate=5e-3, bc_steps=0, target_update_period=1,
        cql_n_actions=10, cql_importance_sample=True, cql_lagrange=False,
        cql_temp=1.0, cql_alpha=10.0, cql_max_target_backup=False, device=device,
    )
    start_step = 0
    best_score = -np.inf
    #Loading checkpoint if it exists
    if checkpoint_path is not None:
        print(f"[CQL] Loading checkpoint state from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        actor.load_state_dict(ckpt["actor_state"])
        critic_1.load_state_dict(ckpt["critic_1_state"])
        critic_2.load_state_dict(ckpt["critic_2_state"])
        trainer.critic_1_optimizer.load_state_dict(ckpt["critic_1_optim_state"])
        trainer.critic_2_optimizer.load_state_dict(ckpt["critic_2_optim_state"])
        trainer.actor_optimizer.load_state_dict(ckpt["actor_optim_state"])
        if "alpha_optim_state" in ckpt and hasattr(trainer, "alpha_optimizer") and trainer.alpha_optimizer is not None:
            trainer.alpha_optimizer.load_state_dict(ckpt["alpha_optim_state"])
        if "log_alpha" in ckpt and hasattr(trainer, "log_alpha"):
            with torch.no_grad():
                next(trainer.log_alpha.parameters()).copy_(ckpt["log_alpha"])
        start_step = ckpt["step"] + 1
        best_score = ckpt.get("best_score", -np.inf)
        print(f"[CQL] Re-entering training loop context at step {start_step:,}")

    def actor_fn(obs, dev):
        return actor.act(obs, dev)

    eval_freq  = max(max_steps // 20, 5_000)

    for t in trange(start_step, max_steps, desc="CQL Training"):
        batch = [b.to(device) for b in buf.sample(256)]
        log_cql = trainer.train(batch)
        if log_cql and isinstance(log_cql, dict):
            wandb.log({f"train/{k}": v for k, v in log_cql.items()}, step=t)
        if (t + 1) % eval_freq == 0:
            raw = eval_gymnasium(actor_fn, eval_env, n_episodes=10, seed=seed, device=device)
            norm = get_normalized_score(raw)
            print(f"  [CQL] step {t+1:>7,}  norm_score={norm:.2f}")

            if norm > best_score:
                best_score = norm
                checkpoint = {
                "algo": "cql", "noise": noise, "seed": seed, "dataset": dataset_id,
                "step": t, "best_score": max(best_score, norm),
                "actor_state": actor.state_dict(),
                "critic_1_state": critic_1.state_dict(), "critic_2_state": critic_2.state_dict(),
                "critic_1_optim_state": trainer.critic_1_optimizer.state_dict(),
                "critic_2_optim_state": trainer.critic_2_optimizer.state_dict(),
                "actor_optim_state": trainer.actor_optimizer.state_dict(),
                "state_mean": state_mean, "state_std": state_std
                }
                if hasattr(trainer, "alpha_optimizer") and trainer.alpha_optimizer is not None:
                    checkpoint["alpha_optim_state"] = trainer.alpha_optimizer.state_dict()
                if hasattr(trainer, "log_alpha"):
                    checkpoint["log_alpha"] = next(trainer.log_alpha.parameters()).clone()

                
                checkpointpath = os.path.join(CHECKPOINT_DIR, f"cql_noise_{noise:.2f}_seed_{seed}.pt")
                torch.save(checkpoint, checkpointpath)
                print(f"  [CQL]  → Saved new best model checkpoint to {checkpointpath}")

            wandb.log({"eval/raw_return": raw, "eval/normalized_score": norm}, step=t)

    #Recording video
    print("\n[CQL] Recording final evaluation video...")
    video_base_env = gym.make("Walker2d-v5", render_mode="rgb_array")
    
    video_env = RecordVideo(
        video_base_env, 
        video_folder=f"videos/cql_seed_{seed}_bestscore{best_score}", 
        episode_trigger=lambda ep: True,
        disable_logger=True
    )
    video_env = NormWrapper(video_env)
    obs, _ = video_env.reset(seed=seed)
    done = False
    while not done:
        action = actor_fn(obs, device)
        obs, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated    
    video_env.close()
    print("[CQL] Video saved.")

    return best_score

# RUN DT

def run_dt(traj_list: list, env, seed: int, device: str, update_steps: int, 
           dataset_id: str, noise: float, checkpoint_path: str = None) -> float:

    set_seed(seed)


    all_obs    = np.concatenate([t["observations"] for t in traj_list])
    state_mean = all_obs.mean(0, keepdims=True)
    state_std  = all_obs.std(0, keepdims=True) + 1e-6

    seq_len      = 20
    reward_scale = 0.001
    traj_lens    = np.array([len(t["actions"]) for t in traj_list])
    sample_prob  = traj_lens / traj_lens.sum()

    def sample_batch(batch_size):
        S, A, R, T, M = [], [], [], [], []
        for _ in range(batch_size):
            traj_idx = np.random.choice(len(traj_list), p=sample_prob)
            traj     = traj_list[traj_idx]
            start_idx = random.randint(0, traj["rewards"].shape[0] - 1)
            
            states = traj["observations"][start_idx : start_idx + seq_len]
            actions = traj["actions"][start_idx : start_idx + seq_len]
            returns = traj["returns"][start_idx : start_idx + seq_len]
            time_steps = np.arange(start_idx, start_idx + seq_len)

            states = (states - state_mean) / state_std
            returns = returns * reward_scale
            
            mask = np.hstack([np.ones(states.shape[0]), np.zeros(seq_len - states.shape[0])])
            if states.shape[0] < seq_len:
                states = pad_along_axis(states, pad_to=seq_len)
                actions = pad_along_axis(actions, pad_to=seq_len)
                returns = pad_along_axis(returns, pad_to=seq_len)
                
            S.append(states); A.append(actions); R.append(returns); T.append(time_steps); M.append(mask)
        
        return (
            torch.tensor(np.stack(S), dtype=torch.float32, device=device),
            torch.tensor(np.stack(A), dtype=torch.float32, device=device),
            torch.tensor(np.stack(R), dtype=torch.float32, device=device),
            torch.tensor(np.stack(T), dtype=torch.long, device=device),
            torch.tensor(np.stack(M), dtype=torch.float32, device=device)
        )

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = DecisionTransformer(
        state_dim=state_dim, action_dim=action_dim, seq_len=seq_len,
        episode_len=1000, embedding_dim=128, num_layers=3, num_heads=1,
        attention_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1,
        max_action=max_action,
    ).to(device)

    optim     = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min((s + 1) / 10_000, 1))

    eval_freq      = max(update_steps // 10, 10_000)
    target_returns = (4500.0, 2250.0)  # Standard D4RL testing intervals mapped to Walker2d scale
    best_score     = -np.inf
    start_step = 0

    if checkpoint_path is not None:
        print(f"[DT] Loading checkpoint state map securely from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = ckpt["step"] + 1
        best_score = ckpt.get("best_score", -np.inf)
        print(f"[DT] Re-entering training loop context at step {start_step:,}")

    class NormObservation(gym.ObservationWrapper):
        def observation(self, obs):
            return ((obs - state_mean.squeeze()) / state_std.squeeze()).astype(np.float32)

    class ScaleReward(gym.RewardWrapper):
        def reward(self, reward):
            return reward * reward_scale

    eval_env = NormObservation(env)
    eval_env = ScaleReward(eval_env)

    @torch.no_grad()
    def clean_eval_rollout(target_return: float, eval_seed: int):
        states = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device)
        actions = torch.zeros(1, model.episode_len, model.action_dim, dtype=torch.float, device=device)
        returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
        time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device).view(1, -1)

        obs, _ = eval_env.reset(seed=eval_seed)
        states[:, 0] = torch.as_tensor(obs, device=device)
        returns[:, 0] = torch.as_tensor(target_return, device=device)

        episode_return, episode_len = 0.0, 0.0
        for step in range(model.episode_len):
            predicted_actions = model(
                states[:, : step + 1][:, -model.seq_len :],
                actions[:, : step + 1][:, -model.seq_len :],
                returns[:, : step + 1][:, -model.seq_len :],
                time_steps[:, : step + 1][:, -model.seq_len :],
            )
            predicted_action = predicted_actions[0, -1].cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = eval_env.step(predicted_action)
            done = terminated or truncated
            
            actions[:, step] = torch.as_tensor(predicted_action, device=device)
            states[:, step + 1] = torch.as_tensor(next_state, device=device)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

            episode_return += reward
            episode_len += 1
            if done:
                break

        return episode_return, episode_len

    model.train()
    for step in trange(start_step, update_steps, desc="DT Training"):
        batch = sample_batch(64)
        states, actions, returns, time_steps, mask = [b.to(device) for b in batch]
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        
        # Loss calculation identical to line 290 in clean template
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        loss = (loss * mask.unsqueeze(-1)).mean()
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        scheduler.step()

        wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=step)

        # Evaluation phase execution logic built out from TrainConfig steps
        if step % eval_freq == 0 or step == update_steps - 1:
            model.eval()
            for target_return in target_returns:
                eval_returns = []
                for ep_i in range(10):  # Performs standard batch trajectory evaluation
                    eval_return, _ = clean_eval_rollout(
                        target_return=target_return * reward_scale, 
                        eval_seed=seed + ep_i
                    )
                    # Unscale raw rewards exactly like line 324 in clean script
                    eval_returns.append(eval_return / reward_scale)
                
                mean_raw_return = float(np.mean(eval_returns))
                norm = get_normalized_score(mean_raw_return)
                
                if target_return == 4500.0:
                    print(f"  [DT]  step {step+1:>7,}  target={target_return}  norm_score={norm:.2f}")

                    #save checkpoint
                    if norm > best_score:
                        best_score = norm
                        checkpoint= {
                            "algo": "dt", "noise": noise, "seed": seed, "dataset": dataset_id,
                            "step": step, "best_score": max(best_score, norm),
                            "model_state": model.state_dict(),
                            "optim_state": optim.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "state_mean": state_mean, "state_std": state_std,
                            "seq_len": seq_len, "reward_scale": reward_scale
                        }
                        checkpointpath = os.path.join(CHECKPOINT_DIR,f"dt_noise_{noise:.2f}_seed_{seed}.pt")
                        torch.save(checkpoint, checkpointpath)
                        print(f"  [DT]  → Saved new best model checkpoint to {checkpointpath}")

                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": mean_raw_return,
                        f"eval/{target_return}_normalized_score_mean": norm,
                    },
                    step=step,
                )
            model.train()

    #Recording video
    print("\n[DT] Recording final evaluation video...")
    video_base_env = gym.make("Walker2d-v5", render_mode="rgb_array")
    video_env = RecordVideo(
        video_base_env, 
        video_folder=f"videos/dt_seed_{seed}_bestscore_{best_score}", 
        episode_trigger=lambda ep: True,
        disable_logger=True
    )
    video_env = NormObservation(video_env)
    video_env = ScaleReward(video_env)

    states = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(1, model.episode_len, model.action_dim, dtype=torch.float, device=device)
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device).view(1, -1)

    obs, _ = video_env.reset(seed=seed)
    states[:, 0] = torch.as_tensor(obs, device=device)
    returns[:, 0] = torch.as_tensor(4500.0 * reward_scale, device=device)

    for step in range(model.episode_len):
        predicted_actions = model(
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        predicted_action = predicted_actions[0, -1].cpu().detach().numpy()
        next_state, reward, terminated, truncated, _ = video_env.step(predicted_action)
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
        if terminated or truncated:
            break

    video_env.close()
    print("[DT] Video saved.")
        
    return best_score

# RUN CDT

def run_cdt(traj_list: list, env, seed: int, device: str, update_steps: int, dataset_id: str, noise: float, checkpoint_path: str = None) -> float:

    set_seed(seed)

    all_obs    = np.concatenate([t["observations"] for t in traj_list])
    state_mean = all_obs.mean(0, keepdims=True)
    state_std  = all_obs.std(0, keepdims=True) + 1e-6

    seq_len      = 20
    reward_scale = 0.001
    traj_lens    = np.array([len(t["actions"]) for t in traj_list])
    sample_prob  = traj_lens / traj_lens.sum()

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    def sample_batch(batch_size):
        S, A, R, C, T, M, EC, Costs = [], [], [], [], [], [], [], []
        for _ in range(batch_size):
            traj_idx  = np.random.choice(len(traj_list), p=sample_prob)
            traj      = traj_list[traj_idx]
            start_idx = random.randint(0, traj["rewards"].shape[0] - 1)

            states     = traj["observations"][start_idx : start_idx + seq_len]
            actions    = traj["actions"][start_idx : start_idx + seq_len]
            returns    = traj["returns"][start_idx : start_idx + seq_len]
            costs_rtg  = np.zeros(len(returns), dtype=np.float32)
            costs_bin  = np.zeros(len(returns), dtype=np.float32)
            time_steps = np.arange(start_idx, start_idx + seq_len)
            episode_cost = np.float32(0.0)

            states  = (states - state_mean) / state_std
            returns = returns * reward_scale

            mask = np.hstack([np.ones(states.shape[0]), np.zeros(seq_len - states.shape[0])])
            if states.shape[0] < seq_len:
                states    = pad_along_axis(states,    pad_to=seq_len)
                actions   = pad_along_axis(actions,   pad_to=seq_len)
                returns   = pad_along_axis(returns,   pad_to=seq_len)
                costs_rtg = pad_along_axis(costs_rtg, pad_to=seq_len)
                costs_bin = pad_along_axis(costs_bin, pad_to=seq_len)

            S.append(states); A.append(actions); R.append(returns)
            C.append(costs_rtg); T.append(time_steps); M.append(mask)
            EC.append(episode_cost); Costs.append(costs_bin)

        to_t = lambda x, dtype=torch.float32: torch.tensor(np.stack(x), dtype=dtype, device=device)
        return (
            to_t(S), to_t(A), to_t(R), to_t(C),
            to_t(T, dtype=torch.long), to_t(M),
            to_t(EC), to_t(Costs),
        )

    model = CDT(
        state_dim=state_dim, action_dim=action_dim, max_action=max_action,
        seq_len=seq_len, episode_len=1000, embedding_dim=128,
        num_layers=3, num_heads=1,
        attention_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1,
        use_rew=True, use_cost=False,
        stochastic=True,
        target_entropy=-action_dim,
    ).to(device)

    class NormObs(gym.ObservationWrapper):
        def observation(self, obs):
            return ((obs - state_mean.squeeze()) / state_std.squeeze()).astype(np.float32)
        
    class ScaleReward(gym.RewardWrapper):
        def reward(self, reward):
            return reward * reward_scale

    eval_env = NormObs(env)
    eval_env = ScaleReward(eval_env)

    trainer = WalkerCDTTrainer(
        model=model, env=eval_env,
        learning_rate=1e-4, weight_decay=1e-4, betas=(0.9, 0.999),
        clip_grad=0.25, lr_warmup_steps=10_000,
        reward_scale=reward_scale, cost_scale=1.0,
        loss_cost_weight=0.0, loss_state_weight=0.0,
        device=device,
    )

    start_step = 0
    best_score = -np.inf
    eval_freq  = max(update_steps // 10, 10_000)

    if checkpoint_path is not None:
        print(f"[CDT] Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        trainer.optim.load_state_dict(ckpt["optim_state"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = ckpt["step"] + 1
        best_score = ckpt.get("best_score", -np.inf)
        print(f"[CDT] Resuming at step {start_step:,}")

    model.train()
    for step in trange(start_step, update_steps, desc="CDT Training"):
        states, actions, returns, costs_rtg, time_steps, mask, episode_cost, costs = sample_batch(64)
        trainer.train_one_step(states, actions, returns, costs_rtg,
                               time_steps, mask, episode_cost, costs)

        if (step + 1) % eval_freq == 0 or step == update_steps - 1:
            mean_ret, mean_cost, mean_len = trainer.evaluate(
                num_rollouts=10,
                target_return=4500.0 * reward_scale,
                target_cost=0.0,
            )
            raw  = mean_ret
            norm = get_normalized_score(raw)
            print(f"  [CDT] step {step+1:>7,}  norm_score={norm:.2f}")

            if norm > best_score:
                best_score = norm
                ckpt = {
                    "algo": "cdt", "noise": noise, "seed": seed, "dataset": dataset_id,
                    "step": step, "best_score": best_score,
                    "model_state": model.state_dict(),
                    "optim_state": trainer.optim.state_dict(),
                    "scheduler_state": trainer.scheduler.state_dict(),
                    "state_mean": state_mean, "state_std": state_std,
                    "seq_len": seq_len, "reward_scale": reward_scale,
                }
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"cdt_noise_{noise:.2f}_seed_{seed}.pt")
                torch.save(ckpt, ckpt_path)
                print(f"  [CDT]  → Saved checkpoint to {ckpt_path}")

            wandb.log({"eval/raw_return": raw, "eval/normalized_score": norm}, step=step)

    print("\n[CDT] Recording final evaluation video...")
    video_base_env = gym.make("Walker2d-v5", render_mode="rgb_array")
    video_env = RecordVideo(
        video_base_env,
        video_folder=f"videos/cdt_seed_{seed}_bestscore_{best_score}",
        episode_trigger=lambda ep: True,
        disable_logger=True,
    )
    video_env = NormObs(video_env)

    model.eval()
    trainer.rollout(model, video_env, target_return=4500.0 * reward_scale, target_cost=0.0)
    video_env.close()
    print("[CDT] Video saved.")

    return best_score

# RESULTS

RESULTS_FILE = "experiment_a_results.csv"

def log_result(algo, noise, seed, score):
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["algo", "noise_fraction", "seed", "normalized_score"])
        w.writerow([algo, noise, seed, f"{score:.4f}"])
    print(f"  → Logged: {algo} | noise={noise:.2f} | seed={seed} | score={score:.2f}")


def summarise_results():
    if not os.path.exists(RESULTS_FILE):
        return
    rows = defaultdict(list)
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            rows[(row["algo"], row["noise_fraction"])].append(float(row["normalized_score"]))
    print("\n" + "="*58)
    print(f"{'Algo':<6} {'Noise':>8}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("="*58)
    for (algo, noise), scores in sorted(rows.items()):
        print(f"{algo:<6} {noise:>8}  {np.mean(scores):>8.2f}  {np.std(scores):>8.2f}  {len(scores):>4}")
    print("="*58)


NOISE_LEVELS = [0.25, 0.50, 0.75]
SEEDS        = [1, 2, 3, 4, 5]
ALGOS        = ["cql", "dt", "cdt"]
STEPS_ALGO   = [1000000, 100000]


def run_single(algo, noise, seed, dataset_id, device, steps, checkpoint_path=None):
    print("Experiment A")
    print(f"  algo={algo}  noise={noise*100:.0f}%  seed={seed}  device={device}")
    print(f"{'─'*55}")
    
    if wandb.run is not None:
        wandb.finish()
      
      
    wandb.init(project = "Experiment-A",
               name = f"{algo}_noise_{noise:.2f}_seed_{seed}_obs" + ("_resumed" if checkpoint_path else ""),
               config ={"algo": algo, "noise_level": noise, "seed": seed, "dataset_id": dataset_id, "device": device, "steps": steps})

    flat, env, trajs = load_minari_dataset(dataset_id)
    noise_dict = generate_noise_dict(flat, noise, seed, True, False)
    noisy_flat  = inject_gaussian_noise(flat,  noise_dict)
    noisy_trajs = inject_noise_into_trajs(trajs, noise_dict)

    if algo == "cql":
        score = run_cql(noisy_flat,  env, seed, device, steps, dataset_id, noise, checkpoint_path)
    elif algo == "dt":
        score = run_dt(noisy_trajs,  env, seed, device, steps, dataset_id, noise, checkpoint_path)
    elif algo == "cdt":
        score = run_cdt(noisy_trajs, env, seed, device, steps, dataset_id, noise, checkpoint_path)

    log_result(algo, noise, seed, score)
    wandb.finish()
    return score

if __name__ == "__main__":
    args   = get_args()
    device = get_device(args.device)
    if args.checkpoint is not None:
        if args.resume:
            print(f"[Harness] Loading historical parameters to resume training perfectly...")
            ckpt = torch.load(args.checkpoint, map_location="cpu")

            # Extract and force parameters to preserve absolute dataset identity
            algo = ckpt["algo"]
            noise = ckpt["noise"]
            seed = ckpt["seed"]
            dataset = ckpt["dataset"]

            print(f"→ Recovered configuration: ALGO={algo}, NOISE={noise}, SEED={seed}, DATASET={dataset}")
            run_single(algo, noise, seed, dataset, device, args.steps, checkpoint_path=args.checkpoint)
        else:
            run_checkpoint_evaluation(args.checkpoint, device)
        sys.exit(0)
    if args.full:
        # need algo and steps as well to be set
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                run_single(algo, noise, seed, args.dataset, device, args.steps)
        summarise_results()
    else:
        run_single(args.algo, args.noise, args.seed,
                   args.dataset, device, args.steps,checkpoint_path=args.checkpoint)
        summarise_results()