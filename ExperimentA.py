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
#from cdt import CDT, CDTTrainer

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
    parser.add_argument("--dataset",   type=str,   default="mujoco/walker2d/expert-v0") #mujoco/walker2d/medium-v0
    parser.add_argument("--dt_steps",  type=int,   default=100_000)
    parser.add_argument("--cql_steps", type=int,   default=1_000_000)
    parser.add_argument("--steps", type=int,   default=100_000)
    parser.add_argument("--full",      action="store_true")
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

def inject_gaussian_noise(dataset: dict, noise_fraction: float, seed: int) -> dict:
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


def inject_noise_into_trajs(traj_list: list, noise_fraction: float, seed: int) -> list:
    """
    Same noise injection but applied to a trajectory list (for DT/CDT).
    Operates on the same random indices as inject_gaussian_noise for consistency.
    """
    if noise_fraction == 0.0:
        return traj_list

    # Rebuild flat index so we can select the same transitions as the flat version
    rng = np.random.default_rng(seed)
    lengths = [len(t["rewards"]) for t in traj_list]
    n = sum(lengths)
    n_corrupt = int(n * noise_fraction)
    corrupt_flat_idx = set(rng.choice(n, size=n_corrupt, replace=False).tolist())

    all_obs = np.concatenate([t["observations"] for t in traj_list])
    obs_std = all_obs.std(axis=0)
    rew_std  = float(np.concatenate([t["rewards"] for t in traj_list]).std())

    new_trajs = []
    flat_cursor = 0
    for traj in traj_list:
        traj = {k: v.copy() for k, v in traj.items()}
        T = len(traj["rewards"])
        for local_i in range(T):
            if flat_cursor + local_i in corrupt_flat_idx:
                traj["observations"][local_i] += rng.normal(
                    0, 0.1 * obs_std
                ).astype(np.float32)
                traj["rewards"][local_i] += float(rng.normal(0, 0.1 * rew_std))
        # Recompute RTG after reward noise
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

# RUN CQL

def run_cql(flat_dataset: dict, env, seed: int, device: str, max_steps: int) -> float:

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

    def actor_fn(obs, dev):
        return actor.act(obs, dev)

    eval_freq  = max(max_steps // 20, 5_000)
    best_score = -np.inf

    for t in trange(max_steps, desc="CQL Training"):
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
                    "algo": "cql",
                    "actor_state": actor.state_dict(),
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "best_score": best_score,
                }
                checkpointpath = os.path.join(CHECKPOINT_DIR, f"cql_seed_{seed}_best.pt")
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

def run_dt(traj_list: list, env, seed: int, device: str, update_steps: int) -> float:

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
            
            actions[:, step] = torch.as_tensor(predicted_action)
            states[:, step + 1] = torch.as_tensor(next_state)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

            episode_return += reward
            episode_len += 1
            if done:
                break

        return episode_return, episode_len

    model.train()
    for step in trange(update_steps, desc="DT Training"):
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
                        checkpoint = {"algo": "dt",
                                    "model_state": model.state_dict(),
                                    "state_mean": state_mean,
                                    "state_std": state_std,
                                    "seq_len": seq_len,
                                    "reward_scale": reward_scale,
                                    "best_score": best_score,}
                        checkpointpath = os.path.join(CHECKPOINT_DIR,f"dt_seed_{seed}_best.pt")
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

def run_cdt(traj_list: list, env, seed: int, device: str, update_steps: int) -> float:

    set_seed(seed)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = CDT(
        state_dim=state_dim, action_dim=action_dim, max_action=max_action,
        seq_len=20, episode_len=1000, embedding_dim=128,
        num_layers=3, num_heads=4,
        use_rew=True, use_cost=True, stochastic=False,
    ).to(device)

    trainer = CDTTrainer(
        model=model, env=env,
        learning_rate=1e-4, weight_decay=1e-4, betas=(0.9, 0.999),
        clip_grad=0.25, lr_warmup_steps=10_000,
        reward_scale=0.001, cost_scale=1.0, device=device,
    )

    # TODO: build batch sampler for train_one_step:
    # trainer.train_one_step(states, actions, returns, costs_return,
    #                        time_steps, mask, episode_cost, costs)
    print("  [CDT] Training loop not yet wired")
    return -1.0

#Checkpoint loader

def evaluate_saved_checkpoint(checkpoint_path: str, env_name: str = "Walker2d-v4", device: str = "cuda"):
    #Load the checkpoint dictionary
    ckpt = torch.load(checkpoint_path, map_location=device)
    algo = ckpt["algo"]
    state_mean = ckpt["state_mean"]
    state_std = ckpt["state_std"]
    
    # Build the environment
    base_env = gym.make(env_name)
    
    class NormObservation(gym.ObservationWrapper):
        def observation(self, obs):
            return ((obs - state_mean.squeeze()) / state_std.squeeze()).astype(np.float32)
            
    eval_env = NormObservation(base_env)
    
    if algo == "dt":
        from dt import DecisionTransformer
        # Reconstruct the DT architecture
        model = DecisionTransformer(
            state_dim=eval_env.observation_space.shape[0],
            action_dim=eval_env.action_space.shape[0],
            seq_len=ckpt["seq_len"],
            episode_len=1000, embedding_dim=128, num_layers=3, num_heads=1,
            max_action=float(eval_env.action_space.high[0])
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"Successfully loaded DT checkpoint. Historical Best Score: {ckpt['best_score']:.2f}")

    elif algo == "cql":
        from cql import TanhGaussianPolicy
        # Reconstruct the CQL Actor
        actor = TanhGaussianPolicy(
            state_dim=eval_env.observation_space.shape[0],
            action_dim=eval_env.action_space.shape[0],
            max_action=float(eval_env.action_space.high[0])
        ).to(device)
        actor.load_state_dict(ckpt["actor_state"])
        print(f"Successfully loaded CQL checkpoint. Historical Best Score: {ckpt['best_score']:.2f}")

    elif algo == "cdt":
        print("todo")

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


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

NOISE_LEVELS = [0.0, 0.25, 0.50, 0.75]
SEEDS        = [0, 1, 2, 3, 4]
ALGOS        = ["cql", "dt"]    #["cql", "dt", "cdt"]
STEPS_ALGO   = [1000000, 100000]


def run_single(algo, noise, seed, dataset_id, device, steps):
    print("Experiment A")
    print(f"\n{'─'*55}")
    print(f"  algo={algo}  noise={noise*100:.0f}%  seed={seed}  device={device}")
    print(f"{'─'*55}")
    
    if wandb.run is not None:
        wandb.finish()
    
    wandb.init(project = "Experiment-A",
               name = f"{algo}_noise_{noise:.2f}_seed_{seed}",
               config ={"algo": algo,
                        "noise_level": noise,
                        "seed": seed,
                        "dataset_id": dataset_id,
                        "device:": device,
                        "steps": steps})

    flat, env, trajs = load_minari_dataset(dataset_id)

    noisy_flat  = inject_gaussian_noise(flat,  noise, seed)
    noisy_trajs = inject_noise_into_trajs(trajs, noise, seed)

    if algo == "cql":
        score = run_cql(noisy_flat,  env, seed, device, steps)
    elif algo == "dt":
        score = run_dt(noisy_trajs,  env, seed, device, steps)
    elif algo == "cdt":
        #TODO score = run_cdt(noisy_trajs, env, seed, device, steps)
        print("test")

    log_result(algo, noise, seed, score)
    wandb.finish()
    return score


if __name__ == "__main__":
    args   = get_args()
    device = get_device(args.device)

    if args.full:
        for algo in ALGOS:
            for noise in NOISE_LEVELS:
                for seed in SEEDS:
                    #run_single(algo, noise, seed, args.dataset, device, args.steps)
                    print("todo")
        summarise_results()
    else:
        run_single(args.algo, args.noise, args.seed,
                   args.dataset, device, args.steps)
        summarise_results()