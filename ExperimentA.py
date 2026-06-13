"""
Experiment A Pipeline - Joshua
===============================
Minari edition — works on Windows, Linux, and macOS.
No D4RL, no mujoco-py, no legacy gym required.

Dataset:  mujoco/walker2d/medium-v0  (Minari)
Env:      Walker2d-v4                (gymnasium + mujoco)
Algos:    DT, CQL, CDT
Noise:    0%, 25%, 50%, 75% Gaussian injection
Seeds:    [0, 1, 2, 3, 4]
Metric:   D4RL-equivalent normalised score (mean ± std over 5 seeds)

AMD RX 6600 XT (Windows):
  PyTorch does NOT support ROCm on Windows. Use --device cpu on Windows,
  or run in WSL2 with ROCm installed there instead.

Usage:
  python experiment_a_pipeline.py --algo dt --noise 0.0 --seed 0
  python experiment_a_pipeline.py --full
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ══════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ══════════════════════════════════════════════════════════════

def get_device(requested: str = "auto"):
    """
    Returns a device object (not always a string for DirectML).
    All .to(device) calls in the pipeline accept both strings and device objects.
    """
    if requested == "directml":
        import torch_directml
        if torch_directml.is_available():
            print(f"[Device] DirectML: {torch_directml.device()}")
            return torch_directml.device()
        else:
            print("[Device] DirectML not available — falling back to CPU.")
            return "cpu"

    if requested == "auto":
        # Try DirectML first on Windows, then CUDA, then CPU
        try:
            import torch_directml
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


# ══════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",      choices=["dt", "cql", "cdt"], default="dt")
    parser.add_argument("--noise",     type=float, default=0.0)
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--dataset",   type=str,   default="mujoco/walker2d/medium-v0")
    parser.add_argument("--dt_steps",  type=int,   default=100_000)
    parser.add_argument("--cql_steps", type=int,   default=1_000_000)
    parser.add_argument("--full",      action="store_true")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# SEED MANAGEMENT
# ══════════════════════════════════════════════════════════════

def set_seed(seed: int, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.reset(seed=seed)   # gymnasium API uses reset(seed=) not env.seed()
        env.action_space.seed(seed)


# ══════════════════════════════════════════════════════════════
# STEP 2 — DATASET LOADING (Minari)
# ══════════════════════════════════════════════════════════════

# D4RL reference scores for walker2d normalisation (from d4rl/infos.py)
# These are fixed constants — no D4RL install needed.
WALKER2D_REF_MIN = 1.629      # average return of random policy
WALKER2D_REF_MAX = 4592.3     # average return of expert policy

def get_normalized_score(raw_return: float) -> float:
    """
    D4RL-equivalent normalised score, computed manually.
    Replaces env.get_normalized_score() from the old d4rl library.
    Returns a value in roughly [0, 100], where 100 = expert level.
    """
    return 100.0 * (raw_return - WALKER2D_REF_MIN) / (WALKER2D_REF_MAX - WALKER2D_REF_MIN)


def load_minari_dataset(dataset_id: str):
    """
    Downloads (first run) and loads a Minari dataset.
    Returns:
      dataset_dict  – flat dict with keys matching D4RL format:
                      observations, actions, rewards, next_observations, terminals
      env           – a live gymnasium env recovered from the dataset
      traj_list     – list of per-episode dicts (for DT/CDT sequence models)
    """
    import minari

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


# ══════════════════════════════════════════════════════════════
# STEP 3 — NOISE INJECTION
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# EVALUATION HELPER (gymnasium API)
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# STATE NORMALISATION HELPERS
# ══════════════════════════════════════════════════════════════

def compute_mean_std(obs: np.ndarray, eps: float = 1e-3):
    return obs.mean(0), obs.std(0) + eps

def normalize_states(obs: np.ndarray, mean, std):
    return (obs - mean) / std


# ══════════════════════════════════════════════════════════════
# RUN CQL
# ══════════════════════════════════════════════════════════════

def run_cql(flat_dataset: dict, env, seed: int, device: str, max_steps: int) -> float:
    import torch
    from CQL.cql import (
        TanhGaussianPolicy, FullyConnectedQFunction, ContinuousCQL, ReplayBuffer
    )

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
    import gymnasium as gym

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

    for t in range(max_steps):
        batch = [b.to(device) for b in buf.sample(256)]
        trainer.train(batch)
        if (t + 1) % eval_freq == 0:
            raw = eval_gymnasium(actor_fn, eval_env, n_episodes=10, seed=seed, device=device)
            norm = get_normalized_score(raw)
            print(f"  [CQL] step {t+1:>7,}  norm_score={norm:.2f}")
            best_score = max(best_score, norm)

    return best_score


# ══════════════════════════════════════════════════════════════
# RUN DT
# ══════════════════════════════════════════════════════════════

def run_dt(traj_list: list, env, seed: int, device: str, update_steps: int) -> float:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from DT.dt import DecisionTransformer, pad_along_axis

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
            idx   = np.random.choice(len(traj_list), p=sample_prob)
            traj  = traj_list[idx]
            start = random.randint(0, len(traj["rewards"]) - 1)
            s = (traj["observations"][start:start + seq_len] - state_mean) / state_std
            a = traj["actions"][start:start + seq_len]
            r = traj["returns"][start:start + seq_len] * reward_scale
            t = np.arange(start, start + seq_len)
            mask = np.hstack([np.ones(s.shape[0]), np.zeros(seq_len - s.shape[0])])
            if s.shape[0] < seq_len:
                s = pad_along_axis(s, seq_len)
                a = pad_along_axis(a, seq_len)
                r = pad_along_axis(r, seq_len)
            S.append(s); A.append(a); R.append(r); T.append(t); M.append(mask)
        
        # Convert to tensor directly on target device
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

    target_return = 3_000.0 * reward_scale
    eval_freq     = max(update_steps // 10, 10)
    best_score    = -np.inf

    import gymnasium as gym

    class NormWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return ((obs - state_mean.squeeze()) / state_std.squeeze()).astype(np.float32)

    eval_env = NormWrapper(env)

    def eval_dt(n_episodes=10):
        model.eval()
        rets = []
        for ep_i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + ep_i)
            
            # Explicitly force variables to live strictly on the chosen device
            states    = torch.zeros(1, 1001, state_dim,  dtype=torch.float32, device=device)
            actions   = torch.zeros(1, 1000, action_dim, dtype=torch.float32, device=device)
            returns   = torch.zeros(1, 1001,              dtype=torch.float32, device=device)
            timesteps = torch.arange(1000, dtype=torch.long, device=device).unsqueeze(0)

            states[0, 0]  = torch.as_tensor(obs, dtype=torch.float32, device=device)
            returns[0, 0] = target_return

            ep_ret, done = 0.0, False
            for step in range(1000):
                # Ensure the slicing output retains device registration explicitly
                s_input = states[:, :step+1][:, -seq_len:].to(device)
                a_input = actions[:, :step+1][:, -seq_len:].to(device)
                r_input = returns[:, :step+1][:, -seq_len:].to(device)
                t_input = timesteps[:, :step+1][:, -seq_len:].to(device)

                pred = model(
                    states=s_input,
                    actions=a_input,
                    returns_to_go=r_input,
                    time_steps=t_input,
                )
                
                act = pred[0, -1].clamp(-max_action, max_action).cpu().detach().numpy()
                obs, reward, terminated, truncated, _ = eval_env.step(act)
                
                # Write back utilizing exact device targets
                actions[0, step]   = torch.as_tensor(act, dtype=torch.float32, device=device)
                states[0, step+1]  = torch.as_tensor(obs, dtype=torch.float32, device=device)
                returns[0, step+1] = returns[0, step] - reward
                
                ep_ret += reward
                if terminated or truncated:
                    break
            rets.append(ep_ret / reward_scale)
        model.train()
        return float(np.mean(rets))

    model.train()
    from tqdm import trange
    for step in trange(update_steps, desc="DT Training"):
        states, actions, returns, timesteps, mask = sample_batch(64)
        
        # Ensure padding mask calculation is directly on-device
        padding_mask = (~mask.to(torch.bool)).to(device)

        pred = model(
            states=states, 
            actions=actions, 
            returns_to_go=returns,
            time_steps=timesteps, 
            padding_mask=padding_mask
        )
        
        loss = (F.mse_loss(pred, actions.detach(), reduction="none") * mask.unsqueeze(-1)).mean()
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        scheduler.step()

        if (step + 1) % eval_freq == 0:
            raw  = eval_dt()
            norm = get_normalized_score(raw)
            print(f"  [DT]  step {step+1:>7,}  norm_score={norm:.2f}")
            best_score = max(best_score, norm)
        
    return best_score


# ══════════════════════════════════════════════════════════════
# RUN CDT (stub)
# ══════════════════════════════════════════════════════════════

def run_cdt(traj_list: list, env, seed: int, device: str, update_steps: int) -> float:
    """
    CDT stub. Model instantiation works; the train_one_step batch loop
    needs wiring once the OSRL dependency is confirmed on your system.
    """
    import torch
    from CDT.cdt import CDT, CDTTrainer

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
    # costs = zeros tensor for standard D4RL (no cost signal)
    print("  [CDT] Training loop not yet wired — returning placeholder.")
    return -1.0


# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════

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


def run_single(algo, noise, seed, dataset_id, device, dt_steps, cql_steps):
    print(f"\n{'─'*55}")
    print(f"  algo={algo}  noise={noise*100:.0f}%  seed={seed}  device={device}")
    print(f"{'─'*55}")

    flat, env, trajs = load_minari_dataset(dataset_id)

    noisy_flat  = inject_gaussian_noise(flat,  noise, seed)
    noisy_trajs = inject_noise_into_trajs(trajs, noise, seed)

    if algo == "cql":
        score = run_cql(noisy_flat,  env, seed, device, cql_steps)
    elif algo == "dt":
        score = run_dt(noisy_trajs,  env, seed, device, dt_steps)
    elif algo == "cdt":
        score = run_cdt(noisy_trajs, env, seed, device, dt_steps)

    log_result(algo, noise, seed, score)
    return score


if __name__ == "__main__":
    args   = get_args()
    device = get_device(args.device)

    if args.full:
        for algo in ALGOS:
            for noise in NOISE_LEVELS:
                for seed in SEEDS:
                    run_single(algo, noise, seed,
                               args.dataset, device, args.dt_steps, args.cql_steps)
        summarise_results()
    else:
        run_single(args.algo, args.noise, args.seed,
                   args.dataset, device, args.dt_steps, args.cql_steps)
        summarise_results()