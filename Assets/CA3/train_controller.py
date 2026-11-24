import json
from pathlib import Path
import time
import cma
import numpy as np
import torch
from torch import nn

from train import build_model

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 14
ACTION = 7
FEAT = STATE
N_PARAMS = ACTION * FEAT + ACTION

HOME = np.array([0, 0.785398, 0, -1.5708, 0, 0, 0], dtype=np.float32)
JOINT_LO = np.array([-2.96706, -2.0944, -2.96706, -2.0944, -2.96706, -2.0944, -3.05433], dtype=np.float32)
JOINT_HI = -JOINT_LO
RANDOM_SCALE = 0.35

VEL_PENALTY = 0.05
ACTION_SMOOTH_PENALTY = 0.1
ACTION_SMOOTH = 0.2
MAX_DELTA = 0.05
HORIZON = 200
EPISODES = 16
CMA_ITERS = 60


def build_dynamics():
    model = build_model() # TODO: fill in the model you used in the train.py file
    path = ROOT / "dynamics.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_norm():
    path = ROOT / "norm.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    data = np.load(path)
    return data["x_mean"], data["x_std"], data["y_mean"], data["y_std"]


def unpack(theta):
    w = theta[: ACTION * FEAT].reshape(ACTION, FEAT)
    b = theta[ACTION * FEAT :]
    return w, b


def policy(state, goal, w, b):
    q, qd = state[:ACTION], state[ACTION:]
    feat = np.concatenate([q - goal, qd])
    raw = goal + w @ feat + b
    return np.clip(raw, JOINT_LO, JOINT_HI).astype(np.float32)


@torch.no_grad()
def step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std):
    #TODO: get the next step of the world model
    x = np.concatenate([state, action]).astype(np.float32)
    x_n = (x - x_mean) / x_std
    delta_n = model(torch.as_tensor(x_n)).numpy()
    return state + delta_n * y_std + y_mean


def sample_goal(rng):
    span = (JOINT_HI - JOINT_LO) * 0.5 * RANDOM_SCALE
    goal = HOME + rng.uniform(-span, span)
    return np.clip(goal, JOINT_LO, JOINT_HI).astype(np.float32)


def smooth_action(prev, raw):
    blended = (1.0 - ACTION_SMOOTH) * prev + ACTION_SMOOTH * raw
    delta = np.clip(blended - prev, -MAX_DELTA, MAX_DELTA)
    return np.clip(prev + delta, JOINT_LO, JOINT_HI).astype(np.float32)


def rollout(model, theta, x_mean, x_std, y_mean, y_std, rng):
    w, b = unpack(theta)

    start = sample_goal(rng)
    goal = sample_goal(rng)
    state = np.concatenate([start, np.zeros(ACTION, np.float32)])
    prev = start.copy()

    cost = 0.0
    for _ in range(HORIZON):
        raw = policy(state, goal, w, b)
        action = smooth_action(prev, raw)
        state = step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std)

        q, qd = state[:ACTION], state[ACTION:]
        cost += np.mean((q - goal) ** 2)
        cost += VEL_PENALTY * np.mean(qd ** 2)
        cost += ACTION_SMOOTH_PENALTY * np.mean((action - prev) ** 2)
        prev = action

    return -cost / HORIZON


def fitness(theta, model, x_mean, x_std, y_mean, y_std, seed):
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(EPISODES):
        returns.append(rollout(model, theta, x_mean, x_std, y_mean, y_std, rng))

    return -float(np.mean(returns))


def save_controller(theta):
    w, b = unpack(theta)
    payload = {
        "feature": "q_minus_goal_and_qd",
        "state_size": STATE,
        "action_size": ACTION,
        "W": w.astype(float).reshape(-1).tolist(),
        "b": b.astype(float).tolist(),
        "joint_lo": JOINT_LO.astype(float).tolist(),
        "joint_hi": JOINT_HI.astype(float).tolist(),
        "home": HOME.astype(float).tolist(),
        "vel_penalty": VEL_PENALTY,
        "horizon": HORIZON,
    }
    path = ROOT / "controller.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def main():
    model = build_dynamics()
    x_mean, x_std, y_mean, y_std = load_norm()
    args = (model, x_mean, x_std, y_mean, y_std)

    def evaluate(theta):
        return -np.mean([fitness(theta, *args, seed=s) for s in (999, 1000, 1001)])

    zero = np.zeros(N_PARAMS)
    print("baseline (W=0, b=0):", evaluate(zero))

    es = cma.CMAEvolutionStrategy(zero, 0.02, {"seed": 0})
    for it in range(CMA_ITERS):
        candidates = es.ask()
        scores = [fitness(np.asarray(c), *args, seed=it) for c in candidates]
        es.tell(candidates, scores)
        print(f"iter {it + 1:2d}  best {-min(scores):.5f}  mean {-np.mean(scores):.5f}")

    options = {"zero": zero, "xbest": es.result.xbest, "xfavorite": es.result.xfavorite}
    results = {name: evaluate(theta) for name, theta in options.items()}
    for name, r in results.items():
        print(f"{name:10s} held-out return {r:.5f}")
    best_name = max(results, key=results.get)
    print("saving:", best_name)
    save_controller(options[best_name])


if __name__ == "__main__":
    current_time = time.time()
    main()
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
