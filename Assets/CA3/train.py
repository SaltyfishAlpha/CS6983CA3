import numpy as np
import torch
import time
from pathlib import Path
from torch import nn

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 14
ACTION = 7


def mean_std(z):
    return z.mean(0), z.std(0) + 1e-6


def norm(z, mean, std):
    return (z - mean) / std


def save_norm_json(x_mean, x_std, y_mean, y_std):
    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    lines = ["{"]
    keys = list(arrays)
    for index, key in enumerate(keys):
        values = ", ".join(f"{float(value):.8f}" for value in arrays[key])
        comma = "," if index < len(keys) - 1 else ""
        lines.append(f'  "{key}": [{values}]{comma}')
    lines.append("}")
    (ROOT / "norm.json").write_text("\n".join(lines) + "\n")


def load_rollout():
    path = ROOT / "rollout.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"No rollout at {path}. Enable RolloutRecorder and Play.")
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != STATE + ACTION + STATE:
        raise ValueError(f"Expected {STATE + ACTION + STATE} columns, found {data.shape[1]}.")
    return data


def predict_delta(model, state, action, x_mean, x_std, y_mean, y_std):
    x = np.concatenate([state, action], axis=-1)
    x_n = (x - x_mean) / x_std
    with torch.no_grad():
        delta_n = model(torch.as_tensor(x_n)).numpy()
    return delta_n * y_std + y_mean


def joint_rmse(pred, true):
    err = pred[..., 0:ACTION] - true[..., 0:ACTION]
    return float(np.sqrt((err ** 2).sum(axis=-1).mean()))


def open_loop_rmse(model, data, start, x_mean, x_std, y_mean, y_std, horizon):
    end = len(data) - horizon
    if end <= start:
        return float("nan")
    idx = np.arange(start, end)
    if len(idx) > 2000:
        idx = idx[:: max(1, len(idx) // 2000)]
    state = data[idx, 0:STATE].copy()
    for t in range(horizon):
        action = data[idx + t, STATE:STATE + ACTION]
        state = state + predict_delta(model, state, action, x_mean, x_std, y_mean, y_std)
    return joint_rmse(state, data[idx + horizon, 0:STATE])

HIDDEN = 256

def build_model():
    return nn.Sequential(
        nn.Linear(STATE + ACTION, HIDDEN), nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
        nn.Linear(HIDDEN, STATE),
    )

def multi_step_loss(model, data, split, x_mean, x_std, y_mean, y_std, horizon, batch, loss_fn):
    lo, hi = split
    idx = np.random.randint(lo, hi - horizon, size=batch)

    state = torch.as_tensor(data[idx, 0:STATE])
    s_scale = x_std[:STATE]
    loss = 0.0

    for t in range(horizon):
        action = torch.as_tensor(data[idx + t, STATE:STATE + ACTION])
        target = torch.as_tensor(data[idx + t, STATE + ACTION:])

        x_n = (torch.cat([state, action], dim=-1) - x_mean) / x_std
        delta = model(x_n) * y_std + y_mean
        state = state + delta

        loss = loss + loss_fn(state / s_scale, target / s_scale)

    return loss / horizon


def train_world_model(data):
    torch.manual_seed(0)
    np.random.seed(0)
    N = len(data)

    # --- 1. split ---
    split = int(N * 0.9)
    train_rng, val_rng = (0, split), (split, N)

    # --- 2. normalization stats ---
    X = data[:, :STATE + ACTION] # [s, a]
    Y = data[:, STATE + ACTION:] - data[:, :STATE] # s' - s
    x_mean, x_std = mean_std(X[:split])
    y_mean, y_std = mean_std(Y[:split])

    save_norm_json(x_mean, x_std, y_mean, y_std)
    np.savez(ROOT / "norm.npz", x_mean=x_mean, x_std=x_std,
             y_mean=y_mean, y_std=y_std)

    tm = [torch.as_tensor(v) for v in (x_mean, x_std, y_mean, y_std)]
    Xn = torch.as_tensor(norm(X, x_mean, x_std))
    Yn = torch.as_tensor(norm(Y, y_mean, y_std))

    # --- 3. model & optimizer ---
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    EPOCHS, BATCH = 30, 512
    HORIZON, MS_BATCH, MS_WEIGHT = 10, 128, 0.5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(split)
        for b in range(0, split, BATCH):
            i = perm[b:b + BATCH]
            one = loss_fn(model(Xn[i]), Yn[i])
            multi = multi_step_loss(model, data, train_rng, *tm,
                                    HORIZON, MS_BATCH, loss_fn)
            loss = one + MS_WEIGHT * multi

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # --- 4. validation ---
        model.eval()
        with torch.no_grad():
            val_one = loss_fn(model(Xn[split:]), Yn[split:]).item()
            val_ms = multi_step_loss(model, data, val_rng, *tm, HORIZON, 512, loss_fn).item()
        print(f"epoch {epoch+1:2d}  train {loss.item():.5f}  val1 {val_one:.5f}  val{HORIZON} {val_ms:.5f}")

    # --- 5. report ---
    model.eval()
    pred_next = data[split:, :STATE] + predict_delta(model, data[split:, :STATE],
                    data[split:, STATE:STATE + ACTION], x_mean, x_std, y_mean, y_std)
    print("one-step joint RMSE:", joint_rmse(pred_next, data[split:, STATE + ACTION:]))
    for h in (10, 50, 100):
        print(f"open-loop {h}-step RMSE:",
              open_loop_rmse(model, data, split, x_mean, x_std, y_mean, y_std, h))

    # --- 6. save ---
    torch.save(model.state_dict(), ROOT / "dynamics.pt")
    dummy = torch.zeros(1, STATE + ACTION)
    torch.onnx.export(model, dummy, ROOT / "dynamics.onnx",
                      input_names=["input"], output_names=["output"],
                      opset_version=15, dynamo=False)
    print("saved dynamics.pt, dynamics.onnx, norm.json, norm.npz")


if __name__ == "__main__":
    current_time = time.time()
    train_world_model(load_rollout())
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
