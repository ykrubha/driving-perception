"""Train the GRU motion model on labeled KITTI car trajectories.

    python scripts/train_motion.py

Uses only TRAIN_SEQUENCES to fit and MOTION_VAL_SEQUENCES for early stopping. Detector
noise and missed frames are simulated, so the model learns to handle both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from perception.boxes import xyxy_to_cxcywh
from perception.kitti import (
    IMAGE_HEIGHT, IMAGE_WIDTH, MOTION_VAL_SEQUENCES, TRAIN_SEQUENCES, car_trajectories, load_sequence,
)
from perception.motion import NUM_FEATURES, GruWeights

WINDOW = 16
WARMUP = 6
NOISE = 0.03


class MotionGru(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.gru = nn.GRU(NUM_FEATURES, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 4)
        self.register_buffer("mean", torch.zeros(NUM_FEATURES))
        self.register_buffer("std", torch.ones(NUM_FEATURES))


def torch_features(previous, current, observed):
    h = previous[..., 3].clamp(min=1.0)
    return torch.stack([
        (current[..., 0] - previous[..., 0]) / h,
        (current[..., 1] - previous[..., 1]) / h,
        torch.log(current[..., 2].clamp(min=1.0) / previous[..., 2].clamp(min=1.0)),
        torch.log(current[..., 3].clamp(min=1.0) / previous[..., 3].clamp(min=1.0)),
        (current[..., 0] - IMAGE_WIDTH / 2) / IMAGE_HEIGHT,
        (current[..., 1] - IMAGE_HEIGHT / 2) / IMAGE_HEIGHT,
        torch.log(current[..., 2].clamp(min=1.0) / IMAGE_HEIGHT),
        torch.log(current[..., 3].clamp(min=1.0) / IMAGE_HEIGHT),
        observed,
    ], dim=-1)


def torch_apply(box, delta):
    h = box[..., 3].clamp(min=1.0)
    return torch.stack([
        box[..., 0] + delta[..., 0] * h,
        box[..., 1] + delta[..., 1] * h,
        (box[..., 2] * torch.exp(delta[..., 2].clamp(-1, 1))).clamp(min=1.0),
        (box[..., 3] * torch.exp(delta[..., 3].clamp(-1, 1))).clamp(min=1.0),
    ], dim=-1)


def windows(trajectories: list[np.ndarray], stride: int) -> torch.Tensor:
    parts = []
    for trajectory in trajectories:
        boxes = xyxy_to_cxcywh(trajectory)
        for start in range(0, len(boxes) - WINDOW + 1, stride):
            parts.append(boxes[start:start + WINDOW])
    return torch.tensor(np.stack(parts), dtype=torch.float32)


def add_noise(boxes: torch.Tensor, scale: float) -> torch.Tensor:
    """Jitter each box in proportion to its size, as a real detector would."""
    h = boxes[..., 3:4]
    w = boxes[..., 2:3]
    noise = torch.randn_like(boxes) * scale
    return torch.cat([
        boxes[..., 0:1] + noise[..., 0:1] * w,
        boxes[..., 1:2] + noise[..., 1:2] * h,
        (boxes[..., 2:3] * (1 + noise[..., 2:3])).clamp(min=1.0),
        (boxes[..., 3:4] * (1 + noise[..., 3:4])).clamp(min=1.0),
    ], dim=-1)


def rollout(model: MotionGru, truth: torch.Tensor, noise: float, drop: float) -> torch.Tensor:
    """Feed boxes one step at a time and return the mean loss on next-frame predictions.

    After WARMUP steps, each frame can be "missed" at random. The model then gets its own
    prediction as input and a flag saying so, matching how the tracker coasts through gaps.
    """
    batch = truth.shape[0]
    measured = add_noise(truth, noise) if noise > 0 else truth
    hidden = None
    previous = measured[:, 0]
    current = measured[:, 0]
    observed = torch.ones(batch)
    losses = []
    for step in range(WINDOW - 1):
        features = torch_features(previous, current, observed)
        features = (features - model.mean) / model.std
        output, hidden = model.gru(features[:, None], hidden)
        delta = model.head(output[:, 0])
        prediction = torch_apply(current, delta)

        target = truth[:, step + 1]
        h = current[:, 3:4].clamp(min=1.0)
        position = (prediction[:, :2] - target[:, :2]) / h
        size = torch.log(prediction[:, 2:].clamp(min=1.0) / target[:, 2:].clamp(min=1.0))
        if step >= WARMUP - 1:
            losses.append(nn.functional.smooth_l1_loss(
                torch.cat([position, size], dim=-1), torch.zeros(batch, 4), beta=0.05))

        previous = current
        if step + 1 < WARMUP or drop == 0:
            keep = torch.ones(batch, dtype=torch.bool)
        else:
            keep = torch.rand(batch) > drop
        current = torch.where(keep[:, None], measured[:, step + 1], prediction.detach())
        observed = keep.float()
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ab3dmot")
    parser.add_argument("--out", default="models/motion_gru.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)

    def collect(indices):
        runs = []
        for index in indices:
            runs += car_trajectories(load_sequence(args.root, index, with_detections=False).labels, WINDOW)
        return runs

    train = windows(collect(TRAIN_SEQUENCES), stride=3)
    val = windows(collect(MOTION_VAL_SEQUENCES), stride=8)
    print(f"{len(train)} training windows, {len(val)} validation windows")

    model = MotionGru(args.hidden)
    with torch.no_grad():
        noisy = add_noise(train, NOISE)
        all_features = torch_features(noisy[:, :-1], noisy[:, 1:], torch.ones(noisy[:, 1:].shape[:2]))
        model.mean.copy_(all_features.reshape(-1, NUM_FEATURES).mean(0))
        std = all_features.reshape(-1, NUM_FEATURES).std(0)
        model.std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_loss, best_state, best_epoch = float("inf"), None, -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train))
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch = train[order[start:start + args.batch]]
            loss = rollout(model, batch, NOISE, drop=0.15)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            torch.manual_seed(1000)  # Same noise every epoch, so validation scores are comparable.
            val_loss = rollout(model, val, NOISE, drop=0.15).item()
            torch.manual_seed(args.seed + epoch + 1)
        history.append({"epoch": epoch, "train": round(total / len(train), 5), "val": round(val_loss, 5)})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(history[-1], flush=True)

    model.load_state_dict(best_state)
    weights = GruWeights(
        w_input=model.gru.weight_ih_l0.detach().numpy(),
        w_hidden=model.gru.weight_hh_l0.detach().numpy(),
        b_input=model.gru.bias_ih_l0.detach().numpy(),
        b_hidden=model.gru.bias_hh_l0.detach().numpy(),
        w_output=model.head.weight.detach().numpy(),
        b_output=model.head.bias.detach().numpy(),
        feature_mean=model.mean.numpy(),
        feature_std=model.std.numpy(),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    weights.save(out)
    log = {"best_epoch": best_epoch, "best_val_loss": round(best_loss, 5),
           "train_windows": len(train), "val_windows": len(val),
           "parameters": sum(p.numel() for p in model.parameters()), "history": history}
    Path("results").mkdir(exist_ok=True)
    Path("results/motion_training.json").write_text(json.dumps(log, indent=1))
    print(f"Best validation loss {best_loss:.5f} at epoch {best_epoch}. Saved {out}.")


if __name__ == "__main__":
    main()
