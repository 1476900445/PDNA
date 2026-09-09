from __future__ import annotations

import argparse
import json

from pdna.config import load_config
from pdna.data import MixedReplayBuffer, ReplayBuffer
from pdna.fql import FQLAgent
from pdna.training import FQLTrainer
from pdna.utils.checkpoint import atomic_torch_save
from pdna.utils.tensor import resolve_device, seed_everything


def load_jsonl(path, buffer: ReplayBuffer) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                buffer.add(row["state"], row["action"], row["reward"],
                           row["next_state"], row["done"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pdna.yaml")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.raw.seed)
    device = resolve_device(config.raw.device)
    f = config.raw.fql
    offline = ReplayBuffer(f.replay_capacity, f.state_dim, f.action_dim, config.raw.seed)
    online = ReplayBuffer(f.replay_capacity, f.state_dim, f.action_dim, config.raw.seed + 1)
    root = config.resolve_path("negmas_data")
    load_jsonl(root / "offline.jsonl", offline)
    load_jsonl(root / "online.jsonl", online)
    if len(offline) < f.batch_size:
        raise RuntimeError(f"Need at least {f.batch_size} records in {root / 'offline.jsonl'}")
    replay = MixedReplayBuffer(offline, online, f.offline_ratio)
    model = FQLAgent(f.state_dim, f.action_dim, list(f.hidden_dims), f.q_networks,
                     f.gamma, f.target_tau, f.flow_steps, f.bc_coefficient,
                     f.q_guidance_coefficient, f.action_min, f.action_max).to(device)
    trainer = FQLTrainer(model, replay, f.learning_rate, f.batch_size)
    steps = args.steps or f.gradient_steps
    for step in range(1, steps + 1):
        metrics = trainer.train_step(device)
        if step % 1000 == 0:
            print(step, metrics)
            atomic_torch_save({"step": step, "model": model.state_dict()},
                              config.resolve_path("checkpoints") / "fql_latest.pt")


if __name__ == "__main__":
    main()

