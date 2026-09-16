import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup

from src.selector.model import LatentSelector


def load_scores(score_dir: str, step: int, num_seeds: int, num_prompts: int) -> np.ndarray:
    scores = []
    for seed in range(num_seeds):
        path = os.path.join(score_dir, f"sd3_num_inference_steps_{step}_seed_{seed}", "clip_similarity_scores.txt")
        with open(path) as f:
            scores.append([float(line) for line in f][:num_prompts])
    return np.array(scores).transpose()  # (num_prompts, num_seeds)


def load_latents(traj_dir: str, step: int, step_idx: int, num_prompts: int, num_seeds: int) -> torch.Tensor:
    latents = []
    for prompt_idx in range(num_prompts):
        per_seed = []
        for seed in range(num_seeds):
            path = os.path.join(
                traj_dir, f"steps{step}_seed{seed}", "gsdiff_gobj83k_sd35m__render",
                "inference", f"000_{prompt_idx:03d}_013020_traj.pt",
            )
            per_seed.append(torch.load(path, weights_only=True)[step_idx].cpu())
        latents.append(torch.stack(per_seed, dim=0))
    return torch.stack(latents, dim=0)  # (num_prompts, num_seeds, V, C, H, W)


def build_pairs(latents: torch.Tensor, scores: np.ndarray, max_per_bin: int = 100, seed: int = 42):
    """Enumerate all seed pairs per prompt, then balance by |score diff| bin (cf. paper's
    11-bin balancing in Algorithm 1 / Appendix A)."""
    num_prompts, num_seeds = latents.shape[0], latents.shape[1]
    lat1, lat2, diff, label = [], [], [], []
    for p in range(num_prompts):
        for i in range(num_seeds):
            for j in range(i + 1, num_seeds):
                s1, s2 = scores[p, i].item(), scores[p, j].item()
                if s1 == s2:
                    continue
                lat1.append(latents[p, i])
                lat2.append(latents[p, j])
                diff.append(s1 - s2)
                label.append(float(s1 > s2))
    lat1, lat2 = torch.stack(lat1), torch.stack(lat2)
    diff = torch.tensor(diff, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)

    bins = np.arange(0.0, 0.11, 0.01).tolist() + [float("inf")]
    rng = np.random.RandomState(seed)
    abs_diff = diff.abs().numpy()
    keep = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = np.where((abs_diff >= lo) & (abs_diff < hi))[0]
        keep.extend(rng.choice(idx, min(len(idx), max_per_bin), replace=False) if len(idx) else [])
    keep = np.array(keep, dtype=int)
    return lat1[keep], lat2[keep], diff[keep], label[keep]


def group_accuracy(diff: np.ndarray, label: np.ndarray, pred: np.ndarray, thresholds=None):
    thresholds = thresholds or [0.01 * i for i in range(1, 11)]
    abs_diff = np.abs(diff)
    accs = []
    lo = 0.0
    for hi in thresholds:
        mask = (abs_diff >= lo) & (abs_diff < hi)
        accs.append((pred[mask] == label[mask]).mean() if mask.any() else float("nan"))
        lo = hi
    mask = abs_diff >= thresholds[-1]
    accs.append((pred[mask] == label[mask]).mean() if mask.any() else float("nan"))
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str, default="data/trajectory_latent_data/score_results_4o")
    parser.add_argument("--traj_dir", type=str, default="data/trajectories")
    parser.add_argument("--cached_latents", type=str, default="data/trajectory_latent_data/latent_data_4o_data100_seed64_step9in20.pt",
                         help="Precomputed (num_prompts, num_seeds, V, C, H, W) tensor; skips --traj_dir if set")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--num_seeds", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=20)
    parser.add_argument("--step_idx", type=int, default=9, help="Denoising step at which the selector is applied")
    parser.add_argument("--train_split", type=int, default=70, help="Number of prompts used for training")
    parser.add_argument("--num_conv_layers", type=int, default=1)
    parser.add_argument("--num_mlp_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=str, default=None,
                         help="Defaults to checkpoints/selector_{step_idx}in{total_steps}/selector.pt")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"checkpoints/selector_{args.step_idx}in{args.total_steps}/selector.pt"

    scores = load_scores(args.score_dir, args.total_steps, args.num_seeds, args.num_prompts)
    if args.cached_latents and os.path.exists(args.cached_latents):
        latents = torch.load(args.cached_latents, weights_only=True)
    else:
        latents = load_latents(args.traj_dir, args.total_steps, args.step_idx, args.num_prompts, args.num_seeds)

    train_lat1, train_lat2, train_diff, train_label = build_pairs(
        latents[: args.train_split], scores[: args.train_split])
    test_lat1, test_lat2, test_diff, test_label = build_pairs(
        latents[args.train_split:], scores[args.train_split:])

    train_loader = DataLoader(TensorDataset(train_lat1, train_lat2, train_label), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_lat1, test_lat2, test_label), batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentSelector(num_conv_layers=args.num_conv_layers, num_mlp_layers=args.num_mlp_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(train_loader) * args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        for z1, z2, label in train_loader:
            z1, z2, label = z1.to(device), z2.to(device), label.to(device)
            loss = criterion(model(z1, z2), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        all_pred, correct, total = [], 0, 0
        with torch.no_grad():
            for z1, z2, label in test_loader:
                z1, z2, label = z1.to(device), z2.to(device), label.to(device)
                pred = (model(z1, z2) > 0).float()
                all_pred.append(pred.cpu())
                correct += (pred == label).sum().item()
                total += label.numel()
        all_pred = torch.cat(all_pred).numpy()
        test_acc = correct / total
        group_acc = group_accuracy(test_diff.numpy(), test_label.numpy(), all_pred)
        print(f"[epoch {epoch}] test acc {test_acc:.4f}  last-bin acc {group_acc[-1]:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved selector to {args.output}")


if __name__ == "__main__":
    main()
