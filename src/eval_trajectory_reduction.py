import argparse
import os

import numpy as np
import torch
import tqdm

from src.selector.model import LatentSelector
from src.train_selector import load_scores


def load_reward_scores(score_dir: str, step: int, num_seeds: int, num_prompts: int) -> np.ndarray:
    scores = []
    for seed in range(num_seeds):
        path = os.path.join(score_dir, f"sd3_num_inference_steps_{step}_seed_{seed}", "image_reward_scores.txt")
        with open(path) as f:
            scores.append([float(line) for line in f][:num_prompts])
    return np.array(scores).transpose()


def tournament_select(model, latents: torch.Tensor, device: str) -> int:
    """Pairwise tournament elimination over the seed dimension (Fig. 3 left)."""
    remaining = list(range(latents.shape[0]))
    while len(remaining) > 1:
        next_round = []
        for i in range(0, len(remaining), 2):
            if i + 1 >= len(remaining):
                next_round.append(remaining[i])
                continue
            idx1, idx2 = remaining[i], remaining[i + 1]
            z1 = latents[idx1].unsqueeze(0).to(device)
            z2 = latents[idx2].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.sigmoid(model(z1, z2))
            next_round.append(idx1 if pred > 0.5 else idx2)
        remaining = next_round
    return remaining[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str, default="data/trajectory_latent_data/score_results_4o")
    parser.add_argument("--cached_latents", type=str, default="data/trajectory_latent_data/latent_data_4o_data100_seed64_step9in20.pt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/selector_9in20/selector.pt")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--num_seeds", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=20)
    parser.add_argument("--num_candidates", type=int, default=8, help="Trajectories per tournament, N in the paper")
    parser.add_argument("--num_repeats", type=int, default=32, help="Repeat the random N-subsample this many times")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentSelector().to(device)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    clip_scores = load_scores(args.score_dir, args.total_steps, args.num_seeds, args.num_prompts)
    reward_scores = load_reward_scores(args.score_dir, args.total_steps, args.num_seeds, args.num_prompts)
    latents = torch.load(args.cached_latents, weights_only=True)

    print(f"random baseline  CLIP-sim {clip_scores.mean():.4f}  ImageReward {reward_scores.mean():.4f}")

    clip_repeat, reward_repeat = [], []
    for _ in tqdm.trange(args.num_repeats):
        subset = np.random.choice(args.num_seeds, args.num_candidates, replace=False)
        clip_sel, reward_sel = [], []
        for p in range(args.num_prompts):
            winner = tournament_select(model, latents[p, subset], device)
            clip_sel.append(clip_scores[p, subset][winner])
            reward_sel.append(reward_scores[p, subset][winner])
        clip_repeat.append(np.mean(clip_sel))
        reward_repeat.append(np.mean(reward_sel))

    print(f"TRIM (selected)  CLIP-sim {np.mean(clip_repeat):.4f} +- {np.std(clip_repeat):.4f}  "
          f"ImageReward {np.mean(reward_repeat):.4f} +- {np.std(reward_repeat):.4f}")


if __name__ == "__main__":
    main()
