import argparse
import glob
import os

from PIL import Image
import tqdm

from src.utils.metrics import TextConditionMetrics


def load_views(image_path: str, num_views: int = 4):
    image = Image.open(image_path)
    width, height = image.size
    view_w = view_h = width // num_views
    y0 = height - view_h
    return [image.crop((i * view_w, y0, (i + 1) * view_w, height)) for i in range(num_views)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_dir", type=str, required=True, help="Dir with `*_gs.png` renders")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    with open(args.prompt_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    metrics = TextConditionMetrics(device_idx=args.gpu_id)

    clip_scores, reward_scores = [], []
    for idx, prompt in enumerate(tqdm.tqdm(prompts)):
        matches = glob.glob(os.path.join(args.inference_dir, f"000_{idx:03d}_*_gs.png"))
        assert len(matches) == 1, f"expected exactly one match for prompt {idx}, got {matches}"
        views = load_views(matches[0])
        clipsim, _, imagereward = metrics.evaluate(views, [prompt] * len(views))
        clip_scores.append(clipsim)
        reward_scores.append(imagereward)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "clip_similarity_scores.txt"), "w") as f:
        f.write("\n".join(f"{s:.6f}" for s in clip_scores) + "\n")
    with open(os.path.join(args.output_dir, "image_reward_scores.txt"), "w") as f:
        f.write("\n".join(f"{s:.6f}" for s in reward_scores) + "\n")


if __name__ == "__main__":
    main()
