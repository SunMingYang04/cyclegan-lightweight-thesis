import argparse
import json
from pathlib import Path

import torch

from models import NLayerDiscriminator
from models.build import build_generator
from utils.count_params import count_parameters
from utils.seed import set_seed
from utils.timer import measure_inference_time


def parse_args():
    parser = argparse.ArgumentParser("Profile CycleGAN models")
    parser.add_argument("--model_type", choices=["baseline", "lightweight", "dwconv9"], required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42, deterministic=False, benchmark=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = {"model_type": args.model_type, "n_blocks": 9 if args.model_type in {"baseline", "dwconv9"} else 6}
    generator = build_generator(cfg)
    discriminator = NLayerDiscriminator()
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    x = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    avg_time, std_time = measure_inference_time(generator, x, warmup=args.warmup, repeat=args.repeat)
    gen_params = count_parameters(generator)
    disc_params = count_parameters(discriminator)
    result = {
        "model_type": args.model_type,
        "image_size": args.image_size,
        "device": str(device),
        "generator_params": gen_params,
        "discriminator_params": disc_params,
        "total_params": gen_params["total"] + disc_params["total"],
        "avg_inference_ms": avg_time * 1000,
        "std_inference_ms": std_time * 1000,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }
    output_path = Path("results") / f"profile_{args.model_type}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
