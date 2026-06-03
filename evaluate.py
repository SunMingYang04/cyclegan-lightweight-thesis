import argparse
from pathlib import Path

from utils.metrics import compute_cmmd, compute_fid_kid, compute_ssim_lpips, save_metrics


def parse_args():
    parser = argparse.ArgumentParser("Evaluate generated images")
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--fake_dir", required=True)
    parser.add_argument("--input_dir", default=None, help="Optional paired input/reference dir for SSIM/LPIPS.")
    parser.add_argument("--output", required=True, help="Path to metrics.json.")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--cmmd", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = {
        "real_dir": str(Path(args.real_dir)),
        "fake_dir": str(Path(args.fake_dir)),
        "input_dir": str(Path(args.input_dir)) if args.input_dir else None,
    }
    metrics.update(compute_fid_kid(args.real_dir, args.fake_dir, max_images=args.max_images))
    metrics.update(compute_ssim_lpips(args.input_dir, args.fake_dir, max_images=args.max_images))
    metrics.update(compute_cmmd(args.real_dir, args.fake_dir) if args.cmmd else {"CMMD": None})
    json_path, csv_path = save_metrics(metrics, args.output)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
