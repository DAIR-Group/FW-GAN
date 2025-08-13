import os
import argparse
from lib.utils import yaml2config
from networks import get_model
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/fw_gan_iam.yml")
    parser.add_argument("--random_lexicon", action="store_true", help="Use random lexicon for fake text")
    args = parser.parse_args()

    cfg = yaml2config(args.config)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4])
    model = get_model(cfg.model)(cfg, logdir)

    if cfg.ckpt and os.path.exists(cfg.ckpt):
        print(f"Loading checkpoint from {cfg}")
        model.load(cfg.ckpt, cfg.device)
    else:
        print("No valid checkpoint found, using random weights.")

    # Generate and save fakes and reals
    model.gen_fakes(guided=True, use_random_lexicon=args.random_lexicon)