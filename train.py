import os
from datetime import datetime
import argparse

from lib.utils import yaml2config
from networks import get_model
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/fw_gan_iam.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    print(f"Config file: {args.config}")

    cfg = yaml2config(args.config)
    run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4] + '-' + str(run_id))
    print(logdir)
    
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if DEVICE.type == "cuda":
    #     print(f"[INFO] CUDA available. Using device: {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    # else:
    #     print("[INFO] CUDA not available. Falling back to CPU.")

    # cfg['device'] = str(DEVICE)


    model = get_model(cfg.model)(cfg, logdir)
    
    # Check and load checkpoint
    epoch_done = 1
    if cfg.ckpt and os.path.exists(cfg.ckpt):
        print(f"Loading checkpoint from {cfg.ckpt}")
        epoch_done = model.load(cfg.ckpt, cfg.device)
    else:
        print("No valid checkpoint found, starting from scratch.")

    model.train(epoch_done=epoch_done)