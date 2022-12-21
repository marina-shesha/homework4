import argparse
import collections
import warnings

import numpy as np
import torch

import hw_nv.loss as module_loss
import hw_nv.model as module_arch
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
import hw_nv.data as data

from hw_nv.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataset = config.init_obj(config["dataset"], data)
    dataloader = config.init_obj(config["dataloader"],  torch.utils.data, dataset=dataset)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    g_params = filter(lambda p: p.requires_grad, model.g_model.parameters())
    mpd_params = filter(lambda p: p.requires_grad, model.mpd.parameters())
    msd_params = filter(lambda p: p.requires_grad, model.msd.parameters())
    optimizer = {}
    optimizer['optimizer_g'] = config.init_obj(config["optimizer_g"], torch.optim, g_params)
    optimizer['optimizer_d'] = config.init_obj(config["optimizer_d"], torch.optim, mpd_params + msd_params)
    lr_scheduler = {}
    lr_scheduler['lr_scheduler_g'] = config.init_obj(config["lr_scheduler_g"], torch.optim.lr_scheduler, optimizer['optimizer_g'])
    lr_scheduler['lr_scheduler_d'] = config.init_obj(config["lr_scheduler_d"], torch.optim.lr_scheduler, optimizer['optimizer_d'])
    trainer = Trainer(
        model,
        optimizer,
        config=config,
        device=device,
        data_loader=dataloader,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
