import argparse
import json
import os
from pathlib import Path
import numpy as np

import torch
from tqdm import tqdm

import hw_nv.model as module_model
from hw_nv.trainer import Trainer
from hw_nv.utils import ROOT_PATH
import hw_nv.data as data
from hw_nv.utils.parse_config import ConfigParser
from hw_nv.data.MelSpec import MelSpectrogram
import torchaudio

dir_wavs = "homework4/wavs"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    # dataset = config.init_obj(config["dataset"], data)
    # dataloader = config.init_obj(config["dataloader"], data, dataset=dataset)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    melspec = MelSpectrogram()

    def get_data_and_log(filename, num):
        wav, sr = torchaudio.load(filename)
        wav = wav.unsqueeze(0)
        mel = melspec(wav)
        with torch.no_grad():
            fake_wav = model(mel.to(device)).cpu()
            torchaudio.save(os.path.join('results', f'fakek_{num}.wav'),  fake_wav.squeeze(0), sr)

    dir = dir_wavs
    for i, name in enumerate(os.listdir(dir)):
        filename = os.path.join(dir, name)
        get_data_and_log(filename, i)


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
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output)
