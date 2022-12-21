import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
import torch
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
from .MelSpec import MelSpectrogram
from .MelSpec import MelSpectrogramConfig
import random
import math

from hw_nv.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
}
#на основе репы ASR


class LJSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, part, max_len, data_dir=None, *args, **kwargs):
        super().__init__()
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._index = self._sort_index(self._get_or_load_index(part))
        self.mel_spec = MelSpectrogram()
        self._max_len = max_len

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wav = self.load_audio(audio_path)
        mel_spec = self.mel_spec(audio_wav.unsqueeze(0)).squeeze(0)
        audio_wav, mel_spec = self._prepare_wav_and_spec(audio_wav, mel_spec)
        return {
            "audio": audio_wav,
            "spectrogram": mel_spec,
        }

    def _prepare_wav_and_spec(self, audio_wav, mel_spec):
        #из оригинальной репы
        frames_per_seg = math.ceil(self._max_len / MelSpectrogramConfig.hop_length)

        if audio_wav.size(1) >= self._max_len:
            mel_start = random.randint(0, mel_spec.size(2) - frames_per_seg - 1)
            mel = mel_spec[:, :, mel_start:mel_start + frames_per_seg]
            audio = audio_wav[:, mel_start * MelSpectrogramConfig.hop_length:(mel_start + frames_per_seg) * MelSpectrogramConfig.hop_length]
        else:
            mel = torch.nn.functional.pad(mel_spec, (0, frames_per_seg - mel_spec.size(2)), 'constant')
            audio = torch.nn.functional.pad(audio_wav, (0, self._max_len - audio_wav.size(1)), 'constant')

        return audio, mel

    def _sort_index(self, index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def _load_data(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print("LJSpeech-1.1.tar.bz2")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir /"LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))
        path_train_wav = self._data_dir / "train"
        path_train_wav.mkdir(exist_ok=True, parents=True)
        path_wav = self._data_dir / "wavs"
        for path in path_wav.iterdir():
            shutil.move(str(path), str(path_train_wav/path.name))

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        return audio_tensor

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_data()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing LJSpeech-1.1 folders: {part}"
        ):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for line in f:
                    w_id = line.split("|")[0]
                    w_text = " ".join(line.split("|")[1:]).strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(wav_path.absolute().resolve()),
                            "text": w_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index

