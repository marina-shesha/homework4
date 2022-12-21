from torch import nn
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from hw_nv.base import BaseTrainer
from hw_nv.utils import inf_loop, MetricTracker
import os
import torchaudio
from ..data.MelSpec import MelSpectrogram

dir_wavs = "homework4/wavs"

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            optimizer,
            config,
            device,
            data_loader,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = config["trainer"]["log_step"]

        self.train_metrics = MetricTracker(
            "gen_loss", "mel_loss", "disc_loss", "mpd_loss", "msd_loss", "grad norm",  writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        batch_idx = 0
        for batch in tqdm(self.train_dataloader):
            self.move_batch_to_device(batch, self.device)
            d_loss, mpd_loss, msd_loss, g_loss, mel_loss = self.model.optimizer_step(
                batch,
                self.optimizer['optimizer_g'],
                self.optimizer['optimizer_d']
            )

            self.train_metrics.update("grad norm", self.get_grad_norm())
            self.train_metrics.update("gen_loss", g_loss.item())
            self.train_metrics.update("mel_loss", mel_loss.item())
            self.train_metrics.update("disc_loss", d_loss.item())
            self.train_metrics.update("mpd_loss", mpd_loss.item())
            self.train_metrics.update("msd_loss", msd_loss.item())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} gen_loss: {:.6f} mel_loss: {:.6f} disc_loss: {:.6f} mpd_loss: {:.6f} msd_loss: {:.6f}".format(
                        epoch, self._progress(batch_idx),
                        g_loss.item(),
                        mel_loss.item(),
                        d_loss.item(),
                        mpd_loss.item(),
                        msd_loss.item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate 1", self.lr_scheduler['lr_scheduler_g'].get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                if batch_idx >= self.len_epoch:
                    break
            batch_idx += 1
            break
        self.lr_scheduler['lr_scheduler_g'].step()
        self.lr_scheduler['lr_scheduler_d'].step()

        log = last_train_metrics
        self.evaluation()

        return log

    def evaluation(self):
        self.model.eval()
        melspec = MelSpectrogram()

        def get_data_and_log(filename, num):
            wav, sr = torchaudio.load(filename)
            self._log_audio(f'true_{num}', wav, sr)
            wav = wav.unsqueeze(0)
            mel = melspec(wav)
            print("mel", mel.shape)
            print("true", wav.shape)
            with torch.no_grad():
                fake_wav = self.model(mel.to(self.device)).cpu()
                print("fake", fake_wav.shape)
                self._log_audio(f'fake_{num}', fake_wav.squeeze(0), sr)
        dir = dir_wavs
        for i, name in enumerate(os.listdir(dir)):
            filename = os.path.join(dir, name)
            get_data_and_log(filename, i)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_audio(self, name, audio, sr):
        self.writer.add_audio(f"audio{name}", audio, sample_rate=sr)

    def move_batch_to_device(self, batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch
