from training.PaSST.passt import PaSST, PatchEmbed, get_model
from training.PaSST.mel_configurable import AugmentMelSTFT
from attacks.filter_pgd import run_pgd_batched

import numpy as np
import pytorch_lightning as L
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class PasstAdv(L.LightningModule):
    def __init__(self, 
                 pretrained_arch:str = "passt_s_swa_p16_128_ap476",
                 num_classes:int = 527,
                 lr:float = 0.00002,
                 s_patchout_t:int = 40,
                 s_patchout_f:int = 4,
                 n_mels:int = 128,
                 sr:int = 32000,
                 win_length:int = 800,
                 hop_size:int = 320,
                 n_fft:int = 1024,
                 freqm:int = 48,
                 timem:int = 192,
                 htk:bool = True,
                 f_min:float = 0.0,
                 f_max:float = None,
                 window_fn:str = "hann",
                 power:int = 2,
                 normalized:bool = False,
                 center:bool = True,
                 pad_mode:str = "reflect",
                 pgd_alpha = 0,
                 pgd_eps = 0,
                 pgd_steps = 0,
                 pgd_restarts = 0,
                 pgd_restarts_val = 0,
                 *args, **kwargs) -> None:
        """
        This class implements adversarial training and evaluation for PaSST.
        It uses Projected Gradient Descent (PGD) to create filter-based adversarial examples during training, validation and testing.
        Args:
            pretrained_arch (str, optional): The architecture of the pretrained PaSST model to use. See `get_model` in `training/PaSST/passt.py` for available architectures. Defaults to 'passt_s_swa_p16_128_ap476'.
            num_classes (int, optional): Number of output classes. Defaults to 527 for AudioSet. 
                Use 50 for ESC-50, 35 for SpeechCommands, and 11 for NSynth.
            lr (float, optional): Learning rate. Defaults to 0.00002.
            s_patchout_t (int, optional): Temporal patchout for PaSST. Defaults to 40.
            s_patchout_f (int, optional): Frequency patchout for PaSST. Defaults to 4.
            n_mels (int, optional): Number of mel bins. Defaults to 128.
            sr (int, optional): Sample rate. Defaults to 32000.
            win_length (int, optional): STFT window length. Defaults to 800.
            hop_size (int, optional): STFT hop size. Defaults to 320.
            n_fft (int, optional): STFT FFT size. Defaults to 1024.
            freqm (int, optional): Frequency masking parameter for SpecAugment. Defaults to 48.
            timem (int, optional): Time masking parameter for SpecAugment. Defaults to 192.
            htk (bool, optional): Use HTK mel scale. Defaults to True.
            f_min (float, optional): Minimum frequency for mel spectrogram. Defaults to 0.0.
            f_max (float, optional): Maximum frequency for mel spectrogram. Defaults to None, which sets it to sr/2.
            window_fn (str, optional): Window function for STFT. Defaults to 'hann'.
            power (int, optional): Power for spectrogram. Defaults to 2.
            normalized (bool, optional): Whether to normalize the spectrogram. Defaults to False.
            center (bool, optional): Whether to center the STFT. Defaults to True.
            pad_mode (str, optional): Padding mode for STFT. Defaults to 'reflect'.
            pgd_alpha (float, optional): Step size for PGD. Set to >0 to enable adversarial training. Defaults to 0.
            pgd_eps (float, optional): Clips the filter perturbation in the range [1 - eps, 1 + eps]. Defaults to 0.
            pgd_steps (int, optional): Number of PGD steps. Defaults to 0.
            pgd_restarts (int, optional): Number of random restarts for PGD during training AND(!) testing.
                Not recommended to be set > 1 for training, as training is substantially slowed down. Defaults to 0.
            pgd_restarts_val (int, optional): Number of random restarts for PGD during validation. 
                Set to > 0 to enable adversarial validation. WARNING: This will make your validation steps take a long time. Defaults to 0.
            """
        super().__init__(*args, **kwargs)
        
        # Load Pretrained
        if pretrained_arch is not None:
            self.passt = get_model(arch=pretrained_arch, pretrained=True, n_classes=num_classes)
        else:
            self.passt = PaSST(num_classes=num_classes, 
                            s_patchout_t=s_patchout_t, 
                            s_patchout_f=s_patchout_f)
            
        self.mel = AugmentMelSTFT(n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hop_size, n_fft=n_fft, freqm=freqm,
                                 timem=timem, htk=htk, f_min=f_min, f_max=f_max, window_fn=window_fn, power=power, 
                                 normalized=normalized, center=center, pad_mode=pad_mode) 
        self.lr = lr
        self.loss = torch.nn.functional.cross_entropy
        self.pgd_alpha = pgd_alpha
        self.pgd_eps = pgd_eps
        self.pgd_steps = pgd_steps
        self.pgd_restarts = pgd_restarts
        self.pgd_restarts_val = pgd_restarts_val
        
        self.y_hats = []
        self.y_trues = []
        self.y_hats_perturbed = []
        if self.training:
            self.save_hyperparameters()
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.passt(x)
        return x[0] # Passt returns a tuple (logits, embeddings)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.mel(x)
        if self.pgd_restarts > 0:
            pgd_res = run_pgd_batched(self,
                                    samples=x,
                                    labels=y,
                                    alpha=self.pgd_alpha,
                                    eps=self.pgd_eps,
                                    max_iters=self.pgd_steps,
                                    restarts=self.pgd_restarts,
                                    verbose=False)
            x_adv = pgd_res["perturbed_inputs"]
        else:
            x_adv = x
        y_hat = self.forward(x_adv)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.mel(x)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("test/clean/loss", loss)
        self.y_hats.append(torch.softmax(y_hat, 1).cpu())
        self.y_trues.append(y.cpu())
        with torch.inference_mode(False):
            pgd_res = run_pgd_batched(self,
                                    samples=x.clone(),
                                    labels=y.clone(),
                                    alpha=self.pgd_alpha,
                                    eps=self.pgd_eps,
                                    max_iters=self.pgd_steps,
                                    restarts=self.pgd_restarts,
                                    verbose=False)
            x_adv = pgd_res["perturbed_inputs"]
            y_hat_adv = self.forward(x_adv)
            loss_adv = self.loss(y_hat_adv, y.clone())
            self.log("test/adv/loss", loss_adv)
            self.y_hats_perturbed.append(y_hat_adv.detach().cpu().numpy())
        return {"y_hat": y_hat, "y": y, "loss": loss}

    def on_test_epoch_end(self):
        y_hats = np.vstack(self.y_hats).argmax(axis=1)
        y_trues = np.vstack(self.y_trues).argmax(axis=1)
        y_hats_perturbed = np.vstack(self.y_hats_perturbed).argmax(axis=1)
        self.log(f"test/clean/accuracy", accuracy_score(y_trues, y_hats), on_epoch=True)
        self.log(f"test/clean/precision", precision_score(y_trues, y_hats, average="micro"), on_epoch=True)
        self.log(f"test/clean/recall", recall_score(y_trues, y_hats, average="micro"), on_epoch=True)
        self.log(f"test/clean/f1_score", f1_score(y_trues, y_hats, average="micro"), on_epoch=True)
        
        # Slightly hacky way to log epsilon as step because 
        # loading and updating wandb tables hurt my feelings
        self.logger.log_metrics({f"test/adv/accuracy": accuracy_score(y_trues, y_hats_perturbed)}, step=self.pgd_eps)
        self.logger.log_metrics({f"test/adv/precision": precision_score(y_trues, y_hats_perturbed, average="micro")}, step=self.pgd_eps)
        self.logger.log_metrics({f"test/adv/recall": recall_score(y_trues, y_hats_perturbed, average="micro")}, step=self.pgd_eps)
        self.logger.log_metrics({f"test/adv/f1_score": f1_score(y_trues, y_hats_perturbed, average="micro")}, step=self.pgd_eps)
        self.y_hats = []
        self.y_trues = []
        self.y_hats_perturbed = []
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.mel(x)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val/clean/loss", loss)
        self.y_hats.append(torch.softmax(y_hat, 1).cpu())
        self.y_trues.append(y.cpu())
        
        with torch.inference_mode(False):
            pgd_res = run_pgd_batched(self,
                                    samples=x,
                                    labels=y,
                                    alpha=self.pgd_alpha,
                                    eps=self.pgd_eps,
                                    max_iters=self.pgd_steps,
                                    restarts=self.pgd_restarts_val,
                                    verbose=False)
            x_adv = pgd_res["perturbed_inputs"]
            y_hat_adv = self.forward(x_adv)
            loss_adv = self.loss(y_hat_adv, y)
            self.log("val/adv/loss", loss_adv)
            self.y_hats_perturbed.append(y_hat_adv.detach().cpu().numpy())
        return {"y_hat": y_hat, "y": y, "loss": loss}

    def on_validation_epoch_end(self):
        y_hats = np.vstack(self.y_hats).argmax(axis=1)
        y_trues = np.vstack(self.y_trues).argmax(axis=1)
        y_hats_perturbed = np.vstack(self.y_hats_perturbed).argmax(axis=1)
        self.log(f"val/clean/accuracy", accuracy_score(y_trues, y_hats), on_epoch=True)
        self.log(f"val/clean/precision", precision_score(y_trues, y_hats, average="micro"), on_epoch=True)
        self.log(f"val/clean/recall", recall_score(y_trues, y_hats, average="micro"), on_epoch=True)
        self.log(f"val/clean/f1_score", f1_score(y_trues, y_hats, average="micro"), on_epoch=True)
        
        self.log(f"val/adv/accuracy", accuracy_score(y_trues, y_hats_perturbed), on_epoch=True)
        self.log(f"val/adv/precision", precision_score(y_trues, y_hats_perturbed, average="micro"), on_epoch=True)
        self.log(f"val/adv/recall", recall_score(y_trues, y_hats_perturbed, average="micro"), on_epoch=True)
        self.log(f"val/adv/f1_score", f1_score(y_trues, y_hats_perturbed, average="micro"), on_epoch=True)
        self.y_hats = []
        self.y_trues = []
        self.y_hats_perturbed = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
