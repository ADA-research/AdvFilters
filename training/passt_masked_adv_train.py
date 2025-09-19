import numpy as np
from pytorch_lightning.cli import LightningCLI 
import torch

from attacks.openmic_pgd import run_pgd_batched_openmic, run_pgd_batched_flip_one_openmic
from data.openmic import OpenMICDataModule
from training.passt_masked import PasstMasked
from data.utils import masked_mean_average_precision

def masked_bce_loss(y_hat, y_true, mask):
    samples_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_hat, y_true, reduction="none"
    )
    samples_loss = mask.float() * samples_loss
    loss = samples_loss.mean()
    samples_loss = samples_loss.detach()
    return loss

class PasstMaskedAdv(PasstMasked):
    def __init__(self, 
                 pretrained_arch:str = None,
                 num_classes = 20, 
                 lr = 0.00001, 
                 s_patchout_t = 40, 
                 s_patchout_f = 4, 
                 pgd_alpha = 0.001,
                 pgd_eps = 0.01,
                 pgd_steps = 10,
                 *args, **kwargs):
        super().__init__(pretrained_arch, num_classes, lr, s_patchout_t, s_patchout_f, *args, **kwargs)
        self.loss = masked_bce_loss
        self.pgd_alpha = pgd_alpha
        self.pgd_eps = pgd_eps
        self.pgd_steps = pgd_steps
        self.pgd_successes = 0
        self.pgd_failures = 0
        
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.passt(x)[0]

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.mel(x)
        pgd_res = run_pgd_batched_openmic(self,
                                samples=x,
                                labels=y,
                                mask=mask,
                                alpha=self.pgd_alpha,
                                eps=self.pgd_eps,
                                max_iters=self.pgd_steps,
                                verbose=False)
        x_perturb = pgd_res["perturbed_inputs"]
        y_hat = self.forward(x_perturb)
        loss = self.loss(y_hat, y, mask)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """x, y, mask = batch
        x = self.mel(x)"""
        #batch[0] = self.mel(batch[0]).unsqueeze(1)
        x, y, mask = batch
        x = self.mel(x)
        # Calc adversarial accuracy
        with torch.inference_mode(False):
            pgd_res = run_pgd_batched_flip_one_openmic(self,
                                    samples=x.clone(),
                                    labels=y.clone(),
                                    mask=mask.clone(),
                                    alpha=self.pgd_alpha,
                                    eps=self.pgd_eps,
                                    max_iters=self.pgd_steps,
                                    verbose=True)
            self.pgd_successes += pgd_res["successes"]
            self.pgd_failures += pgd_res["failures"]
        # Calc loss and log results for mAP
        x = x.unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y, mask)
        self.log("test/loss", loss)
        self.y_hats.append(y_hat.cpu().numpy())
        self.y_trues.append(y.cpu().numpy())
        self.masks.append(mask.cpu().numpy())
        return {"y_hat": y_hat, "y": y, "mask": mask}
    
    def on_test_epoch_end(self):
        y_hats = np.vstack(self.y_hats)
        y_trues = np.vstack(self.y_trues)
        masks = np.vstack(self.masks)
        mAP = masked_mean_average_precision(y_trues, y_hats, masks)
        self.log("test/mAP", mAP)
        self.log("test/pgd_flipone_success_rate", self.pgd_successes / (self.pgd_successes + self.pgd_failures))
        self.y_hats = []
        self.y_trues = []
        self.masks = []
        self.pgd_successes = 0
        self.pgd_failures = 0

    def validation_step(self, batch, batch_idx):
        # Calc loss and log results for mAP
        x, y, mask = batch
        x = self.mel(x)
        # Calc adversarial accuracy
        with torch.inference_mode(False):
            pgd_res = run_pgd_batched_flip_one_openmic(self,
                                    samples=x,
                                    labels=y,
                                    mask=mask,
                                    alpha=self.pgd_alpha,
                                    eps=self.pgd_eps,
                                    max_iters=self.pgd_steps,
                                    verbose=True)
            self.pgd_successes += pgd_res["successes"]
            self.pgd_failures += pgd_res["failures"]
        
        x = x.unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y, mask)
        self.log("val/loss", loss)
        self.y_hats.append(y_hat.cpu().numpy())
        self.y_trues.append(y.cpu().numpy())
        self.masks.append(mask.cpu().numpy())
        
        return {"y_hat": y_hat, "y": y, "mask": mask}

    def on_validation_epoch_end(self):
        y_hats = np.vstack(self.y_hats)
        y_trues = np.vstack(self.y_trues)
        masks = np.vstack(self.masks)
        mAP = masked_mean_average_precision(y_trues, y_hats, masks)
        self.log("val/mAP", mAP)
        self.log("val/pgd_flipone_success_rate", self.pgd_successes / (self.pgd_successes + self.pgd_failures))
        self.y_hats = []
        self.y_trues = []
        self.masks = []
        self.pgd_successes = 0
        self.pgd_failures = 0
if __name__ == "__main__":
    cli = LightningCLI(PasstMasked, OpenMICDataModule, save_config_kwargs={"overwrite": True})