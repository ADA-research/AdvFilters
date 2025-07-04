import numpy as np
from pytorch_lightning.cli import LightningCLI 
from sklearn.metrics import average_precision_score
import torch

from pgd.openmic_pgd import run_pgd_batched_openmic
from training.openmic_datamodule import OpenMICDataModule
from training.passt_masked import PasstMasked
from training.utils import masked_mean_average_precision

ALPHA = 0.01
EPS = 0.1
MAX_ITERS_PGD = 100

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
                 *args, **kwargs):
        super().__init__(pretrained_arch, num_classes, lr, s_patchout_t, s_patchout_f, *args, **kwargs)
        self.loss = masked_bce_loss
        
    def forward(self, x):
        return self.passt(x)[0]

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x = self.mel(x)
        pgd_res = run_pgd_batched_openmic(self,
                                samples=x,
                                labels=y,
                                mask=mask,
                                alpha=ALPHA,
                                eps=EPS,
                                max_iters=MAX_ITERS_PGD,
                                verbose=False)
        x_perturb = pgd_res["perturbed_inputs"]
        y_hat = self.forward(x_perturb)
        loss = self.loss(y_hat, y, mask)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """x, y, mask = batch
        x = self.mel(x)"""
        batch[0] = self.mel(batch[0]).unsqueeze(1)
        super().test_step(batch, batch_idx)
        # Calc robust acc.
        
        """
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y, mask)
        self.log("test_loss", loss)
        return {"y_hat": y_hat, "y": y, "mask": mask}"""
"""
    def test_epoch_end(self, outputs):
        y_hats = []
        y_trues = []
        masks = []
        for res in outputs:
            y_hats.append(torch.sigmoid(res["y_hat"]).cpu().numpy())
            y_trues.append(res["y"].cpu().numpy())
            masks.append(res["mask"].cpu().numpy())
        y_hats = np.vstack(y_hats)
        y_trues = np.vstack(y_trues)
        masks = np.vstack(masks)
        mAP = masked_mean_average_precision(y_trues, y_hats, masks)
        self.log("mAP", mAP)"""

if __name__ == "__main__":
    cli = LightningCLI(PasstMasked, OpenMICDataModule)