from sacred import Experiment
import torch
import numpy as np

from hear21passt.base import get_basic_model, get_model_passt
from openmic_pgd import run_pgd_single_sample_openmic, _calc_accuracy
from passt_openmic_datamodule import OpenMICDataModule

ex = Experiment("openmic_pgd")

@ex.config
def config():
    eps = 0.5
    max_iters = 100
    alpha = eps / max_iters
    verbose = False

@ex.automain
def automain(_run, eps, max_iters, alpha, verbose):
    print(f"Running experiment with alpha={alpha}, eps={eps}, max_iters={max_iters}")
    
    # Load Model for Openmic
    model = get_basic_model(mode="logits")
    model.net = get_model_passt(arch="openmic",  n_classes=20)
    model.eval()
    
    # Load DataModule
    dm = OpenMICDataModule(batch_size=1)
    dm.setup("test")
    loader = dm.test_dataloader()
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    print(f"model is on {model.device()}")
    
    all_labels = []
    all_preds_before = []
    all_preds_after = []
    all_masks = []
    found_filters = []
    for x, y, mask in loader:
        spec = model.mel(x.to(device))
        res = run_pgd_single_sample_openmic(model.net, 
                                            spec, 
                                            y.to(device), 
                                            mask.to(device), 
                                            eps=eps, 
                                            alpha=alpha, 
                                            max_iters=max_iters, 
                                            verbose=verbose)
        
        _run.log_scalar("success", res["success"])
        all_labels.append(y.cpu().numpy())
        pred_before = list(res["pred_before"].detach().cpu().numpy().tolist())
        pred_after = list(res["pred_after"].detach().cpu().numpy().tolist())
        all_preds_before.append(pred_before)
        all_preds_after.append(pred_after)
        _run.log_scalar("pred_before", pred_before)
        _run.log_scalar("pred_after", pred_after)
        all_masks.append(mask.cpu().numpy())
        found_filters.append(res["filter"].cpu().numpy())
        
    # Calc mAP before and after
    mAP_before = _calc_accuracy(
        torch.Tensor(all_labels), 
        torch.Tensor(all_preds_before), 
        torch.Tensor(all_masks))
    mAP_after = _calc_accuracy(
        torch.Tensor(all_labels), 
        torch.Tensor(all_preds_after), 
        torch.Tensor(all_masks))
    _run.log_scalar("map_before", mAP_before)
    _run.log_scalar("map_after", mAP_after)
    # Save filters
    np.savetxt("/tmp/filters.csv", np.array(found_filters), delimiter=",")
    _run.add_artifact("/tmp/filters.csv")