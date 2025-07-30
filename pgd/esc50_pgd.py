from torchmetrics.functional.classification import multiclass_accuracy
import torch
from tqdm import tqdm

from hear21passt.base import get_basic_model, get_model_passt
from data.esc50_datamodule import ESC50DataModule

def _calc_accuracy(label, output):
    return multiclass_accuracy(preds=output.argmax(dim=1), target=label.argmax(dim=1), num_classes=50)

def run_pgd_single_sample(model, sample, labels, loss=None, device="cuda", alpha=0.005, eps=0.5, max_iters=100, verbose=False):
    inputs = torch.Tensor(sample.unsqueeze(0)).to(device)
    inputs.requires_grad = True
    model = model.to(device)
    outputs = model(inputs)[0]
    model.zero_grad()
    
    # Calc precision before pgd
    acc_before = _calc_accuracy(labels, outputs)
        
    # Logging df    
    #log = Log(outputs.detach().cpu().numpy(), labels.cpu().numpy())
    
    #adv_filter = torch.ones(inputs.shape[1]).to(device) # TODO: random init, get acc for step=0
    adv_filter = torch.Tensor(inputs.shape[2]).uniform_(1 - eps, 1 + eps).to(device) # random init
    best_filter = adv_filter
    best_filter_acc = acc_before
    best_pred = outputs
    for j in range(max_iters):
        cost = loss(outputs[0], labels[0])
        cost.backward()

        # Create adv filter
        row_signs = torch.sum(inputs.grad, dim=3).sign().squeeze().to(device) # Row-wise sum, then get sign
        adv_filter = torch.clamp(
            adv_filter + row_signs * alpha, 
            min=1-eps, max=1+eps)
        # Apply filter, then normalise
        inputs = (sample.unsqueeze(0) * adv_filter.reshape((1, 1, 128, 1)))
        
        inputs.requires_grad = True
        outputs = model(inputs)[0]
        model.zero_grad()
        
        acc_current = _calc_accuracy(labels, outputs)
         
        # Logging    
        """log.log_step(cost.detach().cpu().item(),
                     acc_current.cpu().item())"""
        #log.log_extra(adv_filter.detach().cpu().numpy())
        if verbose:
            print(f"Iteration: {j}")
        # Early stopping
        if acc_current < acc_before:
            best_filter = adv_filter.clone().detach()
            best_filter_acc = acc_current
            best_pred = outputs.clone()
            break
    return {"filter": best_filter, 
            "pred": best_pred, 
            "acc_before": acc_before,
            "acc_after": best_filter_acc}

if __name__ == "__main__":
    # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
    model = get_basic_model(mode="logits")
    # replace the transformer with one that outputs 50 classes
    model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476",  n_classes=50)
    model.mel = model.mel.to("cuda")
    # load the pre-trained model state dict
    state_dict = torch.load('./models/esc50-passt-s-n-f128-p16-s10-fold1-acc.967.pt')
    # load the weights into the transformer
    model.net.load_state_dict(state_dict)

    model.eval()

    dm = ESC50DataModule(batch_size=1, 
                        labels_csv="/home/dettmer/testing/esc50/ESC-50/meta/esc50.csv", 
                        wav_dir="/home/dettmer/testing/esc50/ESC-50/audio/",
                        test_fold=1)
    dm.setup("test")
    loader = dm.test_dataloader()
    
    successes = 0
    failures = 0
    for x, y in loader:
        spec = model.mel(x.to("cuda"))
        res = run_pgd_single_sample(model.net, spec, y.to("cuda"), torch.nn.CrossEntropyLoss())
        if res["acc_after"] < res["acc_before"]:
            successes += 1
            print("success")
        else:
            failures += 1
            print("failure")
    
    print(f"Successes: {successes}\nFailures{failures}")