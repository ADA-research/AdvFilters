import torch
import numpy as np
from sklearn.metrics import average_precision_score

from hear21passt.base import get_basic_model, get_model_passt
from training.openmic_datamodule import OpenMICDataModule

def _calc_accuracy(label, output, mask):
    return masked_mean_average_precision(label, 
                                         output, 
                                         mask)

def _was_attack_success_flip_any(label, output_before, output_after, mask):
    pred_before = (output_before > 0.5)[0]
    pred_after = (output_after > 0.5)[0]
    label = (label > 0.5)[0]
    unmasked_idx = mask.nonzero()
    for idx in unmasked_idx[0]:
        if pred_before[idx] == label[idx] and pred_after[idx] != label[idx]:
            return True
    return False

def _was_attack_success_flip_one(labels, output_after, flip_idx):
    labels_after = (output_after.squeeze() >= 0.5).int()
    return labels[flip_idx] == labels_after[flip_idx]

def _flip_labels(model_outputs:torch.Tensor):
    """Chooses the model outputs that are easiest to flip based on their
    absolute distance to 0.5
    """
    flip_distances = (model_outputs.detach().cpu() - 0.5).abs().numpy() # Need to use numpy due to torch issue #55027
    print(f"DEBUG: {flip_distances}")
    flip_idx = flip_distances.argmin(axis=1)
    return flip_idx

def masked_mean_average_precision(targets, preds, masks):
    targets = targets.round()
    ap_scores = []
    for i in range(preds.shape[2]):
        tar = targets[:, 0, i]
        pre = preds[:, 0, i]
        mas = masks[:, 0, i]
        ap_score = average_precision_score(tar, pre, sample_weight=mas) # Koutini et. al. 2022
        ap_scores.append(ap_score)
    # Return the mean of AP across all valid classes (this is mAP)
    return np.mean(ap_scores)

def run_pgd_single_sample_openmic(model, sample, labels, mask, device="cuda", alpha=0.005, eps=0.5, max_iters=100, verbose=False):
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    inputs = torch.Tensor(sample.unsqueeze(0)).to(device)
    inputs.requires_grad = True
    model = model.to(device)
    outputs_before = model(inputs)[0]
    model.zero_grad()
        
    # Logging df    
    #log = Log(outputs.detach().cpu().numpy(), labels.cpu().numpy())
    
    #adv_filter = torch.ones(inputs.shape[1]).to(device) # TODO: random init, get acc for step=0
    adv_filter = torch.Tensor(inputs.shape[2]).uniform_(1 - eps, 1 + eps).to(device) # random init
    best_filter = adv_filter
    best_pred = outputs_before.clone()
    outputs = outputs_before.clone()
    success = False
    for j in range(max_iters):
        cost = loss(outputs[0], labels[0], weight=mask[0])
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
         
        # Logging    
        """log.log_step(cost.detach().cpu().item(),
                     acc_current.cpu().item())"""
        #log.log_extra(adv_filter.detach().cpu().numpy())
        if verbose:
            print(f"Iteration: {j}")
        # Early stopping
        if _was_attack_success_flip_any(labels, outputs_before, outputs, mask):
            best_filter = adv_filter.clone().detach()
            success = True
            best_pred = outputs.clone()
            break
    return {"filter": best_filter, 
            "pred_before": outputs_before,
            "pred_after": best_pred, 
            "success": success}

def run_pgd_batched_openmic(model, samples, labels, mask, device="cuda", alpha=0.005, eps=0.5, max_iters=100, verbose=False):
    if verbose:
        print(f"running batched pgd with input size: {samples.shape}, labels: {labels.shape}, masks: {mask.shape}")
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    batch_size, n_mels, __ = samples.shape
    inputs_before = torch.Tensor(samples.unsqueeze(1)).to(device) # (batchsize, 1, n_mels, 1000)
    inputs = inputs_before.clone()
    inputs.requires_grad = True
    model = model.to(device)
    outputs_before = model(inputs) # (batchsize, n_labels)
    model.zero_grad()
    if verbose:
        print(f"Shape of outputs_before: {outputs_before.shape}")
    # Logging df    
    #log = Log(outputs.detach().cpu().numpy(), labels.cpu().numpy())
    
    adv_filters = torch.ones((batch_size, n_mels)).uniform_(1 - eps, 1 + eps).to(device) # random init
    outputs = outputs_before.clone()
    for j in range(max_iters):
        cost = loss(outputs, labels, weight=mask)
        cost.backward()

        # Create adv filter
        row_signs = torch.sum(inputs.grad, dim=3).sign().squeeze().to(device) # Row-wise sum, then get sign, (batchsize, nmels)
        adv_filters = torch.clamp(
            adv_filters + row_signs * alpha, 
            min=1-eps, max=1+eps)
        # Apply filter, then normalise
        inputs = (inputs_before * adv_filters.reshape((batch_size, 1, n_mels, 1)))
        inputs.requires_grad = True
        outputs = model(inputs)
        model.zero_grad()
        
        if verbose:
            print(f"Iteration: {j}")
            print(f"Loss: {cost}")
        # Not applying early stopping in batched pgd, 
        # as we just maximise loss for adv. training
    return {"filters": adv_filters.reshape((batch_size, 1, n_mels, 1)),
            "perturbed_inputs": inputs}
    
def run_pgd_batched_flip_one_openmic(model, samples, labels, mask, device="cuda", alpha=0.005, eps=0.5, max_iters=100, verbose=False):
    if verbose:
        print(f"running batched pgd with input size: {samples.shape}, labels: {labels.shape}, masks: {mask.shape}")
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    batch_size, n_mels, __ = samples.shape
    inputs_before = torch.Tensor(samples.unsqueeze(1)).to(device) # (batchsize, 1, n_mels, 1000)
    inputs = inputs_before.clone()
    inputs.requires_grad = True
    model = model.to(device)
    outputs_before = model(inputs) # (batchsize, n_labels)
    if type(outputs_before) is tuple:
        outputs_before = outputs_before[0]
    flip_idx = _flip_labels(outputs_before)
    if verbose:
        print(flip_idx)
    labels[flip_idx] = labels[flip_idx] * -1 + 1 # Flip labels
    model.zero_grad()
    if verbose:
        print(f"Shape of outputs_before: {outputs_before.shape}")
    # Store everything to find successes later
    all_filters = []
    all_perturbs = []
    all_outputs = []
    
    adv_filters = torch.ones((batch_size, n_mels)).uniform_(1 - eps, 1 + eps).to(device) # random init
    outputs = outputs_before.clone()
    for j in range(max_iters):
        cost = loss(outputs, labels, weight=mask)
        cost.backward()

        # Create adv filter
        row_signs = torch.sum(inputs.grad, dim=3).sign().squeeze().to(device) # Row-wise sum, then get sign, (batchsize, nmels)
        adv_filters = torch.clamp(
            adv_filters - row_signs * alpha, 
            min=1-eps, max=1+eps) # Sign is minus here for flip one!
        # Apply filter, then normalise
        inputs = (inputs_before * adv_filters.reshape((batch_size, 1, n_mels, 1)))
        inputs.requires_grad = True
        outputs = model(inputs)
        model.zero_grad()
        
        if verbose:
            print(f"Iteration: {j}")
            print(f"Loss: {cost}")
        # Not applying early stopping in batched pgd, 
        # as we just maximise loss for adv. training
        # and evaluate successes after max_iters
        all_filters.append(adv_filters)
        all_perturbs.append(inputs.clone().detach())
        all_outputs.append(outputs.clone().detach())
        
    # Evaluate successes
    idx_vec = torch.Tensor([list(range(batch_size)), flip_idx])
    all_flips = [0] * batch_size
    for i, outputs in enumerate(all_outputs):
        preds = outputs.round()
        preds_before = outputs_before.round()
        flips = (preds[idx_vec] != preds_before[idx_vec]).nonzero()
        for flip in flips:
            all_flips[flip] = 1   
    
    return {"filters": adv_filters.reshape((batch_size, 1, n_mels, 1)),
            "perturbed_inputs": inputs,
            "successes": sum(all_flips)}
    
def run_pgd_single_sample_flip_one_openmic(model, sample, flip_idx=0, device="cuda", alpha=0.005, eps=0.5, max_iters=100, verbose=False):
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    inputs = torch.Tensor(sample.unsqueeze(0)).to(device)
    inputs.requires_grad = True
    model = model.to(device)
    outputs_before = model(inputs)[0]
    model.zero_grad()
    
    # For flip one, the desired labels are the model output with one label flipped
    labels_before = (outputs_before >= 0.5).int()
    labels = labels_before[0].clone()
    labels[flip_idx] = (labels[flip_idx] + 1) % 2 # Flip label
    labels = labels.float().to(device)
    
    # Logging df    
    #log = Log(outputs.detach().cpu().numpy(), labels.cpu().numpy())
    
    #adv_filter = torch.ones(inputs.shape[1]).to(device) # TODO: random init, get acc for step=0
    adv_filter = torch.Tensor(inputs.shape[2]).uniform_(1 - eps, 1 + eps).to(device) # random init
    best_filter = adv_filter
    best_pred = outputs_before.clone()
    outputs = outputs_before.clone()
    success = False
    loss_weight = torch.zeros(labels.shape).to(device)
    loss_weight[flip_idx] = 1
    for j in range(max_iters):
        cost = loss(outputs[0], labels, weight=loss_weight)
        cost.backward()

        # Create adv filter
        row_signs = torch.sum(inputs.grad, dim=3).sign().squeeze().to(device) # Row-wise sum, then get sign
        adv_filter = torch.clamp(
            adv_filter - row_signs * alpha, 
            min=1-eps, max=1+eps) # Row signs are minus here, due to flip-one!
        # Apply filter, then normalise
        inputs = (sample.unsqueeze(0) * adv_filter.reshape((1, 1, 128, 1)))
        
        inputs.requires_grad = True
        outputs = model(inputs)[0]
        model.zero_grad()

        if verbose:
            print(f"Iteration: {j}")
        # Early stopping
        if _was_attack_success_flip_one(labels, outputs, flip_idx):
            best_filter = adv_filter.clone().detach()
            success = True
            best_pred = outputs.clone()
            break
    return {"filters": best_filter, 
            "pred_before": outputs_before,
            "pred_after": best_pred, 
            "success": success}

if __name__ == "__main__":
    model = get_basic_model(mode="logits")
    model.net = get_model_passt(arch="openmic",  n_classes=20)
    model.eval()

    dm = OpenMICDataModule(batch_size=1)
    dm.setup("test")
    loader = dm.test_dataloader()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    print(f"model is on {model.device()}")
    
    successes = 0
    failures = 0
    all_labels = []
    all_preds_before = []
    all_preds_after = []
    all_masks = []
    for x, y, mask in loader:
        spec = model.mel(x.to(device))
        res = run_pgd_single_sample_openmic(model.net, spec, y.to(device), mask.to(device), eps=0.5, alpha=0.005)
        if res["success"]:
            successes += 1
            print("success")
        else:
            failures += 1
            print("failure")
        all_labels.append(y.cpu().numpy())
        all_preds_before.append(res["pred_before"].detach().cpu().numpy())
        all_preds_after.append(res["pred_after"].detach().cpu().numpy())
        all_masks.append(mask.cpu().numpy())
    
    # Calc mAP before and after
    mAP_before = _calc_accuracy(
        torch.Tensor(all_labels), 
        torch.Tensor(all_preds_before), 
        torch.Tensor(all_masks))
    mAP_after = _calc_accuracy(
        torch.Tensor(all_labels), 
        torch.Tensor(all_preds_after), 
        torch.Tensor(all_masks))
    print(f"Successes: {successes}\nFailures{failures}")
    print(f"mAP before: {mAP_before}, mAP after: {mAP_after}")
    