from torchmetrics.functional.classification import multiclass_accuracy
import torch

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

def run_pgd_batched(model, samples, labels, device="cuda", alpha=0.005, eps=0.5, max_iters=100, restarts=1, verbose=False):
    if verbose:
        print(f"running batched pgd with input size: {samples.shape}, labels: {labels.shape}")
    loss = torch.nn.CrossEntropyLoss()
    batch_size, n_mels, __ = samples.shape
    found_filters = torch.ones((batch_size, n_mels)).uniform_(1 - eps, 1 + eps)
    
    for restart_idx in range(restarts):
        inputs_before = torch.Tensor(samples.unsqueeze(1)).to(device) # (batchsize, 1, n_mels, 1000)
        inputs = inputs_before.clone()
        inputs.requires_grad = True
        model = model.to(device)
        outputs_before = model(inputs) # (batchsize, n_labels)
        preds_before = outputs_before.argmax(dim=1)
        model.zero_grad()
        if verbose:
            print(f"Shape of outputs_before: {outputs_before.shape}")
        
        adv_filters = torch.ones((batch_size, n_mels)).uniform_(1 - eps, 1 + eps).to(device) # random init
        outputs = outputs_before.clone()
        for j in range(max_iters):
            cost = loss(outputs, labels)
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
            # Evaluate success
            preds_after = outputs.argmax(dim=1)
            success_indices = torch.where(preds_before != preds_after)
            found_filters[success_indices] = adv_filters[success_indices].detach().cpu()
            if verbose:
                print(f"Iteration: {j}")
                print(f"Loss: {cost}")
            # Not applying early stopping in batched pgd, 
            # as we just maximise loss for adv. training
        del inputs, inputs_before, outputs, outputs_before, adv_filters

    # Where the original pred was already wrong, just use a filter of 1s
    preds_before = model(samples.unsqueeze(1).to(device)).argmax(dim=1)
    wrong_indices = torch.where(preds_before != labels.argmax(dim=1))
    found_filters[wrong_indices] = torch.ones((len(wrong_indices), n_mels))
    
    return {"filters": found_filters,
            "perturbed_inputs": samples.unsqueeze(1) * found_filters.reshape((batch_size, 1, n_mels, 1)).to(device)}
