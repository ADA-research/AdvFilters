import torch

def run_random_search_batched(model, samples, labels, device="cuda", eps=0.5, max_iters=100, verbose=False):
    if verbose:
        print(f"running batched random search with input size: {samples.shape}, labels: {labels.shape}")
    batch_size, n_mels, __ = samples.shape
    found_filters = torch.ones((batch_size, n_mels)).uniform_(1 - eps, 1 + eps)
    
    inputs_before = torch.Tensor(samples.unsqueeze(1)).to(device) # (batchsize, 1, n_mels, 1000)
    model = model.to(device)
    outputs_before = model(inputs_before) # (batchsize, n_labels)
    preds_before = outputs_before.argmax(dim=1)
    if verbose:
        print(f"Shape of outputs_before: {outputs_before.shape}")
    
    for j in range(max_iters):
        # Create adv filter
        adv_filters = torch.ones((batch_size, n_mels)).uniform_(1 - eps, 1 + eps).to(device)
        # Apply filter, then normalise
        inputs = inputs_before * adv_filters.reshape((batch_size, 1, n_mels, 1))
        outputs = model(inputs)
        # Evaluate success
        preds_after = outputs.argmax(dim=1)
        success_indices = torch.where(preds_before != preds_after)
        found_filters[success_indices] = adv_filters[success_indices].detach().cpu()

    # Where the original pred was already wrong, just use a filter of 1s
    preds_before = model(samples.unsqueeze(1).to(device)).argmax(dim=1)
    wrong_indices = torch.where(preds_before != labels.argmax(dim=1))
    found_filters[wrong_indices] = torch.ones((len(wrong_indices), n_mels))
    
    return {"filters": found_filters,
            "perturbed_inputs": samples.unsqueeze(1) * found_filters.reshape((batch_size, 1, n_mels, 1)).to(device)}
