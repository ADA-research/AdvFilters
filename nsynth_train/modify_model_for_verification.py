import torch
import onnx

def modify_model(model:torch.nn.Module):
    """Adds a dimension for a filter vector to an existing model 
    by adding a pre forward hook that applies the filter.
    The filter must be contained in all future invocations of the model
    as a column vector in axis 2.

    Args:
        model (torch.nn.Module): Unmodified torch model
    """
    def pre(module, input):
        input_shape = input[0].shape
        new_shape = (input_shape[0], input_shape[1], input_shape[2] - 1, input_shape[3])
        features = input[0][:, :, :-1, :]
        filter = input[0][0, 0, -1, :]
        filtered_features = (features * filter.T).reshape(new_shape)
        return filtered_features
    
    model.register_forward_pre_hook(pre)

    return model