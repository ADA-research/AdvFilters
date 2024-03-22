from keras import models
import tensorflow as tf
import tf2onnx
import onnx

def convert_model(keras_model_path:str, output_path:str=None):
    """Converts an AAM baseline model from keras to onnx format.

    Args:
        keras_model_path (str): Path to the keras model
        output_path (str, optional): Desired output path of the converted model

    Returns:
        ONNX model_proto: Converted model
    """
    keras_model = models.load_model(keras_model_path)
    spec = (tf.TensorSpec((None, 128, 31, 1), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec)

    # Assuming there’s only one GLOBALmaxpool layer in the model
    for i, node in enumerate(onnx_model.graph.node):
        if node.op_type == 'GlobalMaxPool':
            # Create a new MaxPool node
            maxpool_node = onnx.helper.make_node(
                'MaxPool',
                inputs=node.input,
                outputs=node.output,
                name=node.name,
                kernel_shape=(64, 1),
                strides=(1, 1)
            )
            # Replace the GlobalMaxPool node with the MaxPool node
            onnx_model.graph.node.remove(node)
            onnx_model.graph.node.insert(i, maxpool_node)
    if output_path is not None:
        onnx.save_model(onnx_model, output_path)
    return onnx_model

"""
# Example usage:
if __name__ == "__main__":
    model_path = "./aam_inference/Models/C[9Instr]-Tr[AAM5612]-FP[2s]-NN[VGG16-batch128-epochsEarlyStopping].h5"
    output_path = "vgg16.onnx"
    convert_model(model_path, output_path)
"""