# Neural Network Verification for Instrument Recognition

### Current status
trained_model.onnx is a re-trained version of the `mnist_relu_4_1024.onnx` model trained for only a single epoch on the NSynth dataset for instrument recognition.

Due to its more reasonable size, only the validation set is included in the repo.
Scripts to generate features from the other dataset splits and to train additional models are included in the `nsynth_train` folder.

### Note
In `nsynth_train/class_map.json`, the synth_lead class points to the same class index as the keyboard class. This is due to the output size of the mnist model. I chose two classes that are somewhat similar to retain our testing model's parameters.