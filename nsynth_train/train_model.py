import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import onnx
from onnx2pytorch import ConvertModel
import pickle
import torch
import torchvision
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

class NSynthDataset(torch.utils.data.Dataset):
    def __init__(self, picklefile="./nsynth_train/validation_features.pickle", class_map="./nsynth_train/class_map.json"):
        # Load labels and features
        with open(picklefile , "rb") as fp:
            self.features, self.labels = pickle.load(fp)
        with open(class_map, "r") as fp:
            self.class_map = json.load(fp)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features[idx]
        label = self.labels[idx]
        sample = {'melfeatures': features, 'instrument': self.class_map[label]}

        return sample


def train_one_epoch(epoch_index, tb_writer, training_loader, model):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data['melfeatures'], data['instrument']

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        inputs = inputs.to(torch.float)
        inputs = inputs[None, :, :, :]
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-trains an onnx mnist model for Nsynth instrument recognition.")
    parser.add_argument("model_path", help="Path to the mnist onnx model file.", default="./nsynth_train/mnist_relu_4_1024.onnx")
    parser.add_argument("training_features", help="Path to the training set features pickle file.")
    parser.add_argument("test_features", help="Path to the test set features pickle file.")
    parser.add_argument("class_map", help="Path to the class_map.json file.")
    args = parser.parse_args()

    training_dataset = NSynthDataset(picklefile=args.training_features, class_map=args.class_map)
    training_dataset_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True)

    test_dataset = NSynthDataset(picklefile=args.test_features, class_map=args.class_map)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    onnx_model = onnx.load_model(args.model_path)
    onnx.checker.check_model(onnx_model)
    pytorch_model = ConvertModel(onnx_model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/mnist_to_nsynth_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 1

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        pytorch_model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, training_dataset_loader, pytorch_model)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        pytorch_model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_dataset_loader):
                vinputs, vlabels = vdata['melfeatures'], vdata['instrument']
                vinputs = vinputs.to(torch.float)
                vinputs = vinputs[None, :, :, :]
                voutputs = pytorch_model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(pytorch_model.state_dict(), model_path)

        epoch_number += 1
    
    torch.onnx.export(pytorch_model, 
        vinputs, "trained_model.onnx", 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'}})