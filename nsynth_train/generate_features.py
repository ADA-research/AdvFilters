import argparse
import glob
import json
import librosa
import numpy as np
import pickle
from tqdm import tqdm
from skimage.transform import resize

def generate_mel(y:np.ndarray, sr:int):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=1024, power=1)
    mel_norm = librosa.pcen(mel * (2**31), sr=sr)
    return mel_norm

def rescale(features:np.ndarray, x_len:int, y_len:int):
    # Just a "dumb" rescale for testing purposes
    return resize(features, (x_len, y_len))

def generate_all_features(path:str):
    files = glob.glob(path + "*.wav")
    all_features = dict()
    for file in tqdm(files, desc="Generating features..."):
        filename = file.split("/")[-1]
        y, sr = librosa.load(file, sr=16000)
        rescaled_features = rescale(generate_mel(y, sr), 28, 28)
        all_features[filename] = rescaled_features
    return all_features

def parse_labels(all_features:dict, jsonfile:str):
    features_and_labels = [[], []]
    with open(jsonfile, "r") as fp:
        jsonfile = json.load(fp)
    for filename in tqdm(all_features, desc="Parsing labels..."):
        json_obj = jsonfile[filename[:-4]]
        label = json_obj['instrument_family_str']
        features_and_labels[0].append(all_features[filename])
        features_and_labels[1].append(label)
    return features_and_labels

def make_class_map(features_and_labels):
    labels = set(features_and_labels[1])
    label_map = {label: i for i, label in tqdm(enumerate(list(labels)), desc="Creating class map...")}
    return label_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates sample features for the NSynth dataset as a pickle file and optionally a class_map.")
    parser.add_argument("dataset_path", help="Path to the dataset folder.", default="./nsynth_train/nsynth-valid/")
    parser.add_argument("output_path", help="Path to desired output file.")
    parser.add_argument("--class_map_path", help="Path to class map output if desired.")
    args = parser.parse_args()

    all_features = generate_all_features(path=args.dataset_path + "/audio/")
    all_features_and_labels = parse_labels(all_features, args.dataset_path + "/examples.json")
    
    if args.class_map_path is not None:

        class_map = make_class_map(all_features_and_labels)
        with open("./nsynth_train/class_map.json", "w") as fp:
            json.dump(class_map, fp)
    
    # Save pickle file
    with open(args.output_path, "wb") as fp:
        pickle.dump(all_features_and_labels, fp)
        