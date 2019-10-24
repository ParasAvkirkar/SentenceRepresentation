#pylint: disable = redefined-outer-name, invalid-name
# inbuilt lib imports:
from typing import List, Dict
import json
import os
import argparse

# external lib imports:
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import models

# project imports
from data import read_instances, load_vocabulary, index_instances, generate_batches
from util import load_pretrained_model

def predict(model: models.Model,
            instances: List[Dict],
            batch_size: int,
            save_to_file: str = None) -> List[int]:
    """
    Makes predictions using model on instances and saves them in save_to_file.
    """
    batches = generate_batches(instances, batch_size)
    predicted_labels = []

    all_predicted_labels = []
    print("Making predictions")
    for batch_inputs in tqdm(batches):
        batch_inputs.pop("labels")
        logits = model(**batch_inputs, training=False)["logits"]
        predicted_labels = list(tf.argmax(logits, axis=-1).numpy())
        all_predicted_labels += predicted_labels

    if save_to_file:
        print(f"Saving predictions to filepath: {save_to_file}")
        with open(save_to_file, "w") as file:
            for predicted_label in all_predicted_labels:
                file.write(str(predicted_label) + "\n")
    else:
        for predicted_label in all_predicted_labels:
            print(str(predicted_label) + "\n")
    return all_predicted_labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict with trained Main/Probing Model')

    parser.add_argument('load_serialization_dir', type=str,
                             help='serialization directory from which to load the trained model.')
    parser.add_argument('data_file_path', type=str, help='data file path to predict on.')
    parser.add_argument('--predictions-file', type=str, help='output predictions file.')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')

    args = parser.parse_args()

    # Set some constants
    MAX_NUM_TOKENS = 250

    instances = read_instances(args.data_file_path, MAX_NUM_TOKENS)

    vocabulary_path = os.path.join(args.load_serialization_dir, "vocab.txt")
    vocab_token_to_id, _ = load_vocabulary(vocabulary_path)

    instances = index_instances(instances, vocab_token_to_id)

    # Load Config
    config_path = os.path.join(args.load_serialization_dir, "config.json")
    with open(config_path, "r") as file:
        config = json.load(file)

    # Load Model
    classifier = load_pretrained_model(args.load_serialization_dir)

    predict(classifier, instances, args.batch_size, args.predictions_file)
