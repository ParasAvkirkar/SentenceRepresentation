import os
import json
import copy

# external libs imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# project imports
from data import load_vocabulary, index_instances, generate_batches
from util import load_pretrained_model


if __name__ == '__main__':

    training_commands = []
    choices = {"dan": range(1, 4+1), "gru": range(1, 4+1)}

    models = {"dan": None, "gru": None}
    vocabs = {"dan": None, "gru": None}
    for seq2vec_name, _ in choices.items():

        serialization_dir = os.path.join("serialization_dirs", f"main_{seq2vec_name}_5k_with_emb")

        vocab_path = os.path.join(serialization_dir, "vocab.txt")
        config_path = os.path.join(serialization_dir, "config.json")
        weights_path = os.path.join(serialization_dir, "model.ckpt.index")

        model_files_present = all([os.path.exists(path)
                                   for path in [vocab_path, config_path, weights_path]])
        if not model_files_present:
            epochs = 8 if seq2vec_name == "dan" else 4 # gru is slow, use only 4 epochs
            training_command = (f"python train.py main "
                                f"data/imdb_sentiment_train_5k.jsonl "
                                f"data/imdb_sentiment_dev.jsonl "
                                f"--seq2vec-choice {seq2vec_name} "
                                f"--embedding-dim 50 "
                                f"--num-layers 4 "
                                f"--num-epochs {epochs} "
                                f"--suffix-name _{seq2vec_name}_5k_with_emb "
                                f"--pretrained-embedding-file data/glove.6B.50d.txt ")
            training_commands.append(training_command)
            continue

        model = load_pretrained_model(serialization_dir)
        models[seq2vec_name] = model

        vocab, _ = load_vocabulary(vocab_path)
        vocabs[seq2vec_name] = vocab

    if training_commands:
        print("\nFirst, please finish the missing model training using the following commands:")
        print("\n".join(training_commands))
        exit()


    original_instance = {"text_tokens": "the film performances were awesome".split()}
    updates = ["worst", "okay", "cool"]

    updated_instances = []
    for update in updates:
        updated_instance = copy.deepcopy(original_instance)
        updated_instance["text_tokens"][4] = update
        updated_instances.append(updated_instance)
    all_instances = [original_instance]+updated_instances

    layer_representations = {}
    for seq2vec_name in choices.keys():
        model = models[seq2vec_name]
        vocab = vocabs[seq2vec_name]
        all_indexed_instances = index_instances(copy.deepcopy(all_instances), vocab)
        batches = generate_batches(all_indexed_instances, 4)
        layer_representations[seq2vec_name] = model(**batches[0],
                                                    training=False)["layer_representations"]

    for seq2vec_name, representations in layer_representations.items():
        representations = np.asarray(representations)
        differences_across_layers = {"worst": [], "okay": [], "cool": []}
        for layer_num in choices[seq2vec_name]:
            original_representation = representations[0, layer_num-1, :]
            updated_representations = representations[1:, layer_num-1,:]
            differences = [sum(np.abs(original_representation-updated_representation))
                           for updated_representation in updated_representations]
            differences_across_layers["worst"].append(float(differences[0]))
            differences_across_layers["okay"].append(float(differences[1]))
            differences_across_layers["cool"].append(float(differences[2]))

        # Make the plots
        plt.style.use('seaborn-whitegrid')
        plt.plot(choices[seq2vec_name], differences_across_layers["worst"], label="worst")
        plt.plot(choices[seq2vec_name], differences_across_layers["okay"], label="okay")
        plt.plot(choices[seq2vec_name], differences_across_layers["cool"], label="cool")
        plt.xlabel("Layer")
        plt.ylabel("Perturbation Response")
        plt.legend()
        title = f"{seq2vec_name}: Perturbation Response vs Layer"
        plt.title(title)
        plt.savefig(os.path.join("plots", f"perturbation_response_{seq2vec_name}.png"))
        plt.clf()
