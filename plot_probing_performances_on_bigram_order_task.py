import os
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from evaluate import evaluate


if __name__ == '__main__':

    training_commands, predict_commands = [], []
    seq2vec_name_to_last_layer = {"dan": 4, "gru": 4}
    probing_accuracies = {}

    for seq2vec_name, layer in seq2vec_name_to_last_layer.items():

        # Check if Base Models have been trained first.
        serialization_dir = os.path.join("serialization_dirs", f"main_{seq2vec_name}_5k_with_emb")
        model_files_present = all([os.path.exists(os.path.join(serialization_dir, file_name))
                                   for file_name in ["model.ckpt.index", "config.json", "vocab.txt"]])
        epochs = 8 if seq2vec_name == "dan" else 4 # gru is slow, use only 4 epochs
        if not model_files_present:
            print("\nYour base model hasn't been trained yet.")
            print("Please train it first with the following command:")
            training_command = (f"python train.py main "
                                f"data/imdb_sentiment_train_5k.jsonl "
                                f"data/imdb_sentiment_dev.jsonl "
                                f"--seq2vec-choice {seq2vec_name} "
                                f"--embedding-dim 50 "
                                f"--num-layers 4 "
                                f"--num-epochs {epochs} "
                                f"--suffix-name _{seq2vec_name}_5k_with_emb "
                                f"--pretrained-embedding-file data/glove.6B.50d.txt ")
            print(training_command)
            exit()

        serialization_dir = os.path.join("serialization_dirs", f"probing_bigram_order_{seq2vec_name}_with_emb_on_5k_at_layer_{layer}")
        model_files_present = all([os.path.exists(os.path.join(serialization_dir, file_name))
                                   for file_name in ["model.ckpt.index", "config.json", "vocab.txt"]])
        predictions_file = (f"serialization_dirs/probing_bigram_order_{seq2vec_name}_with_emb_on_5k_at_layer_{layer}/"
                            f"predictions_bigram_order_test.txt")
        predictions_present = os.path.exists(predictions_file)

        if not model_files_present:
            training_command = (f"python train.py probing "
                                f"data/bigram_order_train.jsonl "
                                f"data/bigram_order_dev.jsonl "
                                f"--base-model-dir serialization_dirs/main_{seq2vec_name}_5k_with_emb "
                                f"--layer-num {layer} "
                                f"--num-epochs {epochs} "
                                f"--suffix-name _bigram_order_{seq2vec_name}_with_emb_on_5k_at_layer_{layer}")
            training_commands.append(training_command)
            continue

        if not os.path.exists(predictions_file):
            predict_command = (f"python predict.py "
                               f"serialization_dirs/probing_bigram_order_{seq2vec_name}_with_emb_on_5k_at_layer_{layer} "
                               f"data/bigram_order_test.jsonl "
                               f"--predictions-file serialization_dirs/probing_bigram_order_{seq2vec_name}_with_emb_on_5k_at_layer_{layer}/"
                               f"predictions_bigram_order_test.txt")
            predict_commands.append(predict_command)
            continue

        accuracy = evaluate("data/bigram_order_test.jsonl", predictions_file)
        probing_accuracies[seq2vec_name] = accuracy

    if training_commands:
        print("\nPlease finish the missing model training using the following commands:")
        print("\n".join(training_commands))

    if predict_commands:
        print("\nPlease finish the model predictions using the following commands:")
        print("\n".join(predict_commands))

    if training_commands or predict_commands:
        print("\nCannot plot the results until all the files are present.")
        exit()

    # Make the plots
    seq2vec_names = ["dan", "gru"]
    plt.xticks(range(2), seq2vec_names)
    plt.bar(range(2), [probing_accuracies["dan"], probing_accuracies["gru"]],
            align='center', alpha=0.5)
    plt.ylabel('Accuracy')
    plt.title('BigramOrderTask: Probing Performance at Last Layer')
    plt.savefig(os.path.join("plots", "probing_performance_on_bigram_order_task.png"))
