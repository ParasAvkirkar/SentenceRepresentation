import os
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from evaluate import evaluate


if __name__ == '__main__':

    training_commands, predict_commands = [], []
    choices = {"dan": range(1, 4+1), "gru": range(1, 4+1)}
    probing_accuracies = {"dan": [], "gru": []}

    for seq2vec_name, layers in choices.items():
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


        for layer in layers:
            serialization_dir = os.path.join("serialization_dirs", f"probing_sentiment_{seq2vec_name}_with_emb_on_5k_at_layer_{layer}")
            model_files_present = all([os.path.exists(os.path.join(serialization_dir, file_name))
                                       for file_name in ["model.ckpt.index", "config.json", "vocab.txt"]])
            predictions_file = (f"serialization_dirs/probing_sentiment_{seq2vec_name}_with_emb_on_5k_at_layer_{layer}/"
                                f"predictions_imdb_sentiment_5k_test.txt")
            predictions_present = os.path.exists(predictions_file)

            if not model_files_present:
                training_command = (f"python train.py probing "
                                    f"data/imdb_sentiment_train_5k.jsonl "
                                    f"data/imdb_sentiment_dev.jsonl "
                                    f"--base-model-dir serialization_dirs/main_{seq2vec_name}_5k_with_emb "
                                    f"--layer-num {layer} "
                                    f"--num-epochs {epochs} "
                                    f"--suffix-name _sentiment_{seq2vec_name}_with_emb_on_5k_at_layer_{layer}")
                training_commands.append(training_command)
                continue

            if not predictions_present:
                predict_command = (f"python predict.py "
                                   f"serialization_dirs/probing_sentiment_{seq2vec_name}_with_emb_on_5k_at_layer_{layer} "
                                   f"data/imdb_sentiment_test.jsonl "
                                   f"--predictions-file serialization_dirs/probing_sentiment_{seq2vec_name}_with_emb_on_5k_at_layer_{layer}/"
                                   f"predictions_imdb_sentiment_5k_test.txt")
                predict_commands.append(predict_command)
                continue

            accuracy = evaluate("data/imdb_sentiment_test.jsonl", predictions_file)
            probing_accuracies[seq2vec_name].append(accuracy)

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
    plt.style.use('seaborn-whitegrid')
    for seq2vec_name, layer_range in choices.items():
        plt.plot(layer_range, probing_accuracies[seq2vec_name])
        plt.xlabel("Probing Layer")
        plt.ylabel("Accuracy")
        title = "SentimentTask: Probing Performance vs Probing Layer"
        plt.title(title)
        plt.savefig(os.path.join("plots", f"probing_performance_on_sentiment_task_{seq2vec_name}.png"))
        plt.clf()
