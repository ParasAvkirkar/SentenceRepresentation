#pylint: disable = redefined-outer-name, invalid-name

# inbuilt lib imports:
from typing import List, Dict
import os
import argparse
import random
import json

# external lib imports:
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, optimizers

# project imports
from data import read_instances, save_vocabulary, build_vocabulary, \
                 load_vocabulary, index_instances, generate_batches, load_glove_embeddings
from main_model import MainClassifier
from probing_model import ProbingClassifier
from loss import cross_entropy_loss
from util import load_pretrained_model


def train(model: models.Model,
          optimizer: optimizers.Optimizer,
          train_instances: List[Dict[str, np.ndarray]],
          validation_instances: List[Dict[str, np.ndarray]],
          num_epochs: int,
          batch_size: int,
          serialization_dir: str = None) -> tf.keras.Model:
    """
    Trains a model on the give training instances as configured and stores
    the relevant files in serialization_dir. Returns model and some important metrics.
    """

    print("\nGenerating Training batches:")
    train_batches = generate_batches(train_instances, batch_size)
    print("Generating Validation batches:")
    validation_batches = generate_batches(validation_instances, batch_size)

    train_batch_labels = [batch_inputs.pop("labels") for batch_inputs in train_batches]
    validation_batch_labels = [batch_inputs.pop("labels") for batch_inputs in validation_batches]

    tensorboard_logs_path = os.path.join(serialization_dir, f'tensorboard_logs')
    tensorboard_writer = tf.summary.create_file_writer(tensorboard_logs_path)
    best_epoch_validation_accuracy = float("-inf")
    best_epoch_validation_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")

        total_training_loss = 0
        total_correct_predictions, total_predictions = 0, 0
        generator_tqdm = tqdm(list(zip(train_batches, train_batch_labels)))
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            with tf.GradientTape() as tape:
                logits = model(**batch_inputs, training=True)["logits"]
                loss_value = cross_entropy_loss(logits, batch_labels)
                grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_training_loss += loss_value
            batch_predictions = np.argmax(tf.nn.softmax(logits, axis=-1).numpy(), axis=-1)
            total_correct_predictions += (batch_predictions == batch_labels).sum()
            total_predictions += batch_labels.shape[0]
            description = ("Average training loss: %.2f Accuracy: %.2f "
                           % (total_training_loss/(index+1), total_correct_predictions/total_predictions))
            generator_tqdm.set_description(description, refresh=False)
        average_training_loss = total_training_loss / len(train_batches)
        training_accuracy = total_correct_predictions/total_predictions

        total_validation_loss = 0
        total_correct_predictions, total_predictions = 0, 0
        generator_tqdm = tqdm(list(zip(validation_batches, validation_batch_labels)))
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            logits = model(**batch_inputs, training=False)["logits"]
            loss_value = cross_entropy_loss(logits, batch_labels)
            total_validation_loss += loss_value
            batch_predictions = np.argmax(tf.nn.softmax(logits, axis=-1).numpy(), axis=-1)
            total_correct_predictions += (batch_predictions == batch_labels).sum()
            total_predictions += batch_labels.shape[0]
            description = ("Average validation loss: %.2f Accuracy: %.2f "
                           % (total_validation_loss/(index+1), total_correct_predictions/total_predictions))
            generator_tqdm.set_description(description, refresh=False)
        average_validation_loss = total_validation_loss / len(validation_batches)
        validation_accuracy = total_correct_predictions/total_predictions

        if validation_accuracy > best_epoch_validation_accuracy:
            print("Model with best validation accuracy so far: %.2f. Saving the model."
                  % (validation_accuracy))
            classifier.save_weights(os.path.join(serialization_dir, f'model.ckpt'))
            best_epoch_validation_loss = average_validation_loss
            best_epoch_validation_accuracy = validation_accuracy

        with tensorboard_writer.as_default():
            tf.summary.scalar("loss/training", average_training_loss, step=epoch)
            tf.summary.scalar("loss/validation", average_validation_loss, step=epoch)
            tf.summary.scalar("accuracy/training", training_accuracy, step=epoch)
            tf.summary.scalar("accuracy/validation", validation_accuracy, step=epoch)
        tensorboard_writer.flush()

    metrics = {"training_loss": float(average_training_loss),
               "validation_loss": float(average_validation_loss),
               "training_accuracy": float(training_accuracy),
               "best_epoch_validation_accuracy": float(best_epoch_validation_accuracy),
               "best_epoch_validation_loss": float(best_epoch_validation_loss)}

    print("Best epoch validation accuracy: %.4f, validation loss: %.4f"
          %(best_epoch_validation_accuracy, best_epoch_validation_loss))

    return {"model": model, "metrics": metrics}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Main/Probing Model')

    # Setup common parser arguments for training of either models
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('train_data_file_path', type=str, help='training data file path')
    base_parser.add_argument('validation_data_file_path', type=str, help='validation data file path')
    base_parser.add_argument('--load-serialization-dir', type=str,
                             help='if passed, model will be loaded from this serialization directory.')
    base_parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    base_parser.add_argument('--num-epochs', type=int, default=20, help='max num epochs to train for')
    base_parser.add_argument('--suffix-name', type=str, default="",
                             help='optional model name suffix. can be used to prevent name conflict '
                                  'in experiment output serialization directory')

    subparsers = parser.add_subparsers(title='train_models', dest='model_name')

    # Setup parser arguments for main model
    main_model_subparser = subparsers.add_parser("main", description='Train Main Model',
                                                 parents=[base_parser])
    main_model_subparser.add_argument('--seq2vec-choice', type=str, choices=("dan", "gru"),
                                      help='choice of seq2vec. '
                                      'Required if load_serialization_dir not passed.')
    main_model_subparser.add_argument('--embedding-dim', type=int, help='embedding_dim '
                                      'Required if load_serialization_dir not passed.')
    main_model_subparser.add_argument('--num-layers', type=int, help='num layers. '
                                      'Required if load_serialization_dir not passed.')
    main_model_subparser.add_argument('--pretrained-embedding-file', type=str,
                                      help='if passed, use glove embeddings to initialize. '
                                      'the embedding matrix')

    # Setup parser arguments for probing model
    probing_model_subparser = subparsers.add_parser("probing", description='Train Probing Model',
                                                    parents=[base_parser])
    probing_model_subparser.add_argument('--base-model-dir', type=str, required=True,
                                         help='serialization_dir of the base model to probe on. '
                                         'Required for probing with/without load_serialization_dir.')
    probing_model_subparser.add_argument('--layer-num', type=int,
                                         help='layer_num of pretrained representations '
                                              'on which to probe a linear model')

    args = parser.parse_args()

    # Show help message if subparser wasn't triggered.
    if not args.model_name:
        parser.print_help()
        exit()

    # Make sure required configs have been passed. Otherwise, raise exceptions with messages.
    if args.model_name == "main":
        configs_passed = args.seq2vec_choice and args.embedding_dim and args.num_layers
        if not args.load_serialization_dir and not configs_passed:
            raise Exception("To train main model, either pass load-serialization-dir "
                            "or pass seq2vec-choice, embedding-dim and num-layers")
    elif args.model_name == "probing":
        configs_passed = args.base_model_dir and args.layer_num
        if not args.load_serialization_dir and not args.layer_num:
            raise Exception("To train probing model, either pass load-serialization-dir "
                            "or pass layer-num and base-model-dir")

    # Set numpy, tensorflow and python seeds for reproducibility.
    tf.random.set_seed(1337)
    np.random.seed(1337)
    random.seed(13370)

    # Set some constants
    MAX_NUM_TOKENS = 250
    VOCAB_SIZE = 10000
    GLOVE_COMMON_WORDS_PATH = os.path.join("data", "glove_common_words.txt")

    print("Reading training instances.")
    train_instances = read_instances(args.train_data_file_path, MAX_NUM_TOKENS)
    print("Reading validation instances.")
    validation_instances = read_instances(args.validation_data_file_path, MAX_NUM_TOKENS)

    if args.load_serialization_dir:
        print(f"Ignoring the model arguments and loading the "
              f"model from serialization_dir: {args.load_serialization_dir}")

        # Load Vocab
        vocab_path = os.path.join(args.load_serialization_dir, "vocab.txt")
        vocab_token_to_id, vocab_id_to_token = load_vocabulary(vocab_path)

        # Load Model
        classifier = load_pretrained_model(args.load_serialization_dir)
    else:
        # Build Vocabulary
        with open(GLOVE_COMMON_WORDS_PATH) as file:
            glove_common_words = [line.strip() for line in file.readlines() if line.strip()]
        vocab_token_to_id, vocab_id_to_token = build_vocabulary(train_instances, VOCAB_SIZE,
                                                                glove_common_words)

        # Build Config and Model
        if args.model_name == "main":
            config = {"seq2vec_choice": args.seq2vec_choice,
                      "vocab_size": min(VOCAB_SIZE, len(vocab_token_to_id)),
                      "embedding_dim": args.embedding_dim,
                      "num_layers": args.num_layers}
            classifier = MainClassifier(**config)
            config["type"] = "main"
        else:
            config = {"pretrained_model_path": args.base_model_dir,
                      "layer_num": args.layer_num, "classes_num": 2}
            classifier = ProbingClassifier(**config)
            config["type"] = "probing"

    train_instances = index_instances(train_instances, vocab_token_to_id)
    validation_instances = index_instances(validation_instances, vocab_token_to_id)

    if args.model_name == "main" and args.pretrained_embedding_file:
        embeddings = load_glove_embeddings(args.pretrained_embedding_file,
                                           args.embedding_dim,
                                           vocab_id_to_token)
        classifier._embeddings.assign(tf.convert_to_tensor(embeddings))

    optimizer = optimizers.Adam()

    save_serialization_dir = os.path.join("serialization_dirs", args.model_name + args.suffix_name)
    if not os.path.exists(save_serialization_dir):
        os.makedirs(save_serialization_dir)

    training_output = train(classifier, optimizer, train_instances,
                            validation_instances, args.num_epochs,
                            args.batch_size, save_serialization_dir)
    classifier = training_output["model"]
    metrics = training_output["metrics"]

    # Save the used vocabulary
    vocab_path = os.path.join(save_serialization_dir, "vocab.txt")
    save_vocabulary(vocab_id_to_token, vocab_path)

    # Save the used config
    config_path = os.path.join(save_serialization_dir, "config.json")
    with open(config_path, "w") as file:
        json.dump(config, file)

    # Save the training metrics
    metrics_path = os.path.join(save_serialization_dir, "metrics.json")
    with open(metrics_path, "w") as file:
        json.dump(metrics, file)

    print(f"\nFinal model stored in serialization directory: {save_serialization_dir}")
