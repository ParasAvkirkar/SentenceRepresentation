#pylint: disable = invalid-name
# inbuilt lib imports:
import json
import argparse


def evaluate(gold_data_path: str, prediction_data_path: str) -> float:
    """
    Evaluates accuracy of label predictions in ``prediction_data_path``
    based on gold labels in ``gold_data_path``.
    """
    with open(gold_data_path) as file:
        gold_labels = [int(json.loads(line.strip())["label"])
                       for line in file.readlines() if line.strip()]

    with open(prediction_data_path) as file:
        predicted_labels = [int(line.strip())
                            for line in file.readlines() if line.strip()]

    if len(gold_labels) != len(predicted_labels):
        raise Exception("Number of lines in labels and predictions files don't match.")

    correct_count = sum([1.0 if predicted_label == gold_label else 0.0
                         for predicted_label, gold_label in zip(predicted_labels, gold_labels)])
    total_count = len(predicted_labels)
    return correct_count / total_count

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate classification predictions.')
    parser.add_argument('gold_data_path', type=str, help='gold data file path.')
    parser.add_argument('prediction_data_path', type=str,
                        help='predictions data file path.')

    args = parser.parse_args()
    accuracy = evaluate(args.gold_data_path, args.prediction_data_path)
    print(f"Accuracy: {round(accuracy, 2)}")
