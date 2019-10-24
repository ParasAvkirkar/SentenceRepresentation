import os
import json

from tensorflow.keras import models


def load_pretrained_model(serialization_dir: str) -> models.Model:
    """
    Given serialization directory, returns: model loaded with the pretrained weights.
    """

    # Load Config
    config_path = os.path.join(serialization_dir, "config.json")
    model_path = os.path.join(serialization_dir, "model.ckpt.index")

    model_files_present = all([os.path.exists(path)
                               for path in [config_path, model_path]])
    if not model_files_present:
        raise Exception(f"Model files in serialization_dir ({serialization_dir}) "
                        f" are missing. Cannot load_the_model.")

    model_path = model_path.replace(".index", "")

    with open(config_path, "r") as file:
        config = json.load(file)

    # Load Model
    model_name = config.pop("type")
    if model_name == "main":
        from main_model import MainClassifier # To prevent circular imports
        model = MainClassifier(**config)
        model.load_weights(model_path)
    else:
        from probing_model import ProbingClassifier # To prevent circular imports
        model = ProbingClassifier(**config)
        model.load_weights(model_path)

    return model
