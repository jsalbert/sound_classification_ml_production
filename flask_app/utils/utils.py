import json
import numpy as np


def process_predictions(predictions, classes_dictionary_json):
    with open(classes_dictionary_json, 'r') as file:
        class_dictionary = json.load(file)
    # Sort classes by keys
    classes = [class_dictionary[key] for key in sorted(class_dictionary.keys())]
    classes = np.array(classes)
    top_predictions_indices = np.argsort(predictions)[::-1]

    return predictions[top_predictions_indices], classes[top_predictions_indices]
