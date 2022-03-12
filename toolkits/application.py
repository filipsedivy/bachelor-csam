import numpy as np


def get_labels():
    """Getting labels
    Return:
        List of labels
    """
    return (
        'adult', 'adult_half_naked', 'adult_naked',
        'child', 'child_half_naked', 'child_naked',
        'nature', 'other'
    )


def decode_predictions(preds) -> list:
    """Decoding the prediction
    Args:
        preds: Prediction output from the model
    Return:
        Mapping predictions to labels
        [(label, score), (label, score)]
    """
    if type(preds) is np.ndarray:
        preds = preds[0]

    result = list()
    labels = get_labels()

    for i, prediction in enumerate(preds):
        result.append((labels[i], prediction))

    return result


def decode_predictions_dictionary(preds) -> dict:
    predictions = decode_predictions(preds)
    result = {}

    for label, score in predictions:
        result[label] = score

    return result
