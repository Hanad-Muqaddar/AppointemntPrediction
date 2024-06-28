import json

import joblib
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier

from config import config

from .utils import preprocess_data


def make_prediction(x: pd.DataFrame):
    with open(config.encoder, "r") as openfile:
        encoder = json.load(openfile)
    # print(encoder)
    x_processed = preprocess_data(x, encoder)

    model: BalancedRandomForestClassifier = joblib.load(config.model)
    preddictopn_proba = model.predict_proba(x_processed)
    prediction = model.predict(x_processed)

    output = {"choice": []}
    for pred, prob in zip(prediction, preddictopn_proba):
        output["choice"].append(
            {
                "prediction": int(pred),
                "category": config.classes_dict[int(pred)],
                "probality": {"0": float(prob[0]), "1": float(prob[1])},
            }
        )

    return output
