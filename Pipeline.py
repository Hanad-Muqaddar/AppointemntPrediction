import pandas as pd
import json
import joblib


def map_column_values(column_name, column_value, json_data):
    try:
        for j in json_data:
            if column_name in j.keys():
                mapping_dict = j[column_name]
                lower_mapping_dict = {k.lower(): v for k, v in mapping_dict.items()}
                return lower_mapping_dict[column_value.lower()]
    except Exception as e:
        return sum(lower_mapping_dict.values()) / len(lower_mapping_dict)


def apply_mappings(df, columns_to_encode, json_data):
    for column in columns_to_encode:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda x: map_column_values(column, x, json_data)
            )
    return df


def preprocess_data(df, encoder):
    df["case_linked"] = df["case_linked"].str.lower().replace(("yes", "no"), (1, 0))
    df["referred"] = df["referred"].str.lower().replace(("yes", "no"), (1, 0))
    df["sex"] = df["sex"].str.lower().replace(("female", "male", "other"), (0, 1, 2))
    df["appointment_status"] = (
        df["appointment_status"]
        .str.lower()
        .replace(("not rebooked", "rebooked"), (0, 1))
    )
    df["missed"] = df["missed"].str.lower().replace(("yes", "no"), (1, 0))
    df["appointment_start_time"] = pd.to_datetime(
        df["appointment_start_time"], errors="coerce"
    )
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = df["appointment_start_time"].dt.year - df["date_of_birth"].dt.year
    df["appointment_start_time_day"] = df["appointment_start_time"].dt.day
    df["appointment_start_time_week"] = df["appointment_start_time"].dt.month
    df["appointment_start_time_year"] = df["appointment_start_time"].dt.year
    df["appointment_start_time_hour"] = df["appointment_start_time"].dt.time.apply(
        lambda t: t.hour
    )
    df.drop(["appointment_start_time", "date_of_birth"], axis=1, inplace=True)

    # target encoding
    columns_to_encode = [
        "business_name",
        "patient_status",
        "patient_type",
        "state",
        "occupation",
        "category",
        "billable_item",
        "case_type",
        "appointment_type",
        "customer_type",
    ]
    df = apply_mappings(df, columns_to_encode, encoder)
    return df


def MakePrediction(x):
    with open('Models/Encodings.json', 'r') as openfile:
    # with open("/home/ec2-user/PredictionModel/Models/Encodings.json", "r") as openfile:
        encoder = json.load(openfile)
    x = preprocess_data(x, encoder)
    return x


def result_pred(prediction):
    if prediction == 1:
        return "Not Cancelled"
    elif prediction == 0:
        return "Cancelled"


def main(df):
    prepared_data = MakePrediction(df)
    model = joblib.load('Models/first_model.joblib')
    # model = joblib.load("/home/ec2-user/PredictionModel/Models/first_model.joblib")
    prediction = model.predict(prepared_data)
    return result_pred(prediction)
