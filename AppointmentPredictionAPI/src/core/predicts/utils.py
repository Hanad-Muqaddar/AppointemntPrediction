from typing import Dict, List

import pandas as pd


def map_column_values(column_name, value, encoding_dicts):
    for key in encoding_dicts:
        if column_name in encoding_dicts[key].keys():
            mapping_dict = key[column_name]
            # print(mapping_dict)
            return mapping_dict[value]
    return 0


def apply_mappings(df: pd.DataFrame, columns_to_encode: List[str], encoding_dict: Dict):
    for column in columns_to_encode:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda x: map_column_values(
                    column_name=column, value=x, encoding_dicts=encoding_dict
                )
            )
    return df


def preprocess_data(df: pd.DataFrame, encoder: Dict):

    df["case_linked"] = df["case_linked"].str.lower().replace({"yes": 1, "no": 0})
    df["referred"] = df["referred"].str.lower().replace({"yes": 1, "no": 0})
    df["sex"] = df["sex"].str.lower().replace({"female": 0, "male": 1, "other": 2})
    df["appointment_status"] = (
        df["appointment_status"].str.lower().replace({"not rebooked": 0, "rebooked": 1})
    )
    # df["missed"] = df["missed"].str.lower().replace({"yes": 1, "no": 0})
    df["appointment_start_time"] = pd.to_datetime(df["appointment_start_time"], errors="coerce")
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = df["appointment_start_time"].dt.year - df["date_of_birth"].dt.year
    df["appointment_start_time_day"] = df["appointment_start_time"].dt.day
    df["appointment_start_time_week"] = df["appointment_start_time"].dt.month
    df["appointment_start_time_year"] = df["appointment_start_time"].dt.year
    df["appointment_start_time_hour"] = df["appointment_start_time"].dt.time.apply(lambda t: t.hour)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
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
    df = df.drop(
        [
            "time_of_day",
            "month_period",
            "day_of_week",
            "month_of_year",
            "day_of_month",
            "title",
            "city",
            "hospital_id",
            "next_appointment_time",
            "cancelled_at",
        ],
        errors="ignore",
        axis=1,
    )
    return df
