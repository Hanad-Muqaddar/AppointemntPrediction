import io
import os
import shutil
from ast import Dict
from typing import List

import pandas as pd
from fastapi import UploadFile

from src.core.predicts.appointment import make_prediction


class Appointment:

    record_dirs = "tmp/records"

    async def predict_appointment(self, patient_record: List[dict]):
        # if not os.path.isdir(self.record_dirs):
        #     os.mkdir(self.record_dirs)
        # file_path = "%s/%s" % (self.record_dirs, patient_record.filename)
        # with open(file_path, "wb") as file_buffer:

        #     file_content = await patient_record.read()
        #     file_buffer.write(file_content)
        record_df = pd.json_normalize((patient_record))
        # record_df = pd.read_csv(file_path)
        # os.remove(file_path)
        prediction = make_prediction(record_df)
        return prediction
