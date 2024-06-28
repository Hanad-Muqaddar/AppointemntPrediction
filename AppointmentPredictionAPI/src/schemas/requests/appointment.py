from typing import List

from pydantic import BaseModel


class PredictAppointmentRequest(BaseModel):

    data: List[dict]
