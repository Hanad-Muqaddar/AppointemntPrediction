from typing import Dict, List

from fastapi import APIRouter, UploadFile

from src.schemas.requests.appointment import PredictAppointmentRequest
from src.schemas.response.appointment import PredictAppointment
from src.services.appointment import Appointment

appointment_router = APIRouter()


@appointment_router.post("/")
async def predict_appointment(patient_record: PredictAppointmentRequest) -> PredictAppointment:
    appointment_service = Appointment()
    output = await appointment_service.predict_appointment(patient_record=patient_record.data)
    return PredictAppointment(**output)
