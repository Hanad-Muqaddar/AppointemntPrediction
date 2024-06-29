import os
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.schemas.requests.appointment import PredictAppointmentRequest
from src.schemas.response.appointment import PredictAppointment
from src.services.appointment import Appointment

# Use token based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def return_api_key():
    with open('apikey.json') as f:
        data = json.load(f)
    return data['API_KEY']

# Ensure the request is authenticated
def auth_request(token: str = Depends(oauth2_scheme)) -> bool:
    authenticated = token == return_api_key()
    return authenticated


appointment_router = APIRouter()


@appointment_router.post("/")
async def predict_appointment(
    patient_record: PredictAppointmentRequest, authenticated: bool = Depends(auth_request)
) -> PredictAppointment:
    if not authenticated:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not Authorized")
    appointment_service = Appointment()
    output = await appointment_service.predict_appointment(patient_record=patient_record.data)
    return PredictAppointment(**output)

