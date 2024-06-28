from fastapi import APIRouter

from .appointment import appointment_router

router = APIRouter()

router.include_router(appointment_router, prefix="/predict/appointment")
