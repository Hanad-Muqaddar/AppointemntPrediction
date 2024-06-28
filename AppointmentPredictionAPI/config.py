import os
from enum import Enum
from typing import ClassVar

from dotenv import load_dotenv
from pydantic import PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings

load_dotenv()


class Classes(str, Enum):
    CANCELLED = "cancelled"
    Not_CANCELLED = "not cancelled"


class BaseConfig(BaseSettings):
    class Config:
        case_sensitive = True


class Config(BaseConfig):

    classes: ClassVar = Classes
    model: str = "artifact/first_model.joblib"
    encoder: str = "artifact/Encodings.json"
    classes_dict: dict = {0: classes.CANCELLED.value, 1: classes.Not_CANCELLED.value}


config = Config()
