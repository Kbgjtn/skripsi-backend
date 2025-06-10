from pathlib import Path
from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000


    CLASSIFIER_MODEL_PATH: Path = Field(
        default=Path("assets/models/yolov11n-cls.pt"),
        env="YOLO_CNN_MODEL_PATH",
    )
    DETECTION_MODEL_PATH: Path = Field(
        default=Path("assets/models/yolov8s.pt"),
        env="YOLO_DETECTION_MODEL_PATH",
    )

    UPLOAD_DIRECTORY: Path = Path("assets/uploads")
    PREDICTIONS_DIRECTORY: Path = Path("assets/predictions")
    MODELS_DIRECTORY: Path = Path("assets/models")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
