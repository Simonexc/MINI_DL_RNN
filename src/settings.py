from enum import Enum

ENTITY = "dl-mini"
PROJECT = "DL_PROJECT_RNN"
NAMED_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
ALL_CLASSES = NAMED_CLASSES + ["unknown", "silence"]


class JobType(Enum):
    UPLOAD_DATA = "upload-data"


class ArtifactType(Enum):
    DATASET = "dataset"
