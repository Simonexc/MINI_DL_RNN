import os

from settings import ALL_CLASSES, UNKNOWN_CLASS


def load_annotation_file(path: str, base_dir: str) -> set[str]:
    paths = set()
    with open(path, "r") as file:
        for line in file.readlines():
            paths.add(os.path.join(base_dir, line.replace("\n", "")))

    return paths


def get_class_name_from_audio_path(audio_path: str) -> str:
    class_name = audio_path.split(os.sep)[-2]
    if class_name == "_background_noise_":
        raise ValueError(f"Invalid class name: {class_name}")
    if class_name not in ALL_CLASSES:
        class_name = UNKNOWN_CLASS

    return class_name


def get_class_id_from_audio_path(audio_path: str) -> int:
    class_name = get_class_name_from_audio_path(audio_path)
    return ALL_CLASSES.index(class_name)
