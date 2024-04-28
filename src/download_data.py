import argparse
import wandb

from settings import PROJECT, ENTITY, JobType


def download_data(artifact_name: str):
    with wandb.init(
            project=PROJECT, entity=ENTITY, job_type=JobType.UPLOAD_DATA.value
    ) as run:
        data_artifact = run.use_artifact(f"{artifact_name}:latest")
        audio_dir = data_artifact.download()

        print(f"Downloaded files are in: {audio_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download dataset from W&B"
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Chose dataset to download from W&B."
    )

    # Parse the arguments
    args = parser.parse_args()

    download_data(args.dataset)
