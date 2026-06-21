from __future__ import annotations

import modal

APP_NAME = "evidence-codec"

VOLUME_PREFIX = "evidencecodec-ccalg-20260621"

DATA_MOUNT = "/data"
MODELS_MOUNT = "/models"
RUNS_MOUNT = "/runs"
CACHE_MOUNT = "/cache"

data_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-data", create_if_missing=True)
models_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-models", create_if_missing=True)
runs_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-cache", create_if_missing=True)

volumes = {
    DATA_MOUNT: data_volume,
    MODELS_MOUNT: models_volume,
    RUNS_MOUNT: runs_volume,
    CACHE_MOUNT: cache_volume,
}

app = modal.App(APP_NAME)

base_deps_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "datasets<4",
        "numpy",
        "pandas",
        "pyyaml",
        "rank-bm25",
        "scikit-learn",
        "tiktoken",
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "HF_HOME": f"{CACHE_MOUNT}/huggingface",
            "HF_DATASETS_CACHE": f"{CACHE_MOUNT}/huggingface/datasets",
            "TRANSFORMERS_CACHE": f"{CACHE_MOUNT}/huggingface/transformers",
            "EVIDENCE_CODEC_DATA_DIR": DATA_MOUNT,
            "EVIDENCE_CODEC_MODELS_DIR": MODELS_MOUNT,
            "EVIDENCE_CODEC_RUNS_DIR": RUNS_MOUNT,
            "EVIDENCE_CODEC_CACHE_DIR": CACHE_MOUNT,
        }
    )
)

def with_project_sources(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir("src", "/root/src", copy=False)
        .add_local_dir("configs", "/root/configs", copy=False)
        .add_local_dir("modal_apps", "/root/modal_apps", copy=False)
    )


base_image = with_project_sources(base_deps_image)

training_deps_image = base_deps_image.uv_pip_install(
    "accelerate",
    "evaluate",
    "matplotlib",
    "sentence-transformers",
    "torch<3",
    "transformers",
)

training_image = with_project_sources(training_deps_image)

student_image = with_project_sources(base_deps_image.uv_pip_install("lightgbm"))

serving_image = with_project_sources(
    training_deps_image.uv_pip_install("fastapi", "pydantic", "uvicorn")
)


def commit_all() -> None:
    """Persist writes made to attached Modal Volumes."""
    data_volume.commit()
    models_volume.commit()
    runs_volume.commit()
    cache_volume.commit()


def reload_all() -> None:
    """Refresh attached Modal Volumes before reading writes from other containers."""
    data_volume.reload()
    models_volume.reload()
    runs_volume.reload()
    cache_volume.reload()
