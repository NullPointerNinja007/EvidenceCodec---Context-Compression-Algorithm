from __future__ import annotations

from modal_apps.common import app, commit_all, training_image, volumes


@app.function(image=training_image, volumes=volumes, gpu="B200", timeout=4 * 60 * 60)
def evaluate(config: str = "configs/default.yaml", suite: str = "offline") -> dict:
    from evidence_codec.modal_jobs.evaluate import run_evaluation

    result = run_evaluation(config_path=config, suite=suite)
    commit_all()
    return result


@app.local_entrypoint()
def main(config: str = "configs/default.yaml", suite: str = "offline") -> None:
    evaluate.remote(config=config, suite=suite)
