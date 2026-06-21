from __future__ import annotations

from modal_apps.common import app, commit_all, student_image, volumes


@app.function(image=student_image, volumes=volumes, timeout=4 * 60 * 60)
def distill(config: str = "configs/default.yaml") -> dict:
    from evidence_codec.modal_jobs.distill_student import distill_student

    result = distill_student(config_path=config)
    commit_all()
    return result


@app.local_entrypoint()
def main(config: str = "configs/default.yaml") -> None:
    distill.remote(config=config)
