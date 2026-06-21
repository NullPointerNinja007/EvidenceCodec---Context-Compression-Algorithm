from __future__ import annotations

from modal_apps.common import app, base_image, commit_all, volumes


@app.function(image=base_image, volumes=volumes, timeout=4 * 60 * 60)
def prepare_data(include_longbench: bool = True) -> dict:
    from evidence_codec.modal_jobs.prepare_data import prepare_required_data

    result = prepare_required_data(include_longbench=include_longbench)
    commit_all()
    return result


@app.local_entrypoint()
def main(include_longbench: bool = True) -> None:
    result = prepare_data.remote(include_longbench=include_longbench)
    print(result)
