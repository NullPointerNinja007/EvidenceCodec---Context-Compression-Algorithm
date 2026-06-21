from __future__ import annotations

from pprint import pprint

from modal_apps.common import app, base_image, volumes


@app.function(image=base_image, volumes=volumes, timeout=10 * 60)
def inspect_data() -> dict:
    from evidence_codec.modal_jobs.inspect_data import inspect_raw_examples

    return inspect_raw_examples()


@app.function(image=base_image, volumes=volumes, timeout=10 * 60)
def inspect_qasper() -> dict:
    from evidence_codec.modal_jobs.inspect_data import inspect_qasper_answers

    return inspect_qasper_answers()


@app.local_entrypoint()
def main(qasper: bool = False) -> None:
    pprint(inspect_qasper.remote() if qasper else inspect_data.remote())
