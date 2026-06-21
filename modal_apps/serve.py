from __future__ import annotations

import modal

from modal_apps.common import app, serving_image, volumes


@app.function(image=serving_image, volumes=volumes, gpu="B200", timeout=10 * 60)
@modal.asgi_app()
def api():
    from evidence_codec.modal_jobs.serve import build_asgi_app

    return build_asgi_app()
