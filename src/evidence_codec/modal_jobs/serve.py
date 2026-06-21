from __future__ import annotations


def build_asgi_app():
    from evidence_codec.serving.api import create_app

    return create_app()
