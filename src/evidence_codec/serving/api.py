from __future__ import annotations


def create_app():
    from fastapi import FastAPI

    app = FastAPI(title="EvidenceCodec")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
