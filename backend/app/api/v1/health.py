from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    manager = getattr(request.app.state, "job_manager", None)
    return {
        "status": "ok",
        "api_version": "1",
        "release": "1.0.0-rc1",
        "execution": "bounded-background-jobs",
        "deployment": "local-release-candidate",
        "model_readiness": "not_checked",
        "worker_capacity": {
            "max_workers": getattr(manager, "max_workers", None),
            "max_queue": getattr(manager, "max_queue", None),
        },
        "history": "sqlite",
        "capability_contract": "/api/v1/capabilities",
    }
