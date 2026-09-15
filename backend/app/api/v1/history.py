from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["history"])


@router.get("/history")
def history(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    repo = request.app.state.history_repository
    items = repo.list(limit=limit)
    return {"items": items, "count": len(items)}
