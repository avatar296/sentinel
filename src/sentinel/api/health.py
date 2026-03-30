from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.database import get_db

router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {"status": "ok", "db": db_status}


@router.get("/drift")
async def drift_check(request: Request):
    detector = request.app.state.drift_detector
    report = detector.check()
    return {
        "is_drifted": report.is_drifted,
        "psi": round(report.psi, 4),
        "mean_shift": round(report.mean_shift, 4),
        "std_shift": round(report.std_shift, 4),
        "current_mean": round(report.current_mean, 4),
        "baseline_mean": round(report.baseline_mean, 4),
        "sample_size": report.sample_size,
        "baseline_size": report.baseline_size,
        "ready": detector.is_ready,
    }
