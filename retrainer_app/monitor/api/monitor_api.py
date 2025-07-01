from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from retrainer_app.core.db import get_db
from retrainer_app.monitor.services.monitor_service import MonitorService
from retrainer_app.monitor.schemas.monitor import MonitorResponse

router = APIRouter()

@router.post("/run", response_model=MonitorResponse)
def run_monitoring(db: Session = Depends(get_db)):
    """
    Triggers the monitoring service to check for low-confidence predictions.
    """
    try:
        monitor_service = MonitorService(db)
        response = monitor_service.check_predictions_and_trigger_retraining()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
