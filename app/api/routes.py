"""
REST API Routes for Call Recording & Analytics Dashboard
+ Script/Context Management
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from sqlalchemy import func, desc, cast, Integer, Date, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from datetime import datetime, timedelta, date
from typing import Optional, List
import os
import subprocess
import asyncio

from app.core.database import get_db, Call, Recording, CallOutcome
from pydantic import BaseModel
from app.core import config
from app.core.config_manager import get_config_manager, BotConfig
from app.managers.scripts import get_script_manager

# Absolute Recording Directory (Must match test.py)
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


# ═══════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS (API Request/Response Models)
# ═══════════════════════════════════════════════════════════════════

class CallSchema(BaseModel):
    id: int
    call_uuid: str
    phone_number: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    status: str
    has_recording: bool
    customer_agreed: Optional[bool]
    commitment_date: Optional[date]
    unclear_response: Optional[bool]
    disposition: Optional[str]
    notes: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    structured_data: Optional[str] = None
    
    class Config:
        from_attributes = True


class AnalyticsSummary(BaseModel):
    total_calls: int
    completed_calls: int
    total_agreed: int
    total_declined: int
    total_unclear: int
    agreement_percentage: float
    avg_call_duration: float
    total_recording_size_mb: float


class DailyStats(BaseModel):
    date: str
    total_calls: int
    agreed_calls: int
    agreement_rate: float


# ═══════════════════════════════════════════════════════════════════
# API ROUTER
# ═══════════════════════════════════════════════════════════════════

router = APIRouter(prefix=config.API_PREFIX)


@router.get("/calls", response_model=List[CallSchema])
def get_calls(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=config.MAX_PAGE_SIZE),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    phone_number: Optional[str] = None,
    status: Optional[str] = None,
    outcome: Optional[str] = None,  # agreed, declined, unclear
    db: Session = Depends(get_db)
):
    """
    Get list of calls with filtering and pagination
    """
    query = db.query(Call).outerjoin(Recording).outerjoin(CallOutcome)
    
    # Apply filters
    if start_date:
        query = query.filter(Call.start_time >= start_date)
    if end_date:
        query = query.filter(Call.start_time <= end_date)
    if phone_number:
        query = query.filter(Call.phone_number.like(f"%{phone_number}%"))
    if status:
        query = query.filter(Call.status == status)
    
    # Filter by outcome
    if outcome == "agreed":
        query = query.filter(CallOutcome.customer_agreed == True)
    elif outcome == "declined":
        query = query.filter(CallOutcome.customer_agreed == False)
    elif outcome == "unclear":
        query = query.filter(CallOutcome.unclear_response == True)
    
    # Order by most recent first
    query = query.order_by(Call.start_time.desc())
    
    # Pagination
    calls = query.offset(skip).limit(limit).all()
    
    # Transform to schema
    result = []
    for call in calls:
        result.append(CallSchema(
            id=call.id,
            call_uuid=call.call_uuid,
            phone_number=call.phone_number,
            start_time=call.start_time,
            end_time=call.end_time,
            duration_seconds=call.duration_seconds,
            status=call.status,
            has_recording=call.recording is not None,
            customer_agreed=call.outcome.customer_agreed if call.outcome else None,
            commitment_date=call.outcome.commitment_date if call.outcome else None,
            unclear_response=call.outcome.unclear_response if call.outcome else None,
            disposition=call.outcome.disposition if call.outcome else None,
            notes=call.outcome.notes if call.outcome else None,
            transcript=call.outcome.transcript if call.outcome else None,
            summary=call.outcome.summary if call.outcome else None,
            sentiment=call.outcome.sentiment if call.outcome else None,
            structured_data=call.outcome.structured_data if call.outcome else None,
        ))
    
    return result


@router.get("/calls/{call_id}")
def get_call(call_id: int, db: Session = Depends(get_db)):
    """Get specific call details"""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return CallSchema(
        id=call.id,
        call_uuid=call.call_uuid,
        phone_number=call.phone_number,
        start_time=call.start_time,
        end_time=call.end_time,
        duration_seconds=call.duration_seconds,
        status=call.status,
        has_recording=call.recording is not None,
        customer_agreed=call.outcome.customer_agreed if call.outcome else None,
        commitment_date=call.outcome.commitment_date if call.outcome else None,
        unclear_response=call.outcome.unclear_response if call.outcome else None,
        disposition=call.outcome.disposition if call.outcome else None,
        notes=call.outcome.notes if call.outcome else None,
        transcript=call.outcome.transcript if call.outcome else None,
        summary=call.outcome.summary if call.outcome else None,
        sentiment=call.outcome.sentiment if call.outcome else None,
        structured_data=call.outcome.structured_data if call.outcome else None,
    )


@router.get("/calls/{call_id}/recording")
def get_recording(call_id: int, db: Session = Depends(get_db)):
    """Stream or download call recording"""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call or not call.recording:
        print(f"[API ERROR] Recording record not found in DB for call_id: {call_id}")
        raise HTTPException(status_code=404, detail="Recording not found")
    
    file_path = call.recording.file_path
    print(f"[API] Serving recording from DB path: {file_path}")
    
    # Support for S3 / Remote URLs
    if file_path.startswith("http"):
        # ... (keep S3 streaming logic)
        try:
            import boto3
            from fastapi.responses import StreamingResponse
            
            # Extract key from S3 URL
            object_name = file_path.split("/")[-1]
            
            s3 = boto3.client('s3', 
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION
            )
            
            s3_response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=object_name)
            
            return StreamingResponse(
                s3_response['Body'].iter_chunks(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"inline; filename=call_{call.call_uuid}.wav"
                }
            )
        except Exception as e:
            print(f"[API ERROR] S3 Streaming failed: {e}")
            raise HTTPException(status_code=500, detail=f"S3 Streaming failed: {str(e)}")

    # FALLBACK: If local file not found at path, try looking in absolute RECORDINGS_DIR
    if not os.path.exists(file_path):
        filename = os.path.basename(file_path)
        fallback_path = os.path.join(RECORDINGS_DIR, filename)
        print(f"[API] Original path not found. Trying fallback: {fallback_path}")
        if os.path.exists(fallback_path):
            file_path = fallback_path
        else:
            print(f"[API ERROR] Recording file not found anywhere: {file_path}")
            raise HTTPException(status_code=404, detail="Recording file not found on disk")
    
    # Return as streaming response
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"call_{call.call_uuid}.wav"
    )


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def get_analytics_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get overall analytics summary"""
    query = db.query(Call).outerjoin(CallOutcome)
    
    # Apply date filters
    if start_date:
        query = query.filter(Call.start_time >= start_date)
    if end_date:
        query = query.filter(Call.start_time <= end_date)
    
    # Count totals
    total_calls = query.count()
    completed_calls = query.filter(Call.status == "completed").count()
    
    # Outcome counts
    total_agreed = query.filter(CallOutcome.customer_agreed == True).count()
    total_declined = query.filter(CallOutcome.customer_agreed == False).count()
    total_unclear = query.filter(CallOutcome.unclear_response == True).count()
    
    # Agreement percentage
    agreement_percentage = 0.0
    if completed_calls > 0:
        agreement_percentage = round((total_agreed / completed_calls) * 100, 2)
    
    # Average call duration
    avg_duration = db.query(func.avg(Call.duration_seconds)).filter(
        Call.duration_seconds.isnot(None)
    ).scalar() or 0.0
    
    # Total recording size
    total_size_bytes = db.query(func.sum(Recording.file_size_bytes)).scalar() or 0
    total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
    
    return AnalyticsSummary(
        total_calls=total_calls,
        completed_calls=completed_calls,
        total_agreed=total_agreed,
        total_declined=total_declined,
        total_unclear=total_unclear,
        agreement_percentage=agreement_percentage,
        avg_call_duration=round(avg_duration, 2),
        total_recording_size_mb=total_size_mb
    )


@router.get("/analytics/daily", response_model=List[DailyStats])
async def get_daily_analytics(
    days: int = Query(30, le=365),
    db: Session = Depends(get_db)
):
    """Get daily call statistics for trend charts"""
    start_date = datetime.now() - timedelta(days=days)
    
    # Group by date
    results = db.query(
        func.date(Call.start_time).label("date"),
        func.count(Call.id).label("total_calls"),
        func.sum(func.cast(CallOutcome.customer_agreed, Integer)).label("agreed_calls")
    ).outerjoin(CallOutcome).filter(
        Call.start_time >= start_date
    ).group_by(
        func.date(Call.start_time)
    ).order_by(
        func.date(Call.start_time)
    ).all()
    
    # Calculate agreement rates
    daily_stats = []
    for row in results:
        total = row.total_calls or 0
        agreed = row.agreed_calls or 0
        rate = round((agreed / total) * 100, 2) if total > 0 else 0.0
        
        daily_stats.append(DailyStats(
            date=str(row.date),
            total_calls=total,
            agreed_calls=agreed,
            agreement_rate=rate
        ))
    
    return daily_stats


@router.delete("/calls/{call_id}")
async def delete_call(call_id: int, db: Session = Depends(get_db)):
    """Delete call and associated recording (optional feature)"""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Delete recording file from disk
    if call.recording and os.path.exists(call.recording.file_path):
        try:
            os.remove(call.recording.file_path)
        except Exception as e:
            print(f"[API] Failed to delete recording file: {e}")
    
    # Delete from database (cascade will handle recording and outcome)
    db.delete(call)
    db.commit()
    
    return {"status": "deleted", "call_id": call_id}


# ═══════════════════════════════════════════════════════════════════
# PRODUCTION FEATURES
# ═══════════════════════════════════════════════════════════════════

import io
import csv
import time
import psutil

# Track server start time for uptime
_SERVER_START_TIME = time.time()
_ACTIVE_CALLS = set()  # Track active call UUIDs

def register_active_call(uuid: str):
    _ACTIVE_CALLS.add(uuid)

def unregister_active_call(uuid: str):
    _ACTIVE_CALLS.discard(uuid)


@router.get("/health")
async def health_check():
    """Server health status with uptime and resource usage"""
    uptime_seconds = time.time() - _SERVER_START_TIME
    
    # Memory usage
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        "status": "online",
        "uptime_seconds": round(uptime_seconds),
        "uptime_display": _format_uptime(uptime_seconds),
        "active_calls": len(_ACTIVE_CALLS),
        "memory_mb": round(mem_info.rss / 1024 / 1024, 1),
        "cpu_percent": process.cpu_percent(interval=0.1),
        "timestamp": datetime.utcnow().isoformat()
    }


def _format_uptime(seconds):
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    mins = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {mins}m"
    elif hours > 0:
        return f"{hours}h {mins}m"
    else:
        return f"{mins}m"


@router.get("/analytics/today")
async def get_today_stats(db: Session = Depends(get_db)):
    """Get today-only summary stats"""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    calls = db.query(Call).filter(Call.start_time >= today_start).all()
    
    total = len(calls)
    agreed = sum(1 for c in calls if c.outcome and c.outcome.customer_agreed == True)
    declined = sum(1 for c in calls if c.outcome and c.outcome.customer_agreed == False)
    avg_dur = sum(c.duration_seconds or 0 for c in calls) / max(total, 1)
    
    return {
        "today_calls": total,
        "today_agreed": agreed,
        "today_declined": declined,
        "today_rate": round((agreed / max(total, 1)) * 100, 1),
        "today_avg_duration": round(avg_dur, 1)
    }


@router.get("/analytics/dispositions")
async def get_disposition_breakdown(db: Session = Depends(get_db)):
    """Get disposition breakdown for pie chart"""
    outcomes = db.query(CallOutcome).all()
    
    breakdown = {}
    for o in outcomes:
        disp = o.disposition or "Unknown"
        breakdown[disp] = breakdown.get(disp, 0) + 1
    
    return [{"disposition": k, "count": v} for k, v in sorted(breakdown.items(), key=lambda x: -x[1])]


@router.get("/analytics/hourly")
async def get_hourly_analytics(
    days: int = Query(7, le=30),
    db: Session = Depends(get_db)
):
    """Get calls grouped by hour for heatmap"""
    since = datetime.utcnow() - timedelta(days=days)
    calls = db.query(Call).filter(Call.start_time >= since).all()
    
    hourly = {}
    for c in calls:
        hour = c.start_time.hour
        hourly[hour] = hourly.get(hour, 0) + 1
    
    return [{"hour": h, "calls": hourly.get(h, 0)} for h in range(24)]


@router.get("/calls/export/csv")
async def export_calls_csv(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    outcome: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export filtered call data as CSV download"""
    query = db.query(Call).order_by(desc(Call.start_time))
    
    if start_date:
        query = query.filter(Call.start_time >= start_date)
    if end_date:
        query = query.filter(Call.start_time <= end_date)
    
    calls = query.all()
    
    # Filter by outcome
    if outcome:
        filtered = []
        for c in calls:
            if outcome == "agreed" and c.outcome and c.outcome.customer_agreed == True:
                filtered.append(c)
            elif outcome == "declined" and c.outcome and c.outcome.customer_agreed == False:
                filtered.append(c)
            elif outcome == "unclear" and (not c.outcome or c.outcome.customer_agreed is None):
                filtered.append(c)
        calls = filtered
    
    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Call ID", "UUID", "Phone Number", "Start Time", "End Time",
        "Duration (s)", "Status", "Agreed", "Disposition",
        "Commitment Date", "Notes", "Has Recording"
    ])
    
    for c in calls:
        writer.writerow([
            c.id,
            c.call_uuid,
            c.phone_number or "",
            c.start_time.strftime("%Y-%m-%d %H:%M:%S") if c.start_time else "",
            c.end_time.strftime("%Y-%m-%d %H:%M:%S") if c.end_time else "",
            round(c.duration_seconds, 1) if c.duration_seconds else "",
            c.status,
            "Yes" if c.outcome and c.outcome.customer_agreed else ("No" if c.outcome and c.outcome.customer_agreed == False else "Unclear"),
            c.outcome.disposition if c.outcome else "",
            c.outcome.commitment_date.strftime("%Y-%m-%d") if c.outcome and c.outcome.commitment_date else "",
            c.outcome.notes if c.outcome else "",
            "Yes" if c.recording else "No"
        ])
    
    csv_content = output.getvalue()
    
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=call_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )


@router.post("/calls/bulk-delete")
async def bulk_delete_calls(data: dict, db: Session = Depends(get_db)):
    """Delete multiple calls at once"""
    call_ids = data.get("call_ids", [])
    if not call_ids:
        raise HTTPException(status_code=400, detail="No call IDs provided")
    
    deleted = 0
    for cid in call_ids:
        call = db.query(Call).filter(Call.id == cid).first()
        if call:
            if call.recording and call.recording.file_path:
                try:
                    if os.path.exists(call.recording.file_path):
                        os.remove(call.recording.file_path)
                except Exception:
                    pass
            db.delete(call)
            deleted += 1
    
    db.commit()
    return {"status": "success", "deleted": deleted}


@router.patch("/calls/{call_id}/notes")
def update_call_notes(call_id: int, data: dict, db: Session = Depends(get_db)):
    """Update notes for a call"""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if not call.outcome:
        call.outcome = CallOutcome(call_id=call.id)
        db.add(call.outcome)
    
    call.outcome.notes = data.get("notes", "")
    db.commit()
    
    return {"status": "success", "call_id": call_id}


@router.patch("/calls/{call_id}/disposition")
def update_call_disposition(call_id: int, data: dict, db: Session = Depends(get_db)):
    """Update disposition/outcome for a call"""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if not call.outcome:
        call.outcome = CallOutcome(call_id=call.id)
        db.add(call.outcome)
    
    disposition = data.get("disposition", "")
    call.outcome.disposition = disposition
    
    # Auto-set agreed flag based on disposition
    disp_lower = disposition.lower()
    if any(w in disp_lower for w in ["agreed", "interested", "yes", "accept"]):
        call.outcome.customer_agreed = True
    elif any(w in disp_lower for w in ["declined", "not interested", "no", "reject"]):
        call.outcome.customer_agreed = False
    else:
        call.outcome.customer_agreed = None
        call.outcome.unclear_response = True
    
    db.commit()
    
    return {"status": "success", "call_id": call_id}


# ═══════════════════════════════════════════════════════════════════
# SETTINGS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

@router.get("/settings")
async def get_settings():
    """Get current bot configuration settings"""
    try:
        config_mgr = get_config_manager()
        bot_config = config_mgr.load_config()
        return bot_config.dict()
    except Exception as e:
        print(f"[API ERROR] Failed to load settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load settings: {str(e)}")


@router.post("/settings")
async def update_settings(updates: dict):
    """Update bot settings and trigger server restart"""
    try:
        config_mgr = get_config_manager()
        
        # Validate and save settings
        updated_config = config_mgr.update_settings(updates)
        
        print(f"[API] Settings updated successfully")
        print(f"[API] New SILENCE_TIMEOUT: {updated_config.vad.silence_timeout}s")
        print(f"[API] New INTERRUPTION_THRESHOLD: {updated_config.vad.interruption_threshold_db}dB")
        
        # Schedule server restart (delayed to allow response to send)
        asyncio.create_task(restart_server_delayed())
        
        return {
            "status": "success",
            "message": "Settings saved successfully. Server will restart in 2 seconds...",
            "config": updated_config.dict()
        }
    except Exception as e:
        print(f"[API ERROR] Failed to update settings: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid settings: {str(e)}")


@router.post("/restart")
async def restart_server():
    """Manually restart PM2 server process"""
    try:
        print("[API] Manual server restart requested")
        asyncio.create_task(restart_server_delayed())
        return {
            "status": "success", 
            "message": "Server restarting in 2 seconds..."
        }
    except Exception as e:
        print(f"[API ERROR] Failed to restart server: {e}")
        raise HTTPException(status_code=500, detail=f"Restart failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════
# SCRIPT/CONTEXT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

@router.get("/scripts")
async def list_scripts():
    """List all saved bot scripts"""
    try:
        mgr = get_script_manager()
        scripts = mgr.list_scripts()
        active_id = mgr.get_active_id()
        return {
            "scripts": scripts,
            "active_script_id": active_id
        }
    except Exception as e:
        print(f"[API ERROR] Failed to list scripts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scripts/active")
async def get_active_script():
    """Get the currently active script"""
    try:
        mgr = get_script_manager()
        script = mgr.get_active_script()
        return {
            "script": script,
            "active_script_id": mgr.get_active_id()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scripts/{script_id}")
async def get_script(script_id: str):
    """Get a single script by ID"""
    try:
        mgr = get_script_manager()
        script = mgr.get_script(script_id)
        if not script:
            raise HTTPException(status_code=404, detail="Script not found")
        return script
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scripts")
async def save_script(data: dict):
    """Create or update a script"""
    try:
        mgr = get_script_manager()
        
        # Validate required fields
        if not data.get('name'):
            raise HTTPException(status_code=400, detail="Script name is required")
        if not data.get('opener'):
            raise HTTPException(status_code=400, detail="Opening line is required")
        
        script = mgr.save_script(data)
        return {
            "status": "success",
            "message": f"Script '{script['name']}' saved successfully",
            "script": script
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API ERROR] Failed to save script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/scripts/{script_id}")
async def delete_script(script_id: str):
    """Delete a script"""
    try:
        mgr = get_script_manager()
        mgr.delete_script(script_id)
        return {"status": "success", "message": f"Script '{script_id}' deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scripts/{script_id}/activate")
async def activate_script(script_id: str):
    """Activate a script and restart server to apply"""
    try:
        mgr = get_script_manager()
        script = mgr.set_active(script_id)
        
        # Schedule server restart to apply the new script
        asyncio.create_task(restart_server_delayed())
        
        return {
            "status": "success",
            "message": f"Script '{script['name']}' activated. Server restarting in 2s...",
            "script": script
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[API ERROR] Failed to activate script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def restart_server_delayed():
    """Restart PM2 process after delay to allow HTTP response to send"""
    print("[API] Waiting 2s before restart...")
    await asyncio.sleep(2)
    
    try:
        # Find PM2 process name (try common names)
        process_names = ["callex-AI-AMD", "test", "voice-bot"]
        
        for process_name in process_names:
            try:
                result = subprocess.run(
                    ["pm2", "restart", process_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    print(f"[API] ✅ PM2 process '{process_name}' restarted successfully")
                    return
                else:
                    print(f"[API] PM2 restart '{process_name}' returned: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"[API] PM2 restart '{process_name}' timed out")
            except FileNotFoundError:
                print("[API] ⚠️ PM2 not found in PATH")
                break
        
        print("[API] ⚠️ Could not restart PM2 process automatically")
        
    except Exception as e:
        print(f"[API ERROR] Restart failed: {e}")
