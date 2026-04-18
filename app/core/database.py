"""
Database Models and Configuration for Call Recording System
Using SQLAlchemy ORM
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool
from app.core import config

# Create engine with connection pooling
# Ensure the data directory exists for SQLite
import os as _os
if config.DATABASE_URL.startswith("sqlite"):
    _db_path = config.DATABASE_URL.replace("sqlite:///", "")
    _db_dir = _os.path.dirname(_db_path)
    if _db_dir:
        _os.makedirs(_db_dir, exist_ok=True)
    # SQLite-specific configuration
    engine = create_engine(
        config.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    # PostgreSQL/MySQL configuration
    engine = create_engine(
        config.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ═══════════════════════════════════════════════════════════════════
# DATABASE MODELS
# ═══════════════════════════════════════════════════════════════════

class Call(Base):
    """Main call record table"""
    __tablename__ = "calls"
    
    id = Column(Integer, primary_key=True, index=True)
    call_uuid = Column(String(100), unique=True, index=True, nullable=False)
    phone_number = Column(String(20), index=True)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    status = Column(String(20), default="in_progress")  # in_progress, completed, disconnected, error
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    recording = relationship("Recording", back_populates="call", uselist=False, cascade="all, delete-orphan")
    outcome = relationship("CallOutcome", back_populates="call", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Call(uuid={self.call_uuid}, phone={self.phone_number}, status={self.status})>"


class Recording(Base):
    """Audio recording file metadata"""
    __tablename__ = "recordings"
    
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer)
    format = Column(String(10), default="wav")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    call = relationship("Call", back_populates="recording")
    
    def __repr__(self):
        return f"<Recording(call_id={self.call_id}, size={self.file_size_bytes})>"


class CallOutcome(Base):
    """Customer response and agreement tracking"""
    __tablename__ = "call_outcomes"
    
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    # Outcome tracking
    customer_agreed = Column(Boolean, nullable=True)  # True=agreed, False=declined, None=unclear
    commitment_date = Column(DateTime, nullable=True)  # When customer said they'll recharge
    unclear_response = Column(Boolean, default=False)
    disposition = Column(String(100))  # e.g., "Interested", "NOT Interested", "Busy"
    
    # Additional context
    notes = Column(Text)
    transcript = Column(Text)  # Full conversation transcript (User: ... / Bot: ...)
    summary = Column(Text)  # AI-generated call summary
    sentiment = Column(String(20))  # positive / negative / neutral
    structured_data = Column(Text)  # JSON string of highlighted_points + custom fields
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    call = relationship("Call", back_populates="outcome")
    
    def __repr__(self):
        return f"<CallOutcome(call_id={self.call_id}, agreed={self.customer_agreed})>"


# ═══════════════════════════════════════════════════════════════════
# DATABASE UTILITIES
# ═══════════════════════════════════════════════════════════════════

def init_db():
    """Initialize database schema"""
    Base.metadata.create_all(bind=engine)
    
    # Migrate: add missing columns to existing tables
    # (create_all only creates missing tables, not missing columns)
    _migrate_add_columns()
    
    print("[DATABASE] Schema initialized successfully")


def _migrate_add_columns():
    """Safely add new columns to existing tables"""
    from sqlalchemy import text, inspect
    
    inspector = inspect(engine)
    
    # Check if new columns exist in call_outcomes
    if 'call_outcomes' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('call_outcomes')]
        new_cols = {
            'transcript': 'TEXT',
            'summary': 'TEXT',
            'sentiment': 'VARCHAR(20)',
            'structured_data': 'TEXT'
        }
        for col_name, col_type in new_cols.items():
            if col_name not in columns:
                try:
                    with engine.begin() as conn:
                        conn.execute(text(f"ALTER TABLE call_outcomes ADD COLUMN {col_name} {col_type}"))
                    print(f"[DATABASE] ✅ Migrated: added '{col_name}' column to call_outcomes")
                except Exception as e:
                    print(f"[DATABASE] Migration note: {e}")


def get_db() -> Session:
    """Get database session (use with dependency injection)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get database session (direct usage)"""
    return SessionLocal()


def update_call_outcome(db: Session, call_uuid: str, disposition: str):
    """
    Update or create high-level outcome for a call.
    Automatically parses common dispositions into agreed/declined/unclear flags.
    """
    call = db.query(Call).filter(Call.call_uuid == call_uuid).first()
    if not call:
        print(f"[DB] Error: Cannot update outcome, call {call_uuid} not found.")
        return False
        
    outcome = call.outcome
    if not outcome:
        outcome = CallOutcome(call_id=call.id)
        db.add(outcome)
    
    outcome.disposition = disposition
    
    # Heuristic mapping for boolean flags
    disp_lower = " " + disposition.lower() + " "
    if any(word in disp_lower for word in [" unclear ", " not sure ", " maybe "]):
        outcome.customer_agreed = None
        outcome.unclear_response = True
    elif any(word in disp_lower for word in [" not interested ", " decline ", " reject ", " no "]):
        outcome.customer_agreed = False
        outcome.unclear_response = False
    elif any(word in disp_lower for word in [" interested ", " agree ", " yes ", " accept "]):
        outcome.customer_agreed = True
        outcome.unclear_response = False
        
    db.commit()
    print(f"[DB] Call {call_uuid} disposition updated to: {disposition}")
    return True


# Initialize on import (graceful — don't crash app if DB is unavailable)
try:
    init_db()
except Exception as _db_err:
    print(f"[DATABASE] ⚠️ Schema init deferred (non-fatal): {_db_err}")
