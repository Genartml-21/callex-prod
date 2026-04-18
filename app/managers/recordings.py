"""
Call Recording Manager
Handles real-time audio streaming, buffering, and WAV file creation
"""
import struct
import asyncio
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from app.core import config


class RecordingManager:
    """
    Manages real-time call recording to disk
    Buffers audio chunks and writes to WAV file with proper headers
    """
    
    def __init__(
        self,
        call_uuid: str,
        sample_rate: int = config.RECORDING_SAMPLE_RATE,
        channels: int = config.RECORDING_CHANNELS
    ):
        self.call_uuid = call_uuid
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = config.RECORDING_FORMAT
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{call_uuid}.{self.format}"
        self.file_path = config.RECORDINGS_DIR / filename
        
        # Audio buffer
        self.audio_buffer = bytearray()
        self.total_bytes = 0
        self.is_closed = False
        self.file_handle = None
        
        # Initialize file
        self._init_file()
    
    def _init_file(self):
        """Initialize recording file with WAV header placeholder"""
        if self.format == "wav":
            # Create file with placeholder header (we'll update it on close)
            self.file_handle = open(self.file_path, "wb")
            # Write temporary header (44 bytes)
            self.file_handle.write(self._create_wav_header(0))
        elif self.format == "pcm":
            # Raw PCM - no header needed
            self.file_handle = open(self.file_path, "wb")
        
        print(f"[Recording] Started: {self.file_path}")
    
    def _create_wav_header(self, data_size: int) -> bytes:
        """Create WAV file header"""
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,  # File size - 8
            b"WAVE",
            b"fmt ",
            16,  # Subchunk1Size (PCM)
            1,   # AudioFormat (PCM)
            self.channels,
            self.sample_rate,
            self.sample_rate * self.channels * 2,  # ByteRate
            self.channels * 2,  # BlockAlign
            16,  # BitsPerSample
            b"data",
            data_size
        )
    
    def write_chunk(self, audio_bytes: bytes):
        """
        Write audio chunk to file
        Can be called from async context
        """
        if self.is_closed:
            return
        
        if self.file_handle:
            self.file_handle.write(audio_bytes)
            self.total_bytes += len(audio_bytes)
    
    async def write_chunk_async(self, audio_bytes: bytes):
        """Async version - offload to thread if needed"""
        await asyncio.to_thread(self.write_chunk, audio_bytes)
    
    def finalize(self) -> tuple[str, int]:
        """
        Finalize recording, update WAV header, close file
        Returns: (file_path, file_size_bytes)
        """
        if self.is_closed:
            return str(self.file_path), self.total_bytes
        
        self.is_closed = True
        
        if self.file_handle:
            if self.format == "wav":
                # Update WAV header with actual data size
                self.file_handle.seek(0)
                self.file_handle.write(self._create_wav_header(self.total_bytes))
            
            self.file_handle.close()
        
        file_size = os.path.getsize(self.file_path)
        print(f"[Recording] Finalized: {self.file_path} ({file_size:,} bytes)")
        
        return str(self.file_path), file_size
    
    def __del__(self):
        """Cleanup on deletion"""
        if not self.is_closed and self.file_handle:
            try:
                self.finalize()
            except:
                pass


# ═══════════════════════════════════════════════════════════════════
# CLEANUP UTILITIES
# ═══════════════════════════════════════════════════════════════════

def cleanup_old_recordings(days: int = config.RETENTION_DAYS):
    """
    Delete recordings older than specified days
    Call this periodically (e.g., daily cron job)
    """
    from datetime import timedelta
    import time
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0
    freed_bytes = 0
    
    for file_path in config.RECORDINGS_DIR.glob("*"):
        if file_path.is_file():
            file_mtime = file_path.stat().st_mtime
            if file_mtime < cutoff_time:
                file_size = file_path.stat().st_size
                file_path.unlink()
                deleted_count += 1
                freed_bytes += file_size
    
    if deleted_count > 0:
        print(f"[Cleanup] Deleted {deleted_count} old recordings, freed {freed_bytes:,} bytes")
    
    return deleted_count, freed_bytes


def get_storage_stats() -> dict:
    """Get storage statistics"""
    total_files = 0
    total_bytes = 0
    
    for file_path in config.RECORDINGS_DIR.glob("*"):
        if file_path.is_file():
            total_files += 1
            total_bytes += file_path.stat().st_size
    
    return {
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "total_gb": round(total_bytes / (1024 * 1024 * 1024), 2)
    }
