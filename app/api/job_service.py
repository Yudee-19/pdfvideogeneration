"""
Job management service for tracking job status and metadata.
"""
import json
import logging
# import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from threading import Lock
from app.utils.s3_utils import s3_manager
from app.config import settings

logger = logging.getLogger(__name__)


class JobService:
    """Service for managing job status and metadata."""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        self._load_jobs()
    
    def _load_jobs(self):
        """Load existing jobs from job directories."""
        if not settings.JOBS_OUTPUT_PATH.exists():
            return
        
        for job_dir in settings.JOBS_OUTPUT_PATH.iterdir():
            if not job_dir.is_dir():
                continue
            
            job_id = job_dir.name
            metadata_path = job_dir / "job_metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Use status from metadata if available, otherwise determine from files
                    status = metadata.get("status")
                    if not status:
                        # Determine status based on files
                        # Check for final video file (may have different naming patterns)
                        video_files = list(job_dir.glob("*_final_video.mp4"))
                        if video_files:
                            status = "completed"
                        elif list(job_dir.glob("*_processed_audio.mp3")):
                            status = "processing"
                        elif (job_dir / f"{job_id}_extraction.json").exists() or list(job_dir.glob("*_extraction.json")):
                            status = "processing"
                        else:
                            status = "pending"
                    
                    progress_value = metadata.get("progress")
                    self.jobs[job_id] = {
                        "job_id": job_id,
                        "status": status,
                        "message": metadata.get("message", ""),
                        "created_at": metadata.get("created_at", datetime.now().isoformat()),
                        "metadata": {k: v for k, v in metadata.items() if k not in ["status", "message", "created_at", "progress", "updated_at"]},
                        "progress": progress_value
                    }
                    if progress_value is not None:
                        logger.info(f"Loaded job {job_id} with progress: {progress_value}%")
                except Exception as e:
                    logger.warning(f"Failed to load job {job_id}: {e}")
    
    def _save_job_status(self, job_id: str):
        """Save job status and message to metadata file."""
        if job_id not in self.jobs:
            return
        
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        
        # CRITICAL CHECK: If directory was deleted (cleanup), stop here.
        # Otherwise, we might recreate an empty folder or crash.
        if not job_dir.exists():
            return
        
        metadata_path = job_dir / "job_metadata.json"
        try:
            # Load existing metadata or create new
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Update with current status and message
            metadata["status"] = self.jobs[job_id]["status"]
            metadata["message"] = self.jobs[job_id].get("message", "")
            metadata["updated_at"] = datetime.now().isoformat()
            # Always save progress if it exists (including 0)
            if "progress" in self.jobs[job_id]:
                progress_value = self.jobs[job_id]["progress"]
                metadata["progress"] = progress_value
            
            # Merge with any metadata updates
            if "metadata" in self.jobs[job_id]:
                metadata.update(self.jobs[job_id]["metadata"])
            
            # Save back to file and flush immediately
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                f.flush()  # Force write to disk immediately
                os.fsync(f.fileno())  # Ensure OS writes to disk
            
            logger.debug(f"Saved job status for {job_id}")
        except Exception as e:
            logger.warning(f"Failed to save job status for {job_id}: {e}")

    def _cleanup_local_job(self, job_id: str):
        """Deletes the local job folder to save space."""
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        try:
            if job_dir.exists() and job_dir.is_dir():
                logger.info(f"🧹 CLEANUP: Removing local files for job {job_id}...")
                # shutil.rmtree(job_dir)
                logger.info(f"✅ CLEANUP: Job {job_id} removed from local disk.")
        except Exception as e:
            logger.error(f"Failed to clean up job {job_id}: {e}")
    
    def create_job(self, job_id: str, pdf_path: str, generate_summary: bool = False, 
                   start_page: int = 50, end_page: int = 50):
        """Create a new job record."""
        with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "message": "Job created, waiting to start",
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "pdf_path": pdf_path,
                    "generate_summary": generate_summary,
                    "start_page": start_page,
                    "end_page": end_page
                }
            }
            # Persist job creation to metadata file
            self._save_job_status(job_id)
    
    def update_job(self, job_id: str, status: str, message: str = "", 
                   metadata: Optional[Dict[str, Any]] = None, progress: Optional[int] = None):
        """Update job status, trigger S3 Sync, and Clean up."""
        with self.lock:
            if job_id not in self.jobs:
                self.jobs[job_id] = {
                    "job_id": job_id,
                    "status": status,
                    "message": message,
                    "created_at": datetime.now().isoformat(),
                    "metadata": metadata or {},
                    "progress": progress
                }
            else:
                self.jobs[job_id]["status"] = status
                self.jobs[job_id]["message"] = message
                if progress is not None:
                    self.jobs[job_id]["progress"] = progress
                if metadata:
                    if "metadata" not in self.jobs[job_id]:
                        self.jobs[job_id]["metadata"] = {}
                    self.jobs[job_id]["metadata"].update(metadata)
            
            # Log progress updates for debugging
            if progress is not None:
                logger.info(f"Job {job_id}: Progress updated to {progress}% - {message}")
            
            # 1. Save status to local disk FIRST (so S3 picks up the latest metadata)
            self._save_job_status(job_id)

            # 2. If completed/failed, Sync & Cleanup
            if status in ["completed", "failed"]:
                # Sync to S3
                s3_manager.sync_job_to_s3(job_id)
                
                # Clean up local files ONLY IF S3 is configured and working
                # This prevents data loss if you are running locally without AWS
                if s3_manager.s3:
                    self._cleanup_local_job(job_id)
                else:
                    logger.warning(f"S3 not configured. Keeping local files for job {job_id}")
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def list_jobs(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List jobs with pagination."""
        with self.lock:
            sorted_jobs = sorted(
                self.jobs.values(),
                key=lambda x: x.get("created_at", ""),
                reverse=True
            )
            return sorted_jobs[offset:offset + limit]