"""
FastAPI backend for PDF-to-Video generation service.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import logging
import uuid
import json
from datetime import datetime

from app.config import settings
from app.api.job_service import JobService
from app.api.pipeline_service import PipelineService
from app.api.cartesia_service import CartesiaAPIService
from app.phase1_pdf_processing.service import PDFExtractorService
from app.phase2_ai_services.pdf_summarizer import generate_pdf_summary
from app.tasks import (
    process_pdf_job_task, 
    generate_video_from_text_task, 
    generate_reels_video_task
)
from fastapi.responses import RedirectResponse  # <--- Add this
from app.utils.s3_utils import s3_manager       # <--- Add this

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF to Video API",
    description="API for converting PDF books to video with narration",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services - share the same job_service instance
job_service = JobService()
pipeline_service = PipelineService(job_service=job_service)

# Initialize Cartesia API service (may fail if API key not set, that's ok)
try:
    cartesia_api_service = CartesiaAPIService()
except ValueError:
    cartesia_api_service = None
    logger.warning("Cartesia API service not available (API key not configured)")


class JobRequest(BaseModel):
    """Request model for starting a job."""
    generate_summary: bool = False
    start_page: Optional[int] = None
    end_page: Optional[int] = None


class JobResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    message: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None
    progress: Optional[int] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "PDF to Video API is running"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "jobs_path": str(settings.JOBS_OUTPUT_PATH)
    }


@app.post("/api/upload", response_model=JobResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    generate_summary: bool = Form(False),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
    voice_provider: str = Form("openai"),
    openai_voice: Optional[str] = Form(None),
    cartesia_voice_id: Optional[str] = Form(None),
    cartesia_model_id: Optional[str] = Form(None),
):
    """
    Upload a PDF file and start the video generation pipeline.
    
    Args:
        file: PDF file to upload
        generate_summary: Whether to generate a book summary (optional)
        start_page: Optional start page for main video (default: 50)
        end_page: Optional end page for main video (default: 50)
        background_tasks: FastAPI background tasks
    
    Returns:
        JobResponse with job_id and status
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate unique job ID
    job_id = f"{Path(file.filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate OpenAI voice selection
    if voice_provider.lower() == "openai" and not openai_voice:
        raise HTTPException(status_code=400, detail="Please select an OpenAI voice")
    
    # Log received parameters
    logger.info(f"Upload parameters - generate_summary: {generate_summary}, start_page: {start_page}, end_page: {end_page}, voice_provider: {voice_provider}, openai_voice: {openai_voice}, cartesia_voice_id: {cartesia_voice_id}, cartesia_model_id: {cartesia_model_id}")
    
    # Save uploaded PDF
    pdf_path = job_dir / file.filename
    try:
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"PDF uploaded: {pdf_path} (size: {len(content)} bytes)")
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {str(e)}")
    
    # Create job record
    job_service.create_job(
        job_id=job_id,
        pdf_path=str(pdf_path),
        generate_summary=generate_summary,
        start_page=start_page or 50,
        end_page=end_page or 50
    )
    
    # Start pipeline in background
    logger.info(f"Starting background pipeline for job {job_id}")
    
    process_pdf_job_task.delay(
        job_id=job_id,
        pdf_path_str=str(pdf_path),
        generate_summary=generate_summary,
        start_page=start_page or 50,
        end_page=end_page or 50,
        voice_provider=voice_provider,
                openai_voice=openai_voice,
        cartesia_voice_id=cartesia_voice_id,
        cartesia_model_id=cartesia_model_id
    )

    return JobResponse(
        job_id=job_id,
        status="queued", # Status is now queued initially
        message="PDF uploaded and job queued",
        created_at=datetime.now().isoformat()
    )


# @app.get("/api/jobs/{job_id}", response_model=JobResponse)
# async def get_job_status(job_id: str):
#     """
#     Get the status of a job.
#     Reads directly from disk to ensure latest status from Worker.
#     """
#     # 1. Try reading from disk (The Source of Truth)
#     job_dir = settings.JOBS_OUTPUT_PATH / job_id
#     metadata_path = job_dir / "job_metadata.json"
    
#     if metadata_path.exists():
#         try:
#             with open(metadata_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 return JobResponse(
#                     job_id=job_id,
#                     status=data.get("status", "unknown"),
#                     message=data.get("message", ""),
#                     created_at=data.get("created_at", ""),
#                     metadata=data,  # Contains the S3 paths
#                     progress=data.get("progress")
#                 )
#         except Exception as e:
#             logger.error(f"Error reading metadata for {job_id}: {e}")
#             # Don't crash, try fallback
    
#     # 2. Fallback: Try memory (unlikely to work if worker is separate, but safe)
#     job = job_service.get_job(job_id)
#     if not job:
#         raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
#     return JobResponse(
#         job_id=job_id,
#         status=job.get("status", "unknown"),
#         message=job.get("message", ""),
#         created_at=job.get("created_at", ""),
#         metadata=job.get("metadata", {}),
#         progress=job.get("progress")
#     )


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Get job status. 
    Priority: Disk (Fastest) -> S3 (Reliable) -> Memory (Fallback).
    """
    # # 1. Try Local Disk
    # job_dir = settings.JOBS_OUTPUT_PATH / job_id
    # metadata_path = job_dir / "job_metadata.json"
    
    # if metadata_path.exists():
    #     try:
    #         with open(metadata_path, 'r') as f:
    #             data = json.load(f)
    #             return JobResponse(
    #                 job_id=job_id,
    #                 status=data.get("status", "unknown"),
    #                 message=data.get("message", ""),
    #                 created_at=data.get("created_at", ""),
    #                 metadata=data,
    #                 progress=data.get("progress")
    #             )
    #     except: pass
    
    # 2. Try S3 (The "Cloud Backup")
    # If we cleaned up the local folder, the status is here.
    s3_data = s3_manager.get_job_metadata_from_s3(job_id)
    if s3_data:
        return JobResponse(
            job_id=job_id,
            status=s3_data.get("status", "unknown"),
            message=s3_data.get("message", ""),
            created_at=s3_data.get("created_at", ""),
            metadata=s3_data,
            progress=s3_data.get("progress")
        )

    # 3. Fallback to Memory
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        message=job.get("message", ""),
        created_at=job.get("created_at", ""),
        metadata=job.get("metadata", {}),
        progress=job.get("progress")
    )

@app.get("/api/jobs/{job_id}/download/video")
async def download_video(job_id: str):
    """
    Redirect to S3 for video download.
    Reads metadata directly from disk to ensure latest status.
    """
    # 1. Define job_data (The variable you were asking about)
    job_data = None

    # 2. Try reading from disk (The Source of Truth)
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    metadata_path = job_dir / "job_metadata.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                job_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {job_id}: {e}")
    
    # 3. Fallback: Try memory if file missing (rare, but safe)
    if not job_data:
        job_data = job_service.get_job(job_id)
    
    # 4. Validations
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Now we check the status from the FRESH data we just loaded
    if job_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet (Status: {job_data.get('status')})")
    
    # 5. Generate S3 Link
    try:
        # Get path from metadata
        # It might be in "metadata" sub-dictionary OR at the top level depending on how it was saved
        path_str = job_data.get("metadata", {}).get("final_video_path") or job_data.get("final_video_path")
        
        if not path_str:
             raise HTTPException(status_code=404, detail="Video path not found in job data")

        # Extract just the filename (e.g., "reels_..._final_video.mp4")
        filename = Path(path_str).name
        
        # Construct S3 Key: jobs/{job_id}/{filename}
        s3_key = f"jobs/{job_id}/{filename}"
        
        # Generate VIP Ticket (Presigned URL)
        url = s3_manager.generate_presigned_url(s3_key)
        
        if url:
            return RedirectResponse(url=url)
        else:
            raise HTTPException(status_code=500, detail="Could not generate S3 link")
            
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing download: {str(e)}")
    
    
@app.post("/api/jobs/{job_id}/generate-summary", response_model=JobResponse)
async def generate_summary(
    job_id: str,
    background_tasks: BackgroundTasks
):
    """
    Generate a book summary after main video is complete.
    
    Args:
        job_id: Unique job identifier
        background_tasks: FastAPI background tasks
    
    Returns:
        JobResponse with updated status
    """
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Main video must be completed before generating summary")
    
    # Start summary generation in background
    background_tasks.add_task(
        pipeline_service.generate_summary,
        job_id=job_id
    )
    
    return JobResponse(
        job_id=job_id,
        status="processing",
        message="Summary generation started",
        created_at=job.get("created_at", datetime.now().isoformat()),
        metadata=job.get("metadata")
    )


@app.get("/api/jobs/{job_id}/download/summary")
async def download_summary(job_id: str):
    """
    Download the generated summary file (if available).
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Summary text file
    """
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    summary_path = job.get("metadata", {}).get("summary_path")
    if not summary_path:
        raise HTTPException(status_code=404, detail="Summary not available for this job")
    
    summary_file = Path(summary_path)
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Summary file not found")
    
    return FileResponse(
        path=str(summary_file),
        filename=summary_file.name,
        media_type="text/plain"
    )


@app.post("/api/jobs/{job_id}/generate-summary-video", response_model=JobResponse)
async def generate_summary_video(
    job_id: str,
    background_tasks: BackgroundTasks,
    voice_provider: str = Form("openai"),
    openai_voice: Optional[str] = Form(None),
    cartesia_voice_id: Optional[str] = Form(None),
    cartesia_model_id: Optional[str] = Form(None),
):
    """
    Generate a video from the summary (if summary exists).
    
    Args:
        job_id: Unique job identifier
        voice_provider: Voice provider ("openai" or "cartesia")
        cartesia_voice_id: Cartesia voice ID (if using Cartesia)
        cartesia_model_id: Cartesia model ID (if using Cartesia)
        background_tasks: FastAPI background tasks
    
    Returns:
        JobResponse with updated status
    """
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    summary_path = job.get("metadata", {}).get("summary_path")
    if not summary_path:
        raise HTTPException(status_code=400, detail="Summary not available for this job. Generate summary first.")
    
    summary_file = Path(summary_path)
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Summary file not found")
    
    # Validate OpenAI voice selection
    if voice_provider.lower() == "openai" and not openai_voice:
        raise HTTPException(status_code=400, detail="Please select an OpenAI voice")
    
    # Start summary video generation in background
    background_tasks.add_task(
        pipeline_service.generate_summary_video,
        job_id=job_id,
        voice_provider=voice_provider,
        openai_voice=openai_voice,
        cartesia_voice_id=cartesia_voice_id,
        cartesia_model_id=cartesia_model_id
    )
    
    return JobResponse(
        job_id=job_id,
        status="processing",
        message="Summary video generation started",
        created_at=job.get("created_at", datetime.now().isoformat()),
        metadata=job.get("metadata")
    )


@app.get("/api/jobs/{job_id}/download/summary-video")
async def download_summary_video(job_id: str):
    """
    Download the generated summary video file (if available).
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Summary video file
    """
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    summary_video_path = job.get("metadata", {}).get("summary_video_path")
    if not summary_video_path:
        raise HTTPException(status_code=404, detail="Summary video not available for this job")
    
    summary_video_file = Path(summary_video_path)
    if not summary_video_file.exists():
        raise HTTPException(status_code=404, detail="Summary video file not found")
    
    return FileResponse(
        path=str(summary_video_file),
        filename=summary_video_file.name,
        media_type="video/mp4"
    )


@app.post("/api/summarize-pdf", response_model=JobResponse)
async def summarize_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF file and generate an extensive summary (minimum 10k words).
    
    Args:
        file: PDF file to upload
        background_tasks: FastAPI background tasks
    
    Returns:
        JobResponse with job_id and status
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate unique job ID
    job_id = f"summary_{Path(file.filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded PDF
    pdf_path = job_dir / file.filename
    try:
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"PDF uploaded for summarization: {pdf_path} (size: {len(content)} bytes)")
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {str(e)}")
    
    # Create job record
    job_service.create_job(
        job_id=job_id,
        pdf_path=str(pdf_path),
        generate_summary=False,
        start_page=1,
        end_page=1
    )
    
    # Start summarization in background
    def run_summarization():
        """Background task to generate PDF summary."""
        try:
            logger.info(f"Starting PDF summarization for job {job_id}")
            
            # Update status
            job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Extracting text from PDF...",
                progress=10
            )
            
            # Extract text from PDF
            extractor_service = PDFExtractorService(output_dir=settings.JOBS_OUTPUT_PATH)
            extraction_result = extractor_service.extract_from_pdf(
                pdf_path=str(pdf_path),
                job_id=job_id
            )
            
            pdf_text = extraction_result["text_extraction"]["full_text"]
            pdf_filename = extraction_result["pdf_filename"]
            
            logger.info(f"Extracted {len(pdf_text)} characters from PDF")
            
            # Update status
            job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Generating summary with GPT-4o-mini...",
                progress=30
            )
            
            # Generate summary
            summary_text, stats = generate_pdf_summary(
                pdf_text=pdf_text,
                pdf_filename=pdf_filename,
                min_words=10000
            )
            
            # Save summary to file
            summary_path = job_dir / f"{job_id}_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            
            logger.info(f"Summary generated: {stats['word_count']:,} words")
            
            # Update job with summary
            job_service.update_job(
                job_id=job_id,
                status="completed",
                message=f"Summary generated: {stats['word_count']:,} words (~{stats['estimated_minutes']:.1f} min narration)",
                progress=100,
                metadata={
                    "summary_path": str(summary_path),
                    "summary_stats": stats,
                    "summary_text": summary_text,  # Include summary text in metadata
                    "pdf_filename": pdf_filename
                }
            )
            
            logger.info(f"PDF summarization completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"PDF summarization failed for job {job_id}: {e}", exc_info=True)
            job_service.update_job(
                job_id=job_id,
                status="failed",
                message=f"Summarization failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    background_tasks.add_task(run_summarization)
    
    return JobResponse(
        job_id=job_id,
        status="processing",
        message="PDF uploaded, starting summarization...",
        created_at=datetime.now().isoformat()
    )


@app.post("/api/generate-video-from-text", response_model=JobResponse)
async def generate_video_from_text(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    voice_provider: str = Form("openai"),
    openai_voice: Optional[str] = Form(None),
    cartesia_voice_id: Optional[str] = Form(None),
    cartesia_model_id: Optional[str] = Form(None),
):
    """
    Generate video from text directly (for summary text).
    
    Args:
        text: Text content to generate video from
        voice_provider: Voice provider ("openai" or "cartesia")
        cartesia_voice_id: Cartesia voice ID (if using Cartesia)
        cartesia_model_id: Cartesia model ID (if using Cartesia)
        background_tasks: FastAPI background tasks
    
    Returns:
        JobResponse with job_id and status
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Validate OpenAI voice selection
    if voice_provider.lower() == "openai" and not openai_voice:
        raise HTTPException(status_code=400, detail="Please select an OpenAI voice")
    
    # Generate unique job ID
    job_id = f"text_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text to file
    text_path = job_dir / f"{job_id}_input_text.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Text saved for video generation: {text_path} ({len(text)} characters)")
    
    # Create job record
    job_service.create_job(
        job_id=job_id,
        pdf_path=None,  # No PDF for text-based generation
        generate_summary=False,
        start_page=1,
        end_page=1
    )
    
    # Start video generation in background
    logger.info(f"Queuing text-to-video job {job_id}")
    
    generate_video_from_text_task.delay(
        job_id=job_id,
        text_path_str=str(text_path), # Convert Path to string
        voice_provider=voice_provider,
                openai_voice=openai_voice,
        cartesia_voice_id=cartesia_voice_id,
        cartesia_model_id=cartesia_model_id
    )
    
    return JobResponse(
        job_id=job_id,
        status="queued", # Status changed to queued
        message="Summary text received, job queued for processing",
        created_at=datetime.now().isoformat()
    )


@app.post("/api/generate-reels-video", response_model=JobResponse)
async def generate_reels_video(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    voice_provider: str = Form("openai"),
    openai_voice: Optional[str] = Form(None),
    cartesia_voice_id: Optional[str] = Form(None),
    cartesia_model_id: Optional[str] = Form(None),
):
    """
    Generate reels/shorts video from text directly (skips text cleaning).
    Uses custom background image (1488x1960) and smaller font size.
    
    Args:
        text: Text content to generate video from
        voice_provider: Voice provider ("openai" or "cartesia")
        cartesia_voice_id: Cartesia voice ID (if using Cartesia)
        cartesia_model_id: Cartesia model ID (if using Cartesia)
        background_tasks: FastAPI background tasks
    
    Returns:
        JobResponse with job_id and status
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Validate OpenAI voice selection
    if voice_provider.lower() == "openai" and not openai_voice:
        raise HTTPException(status_code=400, detail="Please select an OpenAI voice")
    
    # Generate unique job ID
    job_id = f"reels_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text to file
    text_path = job_dir / f"{job_id}_input_text.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Text saved for reels video generation: {text_path} ({len(text)} characters)")
    
    # Create job record
    job_service.create_job(
        job_id=job_id,
        pdf_path=None,  # No PDF for reels generation
        generate_summary=False,
        start_page=1,
        end_page=1
    )
    
    logger.info(f"Queuing reels job {job_id}")
    
    generate_reels_video_task.delay(
        job_id=job_id,
        text_path_str=str(text_path), # Convert Path to string
        voice_provider=voice_provider,
                openai_voice=openai_voice,
        cartesia_voice_id=cartesia_voice_id,
        cartesia_model_id=cartesia_model_id
    )
    
    return JobResponse(
        job_id=job_id,
        status="queued", # Status changed to queued
        message="Text received, reels job queued",
        created_at=datetime.now().isoformat()
    )


@app.get("/api/jobs")
async def list_jobs(limit: int = 10, offset: int = 0):
    """
    List all jobs with pagination.
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
    
    Returns:
        List of jobs
    """
    jobs = job_service.list_jobs(limit=limit, offset=offset)
    return {"jobs": jobs, "total": len(jobs)}


# ===== CARTESIA API ENDPOINTS =====

@app.get("/api/cartesia/voices")
async def list_cartesia_voices(language: Optional[str] = None, tags: Optional[str] = None):
    """
    List available Cartesia voices.
    
    Args:
        language: Optional language filter (e.g., 'en', 'fr')
        tags: Optional comma-separated tags to filter by (e.g., 'Emotive,Stable')
    
    Returns:
        List of available voices
    """
    if not cartesia_api_service:
        # Return fallback voices even if service is not initialized
        from app.api.cartesia_service import CartesiaAPIService
        try:
            temp_service = CartesiaAPIService()
            voices = temp_service._get_fallback_voices()
            return {"voices": voices, "note": "Using fallback voices (API service not initialized)"}
        except:
            # Last resort: return hardcoded fallback
            voices = [
                {
                    "id": "98a34ef2-2140-4c28-9c71-663dc4dd7022",
                    "name": "Tessa",
                    "language": "en",
                    "tags": ["Emotive", "Expressive"],
                    "description": "Expressive American English voice, great for emotive characters"
                }
            ]
            return {"voices": voices, "note": "Using minimal fallback (API key not configured)"}
    
    tag_list = tags.split(",") if tags else None
    try:
        voices = cartesia_api_service.list_voices(language=language, tags=tag_list)
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing Cartesia voices: {e}", exc_info=True)
        # Return fallback voices on error
        voices = cartesia_api_service._get_fallback_voices()
        return {"voices": voices, "note": f"Using fallback voices due to error: {str(e)}"}


@app.get("/api/cartesia/voices/{voice_id}")
async def get_cartesia_voice(voice_id: str):
    """
    Get details for a specific Cartesia voice.
    
    Args:
        voice_id: Voice ID to retrieve
    
    Returns:
        Voice details
    """
    if not cartesia_api_service:
        raise HTTPException(
            status_code=503,
            detail="Cartesia API service not available. Please configure CARTESIA_API_KEY."
        )
    
    voice = cartesia_api_service.get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id} not found")
    
    return voice


@app.get("/api/cartesia/models")
async def list_cartesia_models():
    """
    List available Cartesia TTS models.
    
    Returns:
        List of available models
    """
    if not cartesia_api_service:
        raise HTTPException(
            status_code=503,
            detail="Cartesia API service not available. Please configure CARTESIA_API_KEY."
        )
    
    models = cartesia_api_service.list_models()
    return {"models": models}

@app.post("/api/upload-audio", response_model=JobResponse)
async def upload_audio_file(
    file: UploadFile = File(...),
):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="File must be an audio file (mp3, wav, m4a)")
    
    # Generate ID
    job_id = f"audio_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Audio
    audio_path = job_dir / file.filename
    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create Job
    job_service.create_job(job_id=job_id, pdf_path=None)
    
    # Send to Celery
    from app.tasks import generate_video_from_audio_task
    generate_video_from_audio_task.delay(
        job_id=job_id, 
        audio_path_str=str(audio_path)
    )
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Audio uploaded, processing started",
        created_at=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

