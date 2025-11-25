import logging
from pathlib import Path
from app.celery_app import celery_app
from app.api.job_service import JobService
from app.api.pipeline_service import PipelineService

# Setup logger
logger = logging.getLogger(__name__)

# Initialize services
job_service = JobService()
pipeline_service = PipelineService(job_service=job_service)

@celery_app.task(bind=True, name="process_pdf_job")
def process_pdf_job_task(
    self, 
    job_id: str, 
    pdf_path_str: str, 
    generate_summary: bool, 
    start_page: int, 
    end_page: int, 
    voice_provider: str, 
    openai_voice: str = None,  # <--- Added this
    cartesia_voice_id: str = None, 
    cartesia_model_id: str = None
):
    try:
        logger.info(f"Worker processing job: {job_id}")
        pdf_path = Path(pdf_path_str)
        pipeline_service.run_pipeline(
            job_id=job_id,
            pdf_path=pdf_path,
            generate_summary=generate_summary,
            start_page=start_page,
            end_page=end_page,
            voice_provider=voice_provider,
            openai_voice=openai_voice,  # <--- Pass it here
            cartesia_voice_id=cartesia_voice_id,
            cartesia_model_id=cartesia_model_id
        )
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        logger.error(f"Task failed: {e}")
        return {"status": "failed", "error": str(e)}

@celery_app.task(bind=True, name="generate_video_from_text")
def generate_video_from_text_task(
    self, 
    job_id: str, 
    text_path_str: str, 
    voice_provider: str, 
    openai_voice: str = None,  # <--- Added this
    cartesia_voice_id: str = None, 
    cartesia_model_id: str = None
):
    try:
        logger.info(f"Worker processing text-to-video job: {job_id}")
        pipeline_service.run_pipeline_from_text(
            job_id=job_id,
            text_path=Path(text_path_str),
            voice_provider=voice_provider,
            openai_voice=openai_voice,  # <--- Pass it here
            cartesia_voice_id=cartesia_voice_id,
            cartesia_model_id=cartesia_model_id
        )
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        logger.error(f"Text task failed: {e}")
        return {"status": "failed", "error": str(e)}

@celery_app.task(bind=True, name="generate_reels_video")
def generate_reels_video_task(
    self, 
    job_id: str, 
    text_path_str: str, 
    voice_provider: str, 
    openai_voice: str = None,  # <--- Added this
    cartesia_voice_id: str = None, 
    cartesia_model_id: str = None
):
    try:
        logger.info(f"Worker processing reels job: {job_id}")
        pipeline_service.run_pipeline_for_reels(
            job_id=job_id,
            text_path=Path(text_path_str),
            voice_provider=voice_provider,
            openai_voice=openai_voice,  # <--- Pass it here
            cartesia_voice_id=cartesia_voice_id,
            cartesia_model_id=cartesia_model_id
        )
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        logger.error(f"Reels task failed: {e}")
        return {"status": "failed", "error": str(e)}
    
@celery_app.task(bind=True, name="generate_video_from_audio")
def generate_video_from_audio_task(self, job_id: str, audio_path_str: str):
    try:
        logger.info(f"Worker processing audio upload job: {job_id}")
        pipeline_service.run_pipeline_from_audio(
            job_id=job_id,
            audio_path=Path(audio_path_str)
        )
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        logger.error(f"Audio task failed: {e}")
        return {"status": "failed", "error": str(e)}