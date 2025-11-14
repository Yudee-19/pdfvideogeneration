import sys
import logging
from pathlib import Path
import time
from datetime import datetime


script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from app.config import settings
from app.logging_config import setup_logging
from app.phase2_ai_services.openai_client import OpenAIService
from app.phase3_audio_processing.mastering import master_audio
from app.phase4_video_generation.renderer import render_video

def main():
    start_time = time.time()
    
    # --- A. Setup ---
    job_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to go to console AND a file in the job dir
    setup_logging(job_id=job_id, log_level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting test pipeline for Job ID: {job_id}")
    logger.info(f"Job output will be in: {job_dir}")

    try:
        # 0. Reading the script
        script_file_path = script_dir / "test_script.txt"
        logger.info(f"Reading script from: {script_file_path}")
        
        with open(script_file_path, 'r', encoding='utf-8') as f:
            test_script = f.read()
        
        logger.info(f"Script loaded successfully ({len(test_script)} characters)")
        
        # 1. AI Service (Text -> Audio + Timestamps)
        logger.info("Step 1: Calling OpenAIService...")
        openai_service = OpenAIService(voice="onyx")
        raw_audio_path, timestamps_path = openai_service.generate_audio_with_timestamps(
            text=test_script,
            output_dir=job_dir,
            job_id=job_id
        )
        logger.info(f"Raw audio at: {raw_audio_path}")
        logger.info(f"Timestamps at: {timestamps_path}")

        # 2. Audio Mastering (Raw Audio -> Processed Audio)
        logger.info("Step 2: Calling Audio Mastering...")
        processed_audio_path = job_dir / f"{job_id}_processed_audio.mp3"
        processed_audio_path = master_audio(
            raw_audio_path=raw_audio_path,
            processed_audio_path=processed_audio_path
        )
        logger.info(f"Processed audio at: {processed_audio_path}")

        # 3. Video Rendering (Audio + Timestamps -> Video)
        logger.info("Step 3: Calling Video Renderer...")
        final_video_path = job_dir / f"{job_id}_final_video.mp4"
        final_video_path = render_video(
            audio_path=processed_audio_path,
            timestamps_path=timestamps_path,
            output_path=final_video_path
        )
        logger.info(f"Final video at: {final_video_path}")

        end_time = time.time()
        logger.info("--- TEST PIPELINE SUCCESS ---")
        logger.info(f"Total time: {end_time - start_time:.2f} seconds")
        logger.info(f"All outputs saved in {job_dir}")
        
    except Exception as e:
        logger.error(f"--- TEST PIPELINE FAILED:  {e} ---", exc_info=True)

if __name__ == "__main__":
    main()