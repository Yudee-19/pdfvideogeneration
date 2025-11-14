import sys
import logging
from pathlib import Path
import time
from datetime import datetime
import argparse

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from app.config import settings
from app.logging_config import setup_logging

from app.phase1_pdf_processing.service import PDFExtractorService
from app.phase1_pdf_processing.image_extractor import extract_images
from app.phase1_pdf_processing.text_cleaner import clean_text

from app.phase2_ai_services.openai_client import OpenAIService
from app.phase3_audio_processing.mastering import master_audio
from app.phase4_video_generation.renderer import render_video


def main(pdf_file_path: Path):
    start_time = time.time()
    
    # --- A. Setup ---
    job_id = f"{pdf_file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(job_id=job_id, log_level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info(f"--- STARTING FULL PIPELINE FOR JOB: {job_id} ---")
    logger.info(f"Input PDF: {pdf_file_path}")
    logger.info(f"Job output will be in: {job_dir}")

    try:
        # ===== PHASE 1: PDF PROCESSING =====
        logger.info("--- PHASE 1: PDF Processing (with Adaptive Logic) ---")
        
        # 1.1: Run full text/table/index service
        extractor_service = PDFExtractorService(output_dir=settings.JOBS_OUTPUT_PATH)
        extraction_result = extractor_service.extract_from_pdf(
            pdf_path=str(pdf_file_path),
            job_id=job_id
        )
        
        # Get the paths to the files service just created
        raw_text_path = Path(extraction_result["output_files"]["full_text"])
        tables_dir = Path(extraction_result["output_files"]["tables_directory"])
        
        # 1.2: Run image extraction logic
        images_dir = extract_images(pdf_file_path, job_dir)
        
        logger.info(f"Book type detected: {extraction_result['book_type']}")
        logger.info(f"Tables found: {extraction_result['summary']['tables_count']}")
        
        # ===== PHASE 1.5: TEXT CLEANING (The Dummy) =====
        logger.info("--- PHASE 1.5: Text Cleaning ---")
        cleaned_script_path = clean_text(
            raw_text_path=raw_text_path,
            tables_dir=tables_dir,
            images_dir=images_dir,
            job_dir=job_dir
        )
        
        with open(cleaned_script_path, 'r', encoding='utf-8') as f:
            text_script = f.read()
            if not text_script.strip():
                raise ValueError("Cleaned script is empty. Cannot proceed.")
        
        # ===== PHASE 2: AI SERVICES =====
        logger.info("--- PHASE 2: AI Services ---")
        openai_service = OpenAIService(voice="onyx")
        raw_audio_path, timestamps_path = openai_service.generate_audio_with_timestamps(
            text=text_script,
            output_dir=job_dir,
            job_id=job_id
        )

        # ===== PHASE 3: AUDIO PROCESSING =====
        logger.info("--- PHASE 3: Audio Mastering ---")
        processed_audio_path = job_dir / f"{job_id}_processed_audio.mp3"
        processed_audio_path = master_audio(
            raw_audio_path=raw_audio_path,
            processed_audio_path=processed_audio_path
        )

        # ===== PHASE 4: VIDEO GENERATION =====
        logger.info("--- PHASE 4: Video Rendering ---")
        final_video_path = job_dir / f"{job_id}_final_video.mp4"
        final_video_path = render_video(
            audio_path=processed_audio_path,
            timestamps_path=timestamps_path,
            output_path=final_video_path
        )
        
        end_time = time.time()
        logger.info(f"--- FULL PIPELINE SUCCESS (Total time: {end_time - start_time:.2f}s) ---")
        logger.info(f"Final Video: {final_video_path}")
        
    except Exception as e:
        logger.error(f"--- FULL PIPELINE FAILED {e} ---", exc_info=True)
        end_time = time.time()
        logger.error(f"Failed after {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full PDF-to-Video pipeline.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file.")
    args = parser.parse_args()
    
    input_pdf = Path(args.pdf_path)
    if not input_pdf.exists():
        print(f"Error: PDF file not found at {input_pdf}")
        sys.exit(1)
        
    main(pdf_file_path=input_pdf)