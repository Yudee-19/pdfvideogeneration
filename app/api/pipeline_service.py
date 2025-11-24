"""
Pipeline service that runs the PDF-to-Video generation pipeline.
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from app.config import settings
from app.logging_config import setup_logging
from app.phase1_pdf_processing.service import PDFExtractorService
from app.phase1_pdf_processing.image_extractor import extract_images
from app.phase1_pdf_processing.text_cleaner import clean_text
from app.phase2_ai_services.openai_client import OpenAIService, detect_book_genre
from app.phase2_ai_services.cartesia_client import CartesiaService
from app.phase2_ai_services.book_summary import generate_book_summary
from app.phase3_audio_processing.mastering import master_audio
from app.phase4_video_generation.renderer import render_video
from app.api.job_service import JobService

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for running the PDF-to-Video pipeline."""
    
    def __init__(self, job_service: Optional[JobService] = None):
        # Use provided job_service or create a new one
        # This allows sharing the same instance across the application
        self.job_service = job_service if job_service is not None else JobService()
    
    def run_pipeline(
        self,
        job_id: str,
        pdf_path: Path,
        generate_summary: bool = False,
        start_page: int = 50,
        end_page: int = 50,
        voice_provider: str = "openai",
        cartesia_voice_id: Optional[str] = None,
        cartesia_model_id: Optional[str] = None
    ):
        """
        Run the complete PDF-to-Video pipeline.
        
        Args:
            job_id: Unique job identifier
            pdf_path: Path to the PDF file
            generate_summary: Whether to generate summary (optional)
            start_page: Start page for main video
            end_page: End page for main video
        """
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(job_id=job_id, log_level="INFO")
        logger = logging.getLogger(__name__)
        
        logger.info(f"=== PIPELINE STARTED FOR JOB: {job_id} ===")
        logger.info(f"PDF Path: {pdf_path}")
        logger.info(f"Job Directory: {job_dir}")
        logger.info(f"Generate Summary: {generate_summary}")
        logger.info(f"Page Range: {start_page}-{end_page}")
        
        try:
            logger.info(f"Updating job status to processing with progress 0%")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Starting PDF processing...",
                progress=0
            )
            logger.info(f"Job status updated successfully")
            
            # ===== PHASE 1: PDF PROCESSING =====
            logger.info("--- PHASE 1: PDF Processing ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Extracting text from PDF...",
                progress=5
            )
            extractor_service = PDFExtractorService(output_dir=settings.JOBS_OUTPUT_PATH)
            extraction_result = extractor_service.extract_from_pdf(
                pdf_path=str(pdf_path),
                job_id=job_id
            )
            book_type = extraction_result.get("book_type", "unknown")
            
            raw_text_path = Path(extraction_result["output_files"]["full_text"])
            tables_dir_path = extraction_result["output_files"].get("tables_directory")
            if tables_dir_path:
                tables_dir = Path(tables_dir_path)
            else:
                tables_dir = job_dir / "tables"
                tables_dir.mkdir(exist_ok=True)
            
            images_dir = extract_images(pdf_path, job_dir)
            
            logger.info(f"Book type detected: {book_type}")
            logger.info(f"Tables found: {extraction_result['summary']['tables_count']}")
            
            # Update progress after PDF extraction
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="PDF extraction complete, filtering pages...",
                progress=15
            )
            
            # Filter text to specified page range
            json_path = Path(extraction_result["output_files"]["json"])
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filtered_text = ""
            for page in data['text_extraction']['pages']:
                if start_page <= page['page_number'] <= end_page:
                    filtered_text += page['text'] + "\n\n"
            
            if not filtered_text.strip():
                raise ValueError(f"No text found for pages {start_page}-{end_page}.")
            
            filtered_text_path = job_dir / f"filtered_pages_{start_page}_to_{end_page}.txt"
            with open(filtered_text_path, 'w', encoding='utf-8') as f:
                f.write(filtered_text)
            
            # ===== PHASE 1.5: TEXT CLEANING =====
            logger.info("--- PHASE 1.5: Text Cleaning ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Cleaning text for narration...",
                progress=20
            )
            cleaned_script_path = clean_text(
                raw_text_path=filtered_text_path,
                tables_dir=tables_dir,
                images_dir=images_dir,
                job_dir=job_dir
            )
            
            # Update progress after text cleaning
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Text cleaning complete",
                progress=22
            )
            
            with open(cleaned_script_path, 'r', encoding='utf-8') as f:
                text_script = f.read()
                if not text_script.strip():
                    raise ValueError("Cleaned script is empty. Cannot proceed.")
            
            # ===== PHASE 2: AI SERVICES =====
            logger.info("--- PHASE 2: AI Services ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Detecting book genre...",
                progress=25
            )
            
            book_title = pdf_path.stem
            genre = detect_book_genre(book_title)
            logger.info(f"Detected genre: {genre}")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Generating audio narration...",
                progress=30
            )
            
            # Initialize metadata file if it doesn't exist
            metadata_path = job_dir / "job_metadata.json"
            if not metadata_path.exists():
                initial_metadata = {
                    "job_id": job_id,
                    "status": "processing",
                    "message": "Starting PDF processing...",
                    "created_at": datetime.now().isoformat(),
                    "pdf_path": str(pdf_path),
                    "generate_summary": generate_summary,
                    "start_page": start_page,
                    "end_page": end_page
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(initial_metadata, f, indent=2)
            
            job_metadata = {
                "job_id": job_id,
                "pdf_path": str(pdf_path),
                "book_title": book_title,
                "book_type": book_type,
                "genre": genre,
                "created_at": datetime.now().isoformat(),
                "generate_summary": generate_summary,
                "start_page": start_page,
                "end_page": end_page,
                "voice_provider": voice_provider,
            }
            
            # Store Cartesia settings if using Cartesia
            if voice_provider.lower() == "cartesia":
                if cartesia_voice_id:
                    job_metadata["cartesia_voice_id"] = cartesia_voice_id
                if cartesia_model_id:
                    job_metadata["cartesia_model_id"] = cartesia_model_id
            
            metadata_path = job_dir / "job_metadata.json"
            
            def _write_metadata():
                with open(metadata_path, 'w', encoding='utf-8') as meta_file:
                    json.dump(job_metadata, meta_file, indent=2)
            
            _write_metadata()
            
            # Initialize voice service based on provider
            if voice_provider.lower() == "cartesia":
                voice_id = cartesia_voice_id or "98a34ef2-2140-4c28-9c71-663dc4dd7022"  # Default: Tessa
                model_id = cartesia_model_id or "sonic-3"  # Default: sonic-3
                logger.info(f"Using Cartesia for voice generation (Voice: {voice_id}, Model: {model_id})")
                voice_service = CartesiaService(voice_id=voice_id, model_id=model_id)
            else:
                logger.info(f"Using OpenAI for voice generation with voice: onyx")
                voice_service = OpenAIService(voice="onyx")
            
            # Estimate number of chunks to provide progress updates
            # Rough estimate: ~900 tokens per chunk, ~4 chars per token
            estimated_chars_per_chunk = 900 * 4
            num_chunks = max(1, (len(text_script) + estimated_chars_per_chunk - 1) // estimated_chars_per_chunk)
            logger.info(f"Estimated {num_chunks} chunks for audio generation")
            
            raw_audio_path, timestamps_path = voice_service.generate_audio_with_timestamps(
                text=text_script,
                output_dir=job_dir,
                job_id=job_id,
                genre=genre
            )
            
            # Update progress after audio generation
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Audio generation complete, processing audio...",
                progress=50
            )
            
            # ===== PHASE 3: AUDIO PROCESSING =====
            logger.info("--- PHASE 3: Audio Mastering ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Mastering audio quality...",
                progress=55
            )
            
            processed_audio_path = job_dir / f"{job_id}_processed_audio.mp3"
            processed_audio_path = master_audio(
                raw_audio_path=raw_audio_path,
                processed_audio_path=processed_audio_path
            )
            
            # CRITICAL: Regenerate timestamps from processed audio to ensure perfect sync
            # This ensures timestamps match the exact audio used in the video
            # For OpenAI: Uses Whisper timestamps (native to OpenAI)
            # For Cartesia: Uses Whisper timestamps (Cartesia bytes endpoint doesn't provide timestamps)
            # Both providers get final timestamps from processed audio for perfect accuracy
            logger.info(f"Regenerating timestamps from processed audio for perfect sync (provider: {voice_provider})...")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Regenerating timestamps from processed audio for perfect sync...",
                progress=58
            )
            
            # Regenerate timestamps using Whisper on the processed audio
            # This ensures timestamps match the exact processed audio timing
            from openai import OpenAI
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            with open(processed_audio_path, "rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            # Save updated timestamps - these are the final timestamps used for video generation
            timestamps_data = transcription.model_dump()
            with open(timestamps_path, "w", encoding="utf-8") as f:
                json.dump(timestamps_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Timestamps regenerated from processed audio: {len(timestamps_data.get('words', []))} words, {len(timestamps_data.get('segments', []))} segments")
            logger.info(f"These timestamps will be used for frame generation to ensure perfect audio-video sync")
            
            # Update progress after audio mastering
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Audio mastering complete",
                progress=60
            )
            
            # ===== PHASE 4: VIDEO GENERATION =====
            logger.info("--- PHASE 4: Video Rendering ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Rendering video frames...",
                progress=70
            )
            
            final_video_path = job_dir / f"{job_id}_final_video.mp4"
            final_video_path = render_video(
                audio_path=processed_audio_path,
                timestamps_path=timestamps_path,
                output_path=final_video_path
            )
            
            # Update progress after video rendering
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Video frames rendered, encoding final video...",
                progress=90
            )
            
            # Small delay to ensure the update is visible
            import time
            time.sleep(0.5)
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Finalizing video...",
                progress=95
            )
            
            logger.info(f"--- MAIN VIDEO COMPLETE ---")
            logger.info(f"Final Video: {final_video_path}")
            
            # Update metadata with video path
            job_metadata["final_video_path"] = str(final_video_path)
            _write_metadata()
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Main video completed",
                metadata={"final_video_path": str(final_video_path)},
                progress=100
            )
            
            # ===== OPTIONAL SUMMARY GENERATION =====
            if generate_summary:
                try:
                    logger.info("--- SUMMARY GENERATION (Target ~1 hour narration) ---")
                    self.job_service.update_job(
                        job_id=job_id,
                        status="processing",
                        message="Generating book summary..."
                    )
                    
                    # Get full book text for summary (not filtered)
                    with open(raw_text_path, 'r', encoding='utf-8') as f:
                        full_book_text = f.read()
                    
                    summary_text, summary_stats = generate_book_summary(
                        book_text=full_book_text,
                        book_title=book_title,
                        genre=genre,
                        book_type=book_type,
                        target_word_count=settings.SUMMARY_TARGET_WORDS,
                        max_word_count=settings.SUMMARY_MAX_WORDS
                    )
                    
                    summary_path = job_dir / f"{job_id}_summary.txt"
                    summary_path.write_text(summary_text, encoding='utf-8')
                    
                    logger.info(
                        f"Summary saved to {summary_path} (~{summary_stats['word_count']} words, est {summary_stats['estimated_minutes']} min)."
                    )
                    
                    job_metadata["summary"] = {
                        "path": str(summary_path),
                        **summary_stats
                    }
                    job_metadata["summary_path"] = str(summary_path)
                    _write_metadata()
                    
                    self.job_service.update_job(
                        job_id=job_id,
                        status="processing",
                        message="Summary generated. Ready for summary video generation.",
                        metadata={"summary_path": str(summary_path)}
                    )
                    
                except Exception as summary_error:
                    logger.error("Summary generation failed.", exc_info=True)
                    self.job_service.update_job(
                        job_id=job_id,
                        status="completed",
                        message=f"Main video completed, but summary generation failed: {str(summary_error)}",
                        metadata={"summary_error": str(summary_error)}
                    )
            else:
                # No summary requested, mark as completed
                # Ensure final_video_path is in metadata
                if "final_video_path" not in job_metadata:
                    job_metadata["final_video_path"] = str(final_video_path)
                _write_metadata()
                
                self.job_service.update_job(
                    job_id=job_id,
                    status="completed",
                    message="Video generation completed successfully",
                    metadata=job_metadata,
                    progress=100
                )
                logger.info("--- PIPELINE SUCCESS (No summary requested) ---")
            
        except Exception as e:
            error_msg = str(e)
            import traceback
            full_traceback = traceback.format_exc()
            
            logger.error(f"--- PIPELINE FAILED FOR JOB {job_id}: {error_msg} ---", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{full_traceback}")
            
            # Extract more detailed error information if available
            detailed_error = error_msg
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                detailed_error = f"{error_msg}\nResponse: {e.response.text}"
            elif hasattr(e, 'status_code'):
                detailed_error = f"{error_msg} (Status: {e.status_code})"
            
            try:
                self.job_service.update_job(
                    job_id=job_id,
                    status="failed",
                    message=f"Pipeline failed: {error_msg}",
                    metadata={
                        "error": error_msg,
                        "error_type": type(e).__name__,
                        "detailed_error": detailed_error,
                        "traceback": full_traceback
                    }
                )
                logger.info(f"Job status updated to failed")
            except Exception as update_error:
                logger.error(f"Failed to update job status to failed: {update_error}", exc_info=True)
            
            # Don't re-raise - background tasks should handle errors gracefully
            # The error is logged and job status is updated, so we don't need to crash
    
    def run_pipeline_from_text(
        self,
        job_id: str,
        text_path: Path,
        voice_provider: str = "openai",
        cartesia_voice_id: Optional[str] = None,
        cartesia_model_id: Optional[str] = None
    ):
        """
        Run the video generation pipeline from text directly (skips PDF processing).
        Used for generating videos from summary text.
        
        Args:
            job_id: Unique job identifier
            text_path: Path to the text file
            voice_provider: Voice provider ("openai" or "cartesia")
            cartesia_voice_id: Cartesia voice ID (if using Cartesia)
            cartesia_model_id: Cartesia model ID (if using Cartesia)
        """
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(job_id=job_id, log_level="INFO")
        logger = logging.getLogger(__name__)
        
        logger.info(f"=== TEXT-TO-VIDEO PIPELINE STARTED FOR JOB: {job_id} ===")
        logger.info(f"Text Path: {text_path}")
        logger.info(f"Job Directory: {job_dir}")
        
        try:
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Starting video generation from summary text...",
                progress=0
            )
            
            # Read text from file
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if not text_content.strip():
                raise ValueError("Text content is empty. Cannot proceed.")
            
            logger.info(f"Text loaded: {len(text_content)} characters")
            
            # ===== TEXT CLEANING =====
            logger.info("--- TEXT CLEANING ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Cleaning text for narration...",
                progress=10
            )
            
            # Create a temporary directory for tables/images (empty for text-only)
            tables_dir = job_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            images_dir = job_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            # Save text to a temporary file for cleaning
            temp_text_path = job_dir / "temp_input_text.txt"
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            cleaned_script_path = clean_text(
                raw_text_path=temp_text_path,
                tables_dir=tables_dir,
                images_dir=images_dir,
                job_dir=job_dir
            )
            
            with open(cleaned_script_path, 'r', encoding='utf-8') as f:
                text_script = f.read()
                if not text_script.strip():
                    raise ValueError("Cleaned script is empty. Cannot proceed.")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Text cleaning complete",
                progress=15
            )
            
            # ===== AI SERVICES =====
            logger.info("--- AI SERVICES ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Detecting genre...",
                progress=20
            )
            
            # Use a generic title for text-based generation
            book_title = "Summary Video"
            genre = "general"  # Default genre for summaries
            logger.info(f"Using genre: {genre}")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Generating audio narration...",
                progress=25
            )
            
            # Initialize metadata
            job_metadata = {
                "job_id": job_id,
                "book_title": book_title,
                "genre": genre,
                "created_at": datetime.now().isoformat(),
                "voice_provider": voice_provider,
                "source": "text"
            }
            
            if voice_provider.lower() == "cartesia":
                if cartesia_voice_id:
                    job_metadata["cartesia_voice_id"] = cartesia_voice_id
                if cartesia_model_id:
                    job_metadata["cartesia_model_id"] = cartesia_model_id
            
            metadata_path = job_dir / "job_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(job_metadata, f, indent=2)
            
            # Initialize voice service
            if voice_provider.lower() == "cartesia":
                voice_id = cartesia_voice_id or "98a34ef2-2140-4c28-9c71-663dc4dd7022"
                model_id = cartesia_model_id or "sonic-3"
                logger.info(f"Using Cartesia (Voice: {voice_id}, Model: {model_id})")
                voice_service = CartesiaService(voice_id=voice_id, model_id=model_id)
            else:
                logger.info("Using OpenAI for voice generation with voice: onyx")
                voice_service = OpenAIService(voice="onyx")
            
            raw_audio_path, timestamps_path = voice_service.generate_audio_with_timestamps(
                text=text_script,
                output_dir=job_dir,
                job_id=job_id,
                genre=genre
            )
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Audio generation complete, processing audio...",
                progress=50
            )
            
            # ===== AUDIO PROCESSING =====
            logger.info("--- AUDIO PROCESSING ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Mastering audio quality...",
                progress=55
            )
            
            processed_audio_path = job_dir / f"{job_id}_processed_audio.mp3"
            processed_audio_path = master_audio(
                raw_audio_path=raw_audio_path,
                processed_audio_path=processed_audio_path
            )
            
            # Regenerate timestamps from processed audio
            logger.info("Regenerating timestamps from processed audio for perfect sync...")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Regenerating timestamps from processed audio...",
                progress=58
            )
            
            from openai import OpenAI
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            with open(processed_audio_path, "rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            timestamps_data = transcription.model_dump()
            with open(timestamps_path, "w", encoding="utf-8") as f:
                json.dump(timestamps_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Timestamps regenerated: {len(timestamps_data.get('words', []))} words")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Audio mastering complete",
                progress=60
            )
            
            # ===== VIDEO GENERATION =====
            logger.info("--- VIDEO GENERATION ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Rendering video frames...",
                progress=70
            )
            
            final_video_path = job_dir / f"{job_id}_final_video.mp4"
            final_video_path = render_video(
                audio_path=processed_audio_path,
                timestamps_path=timestamps_path,
                output_path=final_video_path
            )
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Video frames rendered, encoding final video...",
                progress=90
            )
            
            import time
            time.sleep(0.5)
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Finalizing video...",
                progress=95
            )
            
            logger.info(f"--- VIDEO COMPLETE ---")
            logger.info(f"Final Video: {final_video_path}")
            
            # Update metadata
            job_metadata["final_video_path"] = str(final_video_path)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(job_metadata, f, indent=2)
            
            self.job_service.update_job(
                job_id=job_id,
                status="completed",
                message="Video generation completed successfully",
                progress=100,
                metadata={
                    "final_video_path": str(final_video_path),
                    "source": "text"
                }
            )
            
            logger.info(f"=== TEXT-TO-VIDEO PIPELINE COMPLETED FOR JOB: {job_id} ===")
            
        except Exception as e:
            logger.error(f"Text-to-video pipeline failed for job {job_id}: {e}", exc_info=True)
            self.job_service.update_job(
                job_id=job_id,
                status="failed",
                message=f"Pipeline failed: {str(e)}",
                metadata={"error": str(e)}
            )
            raise
    
    def run_pipeline_for_reels(
        self,
        job_id: str,
        text_path: Path,
        voice_provider: str = "openai",
        cartesia_voice_id: Optional[str] = None,
        cartesia_model_id: Optional[str] = None
    ):
        """
        Run the video generation pipeline for reels/shorts (skips PDF processing and text cleaning).
        Uses custom background image and smaller font size.
        
        Args:
            job_id: Unique job identifier
            text_path: Path to the text file
            voice_provider: Voice provider ("openai" or "cartesia")
            cartesia_voice_id: Cartesia voice ID (if using Cartesia)
            cartesia_model_id: Cartesia model ID (if using Cartesia)
        """
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(job_id=job_id, log_level="INFO")
        logger = logging.getLogger(__name__)
        
        logger.info(f"=== REELS/SHORTS VIDEO PIPELINE STARTED FOR JOB: {job_id} ===")
        logger.info(f"Text Path: {text_path}")
        logger.info(f"Job Directory: {job_dir}")
        
        try:
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Starting reels/shorts video generation...",
                progress=0
            )
            
            # Read text from file (skip text cleaning step, but remove commas)
            with open(text_path, 'r', encoding='utf-8') as f:
                text_script = f.read()
            
            if not text_script.strip():
                raise ValueError("Text content is empty. Cannot proceed.")
            
            # Remove commas from text for video generation
            text_script = text_script.replace(',', '')
            
            logger.info(f"Text loaded: {len(text_script)} characters (commas removed, skipping other text cleaning)")
            
            # ===== AI SERVICES =====
            logger.info("--- AI SERVICES ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Detecting genre...",
                progress=10
            )
            
            # Use a generic title for reels
            book_title = "Reels/Shorts Video"
            genre = "general"  # Default genre for reels
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Generating audio narration...",
                progress=15
            )
            
            # Initialize metadata
            job_metadata = {
                "job_id": job_id,
                "book_title": book_title,
                "genre": genre,
                "created_at": datetime.now().isoformat(),
                "voice_provider": voice_provider,
                "source": "reels",
                "video_type": "reels_shorts"
            }
            
            if voice_provider.lower() == "cartesia":
                if cartesia_voice_id:
                    job_metadata["cartesia_voice_id"] = cartesia_voice_id
                if cartesia_model_id:
                    job_metadata["cartesia_model_id"] = cartesia_model_id
            
            metadata_path = job_dir / "job_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(job_metadata, f, indent=2)
            
            # Initialize voice service
            if voice_provider.lower() == "cartesia":
                voice_id = cartesia_voice_id or "98a34ef2-2140-4c28-9c71-663dc4dd7022"
                model_id = cartesia_model_id or "sonic-3"
                logger.info(f"Using Cartesia (Voice: {voice_id}, Model: {model_id})")
                voice_service = CartesiaService(voice_id=voice_id, model_id=model_id)
            else:
                logger.info("Using OpenAI for voice generation with voice: onyx")
                voice_service = OpenAIService(voice="onyx")
            
            raw_audio_path, timestamps_path = voice_service.generate_audio_with_timestamps(
                text=text_script,
                output_dir=job_dir,
                job_id=job_id,
                genre=genre
            )
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Audio generation complete, processing audio...",
                progress=50
            )
            
            # ===== AUDIO PROCESSING =====
            logger.info("--- AUDIO PROCESSING ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Mastering audio quality...",
                progress=55
            )
            
            processed_audio_path = job_dir / f"{job_id}_processed_audio.mp3"
            processed_audio_path = master_audio(
                raw_audio_path=raw_audio_path,
                processed_audio_path=processed_audio_path
            )
            
            # Regenerate timestamps from processed audio
            logger.info("Regenerating timestamps from processed audio for perfect sync...")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Regenerating timestamps from processed audio...",
                progress=58
            )
            
            from openai import OpenAI
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            with open(processed_audio_path, "rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            timestamps_data = transcription.model_dump()
            with open(timestamps_path, "w", encoding="utf-8") as f:
                json.dump(timestamps_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Timestamps regenerated: {len(timestamps_data.get('words', []))} words")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Audio mastering complete",
                progress=60
            )
            
            # ===== VIDEO GENERATION =====
            logger.info("--- VIDEO GENERATION (REELS/SHORTS) ---")
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Rendering video frames with custom background...",
                progress=70
            )
            
            # Use custom background for reels - load image to get actual dimensions
            reels_background_path = settings.BACKGROUNDS_PATH / "white-paper-texture-background.jpg"
            
            # Load background image to get actual dimensions (source of truth)
            from PIL import Image
            bg_image = Image.open(str(reels_background_path))
            reels_width, reels_height = bg_image.size
            
            logger.info(f"Reels video settings: Background={reels_background_path}, dimensions={reels_width}x{reels_height}")
            logger.info("Using smart detection: font size and margins will be calculated automatically based on dimensions")
            
            final_video_path = job_dir / f"{job_id}_final_video.mp4"
            # Don't pass font_size - let FrameGeneratorV11 calculate it based on reels mode
            # Don't pass width/height - let render_video get them from the background image
            final_video_path = render_video(
                audio_path=processed_audio_path,
                timestamps_path=timestamps_path,
                output_path=final_video_path,
                background_path=reels_background_path
                # width and height will be detected from background image
                # font_size will be calculated by FrameGeneratorV11 in reels mode
            )
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Video frames rendered, encoding final video...",
                progress=90
            )
            
            import time
            time.sleep(0.5)
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Finalizing video...",
                progress=95
            )
            
            logger.info(f"--- REELS VIDEO COMPLETE ---")
            logger.info(f"Final Video: {final_video_path}")
            
            # Update metadata
            job_metadata["final_video_path"] = str(final_video_path)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(job_metadata, f, indent=2)
            
            self.job_service.update_job(
                job_id=job_id,
                status="completed",
                message="Reels/shorts video generation completed successfully",
                progress=100,
                metadata={
                    "final_video_path": str(final_video_path),
                    "source": "reels",
                    "video_type": "reels_shorts"
                }
            )
            
            logger.info(f"=== REELS/SHORTS VIDEO PIPELINE COMPLETED FOR JOB: {job_id} ===")
            
        except Exception as e:
            logger.error(f"Reels video pipeline failed for job {job_id}: {e}", exc_info=True)
            self.job_service.update_job(
                job_id=job_id,
                status="failed",
                message=f"Pipeline failed: {str(e)}",
                metadata={"error": str(e)}
            )
            raise
    
    def generate_summary(self, job_id: str):
        """
        Generate a book summary after main video is complete.
        
        Args:
            job_id: Unique job identifier
        """
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        setup_logging(job_id=job_id, log_level="INFO")
        logger = logging.getLogger(__name__)
        
        try:
            job = self.job_service.get_job(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job["status"] != "completed":
                raise ValueError("Main video must be completed before generating summary")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Preparing book summary...",
                metadata={
                    "summary_status": "processing",
                    "summary_progress": 5
                }
            )
            
            # Load metadata
            metadata_path = job_dir / "job_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                job_metadata = json.load(f)
            
            # Get full book text from extraction
            extraction_json_path = job_dir / f"{job_id}_extraction.json"
            if not extraction_json_path.exists():
                raise ValueError("Extraction JSON not found. Cannot generate summary.")
            
            with open(extraction_json_path, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Collecting full book text for summary...",
                metadata={
                    "summary_status": "processing",
                    "summary_progress": 20
                }
            )
            
            # Reconstruct full book text
            full_book_text = ""
            for page in extraction_data.get('text_extraction', {}).get('pages', []):
                full_book_text += page.get('text', '') + "\n\n"
            
            if not full_book_text.strip():
                raise ValueError("No text found in extraction data")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Generating long-form summary (this may take a few minutes)...",
                metadata={
                    "summary_status": "processing",
                    "summary_progress": 45
                }
            )
            
            # Get book metadata
            book_title = job_metadata.get("book_title", "Unknown")
            genre = job_metadata.get("genre", "novel")
            book_type = job_metadata.get("book_type", "unknown")
            
            logger.info("--- SUMMARY GENERATION (Target ~1 hour narration) ---")
            
            # Generate summary
            summary_text, summary_stats = generate_book_summary(
                book_text=full_book_text,
                book_title=book_title,
                genre=genre,
                book_type=book_type,
                target_word_count=settings.SUMMARY_TARGET_WORDS,
                max_word_count=settings.SUMMARY_MAX_WORDS
            )
            
            summary_path = job_dir / f"{job_id}_summary.txt"
            summary_path.write_text(summary_text, encoding='utf-8')
            
            logger.info(
                f"Summary saved to {summary_path} (~{summary_stats['word_count']} words, est {summary_stats['estimated_minutes']} min)."
            )
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Summary generated. Finalizing...",
                metadata={
                    "summary_status": "processing",
                    "summary_progress": 85
                }
            )
            
            # Update metadata
            job_metadata["summary"] = {
                "path": str(summary_path),
                **summary_stats
            }
            job_metadata["summary_path"] = str(summary_path)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(job_metadata, f, indent=2)
            
            self.job_service.update_job(
                job_id=job_id,
                status="completed",
                message="Summary generated successfully. Ready for summary video generation.",
                metadata={
                    "summary_path": str(summary_path),
                    "summary_status": "completed",
                    "summary_progress": 100
                }
            )
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            self.job_service.update_job(
                job_id=job_id,
                status="completed",  # Main video is still completed
                message=f"Main video completed, but summary generation failed: {str(e)}",
                metadata={
                    "summary_error": str(e),
                    "summary_status": "failed",
                    "summary_progress": 0
                }
            )
            raise
    
    def generate_summary_video(
        self, 
        job_id: str,
        voice_provider: str = "openai",
        cartesia_voice_id: Optional[str] = None,
        cartesia_model_id: Optional[str] = None
    ):
        """
        Generate video from the summary text.
        
        Args:
            job_id: Unique job identifier
            voice_provider: Voice provider ("openai" or "cartesia")
            cartesia_voice_id: Cartesia voice ID (if using Cartesia)
            cartesia_model_id: Cartesia model ID (if using Cartesia)
        """
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        setup_logging(job_id=job_id, log_level="INFO")
        logger = logging.getLogger(__name__)
        
        try:
            job = self.job_service.get_job(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            summary_path = job.get("metadata", {}).get("summary_path")
            if not summary_path:
                raise ValueError("Summary not available for this job")
            
            summary_file = Path(summary_path)
            if not summary_file.exists():
                raise ValueError("Summary file not found")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Preparing summary video...",
                metadata={
                    "summary_video_status": "processing",
                    "summary_video_progress": 5
                }
            )
            
            # Read summary text
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_text = f.read()
            
            # Get genre and book info from metadata
            metadata_path = job_dir / "job_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                job_metadata = json.load(f)
            
            genre = job_metadata.get("genre", "novel")
            book_title = job_metadata.get("book_title", "Unknown")
            
            # Create summary video directory
            summary_job_dir = job_dir / "summary_video"
            summary_job_dir.mkdir(exist_ok=True)
            summary_job_id = f"{job_id}_summary"
            
            logger.info("--- SUMMARY VIDEO PIPELINE ---")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Generating summary narration...",
                metadata={
                    "summary_video_status": "processing",
                    "summary_video_progress": 25
                }
            )
            
            # Use voice provider from parameters (or fallback to metadata)
            voice_provider = voice_provider or job_metadata.get("voice_provider", "openai")
            
            # Get Cartesia settings from parameters or metadata
            if not cartesia_voice_id:
                cartesia_voice_id = job_metadata.get("cartesia_voice_id") or "98a34ef2-2140-4c28-9c71-663dc4dd7022"
            if not cartesia_model_id:
                cartesia_model_id = job_metadata.get("cartesia_model_id") or "sonic-3"
            
            # Initialize voice service based on provider
            if voice_provider.lower() == "cartesia":
                logger.info(f"Using Cartesia for summary video voice generation (Voice: {cartesia_voice_id}, Model: {cartesia_model_id})")
                voice_service = CartesiaService(voice_id=cartesia_voice_id, model_id=cartesia_model_id)
            else:
                logger.info(f"Using OpenAI for summary video voice generation")
                voice_service = OpenAIService(voice="onyx")
            
            summary_raw_audio_path, summary_timestamps_path = voice_service.generate_audio_with_timestamps(
                text=summary_text,
                output_dir=summary_job_dir,
                job_id=summary_job_id,
                genre=genre
            )
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Mastering summary narration audio...",
                metadata={
                    "summary_video_status": "processing",
                    "summary_video_progress": 50
                }
            )
            
            # Process audio
            summary_processed_audio_path = summary_job_dir / f"{summary_job_id}_processed_audio.mp3"
            summary_processed_audio_path = master_audio(
                raw_audio_path=summary_raw_audio_path,
                processed_audio_path=summary_processed_audio_path
            )
            
            # Regenerate timestamps from processed audio to ensure perfect sync
            logger.info("Regenerating summary timestamps from processed audio for perfect sync...")
            from openai import OpenAI
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            with open(summary_processed_audio_path, "rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            # Save updated timestamps
            timestamps_data = transcription.model_dump()
            with open(summary_timestamps_path, "w", encoding="utf-8") as f:
                json.dump(timestamps_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary timestamps regenerated from processed audio: {len(timestamps_data.get('words', []))} words, {len(timestamps_data.get('segments', []))} segments")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Rendering summary video frames...",
                metadata={
                    "summary_video_status": "processing",
                    "summary_video_progress": 70
                }
            )
            
            # Render video
            summary_final_video_path = summary_job_dir / f"{summary_job_id}_final_video.mp4"
            summary_final_video_path = render_video(
                audio_path=summary_processed_audio_path,
                timestamps_path=summary_timestamps_path,
                output_path=summary_final_video_path
            )
            
            logger.info(f"Summary video created: {summary_final_video_path}")
            
            self.job_service.update_job(
                job_id=job_id,
                status="processing",
                message="Encoding summary video...",
                metadata={
                    "summary_video_status": "processing",
                    "summary_video_progress": 90
                }
            )
            
            # Update metadata
            job_metadata["summary_video_path"] = str(summary_final_video_path)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(job_metadata, f, indent=2)
            
            self.job_service.update_job(
                job_id=job_id,
                status="completed",
                message="Summary video generation completed",
                metadata={
                    "summary_video_path": str(summary_final_video_path),
                    "summary_video_status": "completed",
                    "summary_video_progress": 100
                }
            )
            
        except Exception as e:
            logger.error(f"Summary video generation failed: {e}", exc_info=True)
            self.job_service.update_job(
                job_id=job_id,
                status="completed",  # Main video is still completed
                message=f"Main video completed, but summary video generation failed: {str(e)}",
                metadata={
                    "summary_video_error": str(e),
                    "summary_video_status": "failed",
                    "summary_video_progress": 0
                }
            )
            raise

