import subprocess
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def _run_ffmpeg_command(command: List[str]):
    """Helper function to run an FFmpeg command."""
    try:
        logger.debug(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg command failed!", exc_info=True)
        logger.error(f"FFmpeg STDERR: {e.stderr}")
        raise

def master_audio(raw_audio_path: Path, processed_audio_path: Path) -> Path:
    """
    Applies the "Top-Notch" SOTA Audio Processing Pipeline.
    Denoises and masters the audio *without* changing its length.
    
    Args:
        raw_audio_path: Path to the input audio from OpenAI.
        processed_audio_path: Path to save the final mastered audio.

    Returns:
        The path to the processed audio.
    """
    logger.info(f"Starting SOTA Audio Pipeline for {raw_audio_path.name}...")
    
    denoised_path = raw_audio_path.parent / f"{raw_audio_path.stem}_denoised.mp3"

    try:
        # --- Step 1: Denoise ---
        logger.info("Step 1: Denoising audio...")
        denoise_command = [
            "ffmpeg",
            "-i", str(raw_audio_path),
            "-af", "anlmdn",
            "-b:a", "320k",
            "-y",
            str(denoised_path)
        ]
        _run_ffmpeg_command(denoise_command)
        logger.info(f"Denoising complete. Saved to: {denoised_path}")

        # --- Step 2: Final Mastering ---
        logger.info("Step 2: Applying final voice mastering...")
        filter_chain = (
            "highpass=f=90,lowpass=f=13500,"
            "acompressor=threshold=-18dB:ratio=2.2:attack=10:release=200,"
            "bass=g=3:f=150,treble=g=2:f=4000,"
            "loudnorm=i=-18:tp=-2"
        )
        
        master_command = [
            "ffmpeg",
            "-i", str(denoised_path),      # Input from Step 1
            "-af", filter_chain,
            "-ar", "48000",
            "-ac", "2",
            "-b:a", "256k",
            "-y",
            str(processed_audio_path) # Final output path
        ]
        _run_ffmpeg_command(master_command)
        
        # --- Cleanup ---
        if denoised_path.exists():
            denoised_path.unlink()
            
        logger.info(f"Audio mastering complete! Final file: {processed_audio_path}")
        return processed_audio_path

    except Exception as e:
        logger.error(f"Audio pipeline failed: {e}", exc_info=True)
        # Clean up partial files on error
        if denoised_path.exists():
            denoised_path.unlink()
        if processed_audio_path.exists():
            processed_audio_path.unlink()
        raise