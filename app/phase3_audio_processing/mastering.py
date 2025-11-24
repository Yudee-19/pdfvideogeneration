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
    
    work_dir = raw_audio_path.parent
    temp_wav_path = work_dir / f"{raw_audio_path.stem}_temp.wav"
    static_reduced_wav = work_dir / f"{raw_audio_path.stem}_static.wav"
    denoised_wav = work_dir / f"{raw_audio_path.stem}_denoised.wav"
    mastered_wav = work_dir / f"{raw_audio_path.stem}_mastered.wav"

    try:
        # --- Step 0: Convert to high-quality WAV for processing ---
        logger.info("Step 0: Converting source audio to 48kHz WAV for processing...")
        convert_to_wav = [
            "ffmpeg",
            "-i", str(raw_audio_path),
            "-ar", "48000",  # Upsample for cleaner processing
            "-ac", "1",
            "-y",
            str(temp_wav_path)
        ]
        _run_ffmpeg_command(convert_to_wav)

        # --- Step 1: Static reduction + band limiting ---
        logger.info("Step 1: Reducing broadband static and hum...")
        logger.info("  - Applying highpass filter (90Hz) to remove low-frequency rumble")
        logger.info("  - Applying lowpass filter (16kHz) to remove harsh high frequencies")
        logger.info("  - Applying FFT-based denoiser (afftdn) to remove hiss and static noise")
        static_filter = (
            "highpass=f=90,"        # Remove very low rumble
            "lowpass=f=16000,"      # Remove harsh highs
            "afftdn=nf=-28"         # FFT-based denoiser for hiss/static
        )
        static_reduction_command = [
            "ffmpeg",
            "-i", str(temp_wav_path),
            "-af", static_filter,
            "-y",
            str(static_reduced_wav)
        ]
        _run_ffmpeg_command(static_reduction_command)

        # --- Step 2: Fine denoise & smooth ---
        logger.info("Step 2: Applying fine noise reduction...")
        logger.info("  - Applying non-linear median denoise (anlmdn) for additional static removal")
        fine_denoise_filter = "anlmdn=s=0.00005"  # Non-linear median denoise (gentle)
        fine_denoise_command = [
            "ffmpeg",
            "-i", str(static_reduced_wav),
            "-af", fine_denoise_filter,
            "-y",
            str(denoised_wav)
        ]
        _run_ffmpeg_command(fine_denoise_command)

        # --- Step 3: Mastering chain (de-esser, compressor, EQ, loudness) ---
        logger.info("Step 3: Mastering with sibilance reduction, compressor, and loudness normalization...")
        # Using high-shelf EQ to reduce sibilance instead of deesser (better compatibility)
        mastering_filter = (
            "highshelf=f=6000:width_type=h:width=2000:g=-2,"  # Reduce sibilance in 5-8kHz range
            "acompressor=threshold=-20dB:ratio=1.8:attack=5:release=120,"
            "bass=g=1.3:f=160,"
            "treble=g=1.1:f=3500,"
            "loudnorm=I=-16:TP=-1.5:LRA=10"
        )
        mastering_command = [
            "ffmpeg",
            "-i", str(denoised_wav),
            "-af", mastering_filter,
            "-ar", "44100",
            "-ac", "1",
            "-y",
            str(mastered_wav)
        ]
        _run_ffmpeg_command(mastering_command)

        # --- Step 4: Export back to high quality MP3 ---
        logger.info("Step 4: Exporting mastered audio to MP3...")
        export_command = [
            "ffmpeg",
            "-i", str(mastered_wav),
            "-c:a", "libmp3lame",
            "-b:a", "256k",
            "-ar", "44100",
            "-ac", "1",
            "-y",
            str(processed_audio_path)
        ]
        _run_ffmpeg_command(export_command)
        
        # --- Cleanup ---
        for temp_file in [temp_wav_path, static_reduced_wav, denoised_wav, mastered_wav]:
            if temp_file.exists():
                temp_file.unlink()
            
        logger.info(f"Audio mastering complete! Final file: {processed_audio_path}")
        return processed_audio_path

    except Exception as e:
        logger.error(f"Audio pipeline failed: {e}", exc_info=True)
        # Clean up partial files on error
        for temp_file in [temp_wav_path, static_reduced_wav, denoised_wav, mastered_wav]:
            if temp_file.exists():
                temp_file.unlink()
        if processed_audio_path.exists():
            processed_audio_path.unlink()
        raise