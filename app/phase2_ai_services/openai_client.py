import json
import logging
from pathlib import Path
from openai import OpenAI
from typing import Tuple

from app.config import settings

logger = logging.getLogger(__name__)

class OpenAIService:
    """Service for OpenAI API integration (TTS + Whisper STT)."""

    def __init__(self, voice: str = "onyx"):
        
        api_key = settings.OPENAI_API_KEY 
        
        if not api_key or "sk-" not in api_key:
            raise ValueError(
                "OpenAI API key not configured or invalid. "
                "Please set OPENAI_API_KEY in your .env file."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.voice = voice
        logger.info(f"OpenAIService initialized (Voice: {self.voice})")

    def generate_audio_with_timestamps(
        self, 
        text: str, 
        output_dir: Path,
        job_id: str
    ) -> Tuple[Path, Path]:
        
        logger.info(f"Job {job_id}: Starting OpenAI 2-Call Pipeline...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Define output paths
        audio_path = output_dir / f"{job_id}_raw_audio.mp3"
        timestamps_path = output_dir / f"{job_id}_timestamps.json"

        
        instructions = """
Voice Affect: You are Story teller you worked as a story teller in a book store and you are now a story teller in podcast.
Tone: Friendly, engaging, and conversational, Speak at a moderate pace - never rushed.
Clarity: Use clear enunciation and simple langauge.
Pacing: Measured and deliberate. Speak at a steady pace, using brief pauses to emphasize key concepts and findings. Always have a pause between sentences, paragraphs, sections, chapters, full stop's, etc.
Emotion: Enthusiastic and vivid, embodying a storyteller who brings the narrative to life with expressive intonation and dynamic pacing. Convey curiosity, wonder, and intrigue throughout, as if captivating an audience at a bookstore or on a podcast.
Pronunciation: Embody a natural storyteller; use expressive phrasing, dynamic intonation, and smooth flow. Words should sound vivid and fluent, as if captivating an audience. Allow pronunciations to be naturally engaging, emphasizing clarity for important or scientific terms but always in a warm, inviting, and relatable way.
"""
        
        try:
            # --- Call 1: Generate Audio (TTS) ---
            logger.info(f"Job {job_id}: Calling OpenAI TTS API (Voice: {self.voice})...")
            response = self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=self.voice,
                input=text,
                instructions=instructions,
                response_format="mp3"
            )
            response.stream_to_file(str(audio_path))
            logger.info(f"Job {job_id}: Raw audio saved to {audio_path}")
            
            # --- Call 2: Generate Timestamps (Whisper STT) ---
            logger.info(f"Job {job_id}: Calling OpenAI Whisper API for timestamps...")
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            timestamps_data = transcription.model_dump()
            
            with open(timestamps_path, "w", encoding="utf-8") as f:
                json.dump(timestamps_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Job {job_id}: Timestamps saved to {timestamps_path}")
            
            return audio_path, timestamps_path
        
        except Exception as e:
            logger.error(f"Job {job_id}: Error in OpenAI 2-Call Pipeline!", exc_info=True)
            raise