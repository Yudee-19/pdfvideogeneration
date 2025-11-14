import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, CompositeVideoClip, VideoClip, ImageClip
import moviepy.video.fx as vfx
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None
    probability: Optional[float] = None


class FrameGeneratorV11:
    """
    Generates TRANSPARENT frames with clean "karaoke-style" animated text.
    V11: Robust punctuation alignment, left-aligned, bigger font.
    """
    def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
        logger.info("Initializing FrameGeneratorV11 (Robust Punctuation)...")
        
        self.bg_width = bg_width
        self.bg_height = bg_height

        self.all_words, self.segments = self._load_data(timestamps_path)

        # --- 1. Load settings from config.py ---
        self.font_size = max(40, int(self.bg_height / 6))
        self.line_height = int(self.font_size * 1.25)
        self.regular_font, self.bold_font = self._load_fonts(self.font_size)
        self.max_text_width = int(self.bg_width * 0.90)
        self.min_words_per_slide = 8
        
        self.slides, self.slide_layouts, self.slide_start_times = self._build_grouped_slides(
            self.min_words_per_slide
        )
        logger.info(f"FrameGeneratorV11 initialized: {len(self.segments)} segments grouped into {len(self.slides)} slides.")

    def _load_fonts(self, size: int) -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
        # --- 2. Load fonts from config.py paths ---
        try:
            regular_path = settings.DEFAULT_FONT_REGULAR
            bold_path = settings.DEFAULT_FONT_BOLD
            regular = ImageFont.truetype(regular_path, size)
            bold = ImageFont.truetype(bold_path, size)
            logger.info(f"Loaded custom font: {regular_path}")
            return regular, bold
        except Exception as e:
            logger.error(f"FATAL: Could not load custom font! {e}", exc_info=True)
            logger.warning("Falling back to default font.")
            return ImageFont.load_default(size=size), ImageFont.load_default(size=size)

   
    def _load_data(self, timestamps_path: Path) -> Tuple[List[WordTimestamp], List[Dict]]:
        logger.info(f"Loading data from: {timestamps_path}")
        with open(timestamps_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "words" not in data or "segments" not in data:
            raise ValueError("Invalid timestamp format: expected 'words' and 'segments' keys")
        if not data["segments"]:
            raise ValueError("Timestamp data is missing 'segments'. Please re-run the OpenAI service with timestamp_granularities=['word', 'segment']")
        words = [WordTimestamp(**w) for w in data["words"] if w.get('word', '').strip()]
        segments = data["segments"]
        logger.info(f"Loaded {len(words)} words and {len(segments)} segments.")
        return words, segments

    def _get_words_for_segment(self, segment_index: int) -> List[WordTimestamp]:
        segment = self.segments[segment_index]
        segment_start = segment["start"]
        segment_end = self.segments[segment_index + 1]["start"] if segment_index + 1 < len(self.segments) else float('inf')
        return [w for w in self.all_words if w.start >= segment_start and w.start < segment_end]

    def _build_grouped_slides(self, min_words_per_slide: int) -> Tuple[List[List[List[WordTimestamp]]], Dict[int, Dict[int, Tuple[int, int]]], List[float]]:
        logger.info(f"Building grouped slides (min_words={min_words_per_slide})...")
        slides, layouts, slide_start_times = [], {}, []
        dummy_img = Image.new("RGB", (self.bg_width, self.bg_height))
        draw = ImageDraw.Draw(dummy_img)
        space_bbox = draw.textbbox((0, 0), " ", font=self.bold_font); space_width = space_bbox[2] - space_bbox[0]
        current_slide_words_ts, current_slide_start_time, current_slide_segments = [], -1, []
        for i, segment in enumerate(self.segments):
            segment_words_ts = self._get_words_for_segment(i)
            if not segment_words_ts: continue
            if not current_slide_words_ts: current_slide_start_time = segment["start"]
            current_slide_words_ts.extend(segment_words_ts); current_slide_segments.append(segment)
            if len(current_slide_words_ts) >= min_words_per_slide or i == len(self.segments) - 1:
                if not current_slide_words_ts: continue
                slide_index = len(slides)
                all_clean_words = [word for s in current_slide_segments for word in s["text"].strip().split()]
                if len(all_clean_words) == len(current_slide_words_ts):
                    for j in range(len(all_clean_words)): current_slide_words_ts[j].word = all_clean_words[j]
                current_slide_lines, current_line = [], []
                for word in current_slide_words_ts:
                    word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font); word_width = word_bbox[2] - word_bbox[0]
                    line_bbox = draw.textbbox((0, 0), " ".join(w.word for w in current_line), font=self.bold_font); line_width = line_bbox[2] - line_bbox[0]
                    if line_width + word_width + space_width > self.max_text_width and current_line:
                        current_slide_lines.append(current_line); current_line = [word]
                    else: current_line.append(word)
                if current_line: current_slide_lines.append(current_line)
                slides.append(current_slide_lines); layouts[slide_index] = {}
                total_text_height = len(current_slide_lines) * self.line_height; start_y = (self.bg_height - total_text_height) // 2; current_y = start_y
                for line_of_words in current_slide_lines:
                    current_x = int(self.bg_width * 0.05)
                    for word in line_of_words:
                        layouts[slide_index][id(word)] = (current_x, current_y)
                        word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font); word_width = word_bbox[2] - word_bbox[0]
                        current_x += word_width + space_width
                    current_y += self.line_height
                slide_start_times.append(current_slide_start_time)
                current_slide_words_ts, current_slide_start_time, current_slide_segments = [], -1, []
        logger.info("Grouped slide building complete.")
        return slides, layouts, slide_start_times
    
    def make_frame_function(self, slide_index: int, slide_start_time: float):
        def generate_frame(t_local: float) -> np.ndarray:
            global_t = slide_start_time + t_local
            frame = Image.new("RGBA", (self.bg_width, self.bg_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            slide_lines = self.slides[slide_index]
            layout = self.slide_layouts[slide_index]
            for line in slide_lines:
                for word in line:
                    coords = layout.get(id(word))
                    if not coords: continue
                    
                    # --- 3. Load colors from config.py ---
                    if global_t >= word.start:
                        font = self.bold_font
                        color = settings.TEXT_BOLD_COLOR
                    else:
                        font = self.regular_font
                        color = settings.TEXT_REGULAR_COLOR
                    draw.text(coords, word.word, font=font, fill=color)
            return np.array(frame)
        return generate_frame


# This is the "callable" version 
def render_video(
    audio_path: Path,
    timestamps_path: Path,
    output_path: Path
) -> Path:
    """
    Renders the final karaoke-style video.
    
    Args:
        audio_path: Path to the PROCESSED audio file.
        timestamps_path: Path to the timestamps.json file.
        output_path: Path to save the final .mp4 video.

    Returns:
        The path to the rendered video.
    """
    logger.info("--- Starting Video Rendering Pipeline ---")
    
    try:
        logger.info(f"Loading audio: {audio_path}")
        audio_clip = AudioFileClip(str(audio_path))
        
        # --- 4. Load background/config from settings ---
        background_path = settings.DEFAULT_BACKGROUND
        fps = settings.VIDEO_FPS
        width = settings.VIDEO_WIDTH
        height = settings.VIDEO_HEIGHT
        
        logger.info(f"Loading background: {background_path}")
        bg_clip = ImageClip(background_path).with_duration(audio_clip.duration)

        frame_gen = FrameGeneratorV11(
            timestamps_path=timestamps_path,
            bg_width=width,
            bg_height=height
        )

        logger.info("Generating video clips for each slide...")
        all_slide_clips = []
        fade_duration = 0.25
        start_times = frame_gen.slide_start_times
        num_slides = len(start_times)

        for i in range(num_slides):
            slide_start = start_times[i]
            if i + 1 < num_slides:
                clip_duration = (start_times[i+1] - slide_start) + fade_duration
            else:
                clip_duration = audio_clip.duration - slide_start
            if clip_duration <= fade_duration: continue

            slide_clip = VideoClip(
                frame_function=frame_gen.make_frame_function(
                    slide_index=i,
                    slide_start_time=slide_start
                ),
                duration=clip_duration,
                is_mask=True
            )
            if i > 0:
                slide_clip = vfx.FadeIn(duration=fade_duration, initial_color=[0, 0, 0, 0]).apply(slide_clip)
            if i < num_slides - 1:
                slide_clip = vfx.FadeOut(duration=fade_duration, final_color=[0, 0, 0, 0]).apply(slide_clip)
            
            slide_clip = slide_clip.with_start(slide_start)
            all_slide_clips.append(slide_clip)
        
        logger.info(f"Created {len(all_slide_clips)} slide clips.")
        logger.info("Compositing final video...")
        
        final_video = CompositeVideoClip([bg_clip] + all_slide_clips)
        final_video = final_video.with_audio(audio_clip)

        logger.info(f"Rendering {fps}fps video to: {output_path}")
        
        final_video.write_videofile(
            str(output_path),
            fps=fps,
            codec=settings.VIDEO_CODEC,
            audio_codec="aac",
            preset="medium",
            threads=os.cpu_count(),
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        
        logger.info("--- Video Rendering Complete ---")
        return output_path

    except Exception as e:
        logger.error("Video rendering pipeline failed!", exc_info=True)
        raise