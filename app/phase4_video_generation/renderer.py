import json
import logging
import os
import tempfile
import multiprocessing
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, CompositeVideoClip, VideoClip, ImageClip, ImageSequenceClip
import moviepy.video.fx as vfx
from pydantic import BaseModel
from tqdm import tqdm

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
        self.line_height = int(self.font_size * 1.15)
        self.regular_font, self.bold_font = self._load_fonts(self.font_size)
        self.max_text_width = int(self.bg_width * 0.90)
        # self.min_words_per_slide = 8
        
        self.slides, self.slide_layouts, self.slide_start_times = self._build_grouped_slides()
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

    def _build_grouped_slides(self) -> Tuple[List[List[List[WordTimestamp]]], Dict[int, Dict[int, Tuple[int, int]]], List[float]]:
        logger.info("Building grouped slides (Max Words strategy)...")
        slides, layouts, slide_start_times = [], {}, []
        dummy_img = Image.new("RGB", (self.bg_width, self.bg_height))
        draw = ImageDraw.Draw(dummy_img)
        space_bbox = draw.textbbox((0, 0), " ", font=self.bold_font); space_width = space_bbox[2] - space_bbox[0]

        MAX_WORDS_PER_SLIDE = 10

        current_slide_words_ts, current_slide_start_time, current_slide_segments = [], -1, []

        def build_slide_layout(words_ts, segments_list, start_time):
            """Helper function to avoid repeating the build logic."""
            if not words_ts:
                return

            slide_index = len(slides)
            all_clean_words = [word for s in segments_list for word in s["text"].strip().split()]
            if len(all_clean_words) == len(words_ts):
                for j in range(len(all_clean_words)):
                    words_ts[j].word = all_clean_words[j]

            current_slide_lines, current_line = [], []
            for word in words_ts:
                word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font); word_width = word_bbox[2] - word_bbox[0]
                line_bbox = draw.textbbox((0, 0), " ".join(w.word for w in current_line), font=self.bold_font); line_width = line_bbox[2] - line_bbox[0]
                if line_width + word_width + space_width > self.max_text_width and current_line:
                    current_slide_lines.append(current_line); current_line = [word]
                else:
                    current_line.append(word)
            if current_line: current_slide_lines.append(current_line)

            slides.append(current_slide_lines); layouts[slide_index] = {}
            total_text_height = len(current_slide_lines) * self.line_height
            
            # If text is *still* too tall, warn about it.
            # This can happen if one segment is just massive.
            if total_text_height > self.bg_height * 0.95:
                 logger.warning(f"Slide {slide_index} (starting {start_time}s) may be too tall! Has {len(current_slide_lines)} lines.")

            start_y = (self.bg_height - total_text_height) // 2; current_y = start_y
            for line_of_words in current_slide_lines:
                current_x = int(self.bg_width * 0.05)
                for word in line_of_words:
                    layouts[slide_index][id(word)] = (current_x, current_y)
                    word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font); word_width = word_bbox[2] - word_bbox[0]
                    current_x += word_width + space_width
                current_y += self.line_height
            slide_start_times.append(start_time)

        # main loop
        for i, segment in enumerate(self.segments):
            segment_words_ts = self._get_words_for_segment(i)
            if not segment_words_ts: continue

            # Check if adding this NEW segment will make the slide TOO BIG
            if (len(current_slide_words_ts) + len(segment_words_ts) > MAX_WORDS_PER_SLIDE) and current_slide_words_ts:
                
                # 1. Build the PREVIOUS slide (it's full)
                build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                
                # 2. Reset and start the NEW slide with the current segment
                current_slide_words_ts = segment_words_ts
                current_slide_segments = [segment]
                current_slide_start_time = segment["start"]
            
            else:
                # --- It's not too big, so just add this segment to the current slide ---
                if not current_slide_words_ts: 
                    current_slide_start_time = segment["start"] # Set start time only for the first segment
                current_slide_words_ts.extend(segment_words_ts)
                current_slide_segments.append(segment)
            ### ----------------- ###

        # --- Handle the VERY LAST slide after the loop finishes ---
        if current_slide_words_ts:
            build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)

        logger.info(f"Grouped slide building complete. Created {len(slides)} slides.")
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

    def generate_single_frame(
        self,
        frame_number: int,
        frame_timestamp: float,
        slide_index: int,
        slide_start_time: float
    ) -> Image.Image:
        """
        Generate a single frame at a specific timestamp.
        Used for batch processing.
        """
        global_t = frame_timestamp
        frame = Image.new("RGBA", (self.bg_width, self.bg_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(frame)
        slide_lines = self.slides[slide_index]
        layout = self.slide_layouts[slide_index]
        
        for line in slide_lines:
            for word in line:
                coords = layout.get(id(word))
                if not coords:
                    continue
                
                # State-based logic: check if word has started
                if global_t >= word.start:
                    font = self.bold_font
                    color = settings.TEXT_BOLD_COLOR
                else:
                    font = self.regular_font
                    color = settings.TEXT_REGULAR_COLOR
                
                draw.text(coords, word.word, font=font, fill=color)
        
        return frame


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def _calculate_frame_timestamps(duration: float, fps: int) -> List[Tuple[int, float]]:
    """
    Calculate all frame timestamps for the video.
    These are evenly spaced frame timestamps - word highlighting will use
    Whisper timestamps directly for accuracy.
    
    Returns:
        List of (frame_number, timestamp) tuples
    """
    total_frames = int(duration * fps)
    frame_timestamps = []
    frame_interval = 1.0 / fps
    
    for frame_num in range(total_frames):
        # Calculate precise timestamp for each frame
        timestamp = frame_num * frame_interval
        frame_timestamps.append((frame_num, timestamp))
    
    return frame_timestamps


def _map_frames_to_slides(
    frame_timestamps: List[Tuple[int, float]],
    slide_start_times: List[float],
    audio_duration: float
) -> List[Tuple[int, float, int, float]]:
    """
    Map each frame to its corresponding slide.
    
    Returns:
        List of (frame_number, timestamp, slide_index, slide_start_time) tuples
    """
    mapped_frames = []
    
    for frame_num, timestamp in frame_timestamps:
        # Find which slide this frame belongs to
        slide_index = 0
        slide_start = slide_start_times[0]
        
        for i, slide_start_time in enumerate(slide_start_times):
            if timestamp >= slide_start_time:
                slide_index = i
                slide_start = slide_start_time
            else:
                break
        
        # Handle last slide
        if slide_index == len(slide_start_times) - 1:
            # Check if we're still within the last slide
            if timestamp > audio_duration:
                continue
        
        mapped_frames.append((frame_num, timestamp, slide_index, slide_start))
    
    return mapped_frames


def _create_frame_batches(
    mapped_frames: List[Tuple[int, float, int, float]],
    batch_size: int
) -> List[List[Tuple[int, float, int, float]]]:
    """
    Split frames into batches for parallel processing.
    """
    batches = []
    for i in range(0, len(mapped_frames), batch_size):
        batch = mapped_frames[i:i + batch_size]
        batches.append(batch)
    return batches


def _generate_frame_batch_worker(
    batch_data: Tuple[
        List[Tuple[int, float, int, float]],  # Frame tasks
        Dict[str, Any],  # Frame generator data
        Path,  # Output directory
        int,  # Width
        int,  # Height
    ]
) -> List[str]:
    """
    Worker function to generate a batch of frames in parallel.
    Optimized for performance.
    
    Args:
        batch_data: Tuple containing:
            - List of (frame_number, timestamp, slide_index, slide_start_time)
            - Frame generator serialized data (slides, layouts, fonts, etc.)
            - Output directory for saving frames
            - Width and height
    
    Returns:
        List of generated frame file paths
    """
    frame_tasks, gen_data, output_dir, width, height = batch_data
    
    # Reconstruct frame generator data (cached per worker)
    slides = gen_data['slides']
    slide_layouts = gen_data['slide_layouts']
    font_size = gen_data['font_size']
    
    # Load fonts once per worker (cached)
    try:
        regular_font = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, font_size)
        bold_font = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, font_size)
    except Exception:
        regular_font = ImageFont.load_default(size=font_size)
        bold_font = ImageFont.load_default(size=font_size)
    
    # Pre-extract colors to avoid repeated lookups
    bold_color = settings.TEXT_BOLD_COLOR
    regular_color = settings.TEXT_REGULAR_COLOR
    
    generated_files = []
    
    # Process frames in batch
    for frame_num, timestamp, slide_index, slide_start in frame_tasks:
        # Generate frame
        frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(frame)
        slide_lines = slides[slide_index]
        layout = slide_layouts[slide_index]
        
        # Render all words in the slide
        for line_idx, line in enumerate(slide_lines):
            for word_idx, word in enumerate(line):
                # Look up coordinates using (line_index, word_index) as key
                unique_key = (line_idx, word_idx)
                coords = layout.get(unique_key)
                if not coords:
                    continue
                
                # State-based logic: use Whisper timestamp directly for accuracy
                # This ensures word highlighting matches exactly when words are spoken
                if timestamp >= word['start']:
                    draw.text(coords, word['word'], font=bold_font, fill=bold_color)
                else:
                    draw.text(coords, word['word'], font=regular_font, fill=regular_color)
        
        # Save frame with optimized PNG settings
        frame_filename = output_dir / f"frame_{frame_num:06d}.png"
        # Use optimize=False for faster saving (we'll delete these anyway)
        frame.save(frame_filename, "PNG", optimize=False, compress_level=1)
        generated_files.append(str(frame_filename))
    
    return generated_files


def _detect_hardware_codec() -> Tuple[str, List[str]]:
    """
    Detect available hardware acceleration codec.
    
    Returns:
        Tuple of (codec_name, additional_ffmpeg_params)
    """
    import subprocess
    
    # Try NVIDIA NVENC
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_nvenc" in result.stdout:
            logger.info("Detected NVIDIA GPU - using h264_nvenc")
            return "h264_nvenc", ["-preset", "fast", "-rc", "vbr", "-cq", "23"]
    except Exception:
        pass
    
    # Try Intel QuickSync
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_qsv" in result.stdout:
            logger.info("Detected Intel QuickSync - using h264_qsv")
            return "h264_qsv", ["-preset", "fast", "-global_quality", "23"]
    except Exception:
        pass
    
    # Try AMD AMF
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_amf" in result.stdout:
            logger.info("Detected AMD GPU - using h264_amf")
            return "h264_amf", ["-quality", "balanced", "-rc", "vbr_peak"]
    except Exception:
        pass
    
    # Fallback to software encoding
    logger.info("No hardware acceleration detected - using libx264 (software)")
    return "libx264", ["-preset", "fast", "-crf", "23"]


# This is the "callable" version 
def render_video(
    audio_path: Path,
    timestamps_path: Path,
    output_path: Path
) -> Path:
    """
    Renders the final karaoke-style video using batch processing for faster rendering.
    
    Args:
        audio_path: Path to the PROCESSED audio file.
        timestamps_path: Path to the timestamps.json file.
        output_path: Path to save the final .mp4 video.

    Returns:
        The path to the rendered video.
    """
    logger.info("--- Starting Video Rendering Pipeline (Batch Processing) ---")
    
    temp_frames_dir = None
    
    try:
        logger.info(f"Loading audio: {audio_path}")
        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration
        
        # --- Load background/config from settings ---
        background_path = settings.DEFAULT_BACKGROUND
        fps = settings.VIDEO_FPS
        width = settings.VIDEO_WIDTH
        height = settings.VIDEO_HEIGHT
        
        logger.info(f"Loading background: {background_path}")
        bg_clip = ImageClip(background_path).with_duration(audio_duration)

        # Initialize frame generator
        frame_gen = FrameGeneratorV11(
            timestamps_path=timestamps_path,
            bg_width=width,
            bg_height=height
        )

        # ====================================================================
        # PHASE 1: Pre-calculation
        # ====================================================================
        logger.info("--- Phase 1: Pre-calculating frame timestamps ---")
        frame_timestamps = _calculate_frame_timestamps(audio_duration, fps)
        logger.info(f"Calculated {len(frame_timestamps)} frames for {audio_duration:.2f}s video at {fps}fps")
        
        mapped_frames = _map_frames_to_slides(
            frame_timestamps,
            frame_gen.slide_start_times,
            audio_duration
        )
        logger.info(f"Mapped {len(mapped_frames)} frames to {len(frame_gen.slide_start_times)} slides")
        
        # Calculate optimal batch size - larger batches reduce overhead
        cpu_count = os.cpu_count() or 4
        # Use larger batches to reduce multiprocessing overhead
        # Target: 200-500 frames per batch for better efficiency
        batch_size = max(200, len(mapped_frames) // (cpu_count * 2))
        batches = _create_frame_batches(mapped_frames, batch_size)
        logger.info(f"Created {len(batches)} batches (batch size: {batch_size}, workers: {cpu_count})")
        
        # Serialize frame generator data for multiprocessing
        # Convert WordTimestamp objects to dicts for serialization
        # Use (line_index, word_index) as unique key to prevent collisions
        serialized_slides = []
        serialized_layouts = {}
        
        for slide_idx, slide in enumerate(frame_gen.slides):
            serialized_slide = []
            serialized_layout = {}
            
            # Get the layout for this slide
            layout = frame_gen.slide_layouts.get(slide_idx, {})
            
            for line_idx, line in enumerate(slide):
                serialized_line = []
                for word_idx, word in enumerate(line):
                    # Create unique key: (line_index, word_index)
                    unique_key = (line_idx, word_idx)
                    
                    # Store word data
                    word_data = {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end
                    }
                    serialized_line.append(word_data)
                    
                    # Store layout coordinates using unique key
                    word_id = id(word)
                    if word_id in layout:
                        serialized_layout[unique_key] = layout[word_id]
                
                serialized_slide.append(serialized_line)
            
            serialized_slides.append(serialized_slide)
            serialized_layouts[slide_idx] = serialized_layout
        
        gen_data = {
            'slides': serialized_slides,
            'slide_layouts': serialized_layouts,
            'slide_start_times': frame_gen.slide_start_times,
            'font_size': frame_gen.font_size,
            'line_height': frame_gen.line_height,
            'max_text_width': frame_gen.max_text_width
        }
        
        # ====================================================================
        # PHASE 2: Parallel Frame Generation
        # ====================================================================
        logger.info("--- Phase 2: Generating frames in parallel batches ---")
        
        # Create temporary directory for frames
        temp_frames_dir = Path(tempfile.mkdtemp(prefix="video_frames_"))
        logger.info(f"Temporary frames directory: {temp_frames_dir}")
        
        # Prepare batch data for workers
        batch_data_list = [
            (batch, gen_data, temp_frames_dir, width, height)
            for batch in batches
        ]
        
        # Generate frames in parallel
        all_frame_files = []
        num_workers = min(cpu_count, len(batches))
        
        logger.info(f"Starting parallel frame generation with {num_workers} workers...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use tqdm for progress tracking
            # Use imap_unordered for better performance, then sort
            results = list(tqdm(
                pool.imap_unordered(_generate_frame_batch_worker, batch_data_list),
                total=len(batches),
                desc="Generating frames",
                unit="batch"
            ))
            
            # Flatten results
            for batch_files in results:
                all_frame_files.extend(batch_files)
        
        # Sort frame files by frame number (important for video sequence)
        all_frame_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        logger.info(f"Generated {len(all_frame_files)} frames")
        
        # ====================================================================
        # PHASE 3: Video Assembly
        # ====================================================================
        logger.info("--- Phase 3: Assembling video from frames ---")
        
        # Create video clip from frame images
        frame_clip = ImageSequenceClip(all_frame_files, fps=fps)
        
        # Composite with background
        final_video = CompositeVideoClip([bg_clip, frame_clip])
        final_video = final_video.with_audio(audio_clip)
        
        # ====================================================================
        # PHASE 4: Optimized Encoding
        # ====================================================================
        logger.info("--- Phase 4: Encoding video ---")
        
        # Detect hardware acceleration
        codec, codec_params = _detect_hardware_codec()
        
        # Prepare FFmpeg parameters
        ffmpeg_params = ["-pix_fmt", "yuv420p"] + codec_params
        
        logger.info(f"Rendering {fps}fps video to: {output_path}")
        logger.info(f"Using codec: {codec}")
        
        final_video.write_videofile(
            str(output_path),
            fps=fps,
            codec=codec,
            audio_codec="aac",
            preset="fast",
            threads=os.cpu_count(),
            ffmpeg_params=ffmpeg_params,
            logger=None  # Suppress MoviePy's progress bar (we have tqdm)
        )
        
        logger.info("--- Video Rendering Complete ---")
        return output_path

    except Exception as e:
        logger.error("Video rendering pipeline failed!", exc_info=True)
        raise
    
    finally:
        # ====================================================================
        # PHASE 5: Cleanup
        # ====================================================================
        if temp_frames_dir and temp_frames_dir.exists():
            logger.info(f"Cleaning up temporary frames directory: {temp_frames_dir}")
            try:
                shutil.rmtree(temp_frames_dir)
                logger.info("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")