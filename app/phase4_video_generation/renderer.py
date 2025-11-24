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
    V11: Robust punctuation alignment, justified text alignment, proper margins and padding.
    Supports smart detection for reels/shorts (vertical videos).
    """
    def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int, font_size: Optional[int] = None, is_reels: bool = False):
        logger.info("Initializing FrameGeneratorV11 (Robust Punctuation)...")
        
        self.bg_width = bg_width
        self.bg_height = bg_height
        self.is_reels = is_reels

        self.all_words, self.segments = self._load_data(timestamps_path)

        # --- SMART DETECTION FOR REELS/SHORTS ---
        if is_reels:
            # SHORTS MODE: Vertical video detection
            is_vertical = self.bg_height > self.bg_width
            
            if is_vertical:
                logger.info("Detected vertical video (Shorts/Reels mode)")
                # SHORTS MODE:
                # 1. Big Margins (Top/Bottom) to avoid TikTok/Reels UI overlay
                self.margin_x = int(self.bg_width * 0.12)
                self.margin_y = int(self.bg_height * 0.30)  # 30% down to be safe
                # 2. HUGE Text (Calculated based on WIDTH, not height)
                self.font_size = int(self.bg_width / 8)
                # 3. Constraint: Only 1 or 2 lines per slide max
                self.max_lines = 2
                # Use margin_x for both left and right in reels mode
                self.left_margin = self.margin_x
                self.right_margin = self.margin_x
                self.top_margin = self.margin_y
                self.bottom_margin = self.margin_y
                # 4. Center alignment for reels
                self.text_align = "center"
            else:
                # Horizontal reels (shouldn't happen, but fallback)
                logger.info("Reels mode but horizontal video - using standard settings")
                self.margin_x = int(self.bg_width * 0.10)
                self.margin_y = int(self.bg_height * 0.10)
                self.font_size = int(self.bg_height / 8)
                self.max_lines = 4
                self.left_margin = self.margin_x
                self.right_margin = self.margin_x
                self.top_margin = self.margin_y
                self.bottom_margin = self.margin_y
        else:
            # LAPTOP/LANDSCAPE MODE (Main video generation)
            # Fixed margins - these should NEVER change during rendering
            # Fixed 150px left padding - text should never expand beyond this
            self.margin = 150
            self.left_margin = 150  # Fixed 150px left padding - never changes
            self.right_margin = 200  # Right margin (keeps text area width)
            self.top_margin = 150  # Fixed top margin - never changes
            self.bottom_margin = 200  # Fixed bottom margin - never changes (equal to top for perfect centering)
            # Standard font size calculation
            if font_size is not None:
                self.font_size = font_size
            else:
                self.font_size = max(32, int(self.bg_height / 7))  # Smaller font to fit 5 lines per slide
            # Standard max lines (not explicitly set, uses default logic in _build_grouped_slides)
            self.max_lines = None  # Will use default behavior
            # Left alignment for main videos
            self.text_align = "left"
        
        self.line_height = int(self.font_size * 1.2)  # Slightly more line spacing
        self.regular_font, self.bold_font = self._load_fonts(self.font_size)
        
        # Calculate available text area
        if is_reels:
            self.max_text_width = self.bg_width - (2 * self.margin_x)
            self.max_text_height = self.bg_height - (2 * self.margin_y)
        else:
            self.max_text_width = self.bg_width - self.left_margin - self.right_margin
            self.max_text_height = self.bg_height - (2 * self.margin)
        
        self.slides, self.slide_layouts, self.slide_start_times, self.slide_end_times = self._build_grouped_slides()
        logger.info(f"FrameGeneratorV11 initialized: {len(self.segments)} segments grouped into {len(self.slides)} slides.")

    def _load_fonts(self, size: int) -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
        # --- 2. Load fonts from config.py paths ---
        try:
            regular_path = settings.DEFAULT_FONT_REGULAR
            bold_path = settings.DEFAULT_FONT_BOLD
            
            # Verify font files exist before trying to load
            # Resolve path to handle relative paths and normalize
            regular_path_obj = Path(regular_path).resolve()
            bold_path_obj = Path(bold_path).resolve()
            
            if not regular_path_obj.exists():
                # Try alternative: check if it's a relative path from FONTS_PATH
                alt_path = settings.FONTS_PATH / Path(regular_path).name
                if alt_path.exists():
                    regular_path_obj = alt_path.resolve()
                    logger.info(f"Found font at alternative path: {regular_path_obj}")
                else:
                    raise FileNotFoundError(
                        f"Font file not found: {regular_path} (resolved: {regular_path_obj})\n"
                        f"Also checked: {alt_path}\n"
                        f"FONTS_PATH: {settings.FONTS_PATH}"
                    )
            
            if not bold_path_obj.exists():
                # Try alternative: check if it's a relative path from FONTS_PATH
                alt_path = settings.FONTS_PATH / Path(bold_path).name
                if alt_path.exists():
                    bold_path_obj = alt_path.resolve()
                    logger.info(f"Found font at alternative path: {bold_path_obj}")
                else:
                    raise FileNotFoundError(
                        f"Font file not found: {bold_path} (resolved: {bold_path_obj})\n"
                        f"Also checked: {alt_path}\n"
                        f"FONTS_PATH: {settings.FONTS_PATH}"
                    )
            
            logger.info(f"Loading fonts - Regular: {regular_path_obj}, Bold: {bold_path_obj}")
            regular = ImageFont.truetype(str(regular_path_obj), size)
            bold = ImageFont.truetype(str(bold_path_obj), size)
            logger.info(f"Successfully loaded custom fonts (size: {size})")
            return regular, bold
        except Exception as e:
            logger.error(f"FATAL: Could not load custom font! {e}", exc_info=True)
            logger.error(f"Regular font path: {settings.DEFAULT_FONT_REGULAR}")
            logger.error(f"Bold font path: {settings.DEFAULT_FONT_BOLD}")
            logger.error(f"FONTS_PATH: {settings.FONTS_PATH}")
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
        # Load all words from timestamps - preserve ALL words
        # Only filter out words that are completely empty (whitespace only)
        words = []
        skipped_count = 0
        for w in data["words"]:
            if 'word' in w and w.get('word', '').strip():  # Word exists and is not just whitespace
                words.append(WordTimestamp(**w))
            elif 'word' in w:
                # Word field exists but is empty/whitespace - still include it (might be punctuation)
                words.append(WordTimestamp(**w))
            else:
                skipped_count += 1
                logger.warning(f"Skipping word entry missing 'word' field: {w}")
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} word entries missing 'word' field")
        
        logger.info(f"Loaded {len(words)} words from timestamps file")
        
        segments = data["segments"]
        
        # Map punctuation from segment text to words
        # Whisper word timestamps don't include punctuation, but segment text does
        words_before_mapping = len(words)
        words = self._map_punctuation_to_words(words, segments)
        words_after_mapping = len(words)
        
        if words_after_mapping != words_before_mapping:
            logger.warning(f"Word count changed during punctuation mapping: {words_before_mapping} -> {words_after_mapping}")
        
        # Clean text: remove all punctuation except full stops (periods)
        words = self._clean_text(words)
        
        logger.info(f"Final word count: {len(words)} words and {len(segments)} segments.")
        return words, segments
    
    def _map_punctuation_to_words(self, words: List[WordTimestamp], segments: List[Dict]) -> List[WordTimestamp]:
        """
        Map punctuation from segment text to word timestamps.
        Whisper word timestamps don't include punctuation, but segment text does.
        
        CRITICAL: This function MUST preserve ALL words from timestamps, even if they don't
        match segment text perfectly. Words are the source of truth for what's in the audio.
        """
        import re
        
        # CRITICAL: Start with ALL words in order - these are the source of truth
        # We'll attach punctuation from segments when possible, but never skip words
        updated_words = []
        word_idx = 0
        
        # Create a mapping of segment text to help attach punctuation
        # But we'll process words in order and match them to segments by timestamp
        for segment in segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", float('inf'))
            segment_text = segment.get("text", "").strip()
            
            if not segment_text:
                continue
            
            # Get words that fall within this segment's time range
            segment_words = []
            while word_idx < len(words):
                word = words[word_idx]
                # Check if word falls within segment time range
                # Use a small tolerance to handle edge cases
                if word.start >= segment_start - 0.1 and word.start < segment_end + 0.1:
                    segment_words.append(word)
                    word_idx += 1
                elif word.start >= segment_end:
                    # Word is after this segment, stop collecting
                    break
                else:
                    # Word is before this segment, skip it (shouldn't happen if processing in order)
                    word_idx += 1
            
            if not segment_words:
                continue
            
            # Try to attach punctuation from segment text to words
            # Split segment text into tokens (words with punctuation)
            pattern = r'\b\w+\b[^\w\s]*|[^\w\s]+'
            tokens = re.findall(pattern, segment_text)
            
            # Match tokens to words - but preserve ALL words even if they don't match
            token_idx = 0
            for word_obj in segment_words:
                word_text_clean = word_obj.word.lower().strip()
                
                # Try to find matching token
                matched = False
                for t_idx in range(token_idx, len(tokens)):
                    token = tokens[t_idx]
                    token_word_part = re.sub(r'[^\w]', '', token).lower().strip()
                    
                    if token_word_part == word_text_clean:
                        # Found match - attach punctuation from token
                        punctuation = re.sub(r'\w', '', token)
                        if punctuation:
                            word_obj.word = word_obj.word + punctuation
                        matched = True
                        token_idx = t_idx + 1
                        break
                    elif not token_word_part and t_idx == token_idx:
                        # Pure punctuation token at start - attach to previous word if exists
                        if updated_words:
                            punctuation = token
                            updated_words[-1].word = updated_words[-1].word + punctuation
                        token_idx = t_idx + 1
                        # Don't break, continue to match this word
                
                # CRITICAL: Always add the word, even if no token match found
                # The word timestamp is the source of truth - it exists in the audio
                updated_words.append(word_obj)
                
                # If we didn't match, log it for debugging but still add the word
                if not matched:
                    logger.debug(f"Word '{word_obj.word}' at {word_obj.start}s didn't match any token in segment text")
        
        # CRITICAL: Add any remaining words that weren't in any segment
        # This ensures we never lose words, even if they don't fall within segment boundaries
        while word_idx < len(words):
            logger.debug(f"Adding word '{words[word_idx].word}' at {words[word_idx].start}s that wasn't in any segment")
            updated_words.append(words[word_idx])
            word_idx += 1
        
        # CRITICAL: Verify we didn't lose any words
        original_count = len(words)
        final_count = len(updated_words)
        if final_count < original_count:
            logger.error(f"CRITICAL: Word count decreased! Started with {original_count} words, ended with {final_count} words. {original_count - final_count} words were lost!")
            # This should never happen - if it does, we have a serious bug
            # Try to recover by adding missing words
            words_set = {id(w) for w in updated_words}
            for w in words:
                if id(w) not in words_set:
                    logger.error(f"Recovering lost word: '{w.word}' at {w.start}s")
                    updated_words.append(w)
            final_count = len(updated_words)
            logger.info(f"Recovered to {final_count} words")
        elif final_count > original_count:
            logger.warning(f"Word count increased: Started with {original_count} words, ended with {final_count} words. This may indicate duplicate words.")
        else:
            logger.info(f"✓ All {original_count} words preserved during punctuation mapping.")
        
        # Verify punctuation was preserved - log sample words with punctuation
        sample_words_with_punct = [w.word for w in updated_words[:20] if any(c in w.word for c in [',', '.', '!', '?', ';', ':', '"', "'"])]
        if sample_words_with_punct:
            logger.info(f"Mapped punctuation from segments to {len(updated_words)} words. Sample words with punctuation: {sample_words_with_punct[:5]}")
        else:
            logger.warning(f"Mapped {len(updated_words)} words, but no punctuation detected in first 20 words. This may indicate an issue.")
        
        return updated_words

    def _clean_text(self, words: List[WordTimestamp]) -> List[WordTimestamp]:
        """
        Clean text by removing all punctuation except full stops (periods).
        Normalize spacing to remove extra spaces.
        After every full stop, a new line will start (handled by slide building logic).
        
        Args:
            words: List of WordTimestamp objects to clean
            
        Returns:
            List of WordTimestamp objects with cleaned text (only periods preserved, spacing normalized)
        """
        import re
        
        cleaned_words = []
        for word in words:
            # Remove all punctuation except periods (full stops)
            # Keep periods and alphanumeric characters, remove everything else
            cleaned_word = re.sub(r'[^\w\s.]', '', word.word)
            # Normalize spacing: remove leading/trailing spaces and collapse multiple spaces to single space
            cleaned_word = re.sub(r'\s+', ' ', cleaned_word).strip()
            # Remove any whitespace-only words but keep the word if it has content
            if cleaned_word or word.word.strip():
                # Create new WordTimestamp with cleaned word
                cleaned_word_obj = WordTimestamp(
                    word=cleaned_word,
                    start=word.start,
                    end=word.end,
                    confidence=word.confidence if hasattr(word, 'confidence') else None,
                    probability=word.probability if hasattr(word, 'probability') else None
                )
                cleaned_words.append(cleaned_word_obj)
            else:
                # If word becomes empty after cleaning, still include it to preserve timing
                cleaned_words.append(word)
        
        logger.info(f"Cleaned text: removed all punctuation except periods and normalized spacing from {len(words)} words")
        return cleaned_words

    def _get_words_for_segment(self, segment_index: int, processed_words: set = None) -> List[WordTimestamp]:
        """
        Get words for a segment. Uses lenient matching to ensure no words are missed.
        
        Args:
            segment_index: Index of the segment
            processed_words: Optional set of word IDs that have already been processed (to avoid duplicates)
        """
        segment = self.segments[segment_index]
        segment_start = segment.get("start", 0)
        segment_end = segment.get("end", float('inf'))
        
        # If there's a next segment, use its start as the end boundary
        if segment_index + 1 < len(self.segments):
            next_segment_start = self.segments[segment_index + 1].get("start", float('inf'))
            # Use the earlier of segment end or next segment start
            segment_end = min(segment_end, next_segment_start)
        
        # Use lenient matching with small tolerance to handle edge cases
        # This ensures words aren't missed due to minor timestamp discrepancies
        tolerance = 0.1  # 100ms tolerance
        words = [w for w in self.all_words if w.start >= segment_start - tolerance and w.start < segment_end + tolerance]
        
        # Filter out words that have already been processed (if provided)
        if processed_words is not None:
            words = [w for w in words if id(w) not in processed_words]
        
        return words

    def _build_grouped_slides(self) -> Tuple[List[List[List[WordTimestamp]]], Dict[int, Dict[int, Tuple[int, int]]], List[float], List[float]]:
        logger.info("Building grouped slides (Max Words strategy)...")
        slides, layouts, slide_start_times = [], {}, []
        dummy_img = Image.new("RGB", (self.bg_width, self.bg_height))
        draw = ImageDraw.Draw(dummy_img)
        space_bbox = draw.textbbox((0, 0), " ", font=self.bold_font); space_width = space_bbox[2] - space_bbox[0]

        MAX_WORDS_PER_SLIDE = 12  # Maximum 12 words per slide
        MAX_LINES_PER_SLIDE = 4  # Fixed: Maximum 3 lines per slide (always 3, no dynamic)
        
        logger.info(f"Slide constraints: Maximum {MAX_LINES_PER_SLIDE} lines and {MAX_WORDS_PER_SLIDE} words per slide")

        current_slide_words_ts, current_slide_start_time, current_slide_segments = [], -1, []

        def build_slide_layout(words_ts, segments_list, start_time):
            """Helper function to avoid repeating the build logic.
            Returns: (remaining_words, None) if slide is full, or (None, None) if all words processed.
            """
            if not words_ts:
                return None, None

            slide_index = len(slides)
            
            # CRITICAL: Ensure start_time is set to the first word's start time
            # This ensures slides appear exactly when their first word starts
            if words_ts and start_time < 0:
                start_time = words_ts[0].start
            elif words_ts and start_time != words_ts[0].start:
                # Use the first word's start time as the slide start time
                start_time = words_ts[0].start
            # IMPORTANT: Preserve punctuation from original word timestamps
            # Don't overwrite words with cleaned versions - keep original punctuation
            # The word.word from timestamps already contains proper punctuation

            current_slide_lines, current_line = [], []
            # CRITICAL: Always use MAX_LINES_PER_SLIDE (fixed at 3) - do not calculate dynamically
            max_lines_that_fit = MAX_LINES_PER_SLIDE  # Fixed: Always 3 lines maximum
            
            processed_words = []
            remaining_words = []
            words_to_process = list(words_ts)  # Make a copy to track which words we process
            
            # CRITICAL: Check for sentence boundaries - words after full stop MUST start new slide
            # Find the first word that comes after a full stop (if any)
            def _word_ends_sentence_check(word_text: str) -> bool:
                """Check if word ends with a full stop (sentence ending)."""
                if not word_text:
                    return False
                cleaned = word_text.strip()
                if not cleaned:
                    return False
                # Remove trailing quotes, brackets, etc. but keep periods
                while cleaned and cleaned[-1] in {"'", '"', '"', '"', "'", ")", "]", "}", " "}:
                    cleaned = cleaned[:-1]
                if not cleaned or cleaned[-1] != ".":
                    return False
                # Check for abbreviations
                word_lower = cleaned.lower()
                abbreviations = {
                    'mr.', 'mrs.', 'ms.', 'miss.', 'dr.', 'prof.', 'rev.', 'fr.', 'sr.', 'jr.',
                    'hon.', 'esq.', 'sir.', 'dame.', 'lord.', 'lady.', 'ph.d.', 'm.d.', 'd.d.s.', 'd.v.m.',
                    'rn.', 'cpa.', 'llc.', 'llp.', 'a.m.', 'p.m.', 'am.', 'pm.', 'b.c.', 'a.d.', 'bc.', 'ad.',
                    'jan.', 'feb.', 'mar.', 'apr.', 'may.', 'jun.', 'jul.', 'aug.', 'sep.', 'sept.', 'oct.', 'nov.', 'dec.',
                    'vs.', 'v.', 'etc.', 'e.g.', 'i.e.', 'viz.', 'cf.', 'ex.', 'et al.', 'inc.', 'ltd.', 'corp.', 'co.',
                    'st.', 'ave.', 'blvd.', 'rd.', 'ct.', 'ln.', 'pkwy.', 'sq.', 'n.', 's.', 'e.', 'w.', 'nw.', 'ne.', 'sw.', 'se.',
                    'no.', 'vol.', 'pp.', 'p.', 'ch.', 'sec.', 'fig.', 'art.', 'ft.', 'in.', 'lb.', 'oz.', 'kg.', 'g.', 'ml.', 'l.',
                    'ed.', 'est.', 'approx.', 'min.', 'max.', 'avg.', 'std.', 'gov.', 'pres.', 'sen.', 'rep.', 'gen.', 'col.', 'capt.',
                    'dept.', 'univ.', 'assoc.', 'misc.', 'op.', 'cit.'
                }
                if word_lower in abbreviations or (len(cleaned) == 2 and cleaned[0].isalpha() and cleaned[1] == '.'):
                    return False
                if len(cleaned) >= 3 and '.' in cleaned[:-1]:
                    return False
                return True
            
            # Check if any word in the list comes after a full stop
            # If so, split at that point - words after full stop must start new slide
            for check_idx in range(len(words_to_process) - 1):
                current_word = words_to_process[check_idx]
                next_word = words_to_process[check_idx + 1]
                
                # If current word ends a sentence, next word must start new slide
                if _word_ends_sentence_check(current_word.word):
                    # Split: process words up to and including current word, rest go to next slide
                    words_to_process = words_to_process[:check_idx + 1]
                    remaining_words = words_ts[check_idx + 1:]
                    logger.info(f"Slide {slide_index}: Detected sentence boundary - word '{current_word.word}' ends sentence. Splitting: {len(words_to_process)} words in this slide, {len(remaining_words)} words start next slide")
                    break
            
            # CRITICAL: Check word count constraint
            # If we already have MAX_WORDS_PER_SLIDE words, we need to split
            if len(words_to_process) > MAX_WORDS_PER_SLIDE:
                # Process only first MAX_WORDS_PER_SLIDE words, rest go to next slide
                words_to_process = words_to_process[:MAX_WORDS_PER_SLIDE]
                remaining_words = words_ts[MAX_WORDS_PER_SLIDE:] if not remaining_words else remaining_words
                logger.debug(f"Slide {slide_index}: Word count limit reached ({MAX_WORDS_PER_SLIDE} words), splitting remaining {len(remaining_words)} words to next slide")
            
            for word_idx, word in enumerate(words_to_process):
                # CRITICAL: Check word count constraint FIRST - don't exceed MAX_WORDS_PER_SLIDE
                if len(processed_words) >= MAX_WORDS_PER_SLIDE:
                    # We've reached the word limit - remaining words go to next slide
                    remaining_words = words_to_process[word_idx:]
                    logger.debug(f"Slide {slide_index}: Word count limit reached ({MAX_WORDS_PER_SLIDE} words), {len(remaining_words)} words remaining for next slide")
                    break
                
                # CRITICAL: Check line count constraint - don't exceed MAX_LINES_PER_SLIDE (3)
                if len(current_slide_lines) >= MAX_LINES_PER_SLIDE:
                    # We've reached the line limit - remaining words go to next slide
                    remaining_words = words_to_process[word_idx:]
                    logger.debug(f"Slide {slide_index}: Line count limit reached ({MAX_LINES_PER_SLIDE} lines), {len(remaining_words)} words remaining for next slide")
                    break
                
                word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                word_width = word_bbox[2] - word_bbox[0]
                
                # Calculate current line width including the new word
                if current_line:
                    # Build test line with new word to get accurate width
                    test_line = current_line + [word]
                    line_text = " ".join(w.word for w in test_line)
                    line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
                    test_line_width = line_bbox[2] - line_bbox[0]
                    
                    # Use 99% of max width as threshold (aggressive to fill space completely)
                    threshold = self.max_text_width * 0.99
                    if test_line_width > threshold:
                        # Current line is full, start new line
                        # CRITICAL: Check if adding this line would exceed MAX_LINES_PER_SLIDE
                        if len(current_slide_lines) >= MAX_LINES_PER_SLIDE:
                            # Cannot add another line - slide is full
                            remaining_words = words_to_process[word_idx:]  # All words from current onwards
                            break
                        current_slide_lines.append(current_line)
                        current_line = [word]
                        processed_words.append(word)
                    else:
                        # Word fits on current line - add it
                        current_line.append(word)
                        processed_words.append(word)
                else:
                    # First word in line - check if we can add another line
                    if len(current_slide_lines) >= MAX_LINES_PER_SLIDE:
                        # Slide is full - remaining words go to next slide
                        remaining_words = words_to_process[word_idx:]
                        break
                    # Check if single word fits
                    if word_width > self.max_text_width:
                        # Word is too long, truncate or handle (shouldn't happen often)
                        logger.warning(f"Word '{word.word}' is wider than max width ({word_width} > {self.max_text_width})")
                    current_line.append(word)
                    processed_words.append(word)
            
            # Add current_line if it exists and we have room (strictly enforce MAX_LINES_PER_SLIDE)
            if current_line and len(current_slide_lines) < MAX_LINES_PER_SLIDE:
                current_slide_lines.append(current_line)
            
            # CRITICAL: Redistribute words to fill space better
            # If the last line has very few words and previous lines have space, try to move words up
            # This prevents the issue of one word on a line with huge space
            if len(current_slide_lines) >= 2:
                last_line = current_slide_lines[-1]
                second_last_line = current_slide_lines[-2]
                
                # If last line has only 1-2 words, try to move them to previous line
                if len(last_line) <= 2 and len(second_last_line) > 0:
                    # Test if we can fit last line words into second last line
                    test_line = second_last_line + last_line
                    line_text = " ".join(w.word for w in test_line)
                    line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
                    test_line_width = line_bbox[2] - line_bbox[0]
                    
                    # If it fits (with 99% threshold), merge them
                    if test_line_width <= self.max_text_width * 0.99:
                        current_slide_lines[-2] = test_line
                        current_slide_lines.pop()  # Remove the now-empty last line
                        logger.debug(f"Slide {slide_index}: Redistributed {len(last_line)} words from last line to previous line to fill space")
            
            # Additional optimization: Try to balance lines better by moving words from short lines to longer ones
            # This helps fill space more evenly across all lines
            if len(current_slide_lines) >= 2:
                # Calculate average words per line
                total_words = sum(len(line) for line in current_slide_lines)
                avg_words_per_line = total_words / len(current_slide_lines) if current_slide_lines else 0
                
                # If we have lines with very different lengths, try to balance them
                for line_idx in range(len(current_slide_lines) - 1):
                    current_line = current_slide_lines[line_idx]
                    next_line = current_slide_lines[line_idx + 1]
                    
                    # If current line is much shorter than average and next line exists
                    if len(current_line) < avg_words_per_line * 0.7 and len(next_line) > 0:
                        # Try moving first word(s) from next line to current line
                        words_to_move = min(2, len(next_line))  # Try moving up to 2 words
                        test_line = current_line + next_line[:words_to_move]
                        line_text = " ".join(w.word for w in test_line)
                        line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
                        test_line_width = line_bbox[2] - line_bbox[0]
                        
                        # If it fits, move the words (99% threshold for aggressive filling)
                        if test_line_width <= self.max_text_width * 0.99:
                            current_slide_lines[line_idx] = test_line
                            current_slide_lines[line_idx + 1] = next_line[words_to_move:]
                            logger.debug(f"Slide {slide_index}: Moved {words_to_move} word(s) from line {line_idx+1} to line {line_idx} to balance spacing")

            # CRITICAL: Remove trailing commas from the last word of the last line in the slide
            if current_slide_lines:
                last_line = current_slide_lines[-1]
                if last_line:
                    last_word = last_line[-1]
                    # Remove trailing comma from the last word
                    if last_word.word.endswith(','):
                        # Create a new WordTimestamp with comma removed
                        last_word = WordTimestamp(
                            word=last_word.word.rstrip(','),
                            start=last_word.start,
                            end=last_word.end,
                            confidence=last_word.confidence if hasattr(last_word, 'confidence') else None,
                            probability=last_word.probability if hasattr(last_word, 'probability') else None
                        )
                        last_line[-1] = last_word
                        logger.debug(f"Removed trailing comma from last word '{last_word.word}' in slide {slide_index}")

            # Check if slide exceeds constraints
            total_words_in_slide = len(processed_words)
            total_text_height = len(current_slide_lines) * self.line_height
            
            if MAX_LINES_PER_SLIDE and len(current_slide_lines) > MAX_LINES_PER_SLIDE:
                logger.warning(f"Slide {slide_index} exceeds line limit: {len(current_slide_lines)} lines (max: {MAX_LINES_PER_SLIDE})")
            if total_words_in_slide > MAX_WORDS_PER_SLIDE:
                logger.warning(f"Slide {slide_index} exceeds word limit: {total_words_in_slide} words (max: {MAX_WORDS_PER_SLIDE})")
            
            # Log slide summary
            logger.debug(f"Slide {slide_index}: {len(current_slide_lines)} lines, {total_words_in_slide} words (max: {MAX_LINES_PER_SLIDE} lines, {MAX_WORDS_PER_SLIDE} words)")
            
            if total_text_height > self.bg_height * 0.95:
                 logger.warning(f"Slide {slide_index} (starting {start_time}s) may be too tall! Has {len(current_slide_lines)} lines.")

            # CRITICAL: Remove the last word from the slide if there are remaining words
            # This prevents duplication where the last word of a slide is the same as the first word of the next slide
            # BUT: Only do this if it's NOT the first slide (slide_index > 0) to keep the first slide unchanged
            if remaining_words and slide_index > 0 and current_slide_lines:
                last_line = current_slide_lines[-1]
                if last_line and len(last_line) > 0:
                    # Remove the last word from the last line to prevent duplication
                    removed_word = last_line.pop()
                    logger.debug(f"Slide {slide_index}: Removed last word '{removed_word.word}' to prevent duplication with next slide (next slide starts with '{remaining_words[0].word if remaining_words else 'N/A'}')")
                    
                    # If the last line is now empty, remove it
                    if not last_line:
                        current_slide_lines.pop()
                        logger.debug(f"Slide {slide_index}: Removed empty last line after removing duplicate word")
            
            slides.append(current_slide_lines); layouts[slide_index] = {}

            # ====================================================================
            # APPROACH 1: Dynamic Content-Aware Padding
            # Calculate actual text block dimensions, then center perfectly
            # ====================================================================
            
            # Step 1: Calculate actual content dimensions (bounding box)
            max_line_width = 0
            for line_of_words in current_slide_lines:
                if not line_of_words:
                    continue
                # Calculate actual line width
                line_text = " ".join(w.word for w in line_of_words)
                line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
                line_width = line_bbox[2] - line_bbox[0]
                max_line_width = max(max_line_width, line_width)
            
            # Actual content dimensions
            content_width = max_line_width
            content_height = len(current_slide_lines) * self.line_height
            
            # Step 2: Use the fixed margins defined at class level (equal padding on all sides)
            # These ensure consistent spacing and centering
            min_margin_top = self.top_margin
            min_margin_bottom = self.bottom_margin
            min_margin_left = self.left_margin
            min_margin_right = self.right_margin
            
            # Step 3: Calculate available space (screen - margins)
            available_width = self.bg_width - min_margin_left - min_margin_right
            available_height = self.bg_height - min_margin_top - min_margin_bottom
            
            # Step 4: Fixed 150px left padding - text area starts at exactly 150px from left
            # NO dynamic horizontal padding - left side is fixed at 150px
            text_area_start_x = 150  # Fixed 150px left padding - never changes
            text_area_end_x = self.bg_width - min_margin_right  # Right boundary uses right margin
            text_area_width = text_area_end_x - text_area_start_x
            
            # Step 5: Calculate vertical padding (only vertical centering, horizontal is fixed)
            padding_vertical = max(0, (available_height - content_height) / 2)
            
            # Vertical positioning - centered with equal padding
            start_y = min_margin_top + padding_vertical
            max_y = self.bg_height - min_margin_bottom - padding_vertical
            
            # Ensure text doesn't exceed bounds (safety check)
            if start_y + content_height > max_y:
                # Fallback: use minimum margins if content is too large
                start_y = min_margin_top
                logger.warning(
                    f"Slide {slide_index}: Content height ({content_height}px) exceeds available space. "
                    f"Available: {available_height}px. Using minimum top margin."
                )
            
            # Log padding for debugging
            logger.debug(
                f"Slide {slide_index}: Content={content_width}x{content_height}px, "
                f"Fixed left padding=150px, Padding V={padding_vertical:.1f}px, "
                f"Text area X={text_area_start_x:.1f}-{text_area_end_x:.1f}px, "
                f"Start Y={start_y:.1f}px"
            )
            
            current_y = start_y
            max_x = text_area_end_x
            
            for line_idx, line_of_words in enumerate(current_slide_lines):
                # Calculate text positions
                if len(line_of_words) == 0:
                    current_y += self.line_height
                    continue
                
                # Calculate total width of all words (with spaces)
                word_widths = []
                total_words_width = 0
                for word in line_of_words:
                    word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                    word_width = word_bbox[2] - word_bbox[0]
                    word_widths.append(word_width)
                    total_words_width += word_width
                
                # Add space widths between words
                if len(line_of_words) > 1:
                    total_words_width += (len(line_of_words) - 1) * space_width
                
                # Text alignment: center for reels, left for main videos
                if hasattr(self, 'text_align') and self.text_align == "center":
                    # Center the line within the text area
                    current_x = text_area_start_x + (text_area_width - total_words_width) // 2
                else:
                    # Left-align text (start at text_area_start_x)
                    current_x = text_area_start_x
                
                # CRITICAL: Check if line exceeds bottom margin BEFORE processing words
                # But still process all words to ensure they get layout coordinates
                line_would_exceed = current_y + self.line_height > max_y
                if line_would_exceed:
                    logger.warning(
                        f"Slide {slide_index}: Line {line_idx} would exceed bottom margin. "
                        f"Current y={current_y}, line_height={self.line_height}, max_y={max_y}. "
                        f"Still adding words to layout to ensure all words are rendered."
                    )
                
                for i, word in enumerate(line_of_words):
                    word_width = word_widths[i]
                    # Ensure word fits - adjust position if needed
                    if current_x + word_width > max_x:
                        # Try to fit by positioning at max_x - word_width
                        adjusted_x = max(text_area_start_x, max_x - word_width)
                        if adjusted_x + word_width <= max_x:
                            current_x = adjusted_x
                        else:
                            # Word is too wide, but still left-align it
                            current_x = text_area_start_x
                    
                    # CRITICAL: Convert to int for pixel coordinates
                    x_coord = int(round(current_x))
                    y_coord = int(round(current_y))
                    
                    # CRITICAL: Always add layout coordinates for every word
                    # Use (line_idx, word_idx) as key to match rendering lookup
                    layouts[slide_index][(line_idx, i)] = (x_coord, y_coord)
                    
                    # Debug first few words to verify coordinates
                    if slide_index < 2 and line_idx < 2 and i < 3:
                        logger.debug(
                            f"Slide {slide_index}, Line {line_idx}, Word {i} ('{word.word}'): "
                            f"x={x_coord}, y={y_coord}, width={word_width:.1f}"
                        )
                    
                    current_x += word_width + space_width
                
                # Move to next line (even if it exceeds margin - we want all words to have coordinates)
                current_y += self.line_height
            
            # Verify all words have been assigned positions
            words_in_layout = set(layouts[slide_index].keys())
            # Build set of (line_idx, word_idx) keys for all words in slide
            words_in_slide = set()
            for line_idx, line in enumerate(current_slide_lines):
                for word_idx, word in enumerate(line):
                    words_in_slide.add((line_idx, word_idx))
            
            missing_words = words_in_slide - words_in_layout
            if missing_words:
                logger.error(f"Slide {slide_index}: {len(missing_words)} words missing from layout! This should not happen.")
                # Try to add missing words
                for line_idx, word_idx in missing_words:
                    if line_idx < len(current_slide_lines) and word_idx < len(current_slide_lines[line_idx]):
                        word = current_slide_lines[line_idx][word_idx]
                        word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                        word_width = word_bbox[2] - word_bbox[0]
                        # Position at start of line
                        current_x = text_area_start_x
                        current_y = start_y + line_idx * self.line_height
                        layouts[slide_index][(line_idx, word_idx)] = (current_x, current_y)
                        logger.warning(f"Recovered missing word '{word.word}' at ({line_idx}, {word_idx})")
            
            # Log layout summary for debugging
            total_words_in_layout = len(words_in_layout)
            total_words_in_slide = len(words_in_slide)
            logger.debug(f"Slide {slide_index}: {total_words_in_layout} words in layout, {total_words_in_slide} words in slide")
            
            slide_start_times.append(start_time)
            
            # Return remaining words if any, otherwise None
            return remaining_words if remaining_words else None, None

        def _word_ends_sentence(word_text: str) -> bool:
            """
            Check if word ends with a full stop (period) that actually ends a sentence.
            Excludes common abbreviations like Mr., Mrs., Dr., Ms., etc.
            This is used to ensure sentences after full stops ALWAYS start on a new slide.
            Only full stops (periods) are used - no exclamation or question marks.
            """
            if not word_text:
                return False
            # Strip whitespace but preserve punctuation
            cleaned = word_text.strip()
            if not cleaned:
                return False
            # Remove trailing quotes, brackets, etc. but keep periods
            while cleaned and cleaned[-1] in {"'", '"', '"', '"', "'", ")", "]", "}", " "}:
                cleaned = cleaned[:-1]
            if not cleaned:
                return False
            # Check if last character is a full stop (period)
            if cleaned[-1] != ".":
                return False
            
            # Exclude common abbreviations that end with periods but don't end sentences
            # Convert to lowercase for case-insensitive matching
            word_lower = cleaned.lower()
            
            # Comprehensive list of common abbreviations that should NOT trigger new lines
            abbreviations = {
                # Titles and honorifics
                'mr.', 'mrs.', 'ms.', 'miss.', 'dr.', 'prof.', 'rev.', 'fr.', 'sr.', 'jr.',
                'hon.', 'esq.', 'sir.', 'dame.', 'lord.', 'lady.',
                # Academic and professional
                'ph.d.', 'm.d.', 'd.d.s.', 'd.v.m.', 'rn.', 'cpa.', 'llc.', 'llp.',
                # Time and dates
                'a.m.', 'p.m.', 'am.', 'pm.', 'b.c.', 'a.d.', 'bc.', 'ad.',
                'jan.', 'feb.', 'mar.', 'apr.', 'may.', 'jun.', 'jul.', 'aug.',
                'sep.', 'sept.', 'oct.', 'nov.', 'dec.',
                # Common words
                'vs.', 'v.', 'etc.', 'e.g.', 'i.e.', 'viz.', 'cf.', 'ex.', 'et al.',
                # Business and legal
                'inc.', 'ltd.', 'corp.', 'co.', 'llc.', 'llp.', 'plc.',
                # Addresses and locations
                'st.', 'ave.', 'blvd.', 'rd.', 'dr.', 'ct.', 'ln.', 'pkwy.', 'sq.',
                'n.', 's.', 'e.', 'w.', 'nw.', 'ne.', 'sw.', 'se.',
                # Numbers and measurements
                'no.', 'vol.', 'pp.', 'p.', 'ch.', 'sec.', 'fig.', 'art.',
                'ft.', 'in.', 'lb.', 'oz.', 'kg.', 'g.', 'ml.', 'l.',
                # Other common abbreviations
                'ed.', 'est.', 'approx.', 'min.', 'max.', 'avg.', 'std.',
                'gov.', 'pres.', 'sen.', 'rep.', 'gen.', 'col.', 'capt.',
                'dept.', 'univ.', 'assoc.', 'misc.', 'op.', 'cit.'
            }
            
            # Check if this word is a common abbreviation
            if word_lower in abbreviations:
                return False
            
            # Check if it's a single letter followed by period (like "A." or "I.")
            # These are often abbreviations or initials, not sentence endings
            if len(cleaned) == 2 and cleaned[0].isalpha() and cleaned[1] == '.':
                return False
            
            # Check for common patterns like "U.S.", "U.K.", "Ph.D.", etc.
            if len(cleaned) >= 3 and '.' in cleaned[:-1]:
                # Has period in middle (like "U.S." or "Ph.D.")
                return False
            
            # It's a period that ends a sentence (not an abbreviation)
            return True

        last_word_text = None
        sentence_just_ended = False  # Track if we just finished a sentence
        
        # Track which words have been added to slides to ensure we don't miss any
        # AND to prevent duplicates - each word should only be added once
        words_added_to_slides = set()
        words_processed = set()  # Track words that have already been processed to prevent duplicates

        # CRITICAL: For reels, process ALL words directly word-by-word without segment grouping
        # This ensures every word from timestamps is included
        if self.is_reels:
            # For reels: Process all words directly in order (word-by-word timestamps)
            logger.info("Reels mode: Processing words directly word-by-word (no segment grouping)")
            all_words_ordered = sorted(self.all_words, key=lambda w: w.start)
            
            for word_ts in all_words_ordered:
                # Skip if already processed
                word_id = id(word_ts)
                if word_id in words_processed:
                    continue
                words_processed.add(word_id)
                words_added_to_slides.add(word_id)
                
                # Process word directly (word-by-word)
                word_text = word_ts.word
                
                # If previous word ended a sentence, start new slide
                if sentence_just_ended:
                    if current_slide_words_ts:
                        build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                    current_slide_words_ts = []
                    current_slide_segments = []
                    current_slide_start_time = word_ts.start
                
                sentence_just_ended = False
                
                if not current_slide_words_ts:
                    current_slide_start_time = word_ts.start
                
                # Add word to current slide
                current_slide_words_ts.append(word_ts)
                
                # Check if word ends sentence
                if _word_ends_sentence(word_text):
                    build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                    current_slide_words_ts = []
                    current_slide_segments = []
                    current_slide_start_time = -1
                    sentence_just_ended = True
                    continue
                
                # Check if slide is full (test layout without actually building)
                # CRITICAL: For reels, we need to test if words fit without building the slide
                if current_slide_words_ts and len(current_slide_words_ts) > 1:
                    # Test if current words fit in max_lines
                    test_lines = []
                    test_line = []
                    for w in current_slide_words_ts:
                        word_bbox = draw.textbbox((0, 0), w.word, font=self.bold_font)
                        word_width = word_bbox[2] - word_bbox[0]
                        if test_line:
                            test_line_text = " ".join(w2.word for w2 in test_line + [w])
                            test_line_bbox = draw.textbbox((0, 0), test_line_text, font=self.bold_font)
                            test_line_width = test_line_bbox[2] - test_line_bbox[0]
                            if test_line_width > self.max_text_width * 0.90:
                                test_lines.append(test_line)
                                test_line = [w]
                            else:
                                test_line.append(w)
                        else:
                            test_line.append(w)
                    if test_line:
                        test_lines.append(test_line)
                    
                    # Check if we exceed max lines
                    if len(test_lines) > MAX_LINES_PER_SLIDE:
                        # Slide is full - remove last word and finalize current slide
                        popped_word = current_slide_words_ts.pop()
                        words_added_to_slides.discard(id(popped_word))
                        build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                        # Start new slide with current word
                        current_slide_words_ts = [word_ts]
                        words_added_to_slides.add(id(word_ts))
                        current_slide_start_time = word_ts.start
        else:
            # Main video: Process segment-by-segment (original logic)
            # main loop
            for i, segment in enumerate(self.segments):
                # Pass processed_words to avoid getting words that are already processed
                segment_words_ts = self._get_words_for_segment(i, processed_words=words_processed)
                if not segment_words_ts:
                    continue

                for word_ts in segment_words_ts:
                    # CRITICAL: Skip if this word has already been processed
                    # This prevents duplicates when words appear in multiple segments
                    word_id = id(word_ts)
                    if word_id in words_processed:
                        logger.debug(f"Skipping duplicate word '{word_ts.word}' at {word_ts.start}s (already processed)")
                        continue
                    
                    # Mark word as processed
                    words_processed.add(word_id)
                    words_added_to_slides.add(word_id)
                    
                    # Preserve original word with punctuation - don't strip punctuation
                    word_text = word_ts.word  # Keep original word with all punctuation
                    
                    # CRITICAL: If previous word ended a sentence, we MUST start a new slide
                    # This ensures new sentences ALWAYS start on a new slide, regardless of word/line limits
                    # CRITICAL: If previous word ended a sentence, we MUST start a new slide
                    # Words after a full stop MUST ALWAYS start on a new slide
                    if sentence_just_ended:
                        # Previous word ended a sentence - finish current slide if it has words
                        # CRITICAL: After a full stop, the next sentence MUST start on a completely fresh slide
                        if current_slide_words_ts:
                            # Finish the current slide (don't include the current word yet)
                            # Do NOT carry over remaining words - sentence boundaries are absolute
                            logger.info(f"Starting new slide after sentence boundary. Word '{word_ts.word}' at {word_ts.start}s starts new sentence on new slide.")
                            build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                        
                        # ALWAYS start a completely new slide for the new sentence
                        # This ensures sentences after full stops ALWAYS start on a new slide
                        current_slide_words_ts = []
                        current_slide_segments = []
                        current_slide_start_time = word_ts.start  # Start time for new sentence
                    
                    sentence_just_ended = False  # Reset flag

                    # Initialize new slide start time if needed
                    if not current_slide_words_ts: 
                        current_slide_start_time = word_ts.start

                    # Add word to current slide
                    current_slide_words_ts.append(word_ts)

                    if segment not in current_slide_segments:
                        current_slide_segments.append(segment)

                    # CRITICAL: Check if current word ends a sentence - if so, finish this slide immediately
                    # This takes priority over word/line limits - sentence boundaries are absolute
                    # Words after a full stop MUST ALWAYS start on a new slide
                    if _word_ends_sentence(word_text):
                        # Finish this slide now - the next sentence will ALWAYS start on a new slide
                        # CRITICAL: Do NOT carry over any remaining words to the next slide
                        # After a full stop, the next sentence MUST start on a completely fresh slide
                        logger.info(f"Word '{word_text}' ends sentence at {word_ts.start}s - finishing current slide. Next word will start new slide.")
                        build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                        
                        # ALWAYS start a completely new slide for the next sentence
                        # Do not carry over any remaining words - sentence boundaries are absolute
                        current_slide_words_ts = []
                        current_slide_segments = []
                        current_slide_start_time = -1  # Will be set when next word is added
                        
                        # Mark that a sentence just ended - next word MUST start new slide
                        sentence_just_ended = True
                        last_word_text = word_text
                        continue  # Skip word limit check since we already finished the slide

                    last_word_text = word_text

                    # CRITICAL: Check limits BEFORE testing layout - enforce strict 12 words and 3 lines
                    # Check word count limit first (strict enforcement)
                    if len(current_slide_words_ts) > MAX_WORDS_PER_SLIDE:
                        # Word limit exceeded - move the overflowing word to the next slide
                        overflow_word = current_slide_words_ts.pop()
                        words_added_to_slides.discard(id(overflow_word))
                        
                        logger.debug(f"Word limit exceeded ({MAX_WORDS_PER_SLIDE} words). Moving '{overflow_word.word}' to next slide.")
                        
                        # Finalize current slide without the overflow word
                        build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                        
                        # Start new slide with the overflow word
                        current_slide_words_ts = [overflow_word]
                        words_added_to_slides.add(id(overflow_word))
                        current_slide_segments = [segment] if segment not in current_slide_segments else current_slide_segments
                        current_slide_start_time = overflow_word.start
                        continue
                    
                    # Check word/line limit by testing layout (only if we haven't already finished the slide)
                    # Build a test layout to see if current words fit within 3 lines and 12 words
                    if current_slide_words_ts:
                        # Test if adding this word would exceed limits (12 words or 3 lines)
                        test_words = list(current_slide_words_ts)  # copy current words including the newly added one
                        
                        # CRITICAL: Save current slide count before test
                        slides_count_before_test = len(slides)
                        layouts_count_before_test = len(layouts)
                        slide_start_times_count_before_test = len(slide_start_times)
                        
                        test_remaining, _ = build_slide_layout(test_words, current_slide_segments, current_slide_start_time)
                        
                        # CRITICAL: If a test slide was created, remove it (it was just for testing)
                        if len(slides) > slides_count_before_test:
                            # Remove the test slide that was created
                            test_slide_index = len(slides) - 1  # Index of the test slide (last one)
                            slides.pop()  # Remove last slide (the test slide)
                            
                            # Remove the test layout entry if it exists
                            if test_slide_index in layouts:
                                del layouts[test_slide_index]
                            
                            # Remove the test slide start time if it exists
                            if len(slide_start_times) > slide_start_times_count_before_test:
                                slide_start_times.pop()  # Remove test slide start time
                        
                        # If there are remaining words after building, the slide is full (hit 12 words or 3 lines limit)
                        if test_remaining:
                            # Determine how many words actually fit in the previous slide
                            remaining_count = len(test_remaining)
                            words_that_fit_count = len(test_words) - remaining_count
                            if words_that_fit_count < 0:
                                words_that_fit_count = 0
                            words_that_fit = test_words[:words_that_fit_count]
                            
                            logger.debug(
                                f"Slide full: {len(words_that_fit)} words fit, {remaining_count} words remain for next slide"
                            )
                            
                            # Remove words that moved to the next slide from the tracking set
                            for rem_word in test_remaining:
                                words_added_to_slides.discard(id(rem_word))
                            
                            # Finalize current slide with the words that fit
                            if words_that_fit:
                                build_slide_layout(words_that_fit, current_slide_segments, current_slide_start_time)
                            else:
                                logger.warning(
                                    f"No words fit in slide after capacity test. Remaining words: {remaining_count}"
                                )
                            
                            # Prepare next slide with the remaining words
                            current_slide_words_ts = list(test_remaining)
                            for rem_word in test_remaining:
                                words_added_to_slides.add(id(rem_word))
                            current_slide_segments = (
                                [segment] if segment not in current_slide_segments else current_slide_segments
                            )
                            current_slide_start_time = (
                                test_remaining[0].start if test_remaining else word_ts.start
                            )
                        # If no remaining words, the word fits - keep it in current slide
                        # (word is already added above, so we continue)
                    elif len(current_slide_words_ts) > MAX_WORDS_PER_SLIDE:
                        # Fallback: word count limit (shouldn't happen often with line-based checking)
                        overflow_word = current_slide_words_ts.pop()
                        words_added_to_slides.discard(id(overflow_word))
                        
                        remaining, _ = build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                        if remaining:
                            current_slide_words_ts = remaining
                            remaining_segments = []
                            for seg_idx, seg in enumerate(self.segments):
                                seg_words = self._get_words_for_segment(seg_idx)
                                if any(w in remaining for w in seg_words):
                                    remaining_segments.append(seg)
                            current_slide_segments = remaining_segments
                            current_slide_start_time = remaining[0].start if remaining else -1
                        else:
                            current_slide_words_ts = []
                            current_slide_segments = []
                            current_slide_start_time = -1
                        
                        current_slide_words_ts = [overflow_word]
                        words_added_to_slides.add(id(overflow_word))
                        current_slide_segments = [segment] if segment not in current_slide_segments else current_slide_segments
                        current_slide_start_time = overflow_word.start

        # --- Handle the VERY LAST slide after the loop finishes ---
        # Keep processing remaining words until all are added
        while current_slide_words_ts:
            remaining, _ = build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
            if remaining:
                # There are still remaining words, process them in next slide
                current_slide_words_ts = remaining
                # Find segments that contain remaining words
                remaining_segments = []
                for seg_idx, seg in enumerate(self.segments):
                    seg_words = self._get_words_for_segment(seg_idx)
                    if any(w in remaining for w in seg_words):
                        remaining_segments.append(seg)
                current_slide_segments = remaining_segments
                current_slide_start_time = remaining[0].start if remaining else -1
            else:
                # All words processed
                break
        
        # CRITICAL: Verify all words were added to slides
        total_words = len(self.all_words)
        words_in_slides = len(words_added_to_slides)
        
        # Count words actually in slides (by iterating through slides)
        # Note: word is a WordTimestamp object, not a list, so we count lines instead
        words_in_slide_structures = sum(len(line) for slide in slides for line in slide)
        
        if words_in_slides < total_words:
            missing_count = total_words - words_in_slides
            logger.error(f"CRITICAL: {missing_count} words were NOT added to any slide!")
            logger.error(f"Total words: {total_words}, Words tracked: {words_in_slides}, Words in slide structures: {words_in_slide_structures}")
            
            # Find and add missing words
            missing_words = [w for w in self.all_words if id(w) not in words_added_to_slides]
            missing_words_sample = [f"'{w.word}' at {w.start}s" for w in missing_words[:10]]
            logger.error(f"Missing words (first 10): {missing_words_sample}")
            
            # Try to add missing words to the last slide or create new slides
            if missing_words:
                logger.info(f"Attempting to recover {len(missing_words)} missing words...")
                # Sort missing words by timestamp
                missing_words_sorted = sorted(missing_words, key=lambda w: w.start)
                
                # Add missing words to current slide or create new slides
                for word in missing_words_sorted:
                    if not current_slide_words_ts:
                        current_slide_words_ts = []
                        current_slide_start_time = word.start
                    current_slide_words_ts.append(word)
                    words_added_to_slides.add(id(word))
                    
                    # If slide is getting too long, finalize it
                    if len(current_slide_words_ts) >= MAX_WORDS_PER_SLIDE:
                        build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                        current_slide_words_ts = []
                        current_slide_segments = []
                        current_slide_start_time = -1
                
                # Finalize any remaining words
                if current_slide_words_ts:
                    build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                
                logger.info(f"Recovered {len(missing_words)} missing words")
        elif words_in_slide_structures < total_words:
            logger.warning(f"Words tracked ({words_in_slides}) matches total, but slide structures only contain {words_in_slide_structures} words")
            logger.warning("This may indicate words are being lost during slide building")
        else:
            logger.info(f"✓ All {total_words} words successfully added to slides (verified: {words_in_slide_structures} in structures).")

        logger.info(f"Grouped slide building complete. Created {len(slides)} slides.")
        
        # CRITICAL: Post-process slides to fill space better across slide boundaries
        # Move words from next slide's first line to current slide's last line if they fit
        logger.info("Post-processing slides to optimize space usage across slide boundaries...")
        slides, layouts, slide_start_times = self._optimize_slide_boundaries(slides, layouts, slide_start_times)
        
        # Recompute slide end times after optimization to ensure accuracy
        slide_end_times = []
        for slide_idx, slide_lines in enumerate(slides):
            slide_end_time = slide_start_times[slide_idx] if slide_idx < len(slide_start_times) else 0.0
            for line in reversed(slide_lines):
                if line:
                    slide_end_time = line[-1].end
                    break
            slide_end_times.append(slide_end_time)
        
        return slides, layouts, slide_start_times, slide_end_times
    
    def _optimize_slide_boundaries(self, slides: List[List[List[WordTimestamp]]], layouts: Dict[int, Dict[int, Tuple[int, int]]], slide_start_times: List[float]) -> Tuple[List[List[List[WordTimestamp]]], Dict[int, Dict[int, Tuple[int, int]]], List[float]]:
        """
        Optimize slide boundaries by moving words from next slide to fill current slide's last line.
        CRITICAL: Never moves words across sentence boundaries (after full stops).
        Only optimizes within the same sentence.
        
        Args:
            slides: List of slides (each slide is a list of lines, each line is a list of words)
            layouts: Dictionary of slide layouts
            slide_start_times: List of slide start times
            
        Returns:
            Optimized (slides, layouts, slide_start_times)
        """
        if len(slides) < 2:
            return slides, layouts, slide_start_times
        
        dummy_img = Image.new("RGB", (self.bg_width, self.bg_height))
        draw = ImageDraw.Draw(dummy_img)
        space_bbox = draw.textbbox((0, 0), " ", font=self.bold_font)
        space_width = space_bbox[2] - space_bbox[0]
        
        # Helper function to check if a word ends a sentence
        def _word_ends_sentence(word_text: str) -> bool:
            """Check if word ends with a full stop (sentence ending)."""
            if not word_text:
                return False
            cleaned = word_text.strip()
            if not cleaned:
                return False
            # Remove trailing quotes, brackets, etc. but keep periods
            while cleaned and cleaned[-1] in {"'", '"', '"', '"', "'", ")", "]", "}", " "}:
                cleaned = cleaned[:-1]
            if not cleaned or cleaned[-1] != ".":
                return False
            # Check for abbreviations (same logic as main function)
            word_lower = cleaned.lower()
            abbreviations = {
                'mr.', 'mrs.', 'ms.', 'miss.', 'dr.', 'prof.', 'rev.', 'fr.', 'sr.', 'jr.',
                'hon.', 'esq.', 'sir.', 'dame.', 'lord.', 'lady.', 'ph.d.', 'm.d.', 'd.d.s.', 'd.v.m.',
                'rn.', 'cpa.', 'llc.', 'llp.', 'a.m.', 'p.m.', 'am.', 'pm.', 'b.c.', 'a.d.', 'bc.', 'ad.',
                'jan.', 'feb.', 'mar.', 'apr.', 'may.', 'jun.', 'jul.', 'aug.', 'sep.', 'sept.', 'oct.', 'nov.', 'dec.',
                'vs.', 'v.', 'etc.', 'e.g.', 'i.e.', 'viz.', 'cf.', 'ex.', 'et al.', 'inc.', 'ltd.', 'corp.', 'co.',
                'st.', 'ave.', 'blvd.', 'rd.', 'ct.', 'ln.', 'pkwy.', 'sq.', 'n.', 's.', 'e.', 'w.', 'nw.', 'ne.', 'sw.', 'se.',
                'no.', 'vol.', 'pp.', 'p.', 'ch.', 'sec.', 'fig.', 'art.', 'ft.', 'in.', 'lb.', 'oz.', 'kg.', 'g.', 'ml.', 'l.',
                'ed.', 'est.', 'approx.', 'min.', 'max.', 'avg.', 'std.', 'gov.', 'pres.', 'sen.', 'rep.', 'gen.', 'col.', 'capt.',
                'dept.', 'univ.', 'assoc.', 'misc.', 'op.', 'cit.'
            }
            if word_lower in abbreviations or (len(cleaned) == 2 and cleaned[0].isalpha() and cleaned[1] == '.'):
                return False
            if len(cleaned) >= 3 and '.' in cleaned[:-1]:
                return False
            return True
        
        # Work with a copy to avoid modifying original during iteration
        working_slides = [list(slide) for slide in slides]  # Deep copy of slide structure
        working_layouts = dict(layouts)
        working_start_times = list(slide_start_times)
        
        i = 0
        optimization_count = 0
        skipped_count = 0
        
        while i < len(working_slides) - 1:
            current_slide = working_slides[i]
            next_slide = working_slides[i + 1]
            
            # Only optimize if both slides have content
            if not current_slide or not next_slide:
                logger.debug(f"Skipping optimization for slide {i}: current_slide={bool(current_slide)}, next_slide={bool(next_slide)}")
                i += 1
                continue
            
            current_last_line = current_slide[-1] if current_slide else []
            next_first_line = next_slide[0] if next_slide else []
            
            if not current_last_line or not next_first_line:
                logger.debug(f"Skipping optimization for slide {i}: current_last_line={bool(current_last_line)}, next_first_line={bool(next_first_line)}")
                i += 1
                continue
            
            # CRITICAL: Check if current slide ends with a full stop (sentence boundary)
            # If so, NEVER move words from next slide - it starts a new sentence
            current_slide_ends_with_period = False
            if current_last_line:
                last_word = current_last_line[-1] if current_last_line else None
                if last_word and _word_ends_sentence(last_word.word):
                    current_slide_ends_with_period = True
                    logger.debug(f"Slide {i} ends with full stop (word: '{last_word.word}') - skipping optimization to preserve sentence boundary")
                    skipped_count += 1
                    i += 1
                    continue  # Skip optimization - preserve sentence boundary
            
            
            # Check if we can move words from next slide to current slide's last line
            # Calculate current line width to see how much space is available
            current_line_text = " ".join(w.word for w in current_last_line)
            current_line_bbox = draw.textbbox((0, 0), current_line_text, font=self.bold_font)
            current_line_width = current_line_bbox[2] - current_line_bbox[0]
            available_space = self.max_text_width - current_line_width
            
            logger.debug(f"Slide {i}: Current last line width: {current_line_width:.1f}px, Max width: {self.max_text_width:.1f}px, Available: {available_space:.1f}px")
            
            # Check if last line has only one word - if so, be more aggressive
            last_line_has_one_word = len(current_last_line) == 1
            if last_line_has_one_word:
                logger.debug(f"Slide {i}: Last line has only one word - using aggressive optimization")
            
            # Collect ALL words from next slide that are from the same sentence
            # Check multiple lines, not just the first line
            next_slide_words_same_sentence = []
            for line_idx, line in enumerate(next_slide):
                for word in line:
                    # Stop if we hit a sentence boundary (word ends with period)
                    if _word_ends_sentence(word.word):
                        # Include this word if it's the last word of the sentence
                        next_slide_words_same_sentence.append(word)
                        break
                    next_slide_words_same_sentence.append(word)
                # If we hit a sentence boundary, stop collecting
                if line and _word_ends_sentence(line[-1].word):
                    break
            
            # Try moving words one by one from next slide (from same sentence)
            words_moved = 0
            threshold = 0.99  # Default threshold
            if last_line_has_one_word:
                # If last line has only one word, be more aggressive - allow up to 100% (or even slightly over)
                threshold = 1.0
                logger.debug(f"Slide {i}: Using aggressive threshold (100%) to fill single-word line")
            
            for word_idx in range(len(next_slide_words_same_sentence)):
                test_line = current_last_line + next_slide_words_same_sentence[:word_idx + 1]
                line_text = " ".join(w.word for w in test_line)
                line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
                test_line_width = line_bbox[2] - line_bbox[0]
                
                # Use threshold (99% normally, 100% if last line has only one word)
                if test_line_width <= self.max_text_width * threshold:
                    words_moved = word_idx + 1
                else:
                    break
            
            # If we can move words, do it
            if words_moved > 0:
                logger.info(f"Slide {i}: Moving {words_moved} word(s) from slide {i+1} to fill space on slide {i}")
                optimization_count += 1
                
                # Move words from next slide to current slide
                words_to_move = next_slide_words_same_sentence[:words_moved]
                
                # CRITICAL: Verify all words to move actually exist in next slide
                next_slide_all_words = [w for line in working_slides[i + 1] for w in line]
                words_to_move_ids = {id(w) for w in words_to_move}
                next_slide_word_ids = {id(w) for w in next_slide_all_words}
                missing_from_next = words_to_move_ids - next_slide_word_ids
                if missing_from_next:
                    logger.error(
                        f"CRITICAL: Trying to move {len(missing_from_next)} words that don't exist in slide {i+1}! "
                        f"Words: {[w.word for w in words_to_move if id(w) in missing_from_next]}"
                    )
                
                working_slides[i][-1] = current_last_line + words_to_move
                
                # Update next slide - remove moved words from all lines
                # Track which words we've moved
                words_moved_set = {id(w) for w in words_to_move}
                
                # Remove moved words from next slide
                # CRITICAL: Keep track of which lines become empty to detect missing words
                new_next_slide = []
                lines_removed = 0
                for line_idx, line in enumerate(working_slides[i + 1]):
                    new_line = [w for w in line if id(w) not in words_moved_set]
                    if new_line:  # Only keep non-empty lines
                        new_next_slide.append(new_line)
                    else:
                        # Line became empty - log it for debugging
                        logger.warning(
                            f"Slide {i+1}, Line {line_idx} became empty after moving {len(line)} words. "
                            f"Original line had: {[w.word for w in line]}"
                        )
                        lines_removed += 1
                
                if lines_removed > 0:
                    logger.warning(f"Slide {i+1}: {lines_removed} line(s) became empty during optimization")
                
                # Update next slide
                if new_next_slide:
                    working_slides[i + 1] = new_next_slide
                else:
                    # Next slide is now empty
                    working_slides[i + 1] = []
                
                # Rebuild layouts for affected slides
                # Extract all words from current slide (ensure no duplicates)
                current_slide_words = []
                seen_word_ids = set()
                for line in working_slides[i]:
                    for word in line:
                        word_id = id(word)
                        if word_id not in seen_word_ids:
                            current_slide_words.append(word)
                            seen_word_ids.add(word_id)
                
                # Rebuild current slide layout
                if current_slide_words:
                    working_slides[i], working_layouts[i] = self._rebuild_slide_layout(
                        current_slide_words, i, working_start_times[i]
                    )
                
                # Rebuild next slide layout if it still has content
                if working_slides[i + 1]:
                    next_slide_words = []
                    seen_word_ids = set()
                    for line in working_slides[i + 1]:
                        for word in line:
                            word_id = id(word)
                            if word_id not in seen_word_ids:
                                next_slide_words.append(word)
                                seen_word_ids.add(word_id)
                    
                    if next_slide_words:
                        working_slides[i + 1], working_layouts[i + 1] = self._rebuild_slide_layout(
                            next_slide_words, i + 1, working_start_times[i + 1]
                        )
                    else:
                        # Next slide is now empty - remove it
                        working_slides.pop(i + 1)
                        working_layouts.pop(i + 1, None)
                        if i + 1 < len(working_start_times):
                            working_start_times.pop(i + 1)
                        logger.info(f"Slide {i}: Moved {words_moved} word(s) from slide {i+1} to slide {i}, removed empty slide")
                        continue  # Don't increment i, check same position again
                
                logger.info(f"Slide {i}: Moved {words_moved} word(s) from slide {i+1} to slide {i}")
            else:
                logger.debug(f"Slide {i}: No words can be moved (tested {len(next_first_line)} words, none fit)")
            
            i += 1
        
        logger.info(f"Slide optimization complete: {optimization_count} optimizations performed, {skipped_count} skipped due to sentence boundaries")
        
        # Filter out any empty slides that might remain
        final_slides = []
        final_layouts = {}
        final_start_times = []
        
        # Track all words to ensure none are lost
        # Structure: slides -> lines -> words (WordTimestamp objects)
        original_word_count = sum(len(line) for slide in slides for line in slide)
        final_word_count = 0
        
        for idx, slide in enumerate(working_slides):
            if slide:  # Only include non-empty slides
                final_slides.append(slide)
                final_layouts[len(final_slides) - 1] = working_layouts.get(idx, {})
                if idx < len(working_start_times):
                    final_start_times.append(working_start_times[idx])
                # Count words in this slide (each line is a list of WordTimestamp objects)
                for line in slide:
                    final_word_count += len(line)
        
        # CRITICAL: Verify no words were lost
        if final_word_count != original_word_count:
            logger.error(f"CRITICAL: Word count mismatch during optimization! Original: {original_word_count}, Final: {final_word_count}")
            logger.error(f"Lost {original_word_count - final_word_count} words during slide optimization!")
            # Return original slides if words were lost
            logger.warning("Returning original slides to prevent word loss")
            return slides, layouts, slide_start_times
        else:
            logger.info(f"Optimized slide boundaries: {len(slides)} slides -> {len(final_slides)} slides (all {original_word_count} words preserved)")
        
        return final_slides, final_layouts, final_start_times
    
    def _rebuild_slide_layout(self, words: List[WordTimestamp], slide_index: int, start_time: float) -> Tuple[List[List[WordTimestamp]], Dict[int, Tuple[int, int]]]:
        """
        Rebuild layout for a slide given its words.
        This is a helper for _optimize_slide_boundaries.
        Uses the SAME centering/padding logic as the original build_slide_layout.
        """
        dummy_img = Image.new("RGB", (self.bg_width, self.bg_height))
        draw = ImageDraw.Draw(dummy_img)
        space_bbox = draw.textbbox((0, 0), " ", font=self.bold_font)
        space_width = space_bbox[2] - space_bbox[0]
        
        # Build lines (same logic as original, but with aggressive threshold to prevent single-word last lines)
        lines = []
        current_line = []
        
        for word in words:
            if current_line:
                test_line = current_line + [word]
                line_text = " ".join(w.word for w in test_line)
                line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
                test_line_width = line_bbox[2] - line_bbox[0]
                
                # Use 99% threshold (same as optimization)
                # But if this would create a last line with only one word, be more aggressive
                would_create_single_word_line = len(current_line) == 1 and word == words[-1]
                threshold = 1.0 if would_create_single_word_line else 0.99
                
                if test_line_width > self.max_text_width * threshold:
                    lines.append(current_line)
                    current_line = [word]
                else:
                    current_line.append(word)
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(current_line)
            
            # CRITICAL: If last line has only one word, try to move a word from the previous line
            if len(lines) >= 2 and len(current_line) == 1:
                prev_line = lines[-2]
                if len(prev_line) > 1:  # Previous line has multiple words
                    # Try moving last word from previous line to last line
                    test_line = [prev_line[-1]] + current_line
                    line_text = " ".join(w.word for w in test_line)
                    line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
                    test_line_width = line_bbox[2] - line_bbox[0]
                    
                    if test_line_width <= self.max_text_width * 1.0:  # Allow up to 100%
                        # Move the word
                        moved_word = prev_line.pop()
                        current_line.insert(0, moved_word)
                        logger.debug(f"Rebuild: Moved word '{moved_word.word}' from previous line to fill single-word last line")
        
        # Build layout using SAME centering logic as original build_slide_layout
        layout = {}
        
        # Step 1: Calculate actual content dimensions (bounding box) - SAME AS ORIGINAL
        max_line_width = 0
        for line_of_words in lines:
            if not line_of_words:
                continue
            line_text = " ".join(w.word for w in line_of_words)
            line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
            line_width = line_bbox[2] - line_bbox[0]
            max_line_width = max(max_line_width, line_width)
        
        content_width = max_line_width
        content_height = len(lines) * self.line_height
        
        # Step 2: Use the fixed margins defined at class level - SAME AS ORIGINAL
        min_margin_top = self.top_margin
        min_margin_bottom = self.bottom_margin
        min_margin_left = self.left_margin
        min_margin_right = self.right_margin
        
        # Step 3: Calculate available space (screen - margins)
        available_width = self.bg_width - min_margin_left - min_margin_right
        available_height = self.bg_height - min_margin_top - min_margin_bottom
        
        # Step 4: Fixed 150px left padding - text area starts at exactly 150px from left
        # NO dynamic horizontal padding - left side is fixed at 150px (same as build_slide_layout)
        text_area_start_x = 150  # Fixed 150px left padding - never changes
        text_area_end_x = self.bg_width - min_margin_right  # Right boundary uses right margin
        text_area_width = text_area_end_x - text_area_start_x
        
        # Step 5: Calculate vertical padding (only vertical centering, horizontal is fixed)
        padding_vertical = max(0, (available_height - content_height) / 2)
        
        # Vertical positioning - centered with equal padding - SAME AS ORIGINAL
        start_y = min_margin_top + padding_vertical
        max_y = self.bg_height - min_margin_bottom - padding_vertical
        
        # Ensure text doesn't exceed bounds (safety check)
        if start_y + content_height > max_y:
            start_y = min_margin_top
            logger.warning(f"Slide {slide_index}: Content height exceeds available space, using minimum top margin")
        
        current_y = start_y
        max_x = text_area_end_x
        
        # Position words using SAME logic as original
        for line_idx, line_of_words in enumerate(lines):
            if len(line_of_words) == 0:
                current_y += self.line_height
                continue
            
            # Calculate total width of all words (with spaces) - SAME AS ORIGINAL
            word_widths = []
            total_words_width = 0
            for word in line_of_words:
                word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                word_width = word_bbox[2] - word_bbox[0]
                word_widths.append(word_width)
                total_words_width += word_width
            
            # Add space widths between words
            if len(line_of_words) > 1:
                total_words_width += (len(line_of_words) - 1) * space_width
            
            # Text alignment: center for reels, left for main videos - SAME AS ORIGINAL
            if hasattr(self, 'text_align') and self.text_align == "center":
                current_x = text_area_start_x + (text_area_width - total_words_width) // 2
            else:
                current_x = text_area_start_x
            
            # Store positions for each word - SAME AS ORIGINAL
            for i, word in enumerate(line_of_words):
                word_width = word_widths[i]
                # Ensure word fits - adjust position if needed
                if current_x + word_width > max_x:
                    adjusted_x = max(text_area_start_x, max_x - word_width)
                    if adjusted_x + word_width <= max_x:
                        current_x = adjusted_x
                    else:
                        current_x = text_area_start_x
                
                # CRITICAL: Convert to int for pixel coordinates
                x_coord = int(round(current_x))
                y_coord = int(round(current_y))
                
                layout[(line_idx, i)] = (x_coord, y_coord)
                current_x += word_width + space_width
            
            current_y += self.line_height
        
        return lines, layout
    
    def make_frame_function(self, slide_index: int, slide_start_time: float):
        def generate_frame(t_local: float) -> np.ndarray:
            global_t = slide_start_time + t_local
            frame = Image.new("RGBA", (self.bg_width, self.bg_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            slide_lines = self.slides[slide_index]
            layout = self.slide_layouts[slide_index]
            max_x = self.bg_width - self.right_margin
            
            for line_idx, line in enumerate(slide_lines):
                for word_idx, word in enumerate(line):
                    # Use (line_idx, word_idx) as key to match layout keys
                    coords = layout.get((line_idx, word_idx))
                    if not coords:
                        logger.warning(f"Word '{word.word}' at line {line_idx}, word {word_idx} not found in layout for slide {slide_index}")
                        continue
                    
                    # Safety check: ensure word doesn't overflow screen
                    x, y = coords
                    word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                    word_width = word_bbox[2] - word_bbox[0]
                    
                    if x + word_width > max_x:
                        # Adjust position to prevent overflow
                        x = max(self.left_margin, max_x - word_width)
                    
                    # --- 3. Load colors from config.py ---
                    # Text should be faded (light) before audio starts, then bold when audio plays
                    # Word becomes bold when it starts being spoken and stays bold
                    if global_t >= word.start:
                        font = self.bold_font
                        color = settings.TEXT_BOLD_COLOR
                    else:
                        # Word hasn't been spoken yet - show as faded/light
                        font = self.regular_font
                        color = settings.TEXT_REGULAR_COLOR
                    draw.text((x, y), word.word, font=font, fill=color)
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
        
        max_x = self.bg_width - self.right_margin
        
        # Debug: log layout info
        if slide_index == 0:
            logger.debug(f"Slide {slide_index}: Layout has {len(layout)} entries, slide has {sum(len(line) for line in slide_lines)} words")
        
        for line_idx, line in enumerate(slide_lines):
            for word_idx, word in enumerate(line):
                # Use (line_idx, word_idx) as key to match layout keys
                coords = layout.get((line_idx, word_idx))
                if not coords:
                    logger.error(f"CRITICAL: Word '{word.word}' at line {line_idx}, word {word_idx} not found in layout for slide {slide_index}")
                    # Try to render at a default position to ensure text shows
                    x = self.left_margin
                    y = self.top_margin + line_idx * self.line_height
                    logger.warning(f"Rendering word '{word.word}' at fallback position ({x}, {y})")
                else:
                    x, y = coords
                
                # Safety check: ensure word doesn't overflow screen
                word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                word_width = word_bbox[2] - word_bbox[0]
                
                # Adjust position if word would overflow - don't skip it
                if x + word_width > max_x:
                    # Try to fit by positioning at max_x - word_width
                    adjusted_x = max(self.left_margin, max_x - word_width)
                    if adjusted_x + word_width <= max_x:
                        x = adjusted_x
                    # If still too wide, render it anyway (will be clipped but visible)
                
                # State-based logic: use word-level timestamps (start AND end) for accurate highlighting
                # Text should be faded (light) before audio starts, bold while being spoken, then stays bold
                # Use word.start and word.end to determine if word is currently being spoken
                if global_t >= word.start and global_t < word.end:
                    # Word is currently being spoken - show as bold
                    font = self.bold_font
                    color = settings.TEXT_BOLD_COLOR
                elif global_t >= word.end:
                    # Word has finished being spoken - keep it bold (already spoken)
                    font = self.bold_font
                    color = settings.TEXT_BOLD_COLOR
                else:
                    # Word hasn't been spoken yet - show as faded/light
                    font = self.regular_font
                    color = settings.TEXT_REGULAR_COLOR
                
                # CRITICAL: Always render the word - ensure text is visible
                draw.text((x, y), word.word, font=font, fill=color)
                
                # Debug first word of first slide
                if slide_index == 0 and line_idx == 0 and word_idx == 0:
                    logger.debug(f"Rendered first word '{word.word}' at ({x}, {y}) with color {color}, font size {self.font_size}")
        
        return frame


# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================

def _apply_animations_to_clip(
    frame_clip: VideoClip,
    frame_gen: 'FrameGeneratorV11',
    anim_opts: Dict[str, Any],
    fps: int,
    duration: float
) -> VideoClip:
    """
    Apply animations and transitions to the video clip.
    
    Args:
        frame_clip: The base frame clip
        frame_gen: Frame generator instance
        anim_opts: Animation options dictionary
        fps: Frames per second
        duration: Video duration in seconds
    
    Returns:
        Animated video clip
    """
    logger.info("Applying animations to video clip...")
    
    # Note: Transitions are applied in the main render_video function
    # using CompositeVideoClip.set_opacity() which works with ImageSequenceClip
    # This function only handles text animations (zoom/pan)
    
    # Note: Text animations (zoom/pan) are not applied here because ImageSequenceClip
    # doesn't support fl_image or fl methods. These would need to be applied during
    # frame generation or using a different approach. For now, we'll skip them
    # and only apply fade transitions which work with set_opacity.
    
    if anim_opts.get("enableTextZoom", False) or anim_opts.get("enableTextPan", False):
        logger.info("Text zoom/pan animations requested but not yet implemented for ImageSequenceClip")
        logger.info("These effects would require frame-level processing during generation")
    
    logger.info("Animations setup complete")
    return frame_clip


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def _calculate_frame_timestamps(duration: float, fps: int, word_timestamps: Optional[List[WordTimestamp]] = None) -> List[Tuple[int, float]]:
    """
    Calculate all frame timestamps for the video.
    
    CRITICAL: For word-level accuracy, we generate frames based on:
    1. Evenly spaced frames (FPS-based) for smooth playback
    2. Additional frames at each word start/end timestamp to ensure perfect sync
    
    This ensures every word timestamp has corresponding frames, preventing missing words.
    
    Args:
        duration: Audio duration in seconds
        fps: Frames per second
        word_timestamps: Optional list of WordTimestamp objects to ensure frames at word boundaries
    
    Returns:
        List of (frame_number, timestamp) tuples
    """
    import math
    frame_interval = 1.0 / fps
    frame_timestamps = []
    frame_num = 0
    
    # Step 1: Generate evenly spaced frames (FPS-based)
    total_frames = math.ceil(duration * fps)
    for i in range(total_frames):
        timestamp = i * frame_interval
        if timestamp > duration:
            timestamp = duration
        frame_timestamps.append((frame_num, timestamp))
        frame_num += 1
    
    # Step 2: Add frames at word start/end timestamps for perfect sync
    # This ensures every word has frames generated at its exact timestamps
    if word_timestamps:
        word_timestamps_set = set()
        for word in word_timestamps:
            # Add frame at word start (if not already present)
            word_start = word.start
            if word_start >= 0 and word_start <= duration:
                word_timestamps_set.add(word_start)
            # Add frame at word end (if not already present)
            word_end = word.end
            if word_end >= 0 and word_end <= duration:
                word_timestamps_set.add(word_end)
        
        # Add word timestamps that aren't already covered by FPS frames
        for word_ts in sorted(word_timestamps_set):
            # Check if this timestamp is already close to an existing frame (within 1 frame interval)
            already_covered = any(abs(existing_ts - word_ts) < frame_interval / 2 for _, existing_ts in frame_timestamps)
            if not already_covered:
                frame_timestamps.append((frame_num, word_ts))
                frame_num += 1
        
        logger.info(f"Added {len(word_timestamps_set)} word-level timestamps to frame generation")
    
    # Sort by timestamp to ensure correct order
    frame_timestamps.sort(key=lambda x: x[1])
    # Renumber frames sequentially
    frame_timestamps = [(i, ts) for i, (_, ts) in enumerate(frame_timestamps)]
    
    logger.info(f"Calculated {len(frame_timestamps)} frames for {duration:.3f}s audio at {fps}fps (including word-level timestamps)")
    return frame_timestamps


def _map_frames_to_slides(
    frame_timestamps: List[Tuple[int, float]],
    slide_start_times: List[float],
    slide_end_times: List[float],
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
        # CRITICAL: Iterate backwards to find the correct slide at boundaries
        # This ensures we select the new slide when at a transition boundary, not the old one
        slide_index = len(slide_start_times) - 1  # Default to last slide
        slide_start = slide_start_times[slide_index] if slide_start_times else 0.0
        
        # Find the correct slide for this timestamp
        # CRITICAL: At transition boundaries, we want the NEW slide, not the old one
        epsilon = 1e-5  # Very small epsilon to handle floating point precision
        
        # Find the slide that should be displayed at this timestamp
        # We iterate forward and select the slide whose start time is <= timestamp
        # and whose end time is > timestamp (for non-final slides)
        for i in range(len(slide_start_times)):
            slide_start_time = slide_start_times[i]
            slide_end_time = slide_end_times[i] if i < len(slide_end_times) else audio_duration
            
            # Check if timestamp is before this slide starts
            if timestamp < slide_start_time - epsilon:
                # We've passed all slides that could contain this timestamp
                # Use the previous slide (i-1) or slide 0 if this is the first slide
                if i > 0:
                    slide_index = i - 1
                    slide_start = slide_start_times[slide_index]
                else:
                    slide_index = 0
                    slide_start = slide_start_times[0] if slide_start_times else 0.0
                break
            
            # Check if timestamp is within this slide's range
            if i == len(slide_start_times) - 1:
                # Last slide: extends to end of audio
                if slide_start_time - epsilon <= timestamp:
                    slide_index = i
                    slide_start = slide_start_time
                    break
            else:
                # Non-final slides: timestamp must be < end_time to belong to this slide
                # At the exact boundary (timestamp >= slide_end_time), we want the NEXT slide
                if slide_start_time - epsilon <= timestamp < slide_end_time - epsilon:
                    slide_index = i
                    slide_start = slide_start_time
                    break
                # If timestamp >= slide_end_time, continue to next slide (this slide has ended)
        
        # CRITICAL: Ensure we generate frames for the full audio duration
        # The last slide should persist until the end of audio
        # Don't skip frames - always map them to a slide (use last slide if beyond)
        if slide_index >= len(slide_start_times):
            # If we somehow went beyond slides, use the last slide
            slide_index = len(slide_start_times) - 1
            slide_start = slide_start_times[slide_index] if slide_start_times else 0.0
            logger.warning(f"Frame at {timestamp:.3f}s mapped to last slide {slide_index} (beyond slide count)")
        
        # Always add the frame - don't skip any frames
        # Even if timestamp is slightly beyond audio_duration, include it to ensure full coverage
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
    # IMPORTANT: Use same font loading logic as main renderer
    try:
        regular_path = settings.DEFAULT_FONT_REGULAR
        bold_path = settings.DEFAULT_FONT_BOLD
        
        # Resolve paths and check existence (same logic as main renderer)
        regular_path_obj = Path(regular_path).resolve()
        bold_path_obj = Path(bold_path).resolve()
        
        if not regular_path_obj.exists():
            # Try alternative path
            alt_path = settings.FONTS_PATH / Path(regular_path).name
            if alt_path.exists():
                regular_path_obj = alt_path.resolve()
            else:
                raise FileNotFoundError(f"Font file not found: {regular_path} or {alt_path}")
        
        if not bold_path_obj.exists():
            # Try alternative path
            alt_path = settings.FONTS_PATH / Path(bold_path).name
            if alt_path.exists():
                bold_path_obj = alt_path.resolve()
            else:
                raise FileNotFoundError(f"Font file not found: {bold_path} or {alt_path}")
        
        regular_font = ImageFont.truetype(str(regular_path_obj), font_size)
        bold_font = ImageFont.truetype(str(bold_path_obj), font_size)
        logger.debug(f"Worker loaded fonts: regular={regular_path_obj}, bold={bold_path_obj}")
    except Exception as e:
        logger.error(f"Worker failed to load custom fonts: {e}", exc_info=True)
        logger.error(f"Worker font paths - Regular: {settings.DEFAULT_FONT_REGULAR}, Bold: {settings.DEFAULT_FONT_BOLD}")
        logger.error(f"FONTS_PATH: {settings.FONTS_PATH}")
        logger.warning(f"Worker falling back to default font.")
        regular_font = ImageFont.load_default(size=font_size)
        bold_font = ImageFont.load_default(size=font_size)
    
    # Pre-extract colors to avoid repeated lookups
    bold_color = settings.TEXT_BOLD_COLOR
    regular_color = settings.TEXT_REGULAR_COLOR
    
    generated_files = []
    
    # Process frames in batch
    for frame_num, timestamp, slide_index, slide_start in frame_tasks:
        # CRITICAL: Log if we're missing frames or have invalid indices
        if frame_num % 100 == 0:  # Log every 100th frame for debugging
            logger.debug(f"Worker generating frame {frame_num} at {timestamp:.3f}s, slide {slide_index}")
        # Generate frame
        frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(frame)
        
        # CRITICAL: Ensure slide_index is valid
        if slide_index < 0 or slide_index >= len(slides):
            # Invalid slide index - use last slide or first slide
            if len(slides) > 0:
                slide_index = max(0, min(slide_index, len(slides) - 1))
                logger.warning(f"Frame {frame_num} at {timestamp:.3f}s had invalid slide_index, using {slide_index}")
            else:
                # No slides available - create empty frame
                frame_filename = output_dir / f"frame_{frame_num:06d}.png"
                frame.save(frame_filename, "PNG", optimize=False, compress_level=1)
                generated_files.append(str(frame_filename))
                continue
        
        slide_lines = slides[slide_index]
        layout = slide_layouts.get(slide_index, {})
        
        # Render all words in the slide
        # Use margins from gen_data (dynamic for reels/main video)
        left_margin = gen_data.get('left_margin', 80)
        right_margin = gen_data.get('right_margin', 200)
        max_text_width = width - left_margin - right_margin
        max_x = width - right_margin
        
        for line_idx, line in enumerate(slide_lines):
            for word_idx, word in enumerate(line):
                # Look up coordinates using (line_index, word_index) as key
                unique_key = (line_idx, word_idx)
                coords = layout.get(unique_key)
                if not coords:
                    # CRITICAL: Don't skip words without coordinates - render them at fallback position
                    # This ensures all words are visible even if layout calculation missed them
                    logger.warning(f"Word '{word['word']}' at line {line_idx}, word {word_idx} not found in layout for slide {slide_index} - using fallback position")
                    x = left_margin
                    y = (line_idx * gen_data.get('line_height', 50)) + gen_data.get('top_margin', 150)
                else:
                    x, y = coords
                
                # Safety check: ensure word doesn't overflow screen
                word_bbox = draw.textbbox((0, 0), word['word'], font=bold_font)
                word_width = word_bbox[2] - word_bbox[0]
                
                # Adjust position if word would overflow - don't skip it
                if x + word_width > max_x:
                    # Try to fit by positioning at max_x - word_width
                    adjusted_x = max(left_margin, max_x - word_width)
                    if adjusted_x + word_width <= max_x:
                        x = adjusted_x
                    # If still too wide, render it anyway (will be clipped but visible)
                
                # State-based logic: use word-level timestamps (start AND end) for accurate highlighting
                # Text should be faded (light) before audio starts, bold while being spoken, then stays bold
                # Use word['start'] and word['end'] to determine if word is currently being spoken
                if timestamp >= word['start'] and timestamp < word['end']:
                    # Word is currently being spoken - show as bold
                    draw.text((x, y), word['word'], font=bold_font, fill=bold_color)
                elif timestamp >= word['end']:
                    # Word has finished being spoken - keep it bold (already spoken)
                    draw.text((x, y), word['word'], font=bold_font, fill=bold_color)
                else:
                    # Word hasn't been spoken yet - show as faded/light
                    draw.text((x, y), word['word'], font=regular_font, fill=regular_color)
        
        # Save frame with optimized PNG settings
        frame_filename = output_dir / f"frame_{frame_num:06d}.png"
        # Use optimize=False for faster saving (we'll delete these anyway)
        frame.save(frame_filename, "PNG", optimize=False, compress_level=1)
        generated_files.append(str(frame_filename))
    
    return generated_files


def _get_ffmpeg_path() -> str:
    """Get the path to ffmpeg executable."""
    import shutil
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    
    raise FileNotFoundError("FFmpeg not found. Please install ffmpeg.")


def _detect_hardware_codec() -> Tuple[str, List[str]]:
    """
    Detect available hardware acceleration codec.
    Note: Error handling in _encode_video_with_ffmpeg will fall back to software if hardware fails.
    
    Returns:
        Tuple of (codec_name, additional_ffmpeg_params)
    """
    import subprocess
    
    ffmpeg_path = _get_ffmpeg_path()
    
    # Try NVIDIA NVENC
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_nvenc" in result.stdout:
            logger.info("Detected NVIDIA GPU encoder - will try h264_nvenc (will fallback to software if unavailable)")
            # Optimized NVENC settings for speed: p1 preset (fastest), VBR with CQ
            return "h264_nvenc", [
                "-preset", "p1",  # Fastest preset
                "-rc", "vbr",
                "-cq", "23",
                "-rc-lookahead", "32",
                "-b_ref_mode", "middle"
            ]
    except Exception:
        pass
    
    # Try Intel QuickSync
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_qsv" in result.stdout:
            logger.info("Detected Intel QuickSync - using h264_qsv")
            return "h264_qsv", ["-preset", "veryfast", "-global_quality", "23"]
    except Exception:
        pass
    
    # Try AMD AMF
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_amf" in result.stdout:
            logger.info("Detected AMD GPU - using h264_amf")
            return "h264_amf", ["-quality", "speed", "-rc", "vbr_peak", "-qmin", "18", "-qmax", "28"]
    except Exception:
        pass
    
    # Fallback to software encoding
    logger.info("No hardware acceleration detected - using libx264 (software)")
    return "libx264", ["-preset", "veryfast", "-crf", "23"]


def _encode_video_with_ffmpeg(
    video_clip: VideoClip,
    audio_path: Path,
    output_path: Path,
    fps: int
) -> Path:
    """
    Encode video directly with FFmpeg, bypassing MoviePy's slow encoding.
    
    Args:
        video_clip: The composited video clip from MoviePy
        audio_path: Path to audio file
        output_path: Output video path
        fps: Video FPS
    
    Returns:
        Path to encoded video
    """
    import subprocess
    
    ffmpeg_path = _get_ffmpeg_path()
    codec, codec_params = _detect_hardware_codec()
    
    # Use hardware encoding for temp file if available, otherwise use fastest software encoding
    temp_codec, temp_codec_params = _detect_hardware_codec()
    
    # For temp file, use faster/lower quality settings since it's just intermediate
    if temp_codec == "h264_nvenc":
        temp_params = ["-preset", "p1", "-rc", "vbr", "-cq", "28", "-b:v", "10M"]  # Faster, lower quality temp
    elif temp_codec == "h264_qsv":
        temp_params = ["-preset", "veryfast", "-global_quality", "28", "-b:v", "10M"]
    elif temp_codec == "h264_amf":
        temp_params = ["-quality", "speed", "-rc", "vbr_peak", "-qmin", "24", "-qmax", "32", "-b:v", "10M"]
    else:
        temp_params = ["-preset", "ultrafast", "-crf", "28", "-tune", "zerolatency"]  # Fastest software
    
    temp_video = output_path.parent / f"{output_path.stem}_temp.mp4"
    
    try:
        logger.info("Writing temporary video file (optimized for speed)...")
        # Try hardware encoding for temp file, fall back to software if it fails
        try:
            video_clip.write_videofile(
                str(temp_video),
                fps=fps,
                codec=temp_codec,
                audio=False,
                ffmpeg_params=temp_params,
                logger=None,
                threads=4  # Use multiple threads
            )
        except Exception as e:
            # If hardware encoding fails, fall back to software
            if temp_codec != "libx264":
                logger.warning(f"Hardware encoding for temp file failed: {e}")
                logger.info("Falling back to software encoding for temp file...")
                video_clip.write_videofile(
                    str(temp_video),
                    fps=fps,
                    codec="libx264",
                    audio=False,
                    ffmpeg_params=["-preset", "ultrafast", "-crf", "28", "-tune", "zerolatency"],
                    logger=None,
                    threads=4
                )
            else:
                raise
        
        # Build FFmpeg command for final encoding with hardware acceleration
        # Use higher quality settings for final output
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", str(temp_video),
            "-i", str(audio_path),
            "-c:v", codec,
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest"
        ]
        cmd.extend(codec_params)
        cmd.append(str(output_path))
        
        logger.info(f"Encoding final video with {codec}...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Video encoding complete")
            return output_path
        except subprocess.CalledProcessError as e:
            # If hardware encoding fails, fall back to software encoding
            if codec != "libx264":
                logger.warning(f"Hardware encoding with {codec} failed: {e.stderr}")
                logger.info("Falling back to software encoding (libx264)...")
                
                # Retry with software encoding
                cmd_software = [
                    ffmpeg_path,
                    "-y",
                    "-i", str(temp_video),
                    "-i", str(audio_path),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    str(output_path)
                ]
                subprocess.run(cmd_software, check=True, capture_output=True, text=True)
                logger.info("Video encoding complete (using software encoder)")
                return output_path
            else:
                # Re-raise if software encoding also fails
                raise
        
    finally:
        # Cleanup temporary file
        if temp_video.exists():
            temp_video.unlink()


# This is the "callable" version 
def render_video(
    audio_path: Path,
    timestamps_path: Path,
    output_path: Path,
    background_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    font_size: Optional[int] = None
) -> Path:
    """
    Renders the final karaoke-style video using batch processing for faster rendering.
    
    Args:
        audio_path: Path to the PROCESSED audio file.
        timestamps_path: Path to the timestamps.json file.
        output_path: Path to save the final .mp4 video.
        background_path: Optional custom background image path (for reels/shorts).
        width: Optional custom video width (for reels/shorts).
        height: Optional custom video height (for reels/shorts).
        font_size: Optional custom font size (for reels/shorts - smaller text).

    Returns:
        The path to the rendered video.
    """
    logger.info("--- Starting Video Rendering Pipeline (Dynamic Frame Generation) ---")
    
    temp_bg_path = None
    
    try:
        logger.info(f"Loading audio: {audio_path}")
        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration
        
        # --- Load background/config from settings or use custom values ---
        if background_path is None:
            background_path = Path(settings.DEFAULT_BACKGROUND)
        else:
            background_path = Path(background_path)
        
        fps = settings.VIDEO_FPS
        
        # Load background image FIRST to get actual dimensions (source of truth for reels)
        bg_image = Image.open(str(background_path))
        actual_width, actual_height = bg_image.size
        
        # Use provided dimensions or actual image dimensions
        if width is None:
            width = actual_width
        if height is None:
            height = actual_height
        
        logger.info(f"Loading background: {background_path}")
        logger.info(f"Background image dimensions: {actual_width}x{actual_height}")
        logger.info(f"Using video dimensions: {width}x{height}")
        
        # Resize background image if dimensions don't match
        if bg_image.size != (width, height):
            logger.info(f"Resizing background from {bg_image.size} to ({width}, {height})")
            bg_image = bg_image.resize((width, height), Image.Resampling.LANCZOS)

        # Detect if this is for reels (check if background_path was provided and dimensions are vertical)
        is_reels = background_path is not None and background_path != Path(settings.DEFAULT_BACKGROUND)
        if is_reels:
            # Double-check: if height > width, it's definitely reels
            is_reels = height > width
        
        # Initialize frame generator with optional custom font size and reels mode
        frame_gen = FrameGeneratorV11(
            timestamps_path=timestamps_path,
            bg_width=width,
            bg_height=height,
            font_size=font_size,
            is_reels=is_reels
        )
        
        # CRITICAL: Verify timestamp alignment
        # Check if first word starts at 0 or has an offset
        if frame_gen.all_words:
            first_word_start = frame_gen.all_words[0].start
            if first_word_start > 0.1:  # More than 100ms offset
                logger.warning(f"First word starts at {first_word_start:.3f}s (not at 0.0s). This may cause sync issues.")
            else:
                logger.info(f"First word starts at {first_word_start:.3f}s - timestamp alignment looks good")
        
        # Check if last word ends close to audio duration
        if frame_gen.all_words:
            last_word_end = frame_gen.all_words[-1].end
            if abs(last_word_end - audio_duration) > 0.5:  # More than 500ms difference
                logger.warning(f"Last word ends at {last_word_end:.3f}s but audio duration is {audio_duration:.3f}s. Difference: {abs(last_word_end - audio_duration):.3f}s")
            else:
                logger.info(f"Last word ends at {last_word_end:.3f}s, audio duration is {audio_duration:.3f}s - alignment looks good")

        # ====================================================================
        # PHASE 1: Direct Video Generation (No Frame Files - Pipe to FFmpeg)
        # ====================================================================
        logger.info("--- Phase 1: Setting up direct video generation (piping frames to FFmpeg) ---")
        
        # CRITICAL: Optimization already ran in _build_grouped_slides, so we use optimized layout
        # We'll generate frames on-the-fly and pipe directly to FFmpeg (no disk files)
        
        import subprocess
        import math
        import io
        
        ffmpeg_path = _get_ffmpeg_path()
        codec, codec_params = _detect_hardware_codec()
        
        # Create a function to generate frame at timestamp
        def generate_frame_at_timestamp(t: float) -> Image.Image:
            """
            Generate a single frame at timestamp t using optimized layout.
            Returns PIL Image (RGBA) with text overlay on background.
            """
            # Find which slide this timestamp belongs to
            # CRITICAL: At transition boundaries, we want the NEW slide, not the old one
            slide_index = len(frame_gen.slide_start_times) - 1  # Default to last slide
            epsilon = 1e-5  # Very small epsilon to handle floating point precision
            
            # Find the slide that should be displayed at this timestamp
            # We iterate forward and select the slide whose start time is <= timestamp
            # and whose end time is > timestamp (for non-final slides)
            for i in range(len(frame_gen.slide_start_times)):
                start_time = frame_gen.slide_start_times[i]
                slide_end_time = frame_gen.slide_end_times[i] if i < len(frame_gen.slide_end_times) else audio_duration
                
                # Check if timestamp is before this slide starts
                if t < start_time - epsilon:
                    # We've passed all slides that could contain this timestamp
                    # Use the previous slide (i-1) or slide 0 if this is the first slide
                    if i > 0:
                        slide_index = i - 1
                    else:
                        slide_index = 0
                    break
                
                # Check if timestamp is within this slide's range
                if i == len(frame_gen.slide_start_times) - 1:
                    # Last slide: extends to end of audio
                    if start_time - epsilon <= t:
                        slide_index = i
                        break
                else:
                    # Non-final slides: timestamp must be < end_time to belong to this slide
                    # At the exact boundary (t >= slide_end_time), we want the NEXT slide
                    if start_time - epsilon <= t < slide_end_time - epsilon:
                        slide_index = i
                        break
                    # If t >= slide_end_time, continue to next slide (this slide has ended)
            
            # Ensure slide_index is valid
            if slide_index < 0:
                slide_index = 0
            if slide_index >= len(frame_gen.slides):
                slide_index = len(frame_gen.slides) - 1
            
            # Create frame with background (copy to avoid modifying original)
            frame = bg_image.copy().convert("RGBA")
            draw = ImageDraw.Draw(frame)
            
            slide_lines = frame_gen.slides[slide_index]
            layout = frame_gen.slide_layouts.get(slide_index, {})
            max_x = width - frame_gen.right_margin
            
            # CRITICAL: Validate slide has lines before rendering
            if not slide_lines:
                logger.error(f"Slide {slide_index} has no lines! This should not happen.")
                return frame
            
            # CRITICAL: Log slide structure for debugging
            total_words_in_slide = sum(len(line) for line in slide_lines)
            logger.debug(
                f"Rendering slide {slide_index}: {len(slide_lines)} lines, {total_words_in_slide} words, "
                f"layout has {len(layout)} entries"
            )
            
            # Render all words in the slide using optimized layout
            # CRITICAL: Render ALL lines, even if they appear empty
            for line_idx, line in enumerate(slide_lines):
                # CRITICAL: Check if line is empty and log it
                if not line or len(line) == 0:
                    logger.warning(f"Slide {slide_index}, Line {line_idx} is empty! This may indicate missing words.")
                    continue  # Skip empty lines but log them
                
                for word_idx, word in enumerate(line):
                    # Use (line_idx, word_idx) as key to match layout keys
                    coords = layout.get((line_idx, word_idx))
                    if not coords:
                        # CRITICAL: Don't skip words without coordinates - render them at fallback position
                        # This ensures all words are visible even if layout calculation missed them
                        logger.warning(
                            f"Word '{word.word}' at line {line_idx}, word {word_idx} not found in layout for slide {slide_index} - "
                            f"using fallback position"
                        )
                        # Fallback position: use left margin and calculate y based on line index
                        x = frame_gen.left_margin  # Fixed left padding (150px)
                        y = frame_gen.top_margin + (line_idx * frame_gen.line_height)
                    else:
                        x, y = coords
                    
                    # Safety check: ensure word doesn't overflow screen
                    word_bbox = draw.textbbox((0, 0), word.word, font=frame_gen.bold_font)
                    word_width = word_bbox[2] - word_bbox[0]
                    
                    if x + word_width > max_x:
                        # Adjust position to prevent overflow
                        x = max(frame_gen.left_margin, max_x - word_width)
                    
                    # Word highlighting: bold when spoken, regular before
                    if t >= word.start and t < word.end:
                        # Word is currently being spoken - show as bold
                        font = frame_gen.bold_font
                        color = settings.TEXT_BOLD_COLOR
                    elif t >= word.end:
                        # Word has finished being spoken - keep it bold (already spoken)
                        font = frame_gen.bold_font
                        color = settings.TEXT_BOLD_COLOR
                    else:
                        # Word hasn't been spoken yet - show as faded/light
                        font = frame_gen.regular_font
                        color = settings.TEXT_REGULAR_COLOR
                    
                    draw.text((x, y), word.word, font=font, fill=color)
            
            return frame
        
        # ====================================================================
        # PHASE 2: Direct FFmpeg Encoding (Pipe Frames - No Disk Files)
        # ====================================================================
        logger.info("--- Phase 2: Encoding video directly with FFmpeg (no frame files) ---")
        logger.info(f"Rendering {fps}fps video to: {output_path}")
        
        # Calculate total frames
        total_frames = math.ceil(audio_duration * fps)
        frame_interval = 1.0 / fps
        
        # Build FFmpeg command to receive frames via pipe
        cmd = [
            ffmpeg_path,
            "-y",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-r", str(fps),
            "-i", "-",  # Read from stdin
            "-i", str(audio_path),  # Audio input
            "-c:v", codec,
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-r", str(fps)
        ]
        cmd.extend(codec_params)
        cmd.append(str(output_path))
        
        logger.info(f"Starting FFmpeg encoding with {codec}...")
        logger.info(f"Generating {total_frames} frames on-the-fly and piping to FFmpeg...")
        
        try:
            # Start FFmpeg process with stdin pipe
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            # Generate and pipe frames
            frames_generated = 0
            for frame_num in range(total_frames):
                t = frame_num * frame_interval
                if t > audio_duration:
                    t = audio_duration
                
                # Generate frame
                frame_img = generate_frame_at_timestamp(t)
                
                # Convert to RGB (FFmpeg handles RGB better via pipe)
                frame_rgb = frame_img.convert("RGB")
                
                # Save frame to BytesIO and pipe to FFmpeg
                frame_buffer = io.BytesIO()
                frame_rgb.save(frame_buffer, format="PNG")
                frame_data = frame_buffer.getvalue()
                
                # Write frame to FFmpeg stdin
                process.stdin.write(frame_data)
                process.stdin.flush()
                
                frames_generated += 1
                
                # Progress logging
                if frame_num % max(1, total_frames // 10) == 0:
                    progress = (frame_num / total_frames) * 100
                    logger.info(f"Generated {frame_num}/{total_frames} frames ({progress:.1f}%)")
            
            # Close stdin to signal end of input
            process.stdin.close()
            
            # Wait for FFmpeg to finish
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg encoding failed with return code {process.returncode}")
                logger.error(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr)
            
            logger.info(f"Successfully generated {frames_generated} frames and encoded video directly")
            
        except Exception as e:
            logger.error(f"Direct video encoding failed: {e}", exc_info=True)
            # Fallback: try software encoding
            if codec != "libx264":
                logger.warning("Hardware encoding failed, trying software fallback...")
                cmd_software = [
                    ffmpeg_path,
                    "-y",
                    "-f", "image2pipe",
                    "-vcodec", "png",
                    "-r", str(fps),
                    "-i", "-",
                    "-i", str(audio_path),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-shortest",
                    "-r", str(fps),
                    str(output_path)
                ]
                
                process = subprocess.Popen(
                    cmd_software,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )
                
                for frame_num in range(total_frames):
                    t = frame_num * frame_interval
                    if t > audio_duration:
                        t = audio_duration
                    
                    frame_img = generate_frame_at_timestamp(t)
                    frame_rgb = frame_img.convert("RGB")
                    
                    frame_buffer = io.BytesIO()
                    frame_rgb.save(frame_buffer, format="PNG")
                    frame_data = frame_buffer.getvalue()
                    
                    process.stdin.write(frame_data)
                    process.stdin.flush()
                
                process.stdin.close()
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd_software, stderr=stderr)
                
                logger.info("Video encoding complete (using software encoder)")
            else:
                raise
        
        logger.info("--- Video Rendering Complete ---")
        return output_path

    except Exception as e:
        logger.error("Video rendering pipeline failed!", exc_info=True)
        raise
    
    finally:
        # ====================================================================
        # PHASE 3: Cleanup
        # ====================================================================
        # Cleanup temporary background file if created
        if temp_bg_path and temp_bg_path.exists():
            try:
                temp_bg_path.unlink()
                logger.info(f"Cleaned up temporary background file: {temp_bg_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp background file: {e}")