"""
Chapter processing functionality for PDF-to-Video pipeline.
Handles chapter selection, summary generation, and chapter video creation.
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

from app.config import settings
from app.phase1_pdf_processing.text_cleaner import clean_text
from app.phase1_pdf_processing.image_extractor import extract_images
from app.phase2_ai_services.openai_client import OpenAIService, detect_book_genre
from app.phase3_audio_processing.mastering import master_audio
from app.phase4_video_generation.renderer import render_video

logger = logging.getLogger(__name__)


def extract_chapters_from_headings(extraction_json_path: Path) -> List[Dict]:
    """
    Extract chapters by detecting chapter headings in the text.
    Used as fallback when index is not available.
    
    Args:
        extraction_json_path: Path to the extraction JSON file
        
    Returns:
        List of chapter dictionaries with title and page_reference
    """
    try:
        with open(extraction_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pages = data.get("text_extraction", {}).get("pages", [])
        if not pages:
            return []
        
        chapters = []
        skip_keywords = ["title page", "copyright", "dedication", "epigraph", 
                        "contents", "notes", "suggestions", "about the authors",
                        "bibliography", "references", "index", "table of contents"]
        
        # Patterns for chapter headings - more strict to avoid false positives
        chapter_patterns = [
            # "Chapter 1", "Chapter One", "Chapter I" - must be at start of line
            (r'^chapter\s+([0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b[:\-]?\s*(.+)?$', True),
            # "I.", "II.", "III." (Roman numerals) - must be standalone or followed by short capitalized title
            (r'^([IVX]+)\.?\s+(.+)$', True),
            # "1.", "2.", "3." (Numbers) - must be at start of line
            (r'^(\d+)\.\s+(.+)$', True),
            # "Part I", "Part 1"
            (r'^part\s+([0-9]+|[ivxlcdm]+)\b[:\-]?\s*(.+)?$', True),
            # "Prologue", "Epilogue", "Introduction" - standalone or with subtitle
            (r'^(prologue|epilogue|introduction|preface|foreword)\b[:\-]?\s*(.+)?$', True),
        ]
        
        for page in pages:
            page_text = page.get("text", "")
            page_num = page.get("page_number")
            
            if not page_text:
                continue
            
            # Check first 5 lines of the page for chapter headings (more restrictive)
            lines = page_text.split('\n')[:5]
            
            for line_idx, line in enumerate(lines):
                line_clean = line.strip()
                if not line_clean or len(line_clean) < 3:
                    continue
                
                # Skip if it's too long (probably not a heading)
                if len(line_clean) > 100:
                    continue
                
                # Check against skip keywords
                if any(keyword in line_clean.lower() for keyword in skip_keywords):
                    continue
                
                # Chapter headings are usually at the very start of a page (first 2 lines)
                # or have specific formatting (all caps, centered, etc.)
                is_likely_heading = (
                    line_idx < 2 or  # First 2 lines of page
                    line_clean.isupper() or  # All caps
                    (line_clean[0].isupper() and len(line_clean.split()) <= 10)  # Starts with capital, short
                )
                
                if not is_likely_heading:
                    continue
                
                # Try to match chapter patterns
                for pattern, requires_heading_format in chapter_patterns:
                    match = re.match(pattern, line_clean, re.IGNORECASE)
                    if match:
                        # Extract chapter title
                        groups = match.groups()
                        if len(groups) >= 2 and groups[1]:
                            title = groups[1].strip()
                        elif len(groups) >= 1:
                            # For patterns like "Chapter 1" without subtitle, use the full line
                            if "chapter" in line_clean.lower() or "part" in line_clean.lower():
                                title = line_clean
                            else:
                                title = groups[0].strip()
                        else:
                            title = line_clean
                        
                        # Clean up title (remove extra punctuation, normalize)
                        title = re.sub(r'^[:\-\.\s]+|[:\-\.\s]+$', '', title)
                        title = " ".join(title.split())  # Normalize whitespace
                        
                        # Additional validation: title should look like a chapter title
                        # - Not too long (max 80 chars)
                        # - Not a full sentence (no ending punctuation except ? or !)
                        # - Not dialogue (doesn't start with quote)
                        if len(title) < 3 or len(title) > 80:
                            continue
                        
                        # Skip if it looks like dialogue or a sentence
                        if title.startswith(('"', "'", '"', "'")):
                            continue
                        
                        # Skip if it ends with period and is long (likely a sentence)
                        if title.endswith('.') and len(title) > 30:
                            continue
                        
                        # For Roman numeral patterns, ensure the title part is reasonable
                        if pattern.startswith(r'^([IVX]+)'):
                            # If the "title" part is very long or looks like regular text, skip
                            if len(groups) >= 2 and groups[1]:
                                subtitle = groups[1].strip()
                                # Skip if subtitle is too long or looks like body text
                                if len(subtitle) > 60 or (len(subtitle.split()) > 8):
                                    continue
                                # Skip if it looks like dialogue or a sentence (starts with common sentence starters)
                                subtitle_lower = subtitle.lower()
                                sentence_starters = ['i ', 'he ', 'she ', 'they ', 'we ', 'you ', 'it ', 'the ', 'a ', 'an ']
                                if any(subtitle_lower.startswith(starter) for starter in sentence_starters):
                                    # Only allow if it's very short (likely a title, not a sentence)
                                    if len(subtitle.split()) > 5:
                                        continue
                        
                        # Check if we already have this chapter (avoid duplicates)
                        if any(ch['title'].lower() == title.lower() for ch in chapters):
                            continue
                        
                        chapters.append({
                            "title": title,
                            "entry_number": groups[0] if groups else "",
                            "page_reference": page_num
                        })
                        logger.debug(f"Found chapter heading: '{title}' on page {page_num}")
                        break  # Found a match, move to next line
        
        logger.info(f"Extracted {len(chapters)} chapters from headings")
        return chapters
    
    except Exception as e:
        logger.error(f"Error extracting chapters from headings: {e}", exc_info=True)
        return []


def extract_chapters_from_index(extraction_json_path: Path) -> List[Dict]:
    """
    Extract chapters from the index in the extraction JSON.
    Falls back to heading detection if index is not available.
    
    Args:
        extraction_json_path: Path to the extraction JSON file
        
    Returns:
        List of chapter dictionaries with title, entry_number, and page_reference
    """
    try:
        with open(extraction_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        index_data = data.get("index")
        if not index_data or "entries" not in index_data:
            logger.warning("No index found in extraction JSON, trying heading detection...")
            # Fallback to heading detection
            return extract_chapters_from_headings(extraction_json_path)
        
        chapters = []
        for entry in index_data["entries"]:
            # Filter out non-chapter entries (like "Title Page", "Copyright", etc.)
            title = entry.get("title", "").strip()
            entry_num = entry.get("entry_number", "").strip()
            
            # Skip common non-chapter entries
            skip_keywords = ["title page", "copyright", "dedication", "epigraph", 
                           "contents", "notes", "suggestions", "about the authors",
                           "bibliography", "references", "index"]
            
            if any(keyword in title.lower() for keyword in skip_keywords):
                continue
            
            # Only include entries that look like chapters
            if title and len(title) > 3:
                chapters.append({
                    "title": title,
                    "entry_number": entry_num,
                    "page_reference": entry.get("page_reference")
                })
        
        if chapters:
            logger.info(f"Extracted {len(chapters)} chapters from index")
            return chapters
        else:
            logger.warning("Index found but no valid chapters extracted, trying heading detection...")
            # Fallback to heading detection if index has no valid chapters
            return extract_chapters_from_headings(extraction_json_path)
    
    except Exception as e:
        logger.error(f"Error extracting chapters from index: {e}", exc_info=True)
        # Fallback to heading detection on error
        logger.info("Falling back to heading detection...")
        return extract_chapters_from_headings(extraction_json_path)


def find_chapter_page_range(chapter_title: str, extraction_json_path: Path) -> Optional[Tuple[int, int]]:
    """
    Find the page range for a chapter by searching for its title in the text.
    
    Args:
        chapter_title: Title of the chapter to find
        extraction_json_path: Path to the extraction JSON file
        
    Returns:
        Tuple of (start_page, end_page) or None if not found
    """
    try:
        with open(extraction_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pages = data.get("text_extraction", {}).get("pages", [])
        if not pages:
            return None
        
        # Search for chapter title in pages
        start_page = None
        end_page = None
        
        # Normalize chapter title for matching (remove extra whitespace, lowercase)
        normalized_title = " ".join(chapter_title.lower().split())
        
        for i, page in enumerate(pages):
            page_text = page.get("text", "").lower()
            page_num = page.get("page_number")
            
            # Check if chapter title appears in this page
            if normalized_title in " ".join(page_text.split()):
                if start_page is None:
                    start_page = page_num
                    logger.info(f"Found chapter '{chapter_title}' starting at page {start_page}")
            
            # If we found start page, look for next chapter or end of book
            if start_page is not None and page_num > start_page:
                # Check if this page starts a new chapter (has roman numerals or "Chapter")
                page_lines = page_text.split('\n')[:5]  # Check first few lines
                for line in page_lines:
                    line_clean = line.strip()
                    # Look for patterns like "II.", "III.", "Chapter", etc.
                    if (line_clean and len(line_clean) < 100 and 
                        (any(roman in line_clean[:10] for roman in ["II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X."]) or
                         "chapter" in line_clean[:20].lower())):
                        end_page = page_num - 1
                        break
        
        if start_page and not end_page:
            # Use last page as end if no next chapter found
            end_page = pages[-1].get("page_number")
        
        if start_page and end_page:
            return (start_page, end_page)
        
        return None
    
    except Exception as e:
        logger.error(f"Error finding chapter page range: {e}", exc_info=True)
        return None


def generate_chapter_summary(chapter_text: str, chapter_title: str) -> str:
    """
    Generate a summary of a chapter using OpenAI GPT.
    Handles very long chapters by truncating to fit token limits.
    
    Args:
        chapter_text: Full text of the chapter
        chapter_title: Title of the chapter
        
    Returns:
        Generated summary text
    """
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Estimate tokens (roughly 1 token = 4 characters for English)
        # GPT-4o-mini has 128k context, but we need to leave room for prompt and response
        # Use max ~100k tokens for input to be safe
        max_input_chars = 100000 * 4  # ~400k characters
        
        # Truncate chapter text if too long
        truncated_text = chapter_text
        if len(chapter_text) > max_input_chars:
            logger.warning(f"Chapter text is too long ({len(chapter_text)} chars), truncating to {max_input_chars} chars")
            # Take first part and last part to preserve beginning and end
            first_part = chapter_text[:max_input_chars // 2]
            last_part = chapter_text[-max_input_chars // 2:]
            truncated_text = first_part + "\n\n[... content truncated ...]\n\n" + last_part
        
        prompt = f"""Please provide a comprehensive summary of the following chapter from a book.

Chapter Title: {chapter_title}

Chapter Content:
{truncated_text}

Please provide a detailed summary that includes:
1. Main themes and key concepts
2. Important points and insights
3. Key takeaways
4. How it relates to the overall book topic

Summary:"""

        logger.info(f"Generating summary for chapter: {chapter_title} (text length: {len(truncated_text)} chars)")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates comprehensive, well-structured chapter summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        summary = response.choices[0].message.content.strip()
        logger.info(f"Summary generated successfully ({len(summary)} characters)")
        return summary
    
    except Exception as e:
        logger.error(f"Error generating chapter summary: {e}", exc_info=True)
        raise


def process_chapter_video(
    chapter_title: str,
    chapter_text: str,
    pdf_path: Path,
    main_job_dir: Path,
    genre: str,
    voice: str
) -> Optional[Path]:
    """
    Process a single chapter through the full video generation pipeline.
    
    Args:
        chapter_title: Title of the chapter
        chapter_text: Text content of the chapter
        pdf_path: Path to the original PDF
        main_job_dir: Main job directory (for reference)
        genre: Book genre for voice instructions
        voice: Voice to use for TTS
        
    Returns:
        Path to the generated video, or None if failed
    """
    try:
        from datetime import datetime
        
        # Create chapter-specific job directory
        safe_chapter_name = "".join(c for c in chapter_title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        chapter_job_id = f"{main_job_dir.name}_chapter_{safe_chapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        chapter_job_dir = main_job_dir.parent / chapter_job_id
        chapter_job_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing chapter video: {chapter_title}")
        logger.info(f"Chapter job directory: {chapter_job_dir}")
        
        # Save chapter text
        chapter_text_path = chapter_job_dir / "chapter_text.txt"
        with open(chapter_text_path, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
        
        # Clean chapter text
        tables_dir = main_job_dir / "tables"  # Reuse tables from main job
        images_dir = main_job_dir / "images"  # Reuse images from main job
        
        cleaned_text_path = clean_text(
            raw_text_path=chapter_text_path,
            tables_dir=tables_dir,
            images_dir=images_dir,
            job_dir=chapter_job_dir
        )
        
        with open(cleaned_text_path, 'r', encoding='utf-8') as f:
            cleaned_text = f.read()
        
        if not cleaned_text.strip():
            logger.error("Cleaned chapter text is empty")
            return None
        
        # Generate audio and timestamps
        # Ensure we use "onyx" voice for OpenAI (default if not specified)
        voice_to_use = voice if voice else "onyx"
        logger.info(f"Using OpenAI voice: {voice_to_use} for chapter: {chapter_title}")
        openai_service = OpenAIService(voice=voice_to_use)
        raw_audio_path, timestamps_path = openai_service.generate_audio_with_timestamps(
            text=cleaned_text,
            output_dir=chapter_job_dir,
            job_id=chapter_job_id,
            genre=genre
        )
        
        # Master audio
        processed_audio_path = chapter_job_dir / f"{chapter_job_id}_processed_audio.mp3"
        processed_audio_path = master_audio(
            raw_audio_path=raw_audio_path,
            processed_audio_path=processed_audio_path
        )
        
        # Generate video
        final_video_path = chapter_job_dir / f"{chapter_job_id}_final_video.mp4"
        final_video_path = render_video(
            audio_path=processed_audio_path,
            timestamps_path=timestamps_path,
            output_path=final_video_path
        )
        
        logger.info(f"Chapter video generated successfully: {final_video_path}")
        return final_video_path
    
    except Exception as e:
        logger.error(f"Error processing chapter video: {e}", exc_info=True)
        return None


def get_chapter_text_from_pages(start_page: int, end_page: int, extraction_json_path: Path) -> str:
    """
    Extract text for a chapter from the extraction JSON based on page range.
    
    Args:
        start_page: Starting page number
        end_page: Ending page number
        extraction_json_path: Path to the extraction JSON file
        
    Returns:
        Combined text from the chapter pages
    """
    try:
        with open(extraction_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pages = data.get("text_extraction", {}).get("pages", [])
        chapter_text_parts = []
        
        for page in pages:
            page_num = page.get("page_number")
            if start_page <= page_num <= end_page:
                chapter_text_parts.append(page.get("text", ""))
        
        return "\n\n".join(chapter_text_parts)
    
    except Exception as e:
        logger.error(f"Error extracting chapter text: {e}", exc_info=True)
        return ""

