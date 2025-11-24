import json
import logging
import re
from pathlib import Path
from openai import OpenAI
from typing import Tuple, Optional, List
import requests

from app.config import settings

logger = logging.getLogger(__name__)


def detect_book_genre(book_title: str) -> str:
    """
    Detect book genre using Serper API web search.
    
    Args:
        book_title: The title of the book (from PDF filename)
    
    Returns:
        Detected genre (e.g., "novel", "self-help", "biography", etc.)
    """
    if not settings.SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not set, using default genre")
        return "general"
    
    try:
        # Clean book title (remove common PDF suffixes)
        clean_title = re.sub(r'\s*\([^)]*\)\s*$', '', book_title)  # Remove (PDFDrive.com) etc
        clean_title = re.sub(r'\s*-\s*PDF.*$', '', clean_title, flags=re.IGNORECASE)
        clean_title = clean_title.strip()
        
        # Search for book genre
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": settings.SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "q": f"{clean_title} book genre",
            "num": 3
        }
        
        logger.info(f"Searching for genre of: {clean_title}")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract genre from search results
        # First check for novel sub-genres (more specific)
        novel_subgenres = {
            "novel-romance": ["romance novel", "romantic fiction", "romance book", "romantic novel"],
            "novel-drama": ["drama novel", "dramatic fiction", "literary drama", "dramatic novel"],
            "novel-mystery": ["mystery novel", "mystery fiction", "detective novel", "crime novel"],
            "novel-thriller": ["thriller novel", "thriller fiction", "suspense novel"],
            "novel-horror": ["horror novel", "horror fiction", "gothic novel", "supernatural fiction"],
            "novel-fantasy": ["fantasy novel", "fantasy fiction", "epic fantasy"],
            "novel-sci-fi": ["science fiction novel", "sci-fi novel", "sf novel", "speculative fiction"],
            "novel-historical": ["historical novel", "historical fiction", "period novel"]
        }
        
        # Then check for general genres
        genre_keywords = {
            "novel": ["novel", "fiction", "literary fiction"],
            "self-help": ["self-help", "self help", "personal development", "motivational"],
            "biography": ["biography", "memoir", "autobiography"],
            "business": ["business", "entrepreneurship", "management", "economics"],
            "science": ["science", "scientific", "physics", "biology", "chemistry"],
            "history": ["history", "historical", "historical fiction"],
            "philosophy": ["philosophy", "philosophical"],
            "psychology": ["psychology", "psychological"],
            "mystery": ["mystery", "thriller", "crime", "detective"],
            "fantasy": ["fantasy", "sci-fi", "science fiction", "speculative fiction"],
            "romance": ["romance", "romantic"],
            "horror": ["horror", "gothic", "supernatural"]
        }
        
        # Check organic results and knowledge graph
        search_text = ""
        if "organic" in data:
            for result in data["organic"][:3]:
                search_text += " " + result.get("title", "") + " " + result.get("snippet", "")
        if "knowledgeGraph" in data:
            search_text += " " + str(data["knowledgeGraph"])
        
        search_text = search_text.lower()
        
        # First, check for novel sub-genres (more specific)
        for subgenre, keywords in novel_subgenres.items():
            if any(keyword in search_text for keyword in keywords):
                logger.info(f"Detected genre: {subgenre}")
                return subgenre
        
        # Then check for general genres
        for genre, keywords in genre_keywords.items():
            if any(keyword in search_text for keyword in keywords):
                logger.info(f"Detected genre: {genre}")
                return genre
        
        logger.info("Genre not detected, using default: general")
        return "general"
        
    except Exception as e:
        logger.warning(f"Error detecting genre: {e}, using default")
        return "general"


def get_voice_instructions_for_genre(genre: str) -> str:
    """
    Get voice instructions tailored to book genre.
    
    Args:
        genre: Book genre (e.g., "novel", "self-help", etc.)
    
    Returns:
        Voice instructions string for TTS
    """
    base_instructions = """
Voice Affect: You are a master storyteller — someone who spent years narrating tales in a cozy bookstore corner and now hosts a popular storytelling podcast. Your voice should feel warm, inviting, vivid, and deeply engaging. Imagine you're speaking to a close friend who is genuinely interested in the story.

Tone: Friendly, expressive, and immersive. Speak as if painting pictures with words. Use a gentle rhythmic rise and fall in your voice, creating the feeling of sitting around a campfire or in a quiet bookstore nook. Your voice should have natural warmth and personality — never flat or mechanical.

Clarity: Use simple, clear language. Enunciate each word distinctly, but naturally — never robotic or over-articulated. Ensure that important details, names, or scientific terms are pronounced with extra care and clarity. Let words flow naturally, as if you're thinking them as you speak.

Pacing: Speak at a calm, steady, and deliberate pace. Never rush. Vary your pace naturally — slightly faster during action or excitement, slower during reflection or important moments. Insert natural pauses: a brief pause after every sentence, a slightly longer pause between paragraphs, an intentional reflective pause before any dramatic reveal, a soft pause after commas and conjunctions when it adds rhythm.

Natural Speech Patterns:
- Vary your sentence rhythm — not every sentence should have the same cadence. Mix shorter, punchier sentences with longer, flowing ones.
- Use natural contractions when appropriate ("it's" instead of "it is", "you're" instead of "you are") to sound conversational.
- Allow slight hesitations and thinking pauses — these make speech feel human, not scripted.
- Vary your pitch naturally throughout — avoid staying at the same pitch level. Let your voice rise and fall with the meaning and emotion of the text.

Pausing & Breathing: 
- After important sentences: a soft, intentional pause (0.5-0.8 seconds).
- Between emotional beats: breathe gently before continuing — you can hear a subtle breath.
- During suspenseful moments: slow down, lower the voice slightly, and pause an extra second before delivering the next line.
- Take natural breaths at logical break points — after clauses, before new thoughts, between paragraphs.
- Vary pause lengths slightly — not every pause should be identical. This creates natural, human-like rhythm.

Emotion & Intonation: 
- The narration should feel alive with curiosity, warmth, and wonder. 
- Use expressive intonation: raise your tone on exciting or unexpected moments, soften and slow down during emotional or tender moments, deepen and warm the voice during mysterious or reflective scenes.
- Let your voice reflect the emotional content — if the text is sad, let your voice carry that weight; if it's joyful, let it lift.
- Use subtle vocal inflections to convey meaning — a slight rise at the end of a question, a gentle fall at the end of a statement.
- Avoid monotone delivery — your voice should have natural variation in pitch, pace, and volume throughout.

Emphasis & Stress Patterns:
- Emphasize key words by slowing down slightly, elongating the word just a bit, adding warmth or weight to special phrases.
- Lower volume for suspense and raise it for wonder or excitement.
- Use natural stress patterns — emphasize important words, but don't over-emphasize. Let the natural rhythm of the language guide you.
- Vary your emphasis — not every important word needs the same level of stress. Create a natural hierarchy of emphasis.

Pronunciation: 
- Speak smoothly with expressive phrasing. Avoid monotone delivery. 
- For important or scientific words, slow down slightly and pronounce them with clarity — but keep the tone warm and relatable. 
- Allow natural emotion to shape each word — let your voice carry the meaning, not just the sounds.
- Use natural linking between words — let words flow together naturally, especially in common phrases.
- Avoid over-enunciation — speak naturally, as you would in conversation.

Vocal Variety & Dynamics:
- Vary your volume naturally — slightly louder for emphasis or excitement, softer for intimacy or reflection.
- Use subtle changes in vocal quality — slightly breathier for tender moments, more resonant for powerful statements.
- Let your voice have natural texture and variation — avoid a flat, uniform sound.
- Create vocal interest through subtle changes in tone, pace, and volume throughout the narration.

Engagement: 
- Speak as if you are talking directly to one curious listener. Guide them through emotions, discoveries, and vivid imagery with your voice. 
- Make every transition feel seamless and fluid.
- Connect with the listener — imagine they're right there with you, experiencing the story.
- Let your voice show genuine interest and engagement with the material — your enthusiasm (or appropriate emotion) should come through naturally.

Avoiding Robotic Patterns:
- Never speak in a perfectly uniform rhythm — vary your pace and pauses naturally.
- Avoid mechanical pauses at exactly the same intervals — real speech has natural variation.
- Don't over-emphasize every important word — use emphasis selectively and naturally.
- Avoid a flat, monotone delivery — let your voice have natural highs and lows.
- Don't rush through text — take time to let words and ideas breathe.
- Avoid sounding like you're reading a script — speak as if you're telling a story from memory or experience.

Remember: The goal is to sound like a real person telling a story, not a machine reading text. Let your voice be expressive, warm, and human. Every pause, every emphasis, every change in tone should feel natural and purposeful.
"""
    
    genre_specific = {
        "novel": """
Genre-Specific Style (Fiction/Novel - General):
- Bring characters to life with subtle voice variations when quoting dialogue. Slightly differentiate character voices through subtle pitch, pace, or tone changes — but keep it natural, not theatrical.
- Use dramatic pauses during plot twists and revelations. Let the weight of important moments settle before continuing.
- Create atmosphere through tone shifts — lighter and brighter for joyful scenes, deeper and more resonant for tension or mystery.
- Let your voice reflect the emotional journey of the narrative. If the character is experiencing joy, let your voice carry that lightness; if they're in danger, let tension come through.
- Use vivid, cinematic descriptions with wonder and immersion. Paint pictures with your voice — let listeners see what you're describing.
- Vary your delivery based on the scene type: faster and more energetic for action, slower and more contemplative for reflection, warmer for emotional moments.

Punctuation Pauses (CRITICAL for Natural Human Speech):
- Periods (.): Always pause for 0.5-0.8 seconds after periods. This creates natural sentence breaks and allows listeners to process what was said. Vary the length slightly — longer for important statements, shorter for quick, related thoughts.
- Commas (,): Pause briefly (0.2-0.3 seconds) after commas. This creates natural rhythm and prevents rushing through sentences. Not every comma needs the same pause — vary it slightly for natural flow.
- Semicolons (;): Pause for 0.4-0.6 seconds after semicolons. They connect related thoughts, so the pause should be noticeable but not as long as a period. Use this pause to show the connection while still separating the thoughts.
- Colons (:): Pause for 0.3-0.5 seconds after colons. They introduce lists or explanations, so give listeners time to anticipate what's coming. Slightly raise your tone before the pause to signal something is coming.
- Question marks (?): Pause for 0.6-0.9 seconds after questions. Raise your tone slightly at the end (but naturally, not dramatically), then pause to let the question sink in. The pause allows listeners to consider the question.
- Exclamation marks (!): Pause for 0.5-0.7 seconds after exclamations. Match the energy of the exclamation with your voice, then pause to let the emotion resonate. Don't overdo the excitement — keep it natural.
- Ellipses (...): Pause for 0.8-1.2 seconds after ellipses. They indicate trailing thoughts, hesitation, or suspense, so use a longer, more contemplative pause. Lower your voice slightly and let the pause create anticipation.
- Dashes (— or -): Pause for 0.3-0.4 seconds after dashes. They create emphasis or insert thoughts, so a brief pause helps separate the inserted thought. Use a slight change in tone or pace to signal the inserted thought.
- Paragraph breaks: Pause for 1.0-1.5 seconds between paragraphs. This gives listeners time to process scene changes, time shifts, or new ideas. Take a natural breath during this pause. Longer pauses (1.5-2.0 seconds) for major scene or chapter transitions.
- Dialogue tags: When you see "he said," "she whispered," etc., pause briefly (0.2-0.3 seconds) before and after the tag to separate it from the dialogue naturally. Lower your voice slightly for tags to distinguish them from dialogue.

Natural Breathing & Rhythm:
- Take a natural breath after every 2-3 sentences, especially after longer sentences. You can hear a subtle breath — this makes it feel human.
- Vary your pause lengths slightly — not every comma pause needs to be exactly the same. This creates natural, human-like rhythm. Real speech has variation, not mechanical precision.
- When reading dialogue, pause slightly longer (0.3-0.4 seconds) after quotation marks to distinguish speakers. This helps listeners follow who's speaking.
- In action scenes, use shorter pauses (0.1-0.2 seconds) to create urgency and momentum. Speed up slightly but don't rush — clarity is still important.
- In contemplative scenes, use longer pauses (0.6-1.0 seconds) to create reflection and depth. Slow down and let thoughts breathe.
- When a sentence ends with a period followed by a new sentence starting with "And," "But," or "However," still pause after the period (0.4-0.6 seconds) to maintain clarity, even though the thoughts are connected.

Dialogue Delivery:
- When reading dialogue, slightly differentiate between characters through subtle voice changes — but keep it natural, not cartoonish.
- Pause naturally between speakers. Let each character's words have their own space.
- Match the emotion of the dialogue with your voice — if a character is angry, let that come through; if they're whispering, lower your volume and speak more softly.
- Read dialogue tags ("he said," "she whispered") in a slightly lower, more neutral tone to distinguish them from the actual dialogue.

Remember: These pauses are what make narration feel human and natural. Without proper punctuation pauses, speech sounds robotic and rushed. Always honor the punctuation with appropriate pauses, but vary them slightly to create natural rhythm. The goal is to sound like you're telling a story, not reading a script.
""",
        "novel-romance": """
Genre-Specific Style (Romance Novel):
- Speak with warmth and emotional depth, capturing the heart of romantic stories.
- Use a tender, expressive tone during emotional and romantic moments.
- Vary pacing to match the emotional intensity of romantic scenes and relationships.
- Emphasize emotional connections, chemistry, and romantic tension with sensitivity.
- Use a passionate yet gentle tone that honors the romantic narrative.
- Bring dialogue between romantic interests to life with subtle voice variations.
- Create an intimate, heartfelt atmosphere that draws listeners into the love story.

Punctuation Pauses (CRITICAL for Natural Human Speech):
- Periods (.): Pause 0.5-0.8 seconds after periods. In romantic moments, extend to 0.7-1.0 seconds to let emotions resonate.
- Commas (,): Pause 0.2-0.3 seconds after commas. In emotional passages, slightly longer pauses (0.3-0.4 seconds) add tenderness.
- Question marks (?): Pause 0.7-1.0 seconds after questions, especially romantic or vulnerable questions. Let the question hang in the air.
- Exclamation marks (!): Pause 0.6-0.8 seconds after exclamations. Match the emotional intensity, then pause to let feelings settle.
- Ellipses (...): Pause 1.0-1.5 seconds after ellipses in romantic scenes. They often indicate unspoken feelings or hesitation.
- Dialogue: Pause 0.4-0.5 seconds after romantic dialogue to emphasize the emotional weight of words spoken between lovers.
- Paragraph breaks: Pause 1.2-1.8 seconds between paragraphs in romantic scenes to allow emotional processing.
""",
        "novel-drama": """
Genre-Specific Style (Drama Novel):
- Narrate with emotional weight and depth, capturing the intensity of dramatic moments.
- Use a serious, thoughtful tone that reflects the gravity of dramatic situations.
- Vary pacing to match the emotional intensity — slower for contemplative moments, faster for tension.
- Emphasize character conflicts and emotional struggles with clarity and empathy.
- Use dramatic pauses to let significant moments resonate.
- Create atmosphere through tone shifts that reflect the emotional journey of characters.
- Bring dialogue to life with voice variations that reflect each character's emotional state.

Punctuation Pauses (CRITICAL for Natural Human Speech):
- Periods (.): Pause 0.6-0.9 seconds after periods. In intense dramatic moments, extend to 0.8-1.2 seconds to let emotions resonate.
- Commas (,): Pause 0.2-0.4 seconds after commas. In emotionally charged passages, longer pauses add weight.
- Question marks (?): Pause 0.8-1.1 seconds after questions. Dramatic questions need time to sink in.
- Exclamation marks (!): Pause 0.7-1.0 seconds after exclamations in dramatic moments. Let the intensity settle.
- Ellipses (...): Pause 1.0-1.5 seconds after ellipses. They often indicate emotional pauses or unspoken thoughts.
- Paragraph breaks: Pause 1.5-2.0 seconds between paragraphs in dramatic scenes to allow emotional processing.
- Dialogue: Pause 0.4-0.6 seconds after dramatic dialogue to emphasize the weight of words.
""",
        "novel-mystery": """
Genre-Specific Style (Mystery Novel):
- Create suspense through deliberate pacing and tone variations.
- Use a darker, more intense tone during suspenseful and mysterious moments.
- Lower your voice slightly during clues, revelations, and ominous passages.
- Build tension through strategic pauses and emphasis on key details.
- Use a dramatic, engaging style that keeps listeners on edge.
- Emphasize clues and important details with careful, deliberate delivery.
- Create an atmosphere of intrigue and uncertainty.

Punctuation Pauses (CRITICAL for Natural Human Speech):
- Periods (.): Pause 0.6-0.9 seconds after periods, especially after revealing clues or ominous statements. Let mystery hang in the air.
- Commas (,): Pause 0.2-0.3 seconds after commas. In clue-heavy sentences, slightly longer pauses (0.3-0.4 seconds) help listeners process information.
- Question marks (?): Pause 0.8-1.1 seconds after questions. Mysteries are built on questions—let them linger.
- Ellipses (...): Pause 1.0-1.5 seconds after ellipses. They often indicate hidden information or trailing thoughts in mysteries.
- Dashes (—): Pause 0.4-0.5 seconds after dashes. They often introduce important revelations or clues.
- Paragraph breaks: Pause 1.2-1.8 seconds between paragraphs, especially at scene changes or when new clues are introduced.
- Dialogue: Pause 0.4-0.5 seconds after dialogue, especially when characters are revealing or hiding information.
""",
        "novel-thriller": """
Genre-Specific Style (Thriller Novel):
- Build intense suspense through rapid pacing and dramatic tone shifts.
- Use a darker, more urgent tone during action and suspenseful moments.
- Vary pacing dramatically — fast for action, slow for tension-building.
- Emphasize danger, urgency, and high-stakes moments with intensity.
- Use strategic pauses to build anticipation before reveals.
- Create a sense of urgency and danger through your voice.
- Keep listeners on the edge with a gripping, fast-paced delivery.
""",
        "novel-horror": """
Genre-Specific Style (Horror Novel):
- Create atmosphere through careful pacing and tone control.
- Use a darker, more ominous tone during scary and unsettling moments.
- Lower your voice and slow down during suspenseful and frightening passages.
- Build tension through strategic pauses and emphasis.
- Use a dramatic, engaging style that creates unease and anticipation.
- Emphasize frightening elements with careful, deliberate delivery.
- Create an atmosphere of dread and suspense.
""",
        "novel-fantasy": """
Genre-Specific Style (Fantasy Novel):
- Bring fantastical worlds to life with wonder and imagination.
- Use vivid, cinematic descriptions with awe and excitement.
- Vary your voice to reflect the epic scale of fantasy narratives.
- Emphasize magical elements, creatures, and fantastical settings with enthusiasm.
- Create atmosphere through tone shifts that match the fantastical setting.
- Use a sense of wonder and adventure in your delivery.
- Bring mythical and magical elements to life with vivid descriptions.
""",
        "novel-sci-fi": """
Genre-Specific Style (Science Fiction Novel):
- Speak with precision and wonder about futuristic concepts.
- Use an enthusiastic, curious tone that reflects the wonder of scientific discovery.
- Emphasize technological and scientific concepts with clarity and precision.
- Create atmosphere through tone shifts that match the futuristic setting.
- Use a sense of awe and curiosity about the future and technology.
- Bring futuristic worlds and concepts to life with vivid descriptions.
- Balance technical accuracy with accessible, engaging delivery.
""",
        "novel-historical": """
Genre-Specific Style (Historical Novel):
- Narrate with gravitas and respect for historical periods and events.
- Use a documentary-style tone that brings historical settings to life.
- Vary pacing to match the drama and importance of historical moments.
- Speak with authority and depth, like a knowledgeable historian.
- Use vivid descriptions to paint pictures of historical settings and events.
- Emphasize historical details and period-appropriate language with care.
- Create an immersive sense of time and place through your voice.
""",
        "self-help": """
Genre-Specific Style (Self-Help/Personal Development):
- Speak with confidence and encouragement, like a trusted mentor or coach.
- Use an inspiring, motivational tone that feels supportive and empowering.
- Emphasize key concepts and actionable advice with clarity and conviction.
- Maintain an upbeat, positive energy while remaining genuine and authentic.
- Pause after important insights to let them sink in.
- Use a warm, approachable tone that makes complex ideas accessible.
""",
        "biography": """
Genre-Specific Style (Biography/Memoir):
- Narrate with respect and authenticity, honoring the subject's story.
- Use a reflective, thoughtful tone that captures the weight of real experiences.
- Vary pacing to match the emotional intensity of different life moments.
- Speak with empathy and understanding, especially during challenging periods.
- Use a documentary-style narration that feels both personal and objective.
""",
        "business": """
Genre-Specific Style (Business/Economics):
- Speak with authority and professionalism, like a seasoned business expert.
- Use clear, confident delivery that emphasizes key strategies and insights.
- Maintain an engaging tone that makes business concepts accessible and interesting.
- Emphasize data points and statistics with clarity and precision.
- Use a dynamic, forward-looking tone that inspires action and strategic thinking.
""",
        "science": """
Genre-Specific Style (Science/Technical):
- Speak with precision and clarity, making complex concepts understandable.
- Use an enthusiastic, curious tone that reflects the wonder of discovery.
- Emphasize scientific terms and concepts with careful pronunciation.
- Maintain a balanced tone between technical accuracy and accessibility.
- Use pauses to allow complex ideas to be absorbed.
""",
        "history": """
Genre-Specific Style (History):
- Narrate with gravitas and respect for historical significance.
- Use a documentary-style tone that brings past events to life.
- Vary pacing to match the drama and importance of historical moments.
- Speak with authority and depth, like a knowledgeable historian.
- Use vivid descriptions to paint pictures of historical settings and events.
""",
        "philosophy": """
Genre-Specific Style (Philosophy):
- Speak with thoughtful deliberation, allowing ideas to breathe.
- Use a contemplative, reflective tone that invites deep thinking.
- Pause frequently to let philosophical concepts settle.
- Emphasize key philosophical terms and concepts with clarity.
- Maintain a calm, meditative pace that encourages reflection.
""",
        "psychology": """
Genre-Specific Style (Psychology):
- Speak with empathy and understanding, like a compassionate therapist.
- Use a warm, insightful tone that makes psychological concepts relatable.
- Emphasize key insights about human behavior with clarity and care.
- Maintain a balanced tone between professional expertise and human connection.
- Use pauses to allow important psychological insights to resonate.
""",
        "mystery": """
Genre-Specific Style (Mystery/Thriller):
- Create suspense through deliberate pacing and tone variations.
- Use a darker, more intense tone during suspenseful moments.
- Lower your voice slightly during mysterious or ominous passages.
- Build tension through strategic pauses and emphasis.
- Use a dramatic, engaging style that keeps listeners on edge.
""",
        "fantasy": """
Genre-Specific Style (Fantasy/Sci-Fi):
- Bring fantastical worlds to life with wonder and imagination.
- Use vivid, cinematic descriptions with awe and excitement.
- Vary your voice to reflect the epic scale of fantasy narratives.
- Emphasize magical or futuristic elements with enthusiasm.
- Create atmosphere through tone shifts that match the fantastical setting.
""",
        "romance": """
Genre-Specific Style (Romance):
- Speak with warmth and emotional depth, capturing the heart of romantic stories.
- Use a tender, expressive tone during emotional moments.
- Vary pacing to match the emotional intensity of romantic scenes.
- Emphasize emotional connections and relationships with sensitivity.
- Use a passionate yet gentle tone that honors the romantic narrative.
""",
        "horror": """
Genre-Specific Style (Horror):
- Create atmosphere through careful pacing and tone control.
- Use a darker, more ominous tone during scary moments.
- Lower your voice and slow down during suspenseful passages.
- Build tension through strategic pauses and emphasis.
- Use a dramatic, engaging style that creates unease and anticipation.
"""
    }
    
    # Check for novel sub-genres first (e.g., "novel-romance")
    genre_lower = genre.lower()
    genre_instruction = genre_specific.get(genre_lower, "")
    
    # If no specific instruction found and it's a novel sub-genre, try general novel
    if not genre_instruction and genre_lower.startswith("novel-"):
        genre_instruction = genre_specific.get("novel", "")
    
    return base_instructions + "\n" + genre_instruction + "\nFinal Delivery Style: Your storytelling voice should feel immersive, warm, expressive, rhythmic, cinematic, and deeply human. Every sentence should sound like it belongs in a captivating audiobook or podcast episode."


class OpenAIService:
    """Service for OpenAI API integration (TTS + Whisper STT)."""

    def __init__(self, voice: str = "onyx", max_tokens_per_chunk: int = 900):
        
        api_key = settings.OPENAI_API_KEY 
        
        # Debug: Log what we're getting (masked for security)
        logger.info(f"Loading API key from settings. Length: {len(api_key) if api_key else 0}")
        if api_key:
            masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
            logger.info(f"API key (masked): {masked_key}")
        
        # Check if API key is missing or invalid
        if not api_key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Please set OPENAI_API_KEY in your .env file or environment variables."
            )
        
        # Check if API key starts with "sk-" (required format)
        if not api_key.startswith("sk-"):
            raise ValueError(
                f"OpenAI API key format is invalid. "
                f"API keys must start with 'sk-'. "
                f"Current value (first 10 chars): {api_key[:10] if len(api_key) >= 10 else api_key}. "
                f"Please check your OPENAI_API_KEY in .env file."
            )
        
        # Check for placeholder values
        placeholder_values = ["sk-...", "sk-", "your-api-key-here", "OPENAI_API_KEY"]
        if api_key.lower() in [p.lower() for p in placeholder_values] or len(api_key) < 20:
            raise ValueError(
                f"OpenAI API key appears to be a placeholder or too short (length: {len(api_key)}). "
                f"Current value (first 10 chars): {api_key[:10] if len(api_key) >= 10 else api_key}. "
                f"Please set a valid OPENAI_API_KEY in your .env file. "
                f"You can find your API key at https://platform.openai.com/account/api-keys"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.voice = voice
        self.max_tokens_per_chunk = max(400, min(max_tokens_per_chunk, 1500))  # safety bounds
        logger.info(f"OpenAIService initialized (Voice: {self.voice}, Max tokens/chunk: {self.max_tokens_per_chunk})")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 1 token = 4 characters for English)."""
        return len(text) // 4
    
    def _split_text_into_chunks(self, text: str, max_tokens: int = 900) -> List[str]:
        """
        Split text into chunks that fit within token limit.
        Tries to split at sentence boundaries to maintain natural flow.
        """
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= max_tokens:
            return [text]
        
        chunks = []
        sentences = re.split(r'([.!?]\s+)', text)
        current_chunk = ""

        def flush_current():
            nonlocal current_chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            sentence_tokens = self._estimate_tokens(sentence)

            # If single sentence is larger than max_tokens, split by words
            if sentence_tokens > max_tokens:
                words = sentence.split()
                temp_sentence = ""
                for word in words:
                    test_sentence = (temp_sentence + " " + word).strip()
                    if self._estimate_tokens(test_sentence) > max_tokens:
                        if temp_sentence:
                            if self._estimate_tokens(current_chunk + " " + temp_sentence) > max_tokens:
                                flush_current()
                                current_chunk = temp_sentence
                            else:
                                current_chunk = (current_chunk + " " + temp_sentence).strip()
                        temp_sentence = word
                    else:
                        temp_sentence = test_sentence
                if temp_sentence:
                    if self._estimate_tokens(current_chunk + " " + temp_sentence) > max_tokens:
                        flush_current()
                        current_chunk = temp_sentence
                    else:
                        current_chunk = (current_chunk + " " + temp_sentence).strip()
                continue

            test_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
            if self._estimate_tokens(test_chunk) > max_tokens and current_chunk:
                flush_current()
                current_chunk = sentence.strip()
            else:
                current_chunk = test_chunk.strip()

        flush_current()
        
        return chunks

    def generate_audio_with_timestamps(
        self, 
        text: str, 
        output_dir: Path,
        job_id: str,
        genre: str = "general"
    ) -> Tuple[Path, Path]:
        """
        Generate audio with timestamps using genre-specific voice instructions.
        Handles text chunking if input exceeds token limit.
        
        Args:
            text: Text to convert to speech
            output_dir: Directory to save output files
            job_id: Job identifier
            genre: Book genre (e.g., "novel", "self-help", etc.)
        """
        logger.info(f"Job {job_id}: Starting OpenAI 2-Call Pipeline...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output paths
        audio_path = output_dir / f"{job_id}_raw_audio.mp3"
        timestamps_path = output_dir / f"{job_id}_timestamps.json"

        # Get genre-specific voice instructions
        instructions = get_voice_instructions_for_genre(genre)
        logger.info(f"Using voice instructions for genre: {genre}")
        
        try:
            # Check if text needs to be chunked
            instructions_tokens = self._estimate_tokens(instructions)
            safe_limit = 1900  # Leave ~100 tokens for metadata/overhead
            
            # If instructions themselves exceed the limit, truncate them
            max_instruction_tokens = 1500  # Max tokens for instructions
            if instructions_tokens > max_instruction_tokens:
                logger.warning(
                    f"Voice instructions are too long ({instructions_tokens} tokens). "
                    f"Truncating to {max_instruction_tokens} tokens to fit within API limits."
                )
                # Truncate instructions to fit
                max_instruction_chars = max_instruction_tokens * 4  # Rough estimate
                instructions = instructions[:max_instruction_chars]
                instructions_tokens = self._estimate_tokens(instructions)
            
            dynamic_max_tokens = min(
                self.max_tokens_per_chunk,
                max(200, safe_limit - instructions_tokens)
            )
            if dynamic_max_tokens < 200:
                logger.warning(
                    f"Voice instructions are very long ({instructions_tokens} tokens). "
                    f"Using minimal chunk size of 200 tokens. Consider shortening instructions."
                )
                dynamic_max_tokens = 200

            if dynamic_max_tokens != self.max_tokens_per_chunk:
                logger.info(
                    "Adjusting chunk size to %s tokens to account for instruction length (%s tokens).",
                    dynamic_max_tokens,
                    instructions_tokens
                )

            chunks = self._split_text_into_chunks(text, max_tokens=dynamic_max_tokens)
            
            if len(chunks) > 1:
                logger.info(f"Job {job_id}: Text exceeds token limit, splitting into {len(chunks)} chunks...")
                return self._process_chunked_text(chunks, instructions, audio_path, timestamps_path, job_id)
            else:
                # Single chunk - process normally
                return self._process_single_chunk(text, instructions, audio_path, timestamps_path, job_id)
        
        except Exception as e:
            logger.error(f"Job {job_id}: Error in OpenAI 2-Call Pipeline!", exc_info=True)
            raise
    
    def _process_single_chunk(
        self, 
        text: str, 
        instructions: str, 
        audio_path: Path, 
        timestamps_path: Path,
        job_id: str
    ) -> Tuple[Path, Path]:
        """Process a single text chunk."""
        # --- Call 1: Generate Audio (TTS) ---
        logger.info(f"Job {job_id}: Calling OpenAI TTS API (Voice: {self.voice})...")
        # Use gpt-4o-mini-tts which supports voice instructions (tts-1-hd doesn't)
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
    
    def _process_chunked_text(
        self,
        chunks: List[str],
        instructions: str,
        audio_path: Path,
        timestamps_path: Path,
        job_id: str
    ) -> Tuple[Path, Path]:
        """Process multiple text chunks and combine results."""
        import subprocess
        
        chunk_audio_files = []
        all_words = []
        all_segments = []
        total_duration = 0.0
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Job {job_id}: Processing chunk {i+1}/{len(chunks)}...")
            chunk_audio_raw = audio_path.parent / f"{audio_path.stem}_chunk_{i}_raw.mp3"
            chunk_audio = audio_path.parent / f"{audio_path.stem}_chunk_{i}.mp3"
            
            # Generate audio for chunk
            logger.info(f"Job {job_id}: Generating audio chunk {i+1}/{len(chunks)} using OpenAI voice: {self.voice}")
            response = self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=self.voice,
                input=chunk,
                instructions=instructions,
                response_format="mp3"
            )
            response.stream_to_file(str(chunk_audio_raw))
            
            # Normalize and clean each chunk before concatenation
            # This prevents static noise from level mismatches
            logger.debug(f"Job {job_id}: Processing chunk {i+1} audio - removing static and normalizing...")
            ffmpeg_path = self._get_ffmpeg_path()
            normalize_cmd = [
                ffmpeg_path,
                "-y",
                "-i", str(chunk_audio_raw),
                "-af", "highpass=f=100,lowpass=f=15000,anlmdn=s=0.0001",  # Remove low-freq static, light denoise
                "-ar", "44100",
                "-ac", "1",
                "-b:a", "192k",
                str(chunk_audio)
            ]
            subprocess.run(normalize_cmd, check=True, capture_output=True, text=True)
            logger.debug(f"Job {job_id}: Chunk {i+1} audio processed (static removal and normalization applied)")
            
            # Clean up raw chunk
            if chunk_audio_raw.exists():
                chunk_audio_raw.unlink()
            
            chunk_audio_files.append(chunk_audio)
            
            # Get timestamps for chunk
            with open(chunk_audio, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            chunk_data = transcription.model_dump()
            
            # Adjust timestamps with offset
            if "words" in chunk_data:
                for word in chunk_data["words"]:
                    word["start"] += total_duration
                    word["end"] += total_duration
                all_words.extend(chunk_data["words"])
            
            if "segments" in chunk_data:
                for segment in chunk_data["segments"]:
                    segment["start"] += total_duration
                    segment["end"] += total_duration
                all_segments.extend(chunk_data["segments"])
            
            # Update total duration
            if chunk_data.get("duration"):
                total_duration += chunk_data["duration"]
            elif all_segments:
                total_duration = all_segments[-1]["end"]
        
        # Combine audio files using ffmpeg
        logger.info(f"Job {job_id}: Combining {len(chunk_audio_files)} audio chunks...")
        ffmpeg_path = self._get_ffmpeg_path()
        
        # Create concat file for ffmpeg
        concat_file = audio_path.parent / "concat_list.txt"
        with open(concat_file, "w") as f:
            for chunk_file in chunk_audio_files:
                f.write(f"file '{chunk_file.absolute()}'\n")
        
        # Concatenate audio files with normalization and smooth transitions
        # Normalize all chunks to same level and re-encode for smooth boundaries
        cmd = [
            ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=100",  # Normalize and remove low-freq static
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "1",
            str(audio_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Clean up chunk files and concat file
        for chunk_file in chunk_audio_files:
            if chunk_file.exists():
                chunk_file.unlink()
        if concat_file.exists():
            concat_file.unlink()
        
        # Save combined timestamps
        combined_data = {
            "text": " ".join(chunks),
            "language": chunk_data.get("language", "en"),
            "duration": total_duration,
            "words": all_words,
            "segments": all_segments
        }
        
        with open(timestamps_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Job {job_id}: Combined audio and timestamps saved")
        
        return audio_path, timestamps_path
    
    def _get_ffmpeg_path(self) -> str:
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