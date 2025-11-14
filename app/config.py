import os
from pathlib import Path
from pydantic_settings import BaseSettings

# This is the root directory of *entire* project
PROJECT_ROOT = Path(__file__).parent.parent 

class Settings(BaseSettings):
    """
    Main application settings. Loads from .env file.
    """
    
    # --- Project Paths ---
    ASSETS_PATH: Path = PROJECT_ROOT / "assets"
    FONTS_PATH: Path = ASSETS_PATH / "fonts"
    BACKGROUNDS_PATH: Path = ASSETS_PATH / "backgrounds"
    JOBS_OUTPUT_PATH: Path = PROJECT_ROOT / "jobs"
    
    # --- API Keys (Loaded from .env) ---
    OPENAI_API_KEY: str = "sk-..." # Default,
    
    # --- Video & Text Settings (from your files) ---
    DEFAULT_FONT_REGULAR: str = str(FONTS_PATH / "PlayfairDisplay-Regular.ttf")
    DEFAULT_FONT_BOLD: str = str(FONTS_PATH / "PlayfairDisplay-Medium.ttf")
    DEFAULT_BACKGROUND: str = str(BACKGROUNDS_PATH / "1920x1080-white-solid-color-background.jpg") #for 1080 p quality
    # DEFAULT_BACKGROUND: str = str(BACKGROUNDS_PATH / "854x480-white-background.jpg")     #for 480 p quality
    
    VIDEO_FPS: int = 30
    VIDEO_WIDTH: int = 1920   #for 1080 p quality
    VIDEO_HEIGHT: int = 1080  #for 1080 p quality
    # VIDEO_WIDTH: int = 854    #for 480 p quality
    # VIDEO_HEIGHT: int = 480   #for 480 p quality
    VIDEO_CODEC: str = "libx264"
    
    # Text colors 
    TEXT_REGULAR_COLOR: tuple = (170, 170, 170, 255) # Grey
    TEXT_BOLD_COLOR: tuple = (0, 0, 0, 255) # Black
    
    class Config:
        env_file = PROJECT_ROOT / ".env"
        case_sensitive = False

settings = Settings()