# Env looks like this:

```
OPENAI_API_KEY="sk-"

VIDEO_FPS=30
VIDEO_WIDTH=1920
VIDEO_HEIGHT=1080
VIDEO_CODEC=libx264

# Use the correct JSON format (as you did, which is good!)
TEXT_REGULAR_COLOR="[170, 170, 170, 255]"
TEXT_BOLD_COLOR="[0, 0, 0, 255]"

```

# Mapping:

- All files are changed just be version mismatching now everything is working and tested!
- assets contains the input 
- jobs/ contains the output folder
- scripts/run_full_pipeline.py (contains the complete code of full final integration)
  
```
app/
├── phase1_pdf_processing/
│   ├── __init__.py
│   ├── processor.py             # <-- Adnan's original  pdf_processor.py
│   ├── service.py               # <-- Adnan's original pdf_extractor_service.py (renamed)
│   ├── image_extractor.py       # <-- Adnan's logic from untitled39.py (no change)
│   ├── text_cleaner.py          # <-- The dummy file (no change)
│   │
│   └── utils/                   # <-- NEW FOLDER
│       ├── __init__.py
│       └── pdf_extraction_strategies.py  # <-- The original adnan's service file 
│
├── phase2_ai_services/
│   └── openai_client.py
├── phase3_audio_processing/
│   └── mastering.py
└── phase4_video_generation/
    └── renderer.py

```

