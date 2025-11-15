# Batch Processing Approach for Video Rendering

## Current Problem
- MoviePy generates frames **on-demand** (one at a time) via `frame_function`
- No parallelization - all frames generated sequentially
- For a 5-minute video at 30fps = 9,000 frames generated one-by-one
- This is the main bottleneck

## Solution: Pre-Generate Frames in Parallel Batches

---

## Phase 1: Pre-Calculation & Planning

### Step 1.1: Calculate All Frame Timestamps
- Input: Audio duration, FPS, slide start times
- Calculate: Exact timestamp for each frame (frame 0 = 0.0s, frame 1 = 0.033s, etc.)
- Output: List of (frame_number, timestamp) pairs
- **Time: ~0.1 seconds**

### Step 1.2: Map Frames to Slides
- For each frame timestamp, determine which slide it belongs to
- Create mapping: `{frame_number: slide_index}`
- **Time: ~0.1 seconds**

### Step 1.3: Create Frame Generation Tasks
- Group frames into batches (e.g., 100 frames per batch)
- Each batch contains: frame numbers, timestamps, slide indices
- **Time: ~0.1 seconds**

---

## Phase 2: Parallel Frame Generation

### Step 2.1: Setup Multiprocessing Pool
- Use `multiprocessing.Pool` with `os.cpu_count()` workers
- Each worker will generate a batch of frames
- **Setup time: ~1 second**

### Step 2.2: Worker Function Design
**Input to worker:**
- Batch of frame tasks (frame_number, timestamp, slide_index)
- Shared data: slides, layouts, fonts, colors, dimensions

**Worker process:**
1. Load fonts (once per worker)
2. For each frame in batch:
   - Determine which slide
   - Calculate word states (bold/regular) based on timestamp
   - Render frame as PIL Image
   - Save as PNG: `frame_00001.png`, `frame_00002.png`, etc.
3. Return: list of generated frame file paths

**Output:** List of frame image file paths

### Step 2.3: Generate Frames in Batches
- Submit all batches to process pool
- Process batches in parallel (e.g., 8 workers = 8 batches simultaneously)
- Monitor progress with progress bar
- **Time: ~2-5 minutes** (vs 10-15 minutes sequential)

---

## Phase 3: Video Assembly

### Step 3.1: Create ImageSequenceClip
- Load all generated frame images in order
- Use `ImageSequenceClip` to create video from images
- This is much faster than generating frames on-demand
- **Time: ~10-30 seconds**

### Step 3.2: Apply Background & Effects
- Load background image
- Composite text frames over background
- Apply fade transitions if needed
- **Time: ~5-10 seconds**

### Step 3.3: Add Audio
- Attach audio clip to video
- **Time: ~1 second**

---

## Phase 4: Optimized Encoding

### Step 4.1: Detect Hardware Acceleration
- Check for NVIDIA GPU → use `h264_nvenc`
- Check for Intel QuickSync → use `h264_qsv`
- Check for AMD → use `h264_amf`
- Fallback: `libx264` (software)

### Step 4.2: Use Fast Encoding Preset
- Change from `preset="medium"` to `preset="fast"` or `preset="veryfast"`
- Same quality, faster encoding
- **Time: ~1-3 minutes** (vs 3-5 minutes)

### Step 4.3: Encode Video
- Use detected codec + fast preset
- Multi-threaded encoding
- **Time: ~1-3 minutes**

---

## Phase 5: Cleanup

### Step 5.1: Delete Temporary Frames
- Remove all generated PNG frame images
- Clean up temporary directory
- **Time: ~5 seconds**

---

## Implementation Structure

```
render_video() [Main Function]
├── Phase 1: Pre-calculation
│   ├── calculate_all_frame_timestamps()
│   ├── map_frames_to_slides()
│   └── create_frame_batches()
│
├── Phase 2: Parallel frame generation
│   ├── setup_worker_pool()
│   ├── generate_frame_batch() [Worker function]
│   └── generate_all_frames_parallel()
│
├── Phase 3: Video assembly
│   ├── create_video_from_frames()
│   ├── apply_background()
│   └── add_audio()
│
├── Phase 4: Encoding
│   ├── detect_hardware_codec()
│   └── encode_video()
│
└── Phase 5: Cleanup
    └── cleanup_temp_files()
```

---

## Key Optimizations

### 1. Batch Size
- **Recommended:** 100-200 frames per batch
- Too small: Overhead from process creation
- Too large: Less parallelization benefit
- Formula: `batch_size = total_frames / (cpu_count * 4)`

### 2. Worker Count
- Use `os.cpu_count()` workers
- Typical: 4-16 workers depending on CPU
- More workers = more parallelization

### 3. Memory Management
- Generate frames → Save to disk → Free memory
- Don't keep all frames in RAM
- Use temporary directory for frames

### 4. Progress Tracking
- Show progress bar during frame generation
- Log batch completion
- Estimate remaining time

---

## Expected Performance

### Current (Sequential):
- Frame generation: 10-15 minutes
- Encoding: 3-5 minutes
- **Total: 13-20 minutes**

### With Batch Processing:
- Frame generation: 2-5 minutes (4-8x faster)
- Encoding: 1-3 minutes (2-3x faster with hardware)
- **Total: 3-8 minutes**

### Speedup: **3-5x overall improvement**

---

## Quality Preservation

✅ Same resolution (1080p)
✅ Same frame rate (30 FPS)
✅ Same visual quality
✅ Same text rendering
✅ Same codec quality settings
✅ Only processing method changes

---

## Implementation Steps

1. **Create batch frame generator class**
   - Handle frame timestamp calculation
   - Batch creation logic
   - Worker function for parallel processing

2. **Modify render_video() function**
   - Replace VideoClip with pre-generated frames
   - Use ImageSequenceClip instead
   - Add hardware acceleration detection

3. **Add progress tracking**
   - Progress bar for frame generation
   - Logging improvements

4. **Add cleanup logic**
   - Remove temporary frame images
   - Handle errors gracefully

5. **Test and optimize**
   - Test with different batch sizes
   - Measure performance improvements
   - Fine-tune worker count

---

## Dependencies Needed

- `multiprocessing` (built-in)
- `tqdm` (for progress bars) - may need to add to requirements
- `PIL/Pillow` (already used)
- `moviepy` (already used)
- `numpy` (already used)

---

## Error Handling

- Handle worker crashes gracefully
- Retry failed batches
- Clean up partial files on error
- Log detailed error information

---

## Next Steps

1. Implement Phase 1 (Pre-calculation)
2. Implement Phase 2 (Parallel generation)
3. Implement Phase 3 (Video assembly)
4. Implement Phase 4 (Optimized encoding)
5. Test and benchmark
6. Optimize based on results

