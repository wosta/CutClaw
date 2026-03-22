# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoCuttingAgent is an AI-powered video editing system that automatically creates music-synchronized video montages from source videos. It uses multi-agent architecture with VLM (Vision-Language Models) for intelligent shot selection and editing.

## Core Architecture

### Multi-Agent System
The system uses three specialized agents that work sequentially:

1. **Footage Deconstruction** - Analyzes source video and audio to create searchable databases
   - Video: Shot detection, scene segmentation, visual captioning, character identification
   - Audio: Beat detection, pitch analysis, musical structure analysis

2. **Screenwriter Agent** (`src/Screenwriter_scene_short.py`) - Creates shot plans
   - Takes user instruction and generates a structured editing plan
   - Maps audio segments to video content requirements
   - Outputs `shot_plan.json` with scene recommendations and timing

3. **Editor Agent** (`src/core.py` for films, `src/core_vlog.py` for vlogs) - Selects actual video clips
   - Uses ReAct-style tool calling loop with VLM
   - Tools: `get_related_shot`, `trim_shot`, `review_clip`, `finish`
   - Outputs `shot_point.json` with precise timestamps

4. **Reviewer Agent** (`src/Reviewer.py`) - Validates shot selections
   - Checks for duplicate footage, protagonist presence, face quality
   - Integrated into Editor Core Agent's workflow

### Key Modules

- `src/config.py` - Central configuration (model paths, detection thresholds, API endpoints)
- `src/video/` - Video processing utilities (ASR, shot detection, scene analysis)
- `src/audio/` - Audio analysis with Madmom (beat/pitch/energy detection)
- `src/build_database/` - Database construction for video scenes
- `render/render_video.py` - Final video rendering from shot points

## Common Commands

### Run Full Pipeline
```bash
python local_run.py \
    --Video_Path "Dataset/Video/Movie/The_Dark_Knight.mkv" \
    --Audio_Path "Dataset/Audio/Way_Down_We_Go.mp3" \
    --Instruction "Your editing instruction here" \
    --instruction_type object \
    --type film
```

### Config Overrides
Override any config parameter at runtime:
```bash
python local_run.py \
    --Video_Path "path/to/video.mp4" \
    --Audio_Path "path/to/audio.mp3" \
    --config.MAIN_CHARACTER_NAME "Batman" \
    --config.AUDIO_MIN_SEGMENT_DURATION 2 \
    --config.VIDEO_FPS 2
```

### Render Final Video
```bash
python render/render_video.py \
    --shot-json 'path/to/shot_point.json' \
    --shot-plan 'path/to/shot_plan.json' \
    --video Dataset/Video/Movie/source.mkv \
    --audio Dataset/Audio/music.mp3 \
    --output output/final.mp4 \
    --crop-ratio "9:16"
```


## Pipeline Stages

The `local_run.py` script orchestrates the full pipeline in stages (most are commented out by default for selective execution):

1. **Frame Extraction & Shot Detection** - Decode video to frames, detect shot boundaries
2. **ASR (Automatic Speech Recognition)** - Generate subtitles (films only, not vlogs)
3. **Character Identification** - Identify speakers in subtitles (films only)
4. **Video Captioning** - Generate dense captions for each shot using VLM
5. **Scene Merging** - Group shots into coherent scenes
6. **Scene Analysis** - Detailed video analysis of merged scenes
7. **Audio Analysis** - Detect musical keypoints (beats, pitch changes, energy peaks)
8. **Screenwriter** - Generate shot plan from instruction
9. **Editor Core Agent** - Select precise video clips based on shot plan

## Important Configuration Parameters


### Output JSON Formats

**shot_plan.json** - High-level editing plan
```json
{
  "video_structure": [
    {
      "shot_plan": {
        "shots": [
          {
            "content": "Description of desired content",
            "emotion": "Target emotion",
            "time_duration": 5.2,
            "related_scene": [8, 12]
          }
        ]
      }
    }
  ]
}
```

**shot_point.json** - Precise timestamps for rendering
```json
[
  {
    "status": "success",
    "section_idx": 0,
    "shot_idx": 0,
    "total_duration": 5.2,
    "clips": [
      {
        "shot": 1,
        "start": "00:13:28.0",
        "end": "00:13:33.2",
        "duration": 5.2
      }
    ]
  }
]
```

## Instruction Types

The system supports two editing approaches:

- **object** - Focus on specific character/object, emphasizing visual presence and iconic moments
- **narrative** - Story-driven editing with thematic coherence and emotional arcs

Specify with `--instruction_type object` or `--instruction_type narrative`.

## Development Notes

- The Editor Core Agent uses a ReAct loop with tool calling - when debugging, check tool execution logs
- Shot selection can be resumed from partial results (checks existing `shot_point.json`)
- The agent has duplicate call detection to prevent infinite loops (max 3 duplicate calls)
- VLM protagonist detection is optional but recommended for character-focused edits
- For vlogs, ASR and character identification are automatically skipped

## My Current Goals 

- Reduce the inference cost (token cost, time cost) for easy to use
- Simplify and refine code for open-sourse release
