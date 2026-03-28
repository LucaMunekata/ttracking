# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tennis match computer vision analysis — detects court lines, tracks ball/players, and produces match analytics from broadcast tennis video. Currently in milestone 1: court line detection using classical CV.

## Commands

```bash
# Install/sync dependencies
uv sync --extra dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_court_model.py

# Run a single test
uv run pytest tests/test_court_model.py::test_function_name -v

# Lint
uv run ruff check src/ tests/ scripts/

# Format
uv run ruff format src/ tests/ scripts/

# Import check
uv run python -c "import tennis_tracking"
```

## Architecture

- **`src/tennis_tracking/`** — main package (src layout, installed editable via `uv sync`)
  - **`court/`** — court detection: classical CV pipeline (Canny → Hough → homography), court geometry model, line filtering
  - **`video/`** — video I/O: frame reader (OpenCV wrapper), YouTube downloader (yt-dlp)
  - **`viz/`** — visualization utilities for annotated frame rendering
  - **`ball/`**, **`player/`**, **`analytics/`** — future modules (empty stubs)
  - **`config.py`** — Pydantic models for all tunable parameters
- **`scripts/`** — standalone CLI entry points (`detect_court.py`, `download_video.py`, `tune_params.py`)
- **`tests/`** — pytest tests; `tests/data/` holds small test images committed to repo
- **`data/`** — gitignored; local videos in `data/videos/`, outputs in `data/outputs/`

## Key Patterns

- Config is Pydantic `BaseModel` classes in `config.py` — all CV thresholds are parameterized, not hardcoded
- Modules communicate via typed dataclasses, not by importing each other's internals
- The court homography (3x3 matrix mapping image pixels ↔ real-world court coordinates) is the foundational output — ball/player modules will consume it
- ITF court dimensions are defined in `court/court_model.py` as the ground-truth reference

## Hardware Note

Development targets CPU for classical CV. AMD 9060XT GPU (ROCm, not CUDA) available for future deep learning milestones.
