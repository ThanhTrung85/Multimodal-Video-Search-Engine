# Phase 1 archive (historical snapshot)

> Warning: this document is kept as a historical record of the original Phase 1 work. The code and configuration in this file are legacy and may be outdated relative to the current repository.
>
> Please refer to the current implementation under `src/`, `configs/`, `docker/`, and the active project plan in `PROJECT_PLAN.md` for the latest working state.

## Original Phase 1 note

This archive preserves the early project notes and prototype workflow for the initial tracking + crop generation stage.

- The original concept used a simpler YOLO-based tracking flow and sample-based crop export.
- The current repository has since moved to a more structured metadata pipeline with quality filters, normalized blur scoring, relative crop paths, and a separate embedding stage.
- Historical examples from this phase are retained only for reference; they are not authoritative for the current codebase.

## Legacy summary

- Input: raw video files under `data/raw_videos/`
- Processing: YOLO detection + ByteTrack tracking
- Output: saved crops under `data/crops/`
- Initial workflow: single-step object detection and crop export, before the later normalization and evaluation refinements

This file is intentionally archived and should not be treated as the source of truth for live pipeline configuration.
