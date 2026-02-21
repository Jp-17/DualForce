#!/usr/bin/env python3
"""
CelebV-HQ Dataset Download Script.

Downloads CelebV-HQ dataset from YouTube using the official metadata.

Prerequisites:
    pip install yt-dlp
    git clone https://github.com/CelebV-HQ/CelebV-HQ.git

Usage:
    python scripts/download/download_celebvhq.py \
        --celebvhq_repo /path/to/CelebV-HQ \
        --output_dir /path/to/data/celebvhq_raw \
        --max_workers 8

    # Resume interrupted download
    python scripts/download/download_celebvhq.py \
        --celebvhq_repo /path/to/CelebV-HQ \
        --output_dir /path/to/data/celebvhq_raw \
        --resume
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_celebvhq_metadata(repo_path):
    """Load CelebV-HQ metadata from the repo.

    CelebV-HQ provides celebvhq_info.json with video IDs and annotations.
    """
    info_file = os.path.join(repo_path, "celebvhq_info.json")
    if not os.path.exists(info_file):
        # Try alternate locations
        for alt in ["data/celebvhq_info.json", "metadata/celebvhq_info.json"]:
            alt_path = os.path.join(repo_path, alt)
            if os.path.exists(alt_path):
                info_file = alt_path
                break

    if not os.path.exists(info_file):
        print(f"Error: celebvhq_info.json not found in {repo_path}")
        print("Please ensure the CelebV-HQ repo is properly cloned.")
        return None

    with open(info_file, 'r') as f:
        metadata = json.load(f)

    return metadata


def download_clip(clip_id, info, output_dir, target_fps=25, target_size=512):
    """Download and process a single CelebV-HQ clip."""
    output_path = os.path.join(output_dir, f"{clip_id}.mp4")

    # Skip if already exists
    if os.path.exists(output_path):
        return clip_id, "skipped", output_path

    # Extract YouTube URL and temporal info
    ytb_id = info.get("ytb_id", None)
    if ytb_id is None:
        return clip_id, "no_ytb_id", None

    url = f"https://www.youtube.com/watch?v={ytb_id}"

    # Get temporal boundaries
    start_time = info.get("start_time", None)
    end_time = info.get("end_time", None)
    duration = info.get("duration", None)

    # Build yt-dlp command
    temp_path = os.path.join(output_dir, "_temp", f"{clip_id}.%(ext)s")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    yt_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "-o", temp_path,
        "--no-playlist",
        "--retries", "3",
        "--socket-timeout", "30",
    ]

    # Add download sections if temporal info available
    if start_time is not None and end_time is not None:
        yt_cmd.extend(["--download-sections", f"*{start_time}-{end_time}"])

    yt_cmd.append(url)

    try:
        result = subprocess.run(yt_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return clip_id, f"yt-dlp failed: {result.stderr[:150]}", None

        # Find downloaded file
        temp_dir = os.path.dirname(temp_path)
        downloaded = None
        for ext in ['.mp4', '.mkv', '.webm']:
            candidate = os.path.join(temp_dir, f"{clip_id}{ext}")
            if os.path.exists(candidate):
                downloaded = candidate
                break

        if downloaded is None:
            # Try glob for any matching file
            import glob
            matches = glob.glob(os.path.join(temp_dir, f"{clip_id}.*"))
            if matches:
                downloaded = matches[0]

        if downloaded is None:
            return clip_id, "download_ok_but_file_not_found", None

        # Process: crop face region, resize, normalize fps
        # For now, just resize to target_size and normalize fps
        # Face detection + cropping will be done in preprocessing pipeline
        ffmpeg_cmd = [
            "ffmpeg", "-i", downloaded,
            "-vf", f"fps={target_fps},scale=-2:{target_size}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k", "-ar", "16000",
            "-y", output_path,
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)

        # Clean up temp file
        if os.path.exists(downloaded):
            os.remove(downloaded)

        if result.returncode == 0 and os.path.exists(output_path):
            return clip_id, "success", output_path
        else:
            return clip_id, f"ffmpeg failed: {result.stderr[:150]}", None

    except subprocess.TimeoutExpired:
        return clip_id, "timeout", None
    except Exception as e:
        return clip_id, f"error: {str(e)[:150]}", None


def main():
    parser = argparse.ArgumentParser(description="Download CelebV-HQ dataset")
    parser.add_argument("--celebvhq_repo", type=str, required=True,
                        help="Path to cloned CelebV-HQ repository")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save downloaded videos")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Number of parallel download workers")
    parser.add_argument("--target_fps", type=int, default=25)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous download (skip existing files)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of clips to download (0=all)")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Start from Nth clip (for splitting across machines)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    metadata = load_celebvhq_metadata(args.celebvhq_repo)
    if metadata is None:
        sys.exit(1)

    # CelebV-HQ metadata is a dict: {clip_id: {ytb_id, start_time, end_time, ...}}
    if isinstance(metadata, dict):
        clips = list(metadata.items())
    elif isinstance(metadata, list):
        clips = [(item.get("clip_id", f"clip_{i}"), item) for i, item in enumerate(metadata)]
    else:
        print(f"Unexpected metadata format: {type(metadata)}")
        sys.exit(1)

    # Apply start/limit
    clips = clips[args.start_from:]
    if args.limit > 0:
        clips = clips[:args.limit]

    print("=" * 60)
    print("  CelebV-HQ Dataset Download")
    print("=" * 60)
    print(f"  Total clips in metadata: {len(metadata) if isinstance(metadata, dict) else len(metadata)}")
    print(f"  Clips to download: {len(clips)}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Workers: {args.max_workers}")
    print()

    # Count existing
    if args.resume:
        existing = sum(1 for clip_id, _ in clips
                       if os.path.exists(os.path.join(args.output_dir, f"{clip_id}.mp4")))
        print(f"  Already downloaded: {existing}")
        print(f"  Remaining: {len(clips) - existing}")

    # Download
    stats = {"success": 0, "skipped": 0, "failed": 0}
    results_log = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                download_clip, clip_id, info, args.output_dir,
                args.target_fps, args.target_size
            ): clip_id for clip_id, info in clips
        }

        for i, future in enumerate(as_completed(futures), 1):
            clip_id, status, path = future.result()
            if status == "success":
                stats["success"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
                results_log.append({"clip_id": clip_id, "status": status})

            if i % 100 == 0 or status not in ("success", "skipped"):
                print(f"  [{i}/{len(clips)}] {clip_id}: {status} "
                      f"(ok={stats['success']}, skip={stats['skipped']}, fail={stats['failed']})")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Download Complete")
    print(f"{'='*60}")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed:  {stats['failed']}")

    # Save failed log
    if results_log:
        log_path = os.path.join(args.output_dir, "failed_downloads.json")
        with open(log_path, 'w') as f:
            json.dump(results_log, f, indent=2)
        print(f"  Failed log: {log_path}")

    # Generate metadata for preprocessing pipeline
    metadata_out = []
    for clip_id, info in clips:
        clip_path = os.path.join(args.output_dir, f"{clip_id}.mp4")
        if os.path.exists(clip_path):
            metadata_out.append({
                "clip_id": clip_id,
                "source": "CelebV-HQ",
                "caption": info.get("text", "A person speaking."),
                "fps": args.target_fps,
                "resolution": args.target_size,
            })

    meta_path = os.path.join(args.output_dir, "celebvhq_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata_out, f, indent=2)
    print(f"  Metadata: {meta_path} ({len(metadata_out)} clips)")


if __name__ == "__main__":
    main()
