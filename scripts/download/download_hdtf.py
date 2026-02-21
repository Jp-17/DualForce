#!/usr/bin/env python3
"""
HDTF Dataset Download Script.

Downloads the HDTF (High-Definition Talking Face) dataset from YouTube
using yt-dlp, then processes the videos according to HDTF metadata.

Prerequisites:
    pip install yt-dlp
    # Ensure ffmpeg is installed

Usage:
    # Step 1: Clone HDTF repo (for metadata)
    git clone https://github.com/MRzzm/HDTF.git /path/to/HDTF

    # Step 2: Download videos
    python scripts/data/download_hdtf.py \
        --hdtf_repo /path/to/HDTF \
        --output_dir /path/to/data/hdtf_raw \
        --max_workers 4

    # Step 3: Clip and crop (after download)
    python scripts/data/download_hdtf.py \
        --hdtf_repo /path/to/HDTF \
        --output_dir /path/to/data/hdtf_raw \
        --clip_only
"""

import argparse
import os
import subprocess
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_url_file(filepath):
    """Parse HDTF URL file (format: 'video_name url')."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append({
                    'name': parts[0],
                    'url': parts[1],
                })
    return entries


def parse_annotation_file(filepath):
    """Parse HDTF annotation time file (format: 'video_name start_time end_time')."""
    annotations = {}
    if not os.path.exists(filepath):
        return annotations
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                clips = []
                # Format could be: name start1 end1 start2 end2 ...
                for i in range(1, len(parts) - 1, 2):
                    try:
                        start = float(parts[i])
                        end = float(parts[i + 1])
                        clips.append((start, end))
                    except (ValueError, IndexError):
                        continue
                annotations[name] = clips
    return annotations


def parse_crop_file(filepath):
    """Parse HDTF crop file (format: 'video_name x y w h' or similar)."""
    crops = {}
    if not os.path.exists(filepath):
        return crops
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                name = parts[0]
                try:
                    x, y, w, h = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    crops[name] = (x, y, w, h)
                except ValueError:
                    continue
    return crops


def download_video(name, url, output_dir, resolution=1080):
    """Download a single video using yt-dlp."""
    output_path = os.path.join(output_dir, f"{name}.%(ext)s")
    final_mp4 = os.path.join(output_dir, f"{name}.mp4")

    # Skip if already downloaded
    if os.path.exists(final_mp4):
        return name, "skipped", final_mp4

    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-playlist",
        "--retries", "3",
        "--socket-timeout", "30",
        url,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            # Find the output file (yt-dlp may have used different extension)
            for ext in ['.mp4', '.mkv', '.webm']:
                candidate = os.path.join(output_dir, f"{name}{ext}")
                if os.path.exists(candidate):
                    if ext != '.mp4':
                        # Convert to mp4
                        subprocess.run([
                            "ffmpeg", "-i", candidate, "-c", "copy",
                            final_mp4, "-y"
                        ], capture_output=True, timeout=120)
                        os.remove(candidate)
                    return name, "success", final_mp4
            return name, "success", final_mp4
        else:
            return name, f"failed: {result.stderr[:200]}", None
    except subprocess.TimeoutExpired:
        return name, "timeout", None
    except Exception as e:
        return name, f"error: {str(e)[:200]}", None


def clip_video(input_path, output_dir, name, clips, target_fps=25, target_size=512):
    """Clip and crop video according to annotations."""
    results = []
    for idx, (start, end) in enumerate(clips):
        clip_name = f"{name}_{idx}"
        output_path = os.path.join(output_dir, f"{clip_name}.mp4")

        if os.path.exists(output_path):
            results.append((clip_name, "skipped"))
            continue

        duration = end - start
        if duration < 2.0:
            results.append((clip_name, "too_short"))
            continue

        # Extract clip, resize to target_size, normalize fps
        cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(duration),
            "-vf", f"fps={target_fps},scale={target_size}:{target_size}:force_original_aspect_ratio=decrease,pad={target_size}:{target_size}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k", "-ar", "16000",
            "-y",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and os.path.exists(output_path):
                results.append((clip_name, "success"))
            else:
                results.append((clip_name, f"failed: {result.stderr[:100]}"))
        except Exception as e:
            results.append((clip_name, f"error: {str(e)[:100]}"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Download and process HDTF dataset")
    parser.add_argument("--hdtf_repo", type=str, required=True,
                        help="Path to cloned HDTF repository")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save downloaded/processed videos")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of parallel download workers")
    parser.add_argument("--resolution", type=int, default=1080,
                        help="Max video resolution to download")
    parser.add_argument("--clip_only", action="store_true",
                        help="Skip download, only clip already-downloaded videos")
    parser.add_argument("--target_fps", type=int, default=25,
                        help="Target frame rate for clipped videos")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target resolution (square) for clipped videos")
    parser.add_argument("--groups", nargs="+", default=["RD", "WDA", "WRA"],
                        help="Which HDTF groups to process")
    args = parser.parse_args()

    hdtf_data_dir = os.path.join(args.hdtf_repo, "HDTF_dataset")
    if not os.path.isdir(hdtf_data_dir):
        print(f"Error: HDTF_dataset directory not found at {hdtf_data_dir}")
        print(f"Please clone the HDTF repo: git clone https://github.com/MRzzm/HDTF.git")
        sys.exit(1)

    raw_dir = os.path.join(args.output_dir, "raw")
    clips_dir = os.path.join(args.output_dir, "clips")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    # ========================================
    # Phase 1: Download raw videos
    # ========================================
    if not args.clip_only:
        print("=" * 60)
        print("  Phase 1: Downloading HDTF videos from YouTube")
        print("=" * 60)

        all_entries = []
        for group in args.groups:
            url_file = os.path.join(hdtf_data_dir, f"{group}_video_url.txt")
            if os.path.exists(url_file):
                entries = parse_url_file(url_file)
                print(f"  {group}: {len(entries)} videos")
                all_entries.extend(entries)
            else:
                print(f"  Warning: {url_file} not found, skipping {group}")

        print(f"  Total: {len(all_entries)} videos to download")

        # Download in parallel
        success, failed, skipped = 0, 0, 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    download_video, entry['name'], entry['url'],
                    raw_dir, args.resolution
                ): entry for entry in all_entries
            }

            for future in as_completed(futures):
                name, status, path = future.result()
                if status == "success":
                    success += 1
                    print(f"  [+] {name}: downloaded")
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"  [-] {name}: {status}")

        print(f"\n  Download complete: {success} success, {skipped} skipped, {failed} failed")

    # ========================================
    # Phase 2: Clip and process videos
    # ========================================
    print("\n" + "=" * 60)
    print("  Phase 2: Clipping and processing videos")
    print("=" * 60)

    all_annotations = {}
    for group in args.groups:
        annot_file = os.path.join(hdtf_data_dir, f"{group}_annotion_time.txt")
        if os.path.exists(annot_file):
            annotations = parse_annotation_file(annot_file)
            all_annotations.update(annotations)
            print(f"  {group}: {len(annotations)} annotation entries")

    # Process each video
    total_clips = 0
    clip_results = {"success": 0, "skipped": 0, "failed": 0}
    metadata = []

    for name, clips in all_annotations.items():
        # Find the raw video
        raw_video = None
        for ext in ['.mp4', '.mkv', '.webm']:
            candidate = os.path.join(raw_dir, f"{name}{ext}")
            if os.path.exists(candidate):
                raw_video = candidate
                break

        if raw_video is None:
            continue

        results = clip_video(
            raw_video, clips_dir, name, clips,
            target_fps=args.target_fps, target_size=args.target_size
        )

        for clip_name, status in results:
            if status == "success":
                clip_results["success"] += 1
                clip_path = os.path.join(clips_dir, f"{clip_name}.mp4")
                # Get video info
                try:
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-print_format", "json",
                         "-show_format", "-show_streams", clip_path],
                        capture_output=True, text=True, timeout=30
                    )
                    info = json.loads(probe.stdout)
                    duration = float(info.get("format", {}).get("duration", 0))
                    num_frames = int(duration * args.target_fps)
                except Exception:
                    duration = 0
                    num_frames = 0

                metadata.append({
                    "clip_id": clip_name,
                    "source": "HDTF",
                    "source_video": name,
                    "duration": duration,
                    "num_frames": num_frames,
                    "fps": args.target_fps,
                    "resolution": args.target_size,
                    "caption": "A person speaking.",
                })
            elif status == "skipped":
                clip_results["skipped"] += 1
            else:
                clip_results["failed"] += 1
                print(f"  [-] {clip_name}: {status}")

        total_clips += len(results)

    print(f"\n  Clipping complete: {clip_results['success']} success, "
          f"{clip_results['skipped']} skipped, {clip_results['failed']} failed")

    # Save metadata
    metadata_path = os.path.join(args.output_dir, "hdtf_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_path} ({len(metadata)} clips)")

    print(f"\n  Output directory: {args.output_dir}")
    print(f"  Raw videos: {raw_dir}")
    print(f"  Clipped videos: {clips_dir}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
