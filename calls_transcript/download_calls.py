#!/usr/bin/env python3
"""
Download today's Fireflies call transcripts and save them to calls_transcript directory.

This script fetches all calls from today and saves:
- Full transcript as markdown
- JSON with complete meeting data
- Summary and action items

Usage:
    python3 00_download_today_calls.py
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add chrome-extension-tcs to path to import fireflies_client
chrome_ext_path = Path.home() / "projects" / "chrome-extension-tcs"
sys.path.append(str(chrome_ext_path))

from data_sources.fireflies.fireflies_client import FirefliesClient


def format_transcript(meeting: dict) -> str:
    """
    Format meeting data as readable markdown transcript.

    Args:
        meeting: Meeting data from Fireflies API

    Returns:
        Formatted markdown string
    """
    lines = []

    # Header
    lines.append(f"# {meeting.get('title', 'Untitled Meeting')}")
    lines.append("")

    # Metadata
    date_ms = meeting.get('date', 0)
    if date_ms:
        date_obj = datetime.fromtimestamp(date_ms / 1000)
        date_str = date_obj.strftime('%A, %B %d, %Y at %H:%M')
        lines.append(f"**Date:** {date_str}")

    duration = meeting.get('duration', 0)
    if duration:
        minutes = duration / 60
        lines.append(f"**Duration:** {minutes:.1f} minutes")

    participants = meeting.get('participants', [])
    if participants:
        lines.append(f"**Participants:** {', '.join(participants)}")

    if meeting.get('organizer_email'):
        lines.append(f"**Organizer:** {meeting.get('organizer_email')}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    summary = meeting.get('summary', {})
    if summary:
        lines.append("## Summary")
        lines.append("")

        if summary.get('overview'):
            lines.append("### Overview")
            lines.append(summary['overview'])
            lines.append("")

        if summary.get('keywords'):
            lines.append("### Keywords")
            lines.append(", ".join(summary['keywords']))
            lines.append("")

        if summary.get('action_items') and summary['action_items'] != 'N/A':
            lines.append("### Action Items")
            lines.append(summary['action_items'])
            lines.append("")

        if summary.get('shorthand_bullet'):
            lines.append("### Key Points")
            lines.append(summary['shorthand_bullet'])
            lines.append("")

        if summary.get('outline'):
            lines.append("### Outline")
            lines.append(summary['outline'])
            lines.append("")

    # Full transcript
    sentences = meeting.get('sentences', [])
    if sentences:
        lines.append("---")
        lines.append("")
        lines.append("## Full Transcript")
        lines.append("")

        current_speaker = None
        for sentence in sentences:
            speaker = sentence.get('speaker_name', 'Unknown')
            text = sentence.get('text', '').strip()

            if not text:
                continue

            # Start new speaker section
            if speaker != current_speaker:
                if current_speaker is not None:
                    lines.append("")  # Empty line between speakers
                lines.append(f"**{speaker}:**")
                current_speaker = speaker

            lines.append(text)

    return '\n'.join(lines)


def save_transcript(meeting: dict, output_dir: Path):
    """
    Save meeting transcript and metadata to files.

    Args:
        meeting: Meeting data from Fireflies API
        output_dir: Directory to save files
    """
    # Generate filename from date and title
    date_ms = meeting.get('date', 0)
    meeting_id = meeting.get('id', 'unknown')

    if date_ms:
        date_obj = datetime.fromtimestamp(date_ms / 1000)
        date_prefix = date_obj.strftime('%Y%m%d_%H%M')
    else:
        date_prefix = 'unknown_date'

    # Clean title for filename
    title = meeting.get('title', 'untitled')
    # Remove special characters and limit length
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = safe_title.replace(' ', '_')[:50]

    base_filename = f"{date_prefix}_{safe_title}"

    # Save markdown transcript
    md_path = output_dir / f"{base_filename}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(format_transcript(meeting))
    print(f"   ✓ Saved transcript: {md_path.name}")

    # Save JSON with complete data
    json_path = output_dir / f"{base_filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meeting, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved JSON data: {json_path.name}")


def main():
    """Main execution function."""
    print("🔥 Fireflies Call Transcript Downloader")
    print("=" * 50)
    print()

    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "calls_transcript"
    output_dir.mkdir(exist_ok=True)

    # Initialize Fireflies client
    try:
        client = FirefliesClient()
        print("✓ Connected to Fireflies API")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize Fireflies client: {e}")
        print()
        print("Make sure FIREFLIES_API_KEY is set in environment variables")
        print("or in .env file at chrome-extension-tcs directory")
        sys.exit(1)

    # Get today's date range
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    date_from = today.strftime('%Y-%m-%d')
    date_to = tomorrow.strftime('%Y-%m-%d')

    print(f"📅 Searching for calls from {date_from}")
    print()

    # Search for today's meetings
    try:
        meetings = client.search_meetings(
            date_from=date_from,
            date_to=date_to,
            limit=50
        )

        if not meetings:
            print("ℹ️  No calls found for today")
            return

        print(f"✓ Found {len(meetings)} call(s)")
        print()

        # Download each meeting with full transcript
        for i, meeting_summary in enumerate(meetings, 1):
            meeting_id = meeting_summary.get('id')
            title = meeting_summary.get('title', 'Untitled')

            print(f"[{i}/{len(meetings)}] {title}")

            try:
                # Get full meeting details including transcript
                meeting = client.get_meeting(meeting_id, include_transcript=True)

                if meeting:
                    save_transcript(meeting, output_dir)
                else:
                    print(f"   ⚠️  No data returned for meeting {meeting_id}")

                print()

            except Exception as e:
                print(f"   ❌ Failed to download: {e}")
                print()
                continue

        print("=" * 50)
        print(f"✅ Downloaded {len(meetings)} transcript(s) to: {output_dir}")

    except Exception as e:
        print(f"❌ Search failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
