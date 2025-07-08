"""
Frame Sampling Module - Phase 2A

Extracts video frames at 10-second intervals using ffmpeg.
Stores frames as {videoid}_{timestamp}.jpg with metadata.
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import ffmpeg
from tqdm import tqdm


@dataclass
class FrameMetadata:
    """Metadata for an extracted frame."""
    video_id: str
    start: float
    end: float
    frame_path: str
    timestamp: float


class FrameSampler:
    """
    Extracts frames from video at 10-second intervals using ffmpeg.
    
    Usage:
        sampler = FrameSampler(frames_dir="data/frames")
        metadata = sampler.sample_frames("video.mp4", "video_001")
    """
    
    def __init__(self, frames_dir: str = "data/frames", interval: int = 10):
        """
        Initialize the frame sampler.
        
        Args:
            frames_dir: Directory to store extracted frames
            interval: Time interval between frames in seconds (default: 10)
        """
        self.frames_dir = Path(frames_dir)
        self.interval = interval
        self.frames_dir.mkdir(parents=True, exist_ok=True)
    
    def get_video_duration(self, video_path: str) -> float:
        """
        Get video duration in seconds using ffmpeg-python.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Duration in seconds
        """
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            raise ValueError(f"Could not determine video duration: {e}")
    
    def extract_frame_at_timestamp(self, video_path: str, timestamp: float, output_path: str) -> None:
        """
        Extract a single frame at the specified timestamp.
        
        Args:
            video_path: Path to the input video
            timestamp: Time in seconds to extract frame
            output_path: Path for the output frame image
        """
        try:
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .output(output_path, vframes=1, loglevel='quiet')
                .overwrite_output()
                .run()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract frame at {timestamp}s: {e}")
    
    def sample_frames(self, video_path: str, video_id: str) -> List[FrameMetadata]:
        """
        Sample frames from video at regular intervals.
        
        Args:
            video_path: Path to the input video file
            video_id: Unique identifier for the video
            
        Returns:
            List of FrameMetadata objects for extracted frames
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video duration
        duration = self.get_video_duration(video_path)
        print(f"Video duration: {duration:.2f} seconds")
        
        # Calculate frame timestamps
        timestamps = []
        current_time = 0.0
        while current_time < duration:
            timestamps.append(current_time)
            current_time += self.interval
        
        print(f"Extracting {len(timestamps)} frames at {self.interval}s intervals...")
        
        # Extract frames
        frame_metadata = []
        for timestamp in tqdm(timestamps, desc="Extracting frames"):
            # Create frame filename: {videoid}_{timestamp}.jpg
            frame_filename = f"{video_id}_{int(timestamp)}.jpg"
            frame_path = self.frames_dir / frame_filename
            
            # Extract frame
            self.extract_frame_at_timestamp(video_path, timestamp, str(frame_path))
            
            # Create metadata
            metadata = FrameMetadata(
                video_id=video_id,
                start=timestamp,
                end=min(timestamp + self.interval, duration),
                frame_path=str(frame_path),
                timestamp=timestamp
            )
            frame_metadata.append(metadata)
        
        print(f"Successfully extracted {len(frame_metadata)} frames to {self.frames_dir}")
        return frame_metadata
    
    def save_metadata(self, frame_metadata: List[FrameMetadata], output_path: str) -> None:
        """
        Save frame metadata to JSON file.
        
        Args:
            frame_metadata: List of frame metadata objects
            output_path: Path to save the metadata JSON file
        """
        metadata_dict = {
            "video_id": frame_metadata[0].video_id if frame_metadata else "",
            "total_frames": len(frame_metadata),
            "interval": self.interval,
            "frames": [
                {
                    "video_id": frame.video_id,
                    "start": frame.start,
                    "end": frame.end,
                    "frame_path": frame.frame_path,
                    "timestamp": frame.timestamp
                }
                for frame in frame_metadata
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"Metadata saved to: {output_path}")


def main():
    """Command-line interface for frame sampling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video at 10-second intervals")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("video_id", help="Unique identifier for the video")
    parser.add_argument("--frames-dir", default="data/frames", help="Directory to store frames")
    parser.add_argument("--interval", type=int, default=10, help="Interval between frames in seconds")
    parser.add_argument("--save-metadata", help="Path to save metadata JSON file")
    
    args = parser.parse_args()
    
    # Initialize sampler
    sampler = FrameSampler(frames_dir=args.frames_dir, interval=args.interval)
    
    # Sample frames
    try:
        metadata = sampler.sample_frames(args.video_path, args.video_id)
        
        # Save metadata if requested
        if args.save_metadata:
            sampler.save_metadata(metadata, args.save_metadata)
        
        print(f"\n✅ Frame sampling completed!")
        print(f"   Frames extracted: {len(metadata)}")
        print(f"   Frames directory: {sampler.frames_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 