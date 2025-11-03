"""
Main entry point for the automated video editing system
EventVision - Automated Cinematic Video Editor
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

from config import (INPUT_VIDEOS_DIR, INPUT_MUSIC_DIR, OUTPUT_DIR, 
                   USE_STYLE_TRANSFER, DEVICE)
from emotion_analyzer import EmotionAnalyzer
from video_filters import VideoFilter
from video_processor import VideoProcessor


def setup_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [INPUT_VIDEOS_DIR, INPUT_MUSIC_DIR, OUTPUT_DIR]:
        os.makedirs(dir_path, exist_ok=True)


def get_video_files(directory):
    """Get all video files from a directory"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    video_files = []
    
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(directory, file))
    
    return sorted(video_files)


def main():
    """Main execution function"""
    print("=" * 60)
    print("EventVision - Automated Cinematic Video Editor")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Check for videos
    video_paths = get_video_files(INPUT_VIDEOS_DIR)
    
    if len(video_paths) < 2:
        print(f"\n⛔ Error: Please add at least 2 video files to '{INPUT_VIDEOS_DIR}' directory")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .flv")
        sys.exit(1)
    
    print(f"\n✅ Found {len(video_paths)} video files:")
    for path in video_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Set device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Step 1: Analyze emotions
    print("\n" + "=" * 60)
    print("STEP 1: Emotion Analysis")
    print("=" * 60)
    analyzer = EmotionAnalyzer()
    dominant_emotion, all_emotions = analyzer.analyze_clips(video_paths)
    
    # Step 2: Analyze and sequence important parts
    print("\n" + "=" * 60)
    print("STEP 2: Smart Video Sequencing")
    print("=" * 60)
    from video_sequencer import VideoSequencer
    sequencer = VideoSequencer(min_segment_duration=2.0, 
                             max_segment_duration=10.0,
                             importance_threshold=0.7)
                             
    # Analyze each video for important segments
    for video_path in video_paths:
        sequencer.analyze_video(video_path)
        
    # Get optimized sequence
    sequence = sequencer.get_optimal_sequence()
    if not sequence:
        print("⚠️ No important segments found, using full clips.")
        video_segments = [(path, 0, None) for path in video_paths]
    else:
        print("\n✅ Optimal sequence generated:")
        for i, (path, start, end) in enumerate(sequence):
            print(f"  {i+1}. {os.path.basename(path)} [{start:.1f}s - {end:.1f}s]")
        video_segments = sequence
    
    # Step 3: Setup video filter
    print("\n" + "=" * 60)
    print("STEP 3: Setup Video Filters")
    print("=" * 60)
    video_filter = VideoFilter(use_style_transfer=USE_STYLE_TRANSFER, device=device)
    
    # Step 4: Process videos
    print("\n" + "=" * 60)
    print("STEP 4: Video Processing")
    print("=" * 60)
    processor = VideoProcessor(video_filter)
    processor.process_segments(video_segments)
    
    # Step 4: Select music and assemble
    print("\n" + "=" * 60)
    print("STEP 4: Final Assembly")
    print("=" * 60)
    music_path = processor.select_music(dominant_emotion, INPUT_MUSIC_DIR, analyzer)
    final_video = processor.assemble_video(music_path)
    
    # Step 5: Export
    print("\n" + "=" * 60)
    print("STEP 5: Export")
    print("=" * 60)
    output_path = os.path.join(OUTPUT_DIR, "cinematic_output.mp4")
    success = processor.export_video(final_video, output_path)
    
    # Cleanup
    final_video.close()
    
    if success:
        print("\n" + "=" * 60)
        print("✨ Processing Complete! ✨")
        print("=" * 60)
        print(f"Output video: {output_path}")
        print(f"Dominant emotion: {dominant_emotion}")
        print(f"Total clips processed: {len(video_paths)}")
    else:
        print("\n⛔ Processing failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
