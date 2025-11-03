"""
Main video processing and assembly module
"""

import os
import glob
import random
from moviepy.editor import (VideoFileClip, AudioFileClip, concatenate_videoclips,
                            vfx, afx)
from config import (CLIP_DURATION, TRANSITION_DURATION, TARGET_HEIGHT, 
                   MUSIC_LIBRARY, OUTPUT_FILENAME, VIDEO_CODEC, AUDIO_CODEC)


class VideoProcessor:
    """Handles video processing and assembly"""
    
    def __init__(self, video_filter):
        self.video_filter = video_filter
        self.processed_clips = []
    
    def process_single_clip(self, video_path, start_time=None, end_time=None, target_height=None):
        """
        Process a single video clip with filters and effects
        
        Args:
            video_path: Path to input video
            start_time: Start time for the segment (None = start)
            end_time: End time for the segment (None = end)
            target_height: Target height for resizing (None = use config)
            
        Returns:
            Processed VideoFileClip
        """
        target_height = target_height or TARGET_HEIGHT
        clip = VideoFileClip(video_path)
        
        # 1. Extract segment if specified
        if start_time is not None:
            if end_time is None:
                end_time = start_time + CLIP_DURATION
            clip = clip.subclip(start_time, end_time)
        else:
            # Use middle portion for full clips
            start_time = max(0, (clip.duration / 2) - (CLIP_DURATION / 2))
            clip = clip.subclip(start_time, start_time + CLIP_DURATION)
        
        # 2. Resize to standard height
        clip = clip.resize(height=target_height)
        
        # 3. Apply filter
        filter_func = self.video_filter.get_filter_function()
        print(f"  - Applying filter to '{os.path.basename(video_path)}'...")
        clip = clip.fl_image(filter_func)
        
        # 4. Add fade transitions
        clip = clip.fx(vfx.fadein, TRANSITION_DURATION).fx(vfx.fadeout, TRANSITION_DURATION)
        
        return clip
    
    def process_segments(self, video_segments):
        """
        Process video segments with filters and effects
        
        Args:
            video_segments: List of tuples (video_path, start_time, end_time)
        """
        print("\nProcessing video segments...")
        self.processed_clips = []
        
        for video_path, start_time, end_time in video_segments:
            clip = self.process_single_clip(video_path, start_time, end_time)
            self.processed_clips.append(clip)
        
        print("✅ All segments processed.")
    
    def assemble_video(self, music_path=None):
        """
        Assemble all processed clips into final video
        
        Args:
            music_path: Path to background music file
            
        Returns:
            Final VideoFileClip
        """
        print("\nAssembling final video...")
        
        # Concatenate all clips
        final_video = concatenate_videoclips(self.processed_clips, method="compose")
        
        # Add music if provided
        if music_path and os.path.exists(music_path):
            print("Adding background music...")
            main_audio = AudioFileClip(music_path)
            
            # Loop audio to match video duration
            final_video = final_video.set_audio(
                main_audio.fx(afx.audio_loop, duration=final_video.duration)
            )
            
            # Fade out audio at the end
            final_video = final_video.fx(afx.audio_fadeout, TRANSITION_DURATION)
        
        print("✅ Video assembly complete.")
        return final_video
    
    def export_video(self, final_video, output_path=None):
        """
        Export final video to file
        
        Args:
            final_video: VideoFileClip to export
            output_path: Output file path (None = use config)
        """
        output_path = output_path or OUTPUT_FILENAME
        
        print(f"\nExporting video to '{output_path}'...")
        print("This may take several minutes...")
        
        try:
            final_video.write_videofile(
                output_path,
                codec=VIDEO_CODEC,
                audio_codec=AUDIO_CODEC,
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            print(f"\n🎉 Success! Video saved to: {output_path}")
            return True
        except Exception as e:
            print(f"\n⛔ Error during export: {e}")
            return False
    
    def select_music(self, emotion, music_dir, analyzer=None):
        """
        Select appropriate music file based on detected emotion
        
        Args:
            emotion: Detected emotion string
            music_dir: Directory containing music files
            analyzer: Optional EmotionAnalyzer instance for smart music selection
            
        Returns:
            Path to selected music file or None
        """
        # First check for user-provided music in input_music directory
        if os.path.exists(music_dir):
            music_files = [f for f in os.listdir(music_dir) 
                          if f.endswith(('.mp3', '.wav', '.m4a'))]
            if music_files:
                fallback_path = os.path.join(music_dir, music_files[0])
                print(f"Using user-provided music: '{music_files[0]}'")
                return fallback_path

        # If no user music, use emotion-based selection from default_music
        default_music_dir = os.path.join(os.path.dirname(music_dir), 'input_music', 'default_music', emotion)
        if os.path.exists(default_music_dir):
            default_files = []
            default_files.extend(glob.glob(os.path.join(default_music_dir, '*.mp3')))
            default_files.extend(glob.glob(os.path.join(default_music_dir, '*.wav')))
            if default_files:
                chosen_music = random.choice(default_files)
                print(f"Selected default music for emotion '{emotion}': '{os.path.basename(chosen_music)}'")
                return chosen_music
                
        # Try neutral emotion as fallback
        neutral_dir = os.path.join(os.path.dirname(music_dir), 'input_music', 'default_music', 'neutral')
        if os.path.exists(neutral_dir):
            neutral_files = []
            neutral_files.extend(glob.glob(os.path.join(neutral_dir, '*.mp3')))
            neutral_files.extend(glob.glob(os.path.join(neutral_dir, '*.wav')))
            if neutral_files:
                chosen_music = random.choice(neutral_files)
                print(f"Using neutral fallback music: '{os.path.basename(chosen_music)}'")
                return chosen_music

        print("⚠️ No music files available in default or custom directories.")
        return None
