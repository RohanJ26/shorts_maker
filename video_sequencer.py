"""
Video sequencing module that intelligently detects and sequences important parts of videos
"""

import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import logging

class VideoSequencer:
    """Analyzes and sequences important parts of videos"""
    
    def __init__(self, min_segment_duration=2.0, max_segment_duration=10.0, 
                 importance_threshold=0.7):
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.importance_threshold = importance_threshold
        self.clips = []
        
    def _calculate_frame_importance(self, frame):
        """
        Calculate importance score of a frame based on:
        - Motion intensity
        - Visual complexity
        - Edge density
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # Calculate edge density using Canny
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges) / 255.0
        
        # Calculate visual complexity using standard deviation
        complexity = np.std(gray) / 128.0
        
        # Combine metrics
        importance = (edge_density + complexity) / 2.0
        return importance
        
    def _detect_motion(self, frames):
        """Calculate motion between consecutive frames"""
        if len(frames) < 2:
            return [0]
            
        motion_scores = []
        prev_frame = frames[0]
        
        for frame in frames[1:]:
            # Calculate frame difference
            diff = cv2.absdiff(frame, prev_frame)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append(motion_score)
            prev_frame = frame
            
        motion_scores.append(motion_scores[-1])  # Duplicate last score
        return motion_scores
        
    def analyze_video(self, video_path):
        """
        Analyze a video file and detect important segments
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of tuples (start_time, end_time, importance_score)
        """
        print(f"\nAnalyzing video: {os.path.basename(video_path)}")
        clip = VideoFileClip(video_path)
        
        # Sample frames (1 per second)
        sample_times = np.arange(0, clip.duration, 1.0)
        frames = [clip.get_frame(t) for t in sample_times]
        
        # Calculate importance scores
        importance_scores = []
        for frame in frames:
            score = self._calculate_frame_importance(frame)
            importance_scores.append(score)
            
        # Add motion scores
        motion_scores = self._detect_motion(frames)
        total_scores = [(i + m)/2 for i, m in zip(importance_scores, motion_scores)]
        
        # Find peaks in importance scores
        peaks, _ = find_peaks(total_scores, 
                            height=self.importance_threshold,
                            distance=int(self.min_segment_duration))
                            
        # Extract segments around peaks
        segments = []
        for peak in peaks:
            start_time = max(0, sample_times[peak] - self.min_segment_duration/2)
            end_time = min(clip.duration, 
                         sample_times[peak] + self.min_segment_duration/2)
            
            # Ensure minimum duration
            if end_time - start_time < self.min_segment_duration:
                end_time = min(clip.duration, start_time + self.min_segment_duration)
                
            # Limit maximum duration
            if end_time - start_time > self.max_segment_duration:
                end_time = start_time + self.max_segment_duration
                
            importance = total_scores[peak]
            segments.append((start_time, end_time, importance))
            
        clip.close()
        
        # Store segments with source video
        self.clips.append({
            'path': video_path,
            'segments': segments
        })
        
        print(f"Found {len(segments)} important segments")
        return segments
        
    def _cluster_segments(self, all_segments):
        """Group segments by similarity in importance scores"""
        if not all_segments:
            return []
            
        # Extract importance scores
        X = np.array([score for _, _, score in all_segments]).reshape(-1, 1)
        
        # Determine optimal number of clusters (max 5)
        n_clusters = min(5, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Sort clusters by average importance
        cluster_scores = []
        for i in range(n_clusters):
            mask = clusters == i
            avg_score = np.mean(X[mask])
            cluster_scores.append((i, avg_score))
            
        # Sort clusters by importance score
        cluster_order = [c for c, _ in sorted(cluster_scores, 
                                            key=lambda x: x[1], 
                                            reverse=True)]
                                            
        # Assign sequence numbers
        sequence = []
        for i, (start, end, score) in enumerate(all_segments):
            cluster = clusters[i]
            sequence_pos = cluster_order.index(cluster)
            sequence.append((start, end, score, sequence_pos))
            
        return sorted(sequence, key=lambda x: x[3])
        
    def get_optimal_sequence(self):
        """
        Generate optimal sequence of video segments
        
        Returns:
            List of tuples (video_path, start_time, end_time)
        """
        if not self.clips:
            return []
            
        # Collect all segments with video paths
        all_segments = []
        for clip_info in self.clips:
            for start, end, score in clip_info['segments']:
                all_segments.append((clip_info['path'], start, end, score))
                
        # Sort by importance score initially
        all_segments.sort(key=lambda x: x[3], reverse=True)
        
        # Group segments by importance using clustering
        importance_only = [(0, 0, score) for _, _, _, score in all_segments]
        sequence = self._cluster_segments(importance_only)
        
        # Build final sequence
        final_sequence = []
        for i, (_, _, _, seq_num) in enumerate(sequence):
            video_path, start, end, _ = all_segments[i]
            final_sequence.append((video_path, start, end))
            
        return final_sequence
        
    def extract_sequence(self, output_path=None):
        """
        Extract and concatenate video segments in optimal sequence
        
        Args:
            output_path: Optional path for preview (not saving final video)
            
        Returns:
            VideoFileClip with concatenated segments
        """
        sequence = self.get_optimal_sequence()
        if not sequence:
            return None
            
        print("\nExtracting video segments in optimal sequence:")
        clips = []
        
        for i, (video_path, start, end) in enumerate(sequence):
            print(f"  {i+1}. {os.path.basename(video_path)} [{start:.1f}s - {end:.1f}s]")
            video = VideoFileClip(video_path)
            segment = video.subclip(start, end)
            clips.append(segment)
            video.close()
            
        final_video = concatenate_videoclips(clips)
        
        # Preview if path provided
        if output_path:
            print(f"\nSaving preview to: {output_path}")
            final_video.write_videofile(output_path, 
                                      codec='libx264',
                                      audio_codec='aac')
            
        return final_video
