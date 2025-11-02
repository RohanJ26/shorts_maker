"""
Enhanced emotion analysis module with facial expression detection
"""

import os
import torch
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from moviepy.editor import VideoFileClip
from collections import Counter
from config import VIDEO_MODEL_NAME, KINETICS_TO_EMOTION_MAP, DEFAULT_EMOTION
import random
import glob


class EmotionAnalyzer:
    """Analyzes video clips to determine emotional content using both action and facial expressions"""
    
    def __init__(self):
        # Clear any cached HuggingFace tokens
        os.environ.pop('HF_TOKEN', None)
        os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
        
        self.feature_extractor = None
        self.model = None
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_model()
    
    def load_model(self):
        """Load the VideoMAE model for action recognition"""
        print("Loading VideoMAE model...")
        try:
            self.feature_extractor = VideoMAEImageProcessor.from_pretrained(
                VIDEO_MODEL_NAME,
                use_auth_token=False
            )
            self.model = VideoMAEForVideoClassification.from_pretrained(
                VIDEO_MODEL_NAME,
                use_auth_token=False
            )
            print("✅ Video model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Trying with force_download...")
            self.feature_extractor = VideoMAEImageProcessor.from_pretrained(
                VIDEO_MODEL_NAME,
                use_auth_token=False,
                force_download=True
            )
            self.model = VideoMAEForVideoClassification.from_pretrained(
                VIDEO_MODEL_NAME,
                use_auth_token=False,
                force_download=True
            )
            print("✅ Video model loaded with force_download.")
            
    def get_face_activity(self, frame):
        """
        Detect face presence and activity in a frame
        
        Args:
            frame: numpy array of the frame
            
        Returns:
            activity level: 'excited' if many faces or motion, 'neutral' otherwise
        """
        try:
            # Convert frame to grayscale for face detection
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
                
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Return 'excited' if multiple faces are detected
            if len(faces) > 1:
                return ['excited']
            elif len(faces) == 1:
                return ['neutral']
            
        except Exception as e:
            print(f"Warning: Face detection error - {e}")
        
        return []
    
    def get_clip_emotion(self, video_path, num_frames=16):
        """
        Analyze a video clip to determine its emotional content using both
        action recognition and facial expressions
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample for analysis
            
        Returns:
            Detected emotion string (e.g., 'happy', 'sad', 'angry')
        """
        clip = VideoFileClip(video_path)
        
        # Extract frames evenly spaced throughout the video
        frames = [clip.get_frame((i / num_frames) * clip.duration) 
                  for i in range(num_frames)]
        
        # Detect face activity
        face_activity = []
        for frame in frames:
            face_activity.extend(self.get_face_activity(frame))
        
        # Process frames through the action recognition model
        inputs = self.feature_extractor(list(frames), return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get predicted action label
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class_idx]
        
        # Map action label to emotion
        action_emotion = KINETICS_TO_EMOTION_MAP.get(predicted_label, DEFAULT_EMOTION)
        
        clip.close()
        
        # Combine face activity and action emotions with weights
        all_emotions = face_activity + [action_emotion] * 3  # Give more weight to action emotion
        if all_emotions:
            dominant_emotion = Counter(all_emotions).most_common(1)[0][0]
        else:
            dominant_emotion = DEFAULT_EMOTION
            
        print(f"  - '{video_path}': Action='{predicted_label}', Face Activity={face_activity}, Final='{dominant_emotion}'")
        return dominant_emotion
    
    def analyze_clips(self, video_paths):
        """
        Analyze multiple video clips and determine dominant emotion
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            Tuple of (dominant_emotion, all_emotions_list)
        """
        print("\nAnalyzing video clips for emotional content...")
        clip_emotions = [self.get_clip_emotion(path) for path in video_paths]
        
        # Find the most common emotion
        if clip_emotions:
            dominant_emotion = Counter(clip_emotions).most_common(1)[0][0]
        else:
            dominant_emotion = DEFAULT_EMOTION
        
        print(f"\n✅ Dominant emotion: '{dominant_emotion}'")
        return dominant_emotion, clip_emotions
        
    def get_music_for_emotion(self, emotion, custom_music_path=None):
        """
        Select appropriate background music based on emotion
        
        Args:
            emotion: Detected emotion string
            custom_music_path: Optional path to custom music file
            
        Returns:
            Path to selected music file
        """
        if custom_music_path and os.path.exists(custom_music_path):
            return custom_music_path
            
        # Default music folders
        default_music_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'input_music', 'default_music', emotion)
        
        # Custom music folder
        custom_music_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'input_music')
                                      
        # Look for emotion-specific music in both folders
        music_files = []
        
        # Check default music
        if os.path.exists(default_music_dir):
            music_files.extend(glob.glob(os.path.join(default_music_dir, '*.mp3')))
            music_files.extend(glob.glob(os.path.join(default_music_dir, '*.wav')))
        
        # Check custom music
        emotion_files = [f for f in glob.glob(os.path.join(custom_music_dir, '*.mp3'))
                        if emotion.lower() in os.path.basename(f).lower()]
        music_files.extend(emotion_files)
        
        if music_files:
            return random.choice(music_files)
        
        # If no matching music found, use neutral
        neutral_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'input_music', 'default_music', 'neutral')
        neutral_files = []
        if os.path.exists(neutral_dir):
            neutral_files.extend(glob.glob(os.path.join(neutral_dir, '*.mp3')))
            neutral_files.extend(glob.glob(os.path.join(neutral_dir, '*.wav')))
        
        return random.choice(neutral_files) if neutral_files else None
