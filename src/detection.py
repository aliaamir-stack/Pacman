"""
Player and Ball Detection Module
Uses YOLOv8 pretrained model to detect players and ball in football match videos.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch

class FootballDetector:
    """
    Football player and ball detector using YOLOv8.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # COCO class IDs for person and sports ball
        self.person_class_id = 0  # Person class in COCO
        self.ball_class_id = 32   # Sports ball class in COCO
        
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def detect_frame(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect players and ball in a single frame.
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            Dictionary containing detected players and ball with bounding boxes
        """
        # Run inference
        results = self.model(frame, device=self.device, verbose=False)
        
        players = []
        balls = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get detection data
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    if conf < self.confidence_threshold:
                        continue
                    
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = xyxy
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'confidence': conf,
                        'area': width * height
                    }
                    
                    # Classify detection
                    if cls == self.person_class_id:
                        # Additional filtering for players (remove small detections, refs, etc.)
                        if self._is_likely_player(detection, frame.shape):
                            players.append(detection)
                    elif cls == self.ball_class_id:
                        # Additional filtering for ball
                        if self._is_likely_ball(detection):
                            balls.append(detection)
        
        return {
            'players': players,
            'balls': balls
        }
    
    def _is_likely_player(self, detection: Dict, frame_shape: Tuple) -> bool:
        """
        Filter detections to identify likely players.
        
        Args:
            detection: Detection dictionary
            frame_shape: Shape of the video frame (height, width, channels)
            
        Returns:
            True if detection is likely a player
        """
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Filter by size (players should be reasonably sized)
        min_height = frame_shape[0] * 0.05  # At least 5% of frame height
        max_height = frame_shape[0] * 0.8   # At most 80% of frame height
        min_width = frame_shape[1] * 0.02   # At least 2% of frame width
        max_width = frame_shape[1] * 0.3    # At most 30% of frame width
        
        if not (min_height <= height <= max_height and min_width <= width <= max_width):
            return False
        
        # Filter by aspect ratio (players should be taller than wide)
        aspect_ratio = height / width
        if aspect_ratio < 1.2 or aspect_ratio > 4.0:
            return False
        
        # Filter by position (players should be on the field, not in stands)
        center_y = detection['center'][1]
        if center_y < frame_shape[0] * 0.1:  # Too high in frame (likely spectators)
            return False
        
        return True
    
    def _is_likely_ball(self, detection: Dict) -> bool:
        """
        Filter detections to identify likely football/ball.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            True if detection is likely a ball
        """
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Ball should be roughly circular (similar width and height)
        aspect_ratio = height / width
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        # Ball should be small relative to players
        area = detection['area']
        if area > 10000:  # Too large to be a ball
            return False
        
        return True
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        Process entire video and detect players/ball in each frame.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            
        Returns:
            List of detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in frame
            detections = self.detect_frame(frame)
            
            # Add frame metadata
            timestamp = frame_count / fps
            frame_result = {
                'frame_number': frame_count,
                'timestamp': timestamp,
                'detections': detections
            }
            frame_results.append(frame_result)
            
            # Draw annotations if saving video
            if writer is not None:
                annotated_frame = self._draw_detections(frame, detections)
                writer.write(annotated_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        if writer:
            writer.release()
        
        print(f"Detection complete. Processed {frame_count} frames.")
        return frame_results
    
    def _draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame
            detections: Detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw players
        for i, player in enumerate(detections['players']):
            bbox = player['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            conf = player['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Player {i+1}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw balls
        for i, ball in enumerate(detections['balls']):
            bbox = ball['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            conf = ball['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"Ball: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return annotated_frame

# Test function
def test_detector():
    """Test the detector with a sample video."""
    detector = FootballDetector()
    
    # This would be used with an actual video file
    # results = detector.detect_video("sample_match.mp4", "annotated_output.mp4")
    # print(f"Detected objects in {len(results)} frames")
    
    print("FootballDetector initialized successfully!")
    print(f"Using device: {detector.device}")

if __name__ == "__main__":
    test_detector()