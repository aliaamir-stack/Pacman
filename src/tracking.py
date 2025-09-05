"""
Object Tracking Module
Maintains consistent player IDs across frames using tracking algorithms.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class SimpleTracker:
    """
    Simple object tracker using IoU matching and Kalman filtering.
    """
    
    def __init__(self, max_disappeared: int = 10, iou_threshold: float = 0.3):
        """
        Initialize tracker.
        
        Args:
            max_disappeared: Maximum frames an object can disappear before being removed
            iou_threshold: Minimum IoU for matching detections to tracks
        """
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        
        # Track management
        self.tracks = {}  # track_id -> track_info
        self.next_id = 1
        self.disappeared = defaultdict(int)
        
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary mapping track_id to track info
        """
        if len(detections) == 0:
            # No detections - increment disappeared counter
            for track_id in list(self.tracks.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return self.tracks
        
        # If no existing tracks, initialize with all detections
        if len(self.tracks) == 0:
            for detection in detections:
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'center': detection['center'],
                    'confidence': detection['confidence'],
                    'kalman': self._init_kalman(detection['center'])
                }
                self.next_id += 1
            return self.tracks
        
        # Match detections to existing tracks
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid]['center'] for tid in track_ids]
        detection_centers = [det['center'] for det in detections]
        
        # Compute cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            for j, detection in enumerate(detections):
                iou = self._compute_iou(self.tracks[track_id]['bbox'], detection['bbox'])
                cost_matrix[i, j] = 1 - iou
        
        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Track which detections and tracks are unmatched
        matched_tracks = set()
        matched_detections = set()
        
        # Process matches
        for row_idx, col_idx in zip(row_indices, col_indices):
            track_id = track_ids[row_idx]
            detection = detections[col_idx]
            
            # Check if match is good enough
            iou = 1 - cost_matrix[row_idx, col_idx]
            if iou >= self.iou_threshold:
                # Update track
                self.tracks[track_id]['bbox'] = detection['bbox']
                self.tracks[track_id]['center'] = detection['center']
                self.tracks[track_id]['confidence'] = detection['confidence']
                
                # Update Kalman filter
                self._update_kalman(self.tracks[track_id]['kalman'], detection['center'])
                
                # Reset disappeared counter
                self.disappeared[track_id] = 0
                
                matched_tracks.add(track_id)
                matched_detections.add(col_idx)
        
        # Handle unmatched tracks (increment disappeared counter)
        for i, track_id in enumerate(track_ids):
            if i not in [row for row, _ in zip(row_indices, col_indices) if track_id in matched_tracks]:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
        
        # Handle unmatched detections (create new tracks)
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'center': detection['center'],
                    'confidence': detection['confidence'],
                    'kalman': self._init_kalman(detection['center'])
                }
                self.next_id += 1
        
        return self.tracks
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _init_kalman(self, initial_center: List[float]) -> cv2.KalmanFilter:
        """
        Initialize Kalman filter for position prediction.
        
        Args:
            initial_center: Initial center position [x, y]
            
        Returns:
            Initialized Kalman filter
        """
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements
        
        # State: [x, y, vx, vy]
        kalman.statePre = np.array([initial_center[0], initial_center[1], 0, 0], dtype=np.float32)
        kalman.statePost = np.array([initial_center[0], initial_center[1], 0, 0], dtype=np.float32)
        
        # Transition matrix (constant velocity model)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
        
        # Error covariance
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        return kalman
    
    def _update_kalman(self, kalman: cv2.KalmanFilter, measurement: List[float]):
        """
        Update Kalman filter with new measurement.
        
        Args:
            kalman: Kalman filter instance
            measurement: New measurement [x, y]
        """
        kalman.predict()
        kalman.correct(np.array(measurement, dtype=np.float32))

class PlayerTracker:
    """
    Specialized tracker for football players with team classification.
    """
    
    def __init__(self, max_disappeared: int = 15):
        """
        Initialize player tracker.
        
        Args:
            max_disappeared: Maximum frames a player can disappear
        """
        self.player_tracker = SimpleTracker(max_disappeared=max_disappeared, iou_threshold=0.3)
        self.ball_tracker = SimpleTracker(max_disappeared=5, iou_threshold=0.2)
        
        # Player history for team classification
        self.player_history = defaultdict(list)
        
    def update(self, detections: Dict) -> Dict:
        """
        Update tracking for players and ball.
        
        Args:
            detections: Detection results from detector
            
        Returns:
            Updated tracking results
        """
        # Track players
        player_tracks = self.player_tracker.update(detections['players'])
        
        # Track ball
        ball_tracks = self.ball_tracker.update(detections['balls'])
        
        # Update player history for team classification
        for track_id, track_info in player_tracks.items():
            center = track_info['center']
            self.player_history[track_id].append(center)
            
            # Keep only recent history
            if len(self.player_history[track_id]) > 100:
                self.player_history[track_id] = self.player_history[track_id][-100:]
        
        return {
            'players': player_tracks,
            'balls': ball_tracks
        }
    
    def classify_teams(self, frame_shape: Tuple) -> Dict[int, int]:
        """
        Simple team classification based on field position.
        
        Args:
            frame_shape: Shape of video frame (height, width)
            
        Returns:
            Dictionary mapping player_id to team_id (0 or 1)
        """
        team_assignments = {}
        field_center_x = frame_shape[1] / 2
        
        for player_id, positions in self.player_history.items():
            if len(positions) < 10:  # Need enough history
                continue
            
            # Average x position
            avg_x = np.mean([pos[0] for pos in positions[-20:]])  # Last 20 positions
            
            # Simple left/right team assignment
            team_assignments[player_id] = 0 if avg_x < field_center_x else 1
        
        return team_assignments
    
    def get_player_stats(self) -> Dict[int, Dict]:
        """
        Calculate basic movement statistics for each player.
        
        Returns:
            Dictionary with player statistics
        """
        stats = {}
        
        for player_id, positions in self.player_history.items():
            if len(positions) < 2:
                continue
            
            # Calculate distance covered
            total_distance = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_distance += np.sqrt(dx*dx + dy*dy)
            
            # Calculate average position
            avg_x = np.mean([pos[0] for pos in positions])
            avg_y = np.mean([pos[1] for pos in positions])
            
            # Calculate movement area (bounding box)
            min_x = min(pos[0] for pos in positions)
            max_x = max(pos[0] for pos in positions)
            min_y = min(pos[1] for pos in positions)
            max_y = max(pos[1] for pos in positions)
            
            stats[player_id] = {
                'total_distance': total_distance,
                'avg_position': [avg_x, avg_y],
                'movement_area': [(min_x, min_y), (max_x, max_y)],
                'position_count': len(positions)
            }
        
        return stats

# Test function
def test_tracker():
    """Test the tracker with sample data."""
    tracker = PlayerTracker()
    
    # Simulate some detections
    sample_detections = {
        'players': [
            {'bbox': [100, 200, 150, 300], 'center': [125, 250], 'confidence': 0.9},
            {'bbox': [300, 180, 350, 280], 'center': [325, 230], 'confidence': 0.8}
        ],
        'balls': [
            {'bbox': [200, 220, 220, 240], 'center': [210, 230], 'confidence': 0.7}
        ]
    }
    
    # Update tracker
    tracks = tracker.update(sample_detections)
    print(f"Tracked {len(tracks['players'])} players and {len(tracks['balls'])} balls")
    print("PlayerTracker initialized successfully!")

if __name__ == "__main__":
    test_tracker()