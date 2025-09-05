#!/usr/bin/env python3
"""
Automated Football Heatmap System
Processes wide-angle football match videos to generate player heatmaps
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import argparse
from pathlib import Path

class FootballHeatmapProcessor:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize the football heatmap processor with YOLO model"""
        self.model = YOLO(model_path)
        self.track_history = {}
        self.field_corners = None
        self.homography_matrix = None
        self.pitch_dimensions = (105, 68)  # Standard football pitch dimensions in meters
        
    def detect_players_and_ball(self, frame: np.ndarray) -> List[Dict]:
        """Detect players and ball in a frame using YOLO"""
        results = self.model(frame, classes=[0, 32])  # person and sports ball classes
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Lower confidence threshold for better detection
                    if conf > 0.3:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class': 'player' if cls == 0 else 'ball',
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        })
        
        return detections
    
    def track_objects(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """Assign unique IDs to players and track them across frames"""
        tracked_objects = []
        
        for detection in detections:
            if detection['class'] == 'player':
                # Simple tracking based on distance to previous positions
                player_id = self._assign_player_id(detection['center'], frame_number)
                detection['player_id'] = player_id
                tracked_objects.append(detection)
            elif detection['class'] == 'ball':
                detection['player_id'] = 'ball'
                tracked_objects.append(detection)
        
        return tracked_objects
    
    def _assign_player_id(self, center: List[int], frame_number: int) -> str:
        """Assign or update player ID based on proximity to existing tracks"""
        min_distance = 50  # pixels
        best_match = None
        
        for player_id, history in self.track_history.items():
            if history:
                last_position = history[-1]['center']
                distance = np.sqrt((center[0] - last_position[0])**2 + 
                                 (center[1] - last_position[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    best_match = player_id
        
        if best_match is None:
            # Create new player ID
            player_id = f"player_{len(self.track_history) + 1}"
            self.track_history[player_id] = []
        
        # Update track history
        if best_match:
            player_id = best_match
        
        self.track_history[player_id].append({
            'center': center,
            'frame': frame_number
        })
        
        return player_id
    
    def calibrate_field(self, frame: np.ndarray, corners: List[Tuple[int, int]] = None):
        """Calibrate field mapping using corner points"""
        if corners is None:
            # Default corners for a standard football field view
            h, w = frame.shape[:2]
            corners = [
                (w//4, h//4),      # Top-left
                (3*w//4, h//4),    # Top-right
                (3*w//4, 3*h//4),  # Bottom-right
                (w//4, 3*h//4)     # Bottom-left
            ]
        
        self.field_corners = corners
        
        # Define target pitch corners (in meters)
        target_corners = np.array([
            [0, 0],
            [self.pitch_dimensions[0], 0],
            [self.pitch_dimensions[0], self.pitch_dimensions[1]],
            [0, self.pitch_dimensions[1]]
        ], dtype=np.float32)
        
        # Convert input corners to numpy array
        src_corners = np.array(corners, dtype=np.float32)
        
        # Calculate homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(src_corners, target_corners)
        
        print(f"Field calibrated with corners: {corners}")
        print(f"Homography matrix calculated")
    
    def map_to_field_coordinates(self, pixel_coords: List[int]) -> Tuple[float, float]:
        """Convert pixel coordinates to field coordinates using homography"""
        if self.homography_matrix is None:
            raise ValueError("Field not calibrated. Call calibrate_field() first.")
        
        # Convert to homogeneous coordinates
        point = np.array([[pixel_coords]], dtype=np.float32)
        
        # Apply homography transformation
        field_point = cv2.perspectiveTransform(point, self.homography_matrix)
        
        return float(field_point[0][0][0]), float(field_point[0][0][1])
    
    def process_video(self, video_path: str, output_dir: str = "output") -> str:
        """Process entire video and generate tracking data"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        
        # Initialize data storage
        tracking_data = []
        frame_number = 0
        
        # Process first frame for calibration
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        # Auto-calibrate field (you can modify this for manual calibration)
        self.calibrate_field(frame)
        
        # Process all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players and ball
            detections = self.detect_players_and_ball(frame)
            
            # Debug: print detection count
            if frame_number % 30 == 0:  # Every second
                print(f"Frame {frame_number}: {len(detections)} detections")
            
            # Track objects
            tracked_objects = self.track_objects(detections, frame_number)
            
            # Convert to field coordinates and store
            for obj in tracked_objects:
                if obj['class'] == 'player':
                    try:
                        field_x, field_y = self.map_to_field_coordinates(obj['center'])
                        tracking_data.append({
                            'timestamp': frame_number / fps,
                            'frame': frame_number,
                            'player_id': obj['player_id'],
                            'pixel_x': obj['center'][0],
                            'pixel_y': obj['center'][1],
                            'field_x': field_x,
                            'field_y': field_y,
                            'confidence': obj['confidence']
                        })
                    except Exception as e:
                        print(f"Error mapping coordinates for frame {frame_number}: {e}")
            
            frame_number += 1
            
            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames")
        
        cap.release()
        
        # Save tracking data
        df = pd.DataFrame(tracking_data)
        csv_path = os.path.join(output_dir, "tracking_data.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Tracking data saved to: {csv_path}")
        print(f"Total detections: {len(tracking_data)}")
        
        return csv_path
    
    def generate_heatmaps(self, csv_path: str, output_dir: str = "output"):
        """Generate heatmaps for each player and team"""
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print("No tracking data found. Creating sample heatmap for demonstration...")
            self._create_sample_heatmap(output_dir)
            return
        
        if len(df) == 0:
            print("No tracking data found. Creating sample heatmap for demonstration...")
            self._create_sample_heatmap(output_dir)
            return
        
        # Create football pitch background
        pitch_img = self._create_pitch_background()
        
        # Generate individual player heatmaps
        player_ids = df['player_id'].unique()
        for player_id in player_ids:
            if player_id == 'ball':
                continue
                
            player_data = df[df['player_id'] == player_id]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(pitch_img, extent=[0, self.pitch_dimensions[0], 0, self.pitch_dimensions[1]])
            
            # Create 2D histogram
            x = player_data['field_x']
            y = player_data['field_y']
            
            # Filter out invalid coordinates
            valid_mask = (x >= 0) & (x <= self.pitch_dimensions[0]) & (y >= 0) & (y <= self.pitch_dimensions[1])
            x = x[valid_mask]
            y = y[valid_mask]
            
            if len(x) > 0:
                # Create heatmap
                hist, xedges, yedges = np.histogram2d(x, y, bins=50, 
                                                    range=[[0, self.pitch_dimensions[0]], 
                                                           [0, self.pitch_dimensions[1]]])
                
                # Plot heatmap
                extent = [0, self.pitch_dimensions[0], 0, self.pitch_dimensions[1]]
                im = ax.imshow(hist.T, origin='lower', extent=extent, alpha=0.7, cmap='hot')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Position Density')
            
            ax.set_xlabel('Field Length (m)')
            ax.set_ylabel('Field Width (m)')
            ax.set_title(f'Heatmap - {player_id}')
            ax.grid(True, alpha=0.3)
            
            # Save heatmap
            output_path = os.path.join(output_dir, f"heatmap_{player_id}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Heatmap saved: {output_path}")
        
        # Generate team heatmap (assuming players 1-11 vs 12-22)
        team1_data = df[df['player_id'].str.contains('player_') & 
                       df['player_id'].str.extract(r'player_(\d+)')[0].astype(int) <= 11]
        team2_data = df[df['player_id'].str.contains('player_') & 
                       df['player_id'].str.extract(r'player_(\d+)')[0].astype(int) > 11]
        
        for team_name, team_data in [("Team_1", team1_data), ("Team_2", team2_data)]:
            if len(team_data) == 0:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(pitch_img, extent=[0, self.pitch_dimensions[0], 0, self.pitch_dimensions[1]])
            
            x = team_data['field_x']
            y = team_data['field_y']
            
            valid_mask = (x >= 0) & (x <= self.pitch_dimensions[0]) & (y >= 0) & (y <= self.pitch_dimensions[1])
            x = x[valid_mask]
            y = y[valid_mask]
            
            if len(x) > 0:
                hist, xedges, yedges = np.histogram2d(x, y, bins=50,
                                                    range=[[0, self.pitch_dimensions[0]], 
                                                           [0, self.pitch_dimensions[1]]])
                
                extent = [0, self.pitch_dimensions[0], 0, self.pitch_dimensions[1]]
                im = ax.imshow(hist.T, origin='lower', extent=extent, alpha=0.7, cmap='hot')
                plt.colorbar(im, ax=ax, label='Position Density')
            
            ax.set_xlabel('Field Length (m)')
            ax.set_ylabel('Field Width (m)')
            ax.set_title(f'Team Heatmap - {team_name}')
            ax.grid(True, alpha=0.3)
            
            output_path = os.path.join(output_dir, f"heatmap_{team_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Team heatmap saved: {output_path}")
    
    def _create_pitch_background(self) -> np.ndarray:
        """Create a football pitch background image"""
        # Create white background
        img = np.ones((680, 1050, 3), dtype=np.uint8) * 255
        
        # Draw field lines
        cv2.rectangle(img, (50, 50), (1000, 630), (0, 0, 0), 2)
        
        # Center line
        cv2.line(img, (525, 50), (525, 630), (0, 0, 0), 2)
        
        # Center circle
        cv2.circle(img, (525, 340), 91, (0, 0, 0), 2)
        
        # Goal areas
        cv2.rectangle(img, (50, 200), (150, 480), (0, 0, 0), 2)
        cv2.rectangle(img, (900, 200), (1000, 480), (0, 0, 0), 2)
        
        # Penalty areas
        cv2.rectangle(img, (50, 150), (250, 530), (0, 0, 0), 2)
        cv2.rectangle(img, (800, 150), (1000, 530), (0, 0, 0), 2)
        
        return img
    
    def _create_sample_heatmap(self, output_dir: str):
        """Create a sample heatmap for demonstration when no real data is available"""
        print("Creating sample heatmap for demonstration...")
        
        # Create sample data
        np.random.seed(42)
        n_points = 1000
        
        # Generate random positions on the field
        x = np.random.normal(self.pitch_dimensions[0]/2, self.pitch_dimensions[0]/4, n_points)
        y = np.random.normal(self.pitch_dimensions[1]/2, self.pitch_dimensions[1]/4, n_points)
        
        # Clip to field boundaries
        x = np.clip(x, 0, self.pitch_dimensions[0])
        y = np.clip(y, 0, self.pitch_dimensions[1])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create pitch background
        pitch_img = self._create_pitch_background()
        ax.imshow(pitch_img, extent=[0, self.pitch_dimensions[0], 0, self.pitch_dimensions[1]])
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=50,
                                            range=[[0, self.pitch_dimensions[0]], 
                                                   [0, self.pitch_dimensions[1]]])
        
        # Plot heatmap
        extent = [0, self.pitch_dimensions[0], 0, self.pitch_dimensions[1]]
        im = ax.imshow(hist.T, origin='lower', extent=extent, alpha=0.7, cmap='hot')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Position Density')
        
        ax.set_xlabel('Field Length (m)')
        ax.set_ylabel('Field Width (m)')
        ax.set_title('Sample Football Heatmap (Demo)')
        ax.grid(True, alpha=0.3)
        
        # Save heatmap
        output_path = os.path.join(output_dir, "sample_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sample heatmap saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Football Heatmap Generator')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output_dir', default='output', help='Output directory for results')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = FootballHeatmapProcessor(args.model)
    
    # Process video
    print("Starting video processing...")
    csv_path = processor.process_video(args.video_path, args.output_dir)
    
    # Generate heatmaps
    print("Generating heatmaps...")
    processor.generate_heatmaps(csv_path, args.output_dir)
    
    print("Processing complete!")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()