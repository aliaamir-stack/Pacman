#!/usr/bin/env python3
"""
Demo script for the Football Heatmap System
Creates a sample video and processes it to demonstrate the system
"""

import cv2
import numpy as np
import os
from football_heatmap import FootballHeatmapProcessor

def create_sample_video(output_path: str = "sample_match.mp4", duration: int = 10):
    """Create a sample football video for testing"""
    print("Creating sample football video...")
    
    # Video properties
    fps = 30
    width, height = 1280, 720
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create field background
    field_img = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark green field
    
    # Draw field lines
    cv2.rectangle(field_img, (100, 100), (width-100, height-100), (255, 255, 255), 3)
    cv2.line(field_img, (width//2, 100), (width//2, height-100), (255, 255, 255), 3)
    cv2.circle(field_img, (width//2, height//2), 100, (255, 255, 255), 3)
    
    # Goal areas
    cv2.rectangle(field_img, (100, 200), (300, height-200), (255, 255, 255), 3)
    cv2.rectangle(field_img, (width-300, 200), (width-100, height-200), (255, 255, 255), 3)
    
    def draw_person(frame, x, y, color, number):
        """Draw a simple person-like figure"""
        # Body (rectangle)
        cv2.rectangle(frame, (x-8, y-20), (x+8, y+10), color, -1)
        # Head (circle)
        cv2.circle(frame, (x, y-25), 8, color, -1)
        # Arms
        cv2.line(frame, (x-8, y-10), (x-15, y-5), color, 3)
        cv2.line(frame, (x+8, y-10), (x+15, y-5), color, 3)
        # Legs
        cv2.line(frame, (x-5, y+10), (x-8, y+25), color, 3)
        cv2.line(frame, (x+5, y+10), (x+8, y+25), color, 3)
        # Number
        cv2.putText(frame, str(number), (x-5, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for frame_num in range(total_frames):
        # Create frame
        frame = field_img.copy()
        
        # Add moving players (person-like figures)
        t = frame_num / fps
        
        # Team 1 players (blue)
        for i in range(5):
            x = int(200 + 100 * i + 50 * np.sin(t + i))
            y = int(200 + 100 * i + 30 * np.cos(t + i))
            draw_person(frame, x, y, (255, 0, 0), i+1)
        
        # Team 2 players (red)
        for i in range(5):
            x = int(800 + 100 * i + 50 * np.sin(t + i + 1))
            y = int(200 + 100 * i + 30 * np.cos(t + i + 1))
            draw_person(frame, x, y, (0, 0, 255), i+6)
        
        # Add ball
        ball_x = int(width//2 + 100 * np.sin(t * 2))
        ball_y = int(height//2 + 50 * np.cos(t * 2))
        cv2.circle(frame, (ball_x, ball_y), 8, (0, 255, 255), -1)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {output_path}")

def run_demo():
    """Run the complete demo"""
    print("=== Football Heatmap System Demo ===\n")
    
    # Create sample video
    video_path = "sample_match.mp4"
    if not os.path.exists(video_path):
        create_sample_video(video_path, duration=5)  # 5-second demo
    
    # Initialize processor
    print("\nInitializing processor...")
    processor = FootballHeatmapProcessor()
    
    # Process video
    print("Processing video...")
    output_dir = "demo_output"
    csv_path = processor.process_video(video_path, output_dir)
    
    # Generate heatmaps
    print("Generating heatmaps...")
    processor.generate_heatmaps(csv_path, output_dir)
    
    print(f"\nDemo complete! Check the '{output_dir}' directory for results.")
    print("Files generated:")
    
    # List generated files
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            print(f"  - {file}")

if __name__ == "__main__":
    run_demo()