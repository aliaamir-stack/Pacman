"""
Main Football Analyzer Module
Integrates all components for end-to-end video processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from .detection import FootballDetector
from .tracking import PlayerTracker
from .field_mapping import FieldMapper
from .data_storage import TrackingDataManager
from .heatmap import FootballHeatmapGenerator

class FootballAnalyzer:
    """
    Main analyzer class that orchestrates the entire football analysis pipeline.
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 pitch_dimensions: Tuple[float, float] = (100.0, 64.0)):
        """
        Initialize the football analyzer.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Detection confidence threshold
            pitch_dimensions: Pitch dimensions in meters (length, width)
        """
        self.detector = FootballDetector(model_path, confidence_threshold)
        self.tracker = PlayerTracker()
        self.field_mapper = FieldMapper(pitch_dimensions[0], pitch_dimensions[1])
        self.data_manager = TrackingDataManager()
        self.heatmap_generator = FootballHeatmapGenerator(pitch_dimensions[0], pitch_dimensions[1])
        
        self.pitch_dimensions = pitch_dimensions
        self.is_calibrated = False
        
    def calibrate_field(self, video_path: str, frame_number: int = 0) -> bool:
        """
        Calibrate field mapping using a specific frame from the video.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to use for calibration
            
        Returns:
            True if calibration successful
        """
        print("Starting field calibration...")
        
        # Extract frame for calibration
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return False
        
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: Cannot read frame {frame_number}")
            return False
        
        # Perform interactive calibration
        success = self.field_mapper.calibrate_interactive(frame)
        
        if success:
            self.is_calibrated = True
            print("Field calibration completed successfully!")
            
            # Save calibration
            calib_path = Path(video_path).stem + "_calibration.json"
            self.field_mapper.save_calibration(calib_path)
            
            # Create visualization
            vis_path = Path(video_path).stem + "_calibration_viz.jpg"
            self.field_mapper.visualize_calibration(frame, vis_path)
        else:
            print("Field calibration failed!")
        
        return success
    
    def load_calibration(self, calibration_path: str) -> bool:
        """
        Load existing calibration data.
        
        Args:
            calibration_path: Path to calibration file
            
        Returns:
            True if calibration loaded successfully
        """
        success = self.field_mapper.load_calibration(calibration_path)
        if success:
            self.is_calibrated = True
        return success
    
    def process_video(self, 
                     video_path: str,
                     output_dir: Optional[str] = None,
                     save_annotated_video: bool = False,
                     frame_skip: int = 1) -> Dict:
        """
        Process entire video and generate tracking data and heatmaps.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs (default: same as video)
            save_annotated_video: Whether to save annotated video
            frame_skip: Process every Nth frame (for speed)
            
        Returns:
            Dictionary with processing results
        """
        if not self.is_calibrated:
            print("Warning: Field not calibrated. Coordinates will be in image space.")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(video_path).parent / f"{Path(video_path).stem}_analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup data manager metadata
        self.data_manager.set_metadata(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            resolution=(width, height),
            pitch_dimensions=self.pitch_dimensions
        )
        
        # Setup annotated video writer if requested
        writer = None
        if save_annotated_video:
            output_video_path = output_dir / f"{Path(video_path).stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Processing loop
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            timestamp = frame_count / fps
            
            # Detect objects
            detections = self.detector.detect_frame(frame)
            
            # Track objects
            tracks = self.tracker.update(detections)
            
            # Transform coordinates to pitch space if calibrated
            if self.is_calibrated:
                pitch_player_tracks = {}
                pitch_ball_tracks = {}
                
                # Transform player positions
                for track_id, track_data in tracks['players'].items():
                    center = track_data['center']
                    pitch_center = self.field_mapper.transform_point(center)
                    
                    if pitch_center is not None:
                        # Normalize to 0-1 range
                        norm_center = self.field_mapper.normalize_coordinates([pitch_center])[0]
                        
                        pitch_track_data = track_data.copy()
                        pitch_track_data['center'] = norm_center
                        pitch_player_tracks[track_id] = pitch_track_data
                
                # Transform ball positions
                for track_id, track_data in tracks['balls'].items():
                    center = track_data['center']
                    pitch_center = self.field_mapper.transform_point(center)
                    
                    if pitch_center is not None:
                        norm_center = self.field_mapper.normalize_coordinates([pitch_center])[0]
                        
                        pitch_track_data = track_data.copy()
                        pitch_track_data['center'] = norm_center
                        pitch_ball_tracks[track_id] = pitch_track_data
                
                # Store pitch coordinates
                self.data_manager.add_frame_data(
                    frame_count, timestamp, pitch_player_tracks, pitch_ball_tracks, 
                    pitch_coordinates=True
                )
            else:
                # Store image coordinates
                self.data_manager.add_frame_data(
                    frame_count, timestamp, tracks['players'], tracks['balls'],
                    pitch_coordinates=False
                )
            
            # Draw annotations if saving video
            if writer is not None:
                annotated_frame = self._draw_tracking_annotations(frame, tracks)
                writer.write(annotated_frame)
            
            processed_frames += 1
            frame_count += 1
            
            # Progress update
            if processed_frames % 100 == 0:
                elapsed = time.time() - start_time
                fps_processed = processed_frames / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({processed_frames} frames, {fps_processed:.1f} FPS)")
        
        cap.release()
        if writer:
            writer.release()
        
        # Save tracking data
        csv_path = output_dir / f"{Path(video_path).stem}_tracking.csv"
        json_path = output_dir / f"{Path(video_path).stem}_tracking.json"
        
        self.data_manager.save_csv(str(csv_path))
        self.data_manager.save_json(str(json_path))
        
        # Generate heatmaps
        print("Generating heatmaps...")
        self._generate_all_heatmaps(output_dir, Path(video_path).stem)
        
        # Generate statistics
        stats = self.data_manager.get_statistics()
        player_summary_path = output_dir / f"{Path(video_path).stem}_player_summary.csv"
        self.data_manager.export_player_summary(str(player_summary_path))
        
        elapsed_total = time.time() - start_time
        print(f"Processing complete! Total time: {elapsed_total:.1f}s")
        print(f"Processed {processed_frames} frames at {processed_frames/elapsed_total:.1f} FPS")
        
        return {
            'output_dir': str(output_dir),
            'tracking_csv': str(csv_path),
            'tracking_json': str(json_path),
            'player_summary': str(player_summary_path),
            'statistics': stats,
            'processing_time': elapsed_total,
            'frames_processed': processed_frames
        }
    
    def _draw_tracking_annotations(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Draw tracking annotations on frame.
        
        Args:
            frame: Input frame
            tracks: Tracking results
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw player tracks
        for track_id, track_data in tracks['players'].items():
            bbox = track_data['bbox']
            center = track_data['center']
            confidence = track_data['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(annotated, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
            
            # Draw label
            label = f"P{track_id}: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw ball tracks
        for track_id, track_data in tracks['balls'].items():
            bbox = track_data['bbox']
            center = track_data['center']
            confidence = track_data['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw center point
            cv2.circle(annotated, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            
            # Draw label
            label = f"Ball: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return annotated
    
    def _generate_all_heatmaps(self, output_dir: Path, base_name: str):
        """
        Generate all heatmap visualizations.
        
        Args:
            output_dir: Output directory
            base_name: Base filename
        """
        if not self.is_calibrated:
            print("Skipping heatmap generation - field not calibrated")
            return
        
        # Get all player data
        all_player_data = {}
        for entry in self.data_manager.tracking_data:
            if entry['object_type'] == 'player' and entry['coordinate_type'] == 'pitch':
                player_id = entry['object_id']
                if player_id not in all_player_data:
                    all_player_data[player_id] = []
                
                # Convert normalized coordinates back to pitch coordinates
                norm_pos = [entry['x'], entry['y']]
                pitch_pos = self.field_mapper.denormalize_coordinates([norm_pos])[0]
                all_player_data[player_id].append(pitch_pos)
        
        if not all_player_data:
            print("No player data available for heatmap generation")
            return
        
        # Generate individual player heatmaps
        for player_id, positions in all_player_data.items():
            if len(positions) < 10:  # Skip players with too few positions
                continue
            
            output_path = output_dir / f"{base_name}_player_{player_id}_heatmap.png"
            fig = self.heatmap_generator.generate_player_heatmap(
                positions, player_id, str(output_path)
            )
            fig.close()
        
        # Simple team classification (left/right half)
        team1_data = {}
        team2_data = {}
        
        for player_id, positions in all_player_data.items():
            if len(positions) < 10:
                continue
            
            # Average x position to determine team
            avg_x = np.mean([pos[0] for pos in positions])
            
            if avg_x < self.pitch_dimensions[0] / 2:
                team1_data[player_id] = positions
            else:
                team2_data[player_id] = positions
        
        # Generate team heatmaps
        if team1_data:
            team1_path = output_dir / f"{base_name}_team1_heatmap.png"
            fig = self.heatmap_generator.generate_team_heatmap(
                team1_data, "Team 1", str(team1_path)
            )
            fig.close()
        
        if team2_data:
            team2_path = output_dir / f"{base_name}_team2_heatmap.png"
            fig = self.heatmap_generator.generate_team_heatmap(
                team2_data, "Team 2", str(team2_path)
            )
            fig.close()
        
        # Generate comparison heatmap
        if team1_data and team2_data:
            comparison_path = output_dir / f"{base_name}_teams_comparison.png"
            fig = self.heatmap_generator.generate_comparison_heatmap(
                team1_data, team2_data, "Team 1", "Team 2", str(comparison_path)
            )
            fig.close()
        
        # Generate movement paths
        if all_player_data:
            paths_path = output_dir / f"{base_name}_movement_paths.png"
            fig = self.heatmap_generator.generate_movement_paths(
                all_player_data, str(paths_path)
            )
            fig.close()
        
        print(f"Generated heatmaps for {len(all_player_data)} players")

# Test function
def test_analyzer():
    """Test the analyzer with sample data."""
    analyzer = FootballAnalyzer()
    print("FootballAnalyzer initialized successfully!")
    print(f"Components loaded:")
    print(f"  - Detector: {type(analyzer.detector).__name__}")
    print(f"  - Tracker: {type(analyzer.tracker).__name__}")
    print(f"  - Field Mapper: {type(analyzer.field_mapper).__name__}")
    print(f"  - Data Manager: {type(analyzer.data_manager).__name__}")
    print(f"  - Heatmap Generator: {type(analyzer.heatmap_generator).__name__}")

if __name__ == "__main__":
    test_analyzer()