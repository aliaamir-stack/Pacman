"""
Data Storage Module
Handles saving and loading of tracking data in various formats (CSV, JSON).
"""

import pandas as pd
import json
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import csv

class TrackingDataManager:
    """
    Manages storage and retrieval of football tracking data.
    """
    
    def __init__(self):
        """Initialize the data manager."""
        self.tracking_data = []
        self.metadata = {}
        
    def add_frame_data(self, frame_number: int, timestamp: float, 
                      player_tracks: Dict[int, Dict], ball_tracks: Dict[int, Dict],
                      pitch_coordinates: bool = True):
        """
        Add tracking data for a single frame.
        
        Args:
            frame_number: Frame number in video
            timestamp: Timestamp in seconds
            player_tracks: Dictionary of player tracking data
            ball_tracks: Dictionary of ball tracking data
            pitch_coordinates: Whether coordinates are in pitch space (True) or image space (False)
        """
        # Process player data
        for player_id, track_data in player_tracks.items():
            center = track_data['center']
            bbox = track_data['bbox']
            
            entry = {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'object_type': 'player',
                'object_id': player_id,
                'x': center[0],
                'y': center[1],
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3],
                'confidence': track_data.get('confidence', 0.0),
                'coordinate_type': 'pitch' if pitch_coordinates else 'image'
            }
            
            self.tracking_data.append(entry)
        
        # Process ball data
        for ball_id, track_data in ball_tracks.items():
            center = track_data['center']
            bbox = track_data['bbox']
            
            entry = {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'object_type': 'ball',
                'object_id': ball_id,
                'x': center[0],
                'y': center[1],
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3],
                'confidence': track_data.get('confidence', 0.0),
                'coordinate_type': 'pitch' if pitch_coordinates else 'image'
            }
            
            self.tracking_data.append(entry)
    
    def set_metadata(self, video_path: str, fps: float, total_frames: int,
                    resolution: tuple, pitch_dimensions: tuple = (100.0, 64.0)):
        """
        Set metadata for the tracking session.
        
        Args:
            video_path: Path to source video
            fps: Frames per second of video
            total_frames: Total number of frames
            resolution: Video resolution (width, height)
            pitch_dimensions: Pitch dimensions in meters (length, width)
        """
        self.metadata = {
            'video_path': str(video_path),
            'fps': fps,
            'total_frames': total_frames,
            'resolution': resolution,
            'pitch_dimensions': pitch_dimensions,
            'processing_date': datetime.now().isoformat(),
            'total_entries': 0  # Will be updated when saving
        }
    
    def save_csv(self, filepath: str, include_metadata: bool = True) -> bool:
        """
        Save tracking data to CSV format.
        
        Args:
            filepath: Output CSV file path
            include_metadata: Whether to include metadata in separate file
            
        Returns:
            True if save successful
        """
        try:
            if not self.tracking_data:
                print("No tracking data to save")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(self.tracking_data)
            
            # Sort by frame number and object type
            df = df.sort_values(['frame_number', 'object_type', 'object_id'])
            
            # Save CSV
            df.to_csv(filepath, index=False)
            
            # Update metadata
            self.metadata['total_entries'] = len(self.tracking_data)
            
            # Save metadata if requested
            if include_metadata:
                metadata_path = Path(filepath).with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                print(f"Metadata saved to {metadata_path}")
            
            print(f"Tracking data saved to {filepath}")
            print(f"Total entries: {len(self.tracking_data)}")
            return True
            
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    def save_json(self, filepath: str) -> bool:
        """
        Save tracking data to JSON format.
        
        Args:
            filepath: Output JSON file path
            
        Returns:
            True if save successful
        """
        try:
            if not self.tracking_data:
                print("No tracking data to save")
                return False
            
            # Update metadata
            self.metadata['total_entries'] = len(self.tracking_data)
            
            # Organize data by frames
            frames_data = {}
            for entry in self.tracking_data:
                frame_num = entry['frame_number']
                if frame_num not in frames_data:
                    frames_data[frame_num] = {
                        'frame_number': frame_num,
                        'timestamp': entry['timestamp'],
                        'players': [],
                        'balls': []
                    }
                
                obj_data = {
                    'id': entry['object_id'],
                    'position': [entry['x'], entry['y']],
                    'bbox': [entry['bbox_x1'], entry['bbox_y1'], 
                            entry['bbox_x2'], entry['bbox_y2']],
                    'confidence': entry['confidence'],
                    'coordinate_type': entry['coordinate_type']
                }
                
                if entry['object_type'] == 'player':
                    frames_data[frame_num]['players'].append(obj_data)
                else:
                    frames_data[frame_num]['balls'].append(obj_data)
            
            # Create final structure
            output_data = {
                'metadata': self.metadata,
                'frames': list(frames_data.values())
            }
            
            # Save JSON
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Tracking data saved to {filepath}")
            print(f"Total frames: {len(frames_data)}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def load_csv(self, filepath: str) -> bool:
        """
        Load tracking data from CSV format.
        
        Args:
            filepath: Input CSV file path
            
        Returns:
            True if load successful
        """
        try:
            df = pd.read_csv(filepath)
            self.tracking_data = df.to_dict('records')
            
            # Try to load metadata
            metadata_path = Path(filepath).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            print(f"Loaded {len(self.tracking_data)} tracking entries from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def load_json(self, filepath: str) -> bool:
        """
        Load tracking data from JSON format.
        
        Args:
            filepath: Input JSON file path
            
        Returns:
            True if load successful
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.metadata = data.get('metadata', {})
            frames = data.get('frames', [])
            
            # Convert back to flat format
            self.tracking_data = []
            for frame in frames:
                frame_num = frame['frame_number']
                timestamp = frame['timestamp']
                
                # Process players
                for player in frame.get('players', []):
                    entry = {
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'object_type': 'player',
                        'object_id': player['id'],
                        'x': player['position'][0],
                        'y': player['position'][1],
                        'bbox_x1': player['bbox'][0],
                        'bbox_y1': player['bbox'][1],
                        'bbox_x2': player['bbox'][2],
                        'bbox_y2': player['bbox'][3],
                        'confidence': player['confidence'],
                        'coordinate_type': player['coordinate_type']
                    }
                    self.tracking_data.append(entry)
                
                # Process balls
                for ball in frame.get('balls', []):
                    entry = {
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'object_type': 'ball',
                        'object_id': ball['id'],
                        'x': ball['position'][0],
                        'y': ball['position'][1],
                        'bbox_x1': ball['bbox'][0],
                        'bbox_y1': ball['bbox'][1],
                        'bbox_x2': ball['bbox'][2],
                        'bbox_y2': ball['bbox'][3],
                        'confidence': ball['confidence'],
                        'coordinate_type': ball['coordinate_type']
                    }
                    self.tracking_data.append(entry)
            
            print(f"Loaded {len(self.tracking_data)} tracking entries from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return False
    
    def get_player_data(self, player_id: int) -> List[Dict]:
        """
        Get all tracking data for a specific player.
        
        Args:
            player_id: Player ID to filter by
            
        Returns:
            List of tracking entries for the player
        """
        return [entry for entry in self.tracking_data 
                if entry['object_type'] == 'player' and entry['object_id'] == player_id]
    
    def get_ball_data(self) -> List[Dict]:
        """
        Get all ball tracking data.
        
        Returns:
            List of ball tracking entries
        """
        return [entry for entry in self.tracking_data if entry['object_type'] == 'ball']
    
    def get_frame_data(self, frame_number: int) -> Dict:
        """
        Get all tracking data for a specific frame.
        
        Args:
            frame_number: Frame number to filter by
            
        Returns:
            Dictionary with players and balls data for the frame
        """
        frame_entries = [entry for entry in self.tracking_data 
                        if entry['frame_number'] == frame_number]
        
        players = [entry for entry in frame_entries if entry['object_type'] == 'player']
        balls = [entry for entry in frame_entries if entry['object_type'] == 'ball']
        
        return {
            'frame_number': frame_number,
            'players': players,
            'balls': balls
        }
    
    def get_statistics(self) -> Dict:
        """
        Get basic statistics about the tracking data.
        
        Returns:
            Dictionary with statistics
        """
        if not self.tracking_data:
            return {}
        
        df = pd.DataFrame(self.tracking_data)
        
        stats = {
            'total_entries': len(self.tracking_data),
            'total_frames': df['frame_number'].nunique(),
            'unique_players': df[df['object_type'] == 'player']['object_id'].nunique(),
            'unique_balls': df[df['object_type'] == 'ball']['object_id'].nunique(),
            'frame_range': [int(df['frame_number'].min()), int(df['frame_number'].max())],
            'time_range': [float(df['timestamp'].min()), float(df['timestamp'].max())],
            'average_players_per_frame': df[df['object_type'] == 'player'].groupby('frame_number').size().mean(),
            'average_balls_per_frame': df[df['object_type'] == 'ball'].groupby('frame_number').size().mean()
        }
        
        return stats
    
    def export_player_summary(self, filepath: str) -> bool:
        """
        Export a summary of player statistics to CSV.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export successful
        """
        try:
            if not self.tracking_data:
                return False
            
            df = pd.DataFrame(self.tracking_data)
            player_df = df[df['object_type'] == 'player']
            
            if player_df.empty:
                return False
            
            # Calculate per-player statistics
            player_stats = []
            for player_id in player_df['object_id'].unique():
                player_data = player_df[player_df['object_id'] == player_id]
                
                # Calculate distance covered (simplified)
                positions = player_data[['x', 'y']].values
                distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                total_distance = np.sum(distances)
                
                stats = {
                    'player_id': player_id,
                    'total_detections': len(player_data),
                    'first_appearance': player_data['timestamp'].min(),
                    'last_appearance': player_data['timestamp'].max(),
                    'time_on_field': player_data['timestamp'].max() - player_data['timestamp'].min(),
                    'avg_x_position': player_data['x'].mean(),
                    'avg_y_position': player_data['y'].mean(),
                    'x_range': player_data['x'].max() - player_data['x'].min(),
                    'y_range': player_data['y'].max() - player_data['y'].min(),
                    'estimated_distance': total_distance,
                    'avg_confidence': player_data['confidence'].mean()
                }
                
                player_stats.append(stats)
            
            # Save to CSV
            summary_df = pd.DataFrame(player_stats)
            summary_df.to_csv(filepath, index=False)
            
            print(f"Player summary exported to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error exporting player summary: {e}")
            return False
    
    def clear_data(self):
        """Clear all tracking data and metadata."""
        self.tracking_data = []
        self.metadata = {}
        print("Tracking data cleared")

# Test function
def test_data_manager():
    """Test the data manager with sample data."""
    manager = TrackingDataManager()
    
    # Set sample metadata
    manager.set_metadata(
        video_path="test_video.mp4",
        fps=30.0,
        total_frames=1000,
        resolution=(1920, 1080)
    )
    
    # Add sample tracking data
    sample_players = {
        1: {
            'center': [50.0, 32.0],
            'bbox': [45, 25, 55, 40],
            'confidence': 0.9
        },
        2: {
            'center': [60.0, 30.0],
            'bbox': [55, 23, 65, 38],
            'confidence': 0.8
        }
    }
    
    sample_balls = {
        1: {
            'center': [55.0, 31.0],
            'bbox': [53, 29, 57, 33],
            'confidence': 0.7
        }
    }
    
    manager.add_frame_data(0, 0.0, sample_players, sample_balls)
    manager.add_frame_data(1, 0.033, sample_players, sample_balls)
    
    # Test statistics
    stats = manager.get_statistics()
    print("Statistics:", stats)
    
    # Test CSV save/load
    manager.save_csv("test_tracking.csv")
    
    print("TrackingDataManager test completed!")

if __name__ == "__main__":
    test_data_manager()