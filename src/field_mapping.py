"""
Field Mapping Module
Handles homography transformation from camera coordinates to pitch coordinates.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
import matplotlib.pyplot as plt

class FieldMapper:
    """
    Maps camera coordinates to real pitch coordinates using homography.
    """
    
    def __init__(self, pitch_length: float = 100.0, pitch_width: float = 64.0):
        """
        Initialize field mapper.
        
        Args:
            pitch_length: Length of football pitch in meters (default FIFA standard)
            pitch_width: Width of football pitch in meters (default FIFA standard)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.homography_matrix = None
        self.calibration_points = None
        
        # Standard pitch dimensions and markings (in meters)
        self.pitch_markings = self._get_pitch_markings()
        
    def _get_pitch_markings(self) -> Dict:
        """
        Get standard football pitch markings in meters.
        
        Returns:
            Dictionary with pitch marking coordinates
        """
        length = self.pitch_length
        width = self.pitch_width
        
        return {
            # Field corners
            'corners': [
                [0, 0],           # Bottom-left
                [length, 0],      # Bottom-right  
                [length, width],  # Top-right
                [0, width]        # Top-left
            ],
            # Center circle
            'center_circle': {
                'center': [length/2, width/2],
                'radius': 9.15
            },
            # Goal areas (6-yard box)
            'goal_area_left': [
                [0, (width-18.32)/2],
                [5.5, (width-18.32)/2],
                [5.5, (width+18.32)/2],
                [0, (width+18.32)/2]
            ],
            'goal_area_right': [
                [length-5.5, (width-18.32)/2],
                [length, (width-18.32)/2],
                [length, (width+18.32)/2],
                [length-5.5, (width+18.32)/2]
            ],
            # Penalty areas (18-yard box)
            'penalty_area_left': [
                [0, (width-40.32)/2],
                [16.5, (width-40.32)/2],
                [16.5, (width+40.32)/2],
                [0, (width+40.32)/2]
            ],
            'penalty_area_right': [
                [length-16.5, (width-40.32)/2],
                [length, (width-40.32)/2],
                [length, (width+40.32)/2],
                [length-16.5, (width+40.32)/2]
            ]
        }
    
    def calibrate_interactive(self, frame: np.ndarray) -> bool:
        """
        Interactive calibration using manual point selection.
        
        Args:
            frame: Sample frame from the video for calibration
            
        Returns:
            True if calibration successful
        """
        print("=== Interactive Field Calibration ===")
        print("Click on the four corners of the football pitch in this order:")
        print("1. Bottom-left corner")
        print("2. Bottom-right corner") 
        print("3. Top-right corner")
        print("4. Top-left corner")
        print("Press 'r' to reset, 'q' to quit, 'c' to confirm calibration")
        
        # Storage for clicked points
        clicked_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(clicked_points) < 4:
                    clicked_points.append([x, y])
                    print(f"Point {len(clicked_points)}: ({x}, {y})")
        
        # Create window and set callback
        cv2.namedWindow('Field Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Field Calibration', mouse_callback)
        
        while True:
            display_frame = frame.copy()
            
            # Draw clicked points
            for i, point in enumerate(clicked_points):
                cv2.circle(display_frame, tuple(point), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"{i+1}", (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw lines between points if we have enough
            if len(clicked_points) >= 2:
                for i in range(len(clicked_points)):
                    start_point = tuple(clicked_points[i])
                    end_point = tuple(clicked_points[(i+1) % len(clicked_points)])
                    if i < len(clicked_points) - 1 or len(clicked_points) == 4:
                        cv2.line(display_frame, start_point, end_point, (255, 0, 0), 2)
            
            # Display instructions
            cv2.putText(display_frame, f"Points: {len(clicked_points)}/4", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Field Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            elif key == ord('r'):
                clicked_points = []
                print("Reset calibration points")
            elif key == ord('c') and len(clicked_points) == 4:
                break
        
        cv2.destroyAllWindows()
        
        if len(clicked_points) != 4:
            print("Calibration cancelled - need exactly 4 points")
            return False
        
        # Calculate homography
        return self.calculate_homography(clicked_points)
    
    def calculate_homography(self, image_points: List[List[float]]) -> bool:
        """
        Calculate homography matrix from image points to pitch coordinates.
        
        Args:
            image_points: List of 4 corner points in image coordinates
            
        Returns:
            True if homography calculation successful
        """
        if len(image_points) != 4:
            print("Error: Need exactly 4 calibration points")
            return False
        
        # Convert to numpy arrays
        src_points = np.array(image_points, dtype=np.float32)
        
        # Corresponding pitch coordinates (in meters)
        dst_points = np.array([
            [0, 0],                              # Bottom-left
            [self.pitch_length, 0],              # Bottom-right
            [self.pitch_length, self.pitch_width], # Top-right
            [0, self.pitch_width]                # Top-left
        ], dtype=np.float32)
        
        # Calculate homography matrix
        self.homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        
        if self.homography_matrix is not None:
            self.calibration_points = image_points
            print("Homography calculation successful!")
            return True
        else:
            print("Error: Failed to calculate homography matrix")
            return False
    
    def transform_point(self, image_point: List[float]) -> Optional[List[float]]:
        """
        Transform a single point from image coordinates to pitch coordinates.
        
        Args:
            image_point: Point in image coordinates [x, y]
            
        Returns:
            Point in pitch coordinates [x, y] or None if not calibrated
        """
        if self.homography_matrix is None:
            return None
        
        # Convert to homogeneous coordinates
        point = np.array([[image_point]], dtype=np.float32)
        
        # Apply transformation
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        
        return transformed[0][0].tolist()
    
    def transform_points(self, image_points: List[List[float]]) -> List[List[float]]:
        """
        Transform multiple points from image coordinates to pitch coordinates.
        
        Args:
            image_points: List of points in image coordinates
            
        Returns:
            List of points in pitch coordinates
        """
        if self.homography_matrix is None or len(image_points) == 0:
            return []
        
        # Convert to numpy array
        points = np.array([image_points], dtype=np.float32)
        
        # Apply transformation
        transformed = cv2.perspectiveTransform(points, self.homography_matrix)
        
        return transformed[0].tolist()
    
    def normalize_coordinates(self, pitch_points: List[List[float]]) -> List[List[float]]:
        """
        Normalize pitch coordinates to 0-1 range.
        
        Args:
            pitch_points: Points in pitch coordinates (meters)
            
        Returns:
            Normalized coordinates (0-1 range)
        """
        normalized = []
        for point in pitch_points:
            norm_x = point[0] / self.pitch_length
            norm_y = point[1] / self.pitch_width
            normalized.append([norm_x, norm_y])
        
        return normalized
    
    def denormalize_coordinates(self, normalized_points: List[List[float]]) -> List[List[float]]:
        """
        Convert normalized coordinates back to pitch coordinates.
        
        Args:
            normalized_points: Points in normalized coordinates (0-1)
            
        Returns:
            Points in pitch coordinates (meters)
        """
        denormalized = []
        for point in normalized_points:
            pitch_x = point[0] * self.pitch_length
            pitch_y = point[1] * self.pitch_width
            denormalized.append([pitch_x, pitch_y])
        
        return denormalized
    
    def save_calibration(self, filepath: str) -> bool:
        """
        Save calibration data to file.
        
        Args:
            filepath: Path to save calibration data
            
        Returns:
            True if save successful
        """
        if self.homography_matrix is None:
            return False
        
        calibration_data = {
            'homography_matrix': self.homography_matrix.tolist(),
            'calibration_points': self.calibration_points,
            'pitch_length': self.pitch_length,
            'pitch_width': self.pitch_width
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"Calibration saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration data from file.
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            True if load successful
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.homography_matrix = np.array(data['homography_matrix'], dtype=np.float32)
            self.calibration_points = data['calibration_points']
            self.pitch_length = data['pitch_length']
            self.pitch_width = data['pitch_width']
            
            print(f"Calibration loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def visualize_calibration(self, frame: np.ndarray, output_path: str = None):
        """
        Visualize the calibration by overlaying pitch markings on the frame.
        
        Args:
            frame: Input frame to overlay markings on
            output_path: Optional path to save visualization
        """
        if self.homography_matrix is None:
            print("No calibration data available")
            return
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw calibration points
        if self.calibration_points:
            for i, point in enumerate(self.calibration_points):
                cv2.circle(vis_frame, tuple(map(int, point)), 8, (0, 255, 0), -1)
                cv2.putText(vis_frame, f"{i+1}", (int(point[0])+15, int(point[1])-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Transform and draw pitch markings
        self._draw_pitch_overlay(vis_frame)
        
        if output_path:
            cv2.imwrite(output_path, vis_frame)
            print(f"Calibration visualization saved to {output_path}")
        else:
            cv2.imshow('Calibration Visualization', vis_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def _draw_pitch_overlay(self, frame: np.ndarray):
        """
        Draw pitch markings overlay on frame.
        
        Args:
            frame: Frame to draw on
        """
        if self.homography_matrix is None:
            return
        
        # Draw field boundary
        corners = self.pitch_markings['corners']
        image_corners = self.transform_points_inverse(corners)
        
        if image_corners:
            pts = np.array(image_corners, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
        
        # Draw center line
        center_line = [
            [self.pitch_length/2, 0],
            [self.pitch_length/2, self.pitch_width]
        ]
        image_center_line = self.transform_points_inverse(center_line)
        if len(image_center_line) == 2:
            pt1 = tuple(map(int, image_center_line[0]))
            pt2 = tuple(map(int, image_center_line[1]))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
        
        # Draw center circle
        center = self.transform_points_inverse([[self.pitch_length/2, self.pitch_width/2]])
        if center:
            center_img = tuple(map(int, center[0]))
            # Approximate radius in image coordinates
            radius_point = [self.pitch_length/2 + 9.15, self.pitch_width/2]
            radius_img = self.transform_points_inverse([radius_point])
            if radius_img:
                radius = int(np.linalg.norm(np.array(center_img) - np.array(radius_img[0])))
                cv2.circle(frame, center_img, radius, (255, 255, 255), 2)
    
    def transform_points_inverse(self, pitch_points: List[List[float]]) -> List[List[float]]:
        """
        Transform points from pitch coordinates back to image coordinates.
        
        Args:
            pitch_points: Points in pitch coordinates
            
        Returns:
            Points in image coordinates
        """
        if self.homography_matrix is None or len(pitch_points) == 0:
            return []
        
        try:
            # Use inverse homography
            inv_homography = np.linalg.inv(self.homography_matrix)
            points = np.array([pitch_points], dtype=np.float32)
            transformed = cv2.perspectiveTransform(points, inv_homography)
            return transformed[0].tolist()
        except:
            return []

# Test function
def test_field_mapper():
    """Test the field mapper with sample data."""
    mapper = FieldMapper()
    
    # Test with sample calibration points
    sample_points = [
        [100, 400],   # Bottom-left
        [700, 450],   # Bottom-right
        [650, 150],   # Top-right
        [150, 100]    # Top-left
    ]
    
    if mapper.calculate_homography(sample_points):
        # Test point transformation
        test_point = [400, 300]
        pitch_point = mapper.transform_point(test_point)
        print(f"Image point {test_point} -> Pitch point {pitch_point}")
        
        # Test normalization
        if pitch_point:
            normalized = mapper.normalize_coordinates([pitch_point])
            print(f"Normalized: {normalized[0]}")
    
    print("FieldMapper test completed!")

if __name__ == "__main__":
    test_field_mapper()