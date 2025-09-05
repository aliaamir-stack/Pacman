# Football Heatmap System - Usage Guide

## 🚀 Quick Start

### 1. Command Line Usage
```bash
# Process a video file
python3 football_heatmap.py path/to/your/video.mp4

# Specify output directory
python3 football_heatmap.py path/to/your/video.mp4 --output_dir results

# Use different YOLO model
python3 football_heatmap.py path/to/your/video.mp4 --model yolov8s.pt
```

### 2. Web Interface
```bash
# Start the web server
python3 web_interface.py

# Open your browser to http://localhost:5000
```

### 3. Demo
```bash
# Run the demo with sample video
python3 demo.py
```

## 📁 Output Files

The system generates several output files:

- **`tracking_data.csv`**: Player positions with timestamps
- **`heatmap_player_X.png`**: Individual player heatmaps
- **`heatmap_Team_1.png`** / **`heatmap_Team_2.png`**: Team heatmaps
- **`sample_heatmap.png`**: Demo heatmap (when no players detected)

## 📊 CSV Data Format

| Column | Description |
|--------|-------------|
| `timestamp` | Time in seconds from video start |
| `frame` | Frame number |
| `player_id` | Unique player identifier |
| `pixel_x`, `pixel_y` | Original pixel coordinates |
| `field_x`, `field_y` | Mapped field coordinates (meters) |
| `confidence` | Detection confidence score |

## ⚙️ Configuration

### Field Calibration
The system uses automatic field calibration. For better accuracy:

1. **Manual Calibration**: Modify the `calibrate_field()` method
2. **Corner Points**: Adjust the default corner points for your camera angle
3. **Pitch Dimensions**: Change `pitch_dimensions` for different field sizes

### Detection Settings
- **Confidence Threshold**: Adjust in `detect_players_and_ball()` method
- **YOLO Model**: Use different models (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
- **Classes**: Modify detection classes (0=person, 32=sports ball)

## 🔧 Troubleshooting

### No Players Detected
- Lower confidence threshold (currently 0.3)
- Try different YOLO model
- Check video quality and lighting
- Ensure players are visible and not too small

### Poor Tracking
- Adjust tracking distance threshold in `_assign_player_id()`
- Improve video quality
- Use higher resolution videos

### Field Mapping Issues
- Manually calibrate field corners
- Adjust homography matrix calculation
- Check camera angle and field visibility

## 📈 Advanced Features

### Custom Heatmap Styles
Modify the `_create_pitch_background()` method to:
- Change field colors
- Add team logos
- Customize field markings

### Additional Statistics
Extend the system to calculate:
- Distance covered per player
- Zone occupancy percentages
- Pass completion rates
- Speed analysis

### Real-time Processing
For live video processing:
- Use video capture instead of file input
- Implement frame buffering
- Add real-time visualization

## 🎯 Best Practices

1. **Video Quality**: Use high-resolution videos (1080p or higher)
2. **Camera Position**: Wide-angle view showing full field
3. **Lighting**: Good lighting conditions for better detection
4. **Field Visibility**: Clear field markings and boundaries
5. **Player Size**: Players should be clearly visible (not too small)

## 📝 Notes

- The system works best with wide-angle videos showing the full field
- Player tracking may need adjustment for different camera positions
- Field calibration can be improved with manual corner selection
- For production use, consider using more robust tracking algorithms