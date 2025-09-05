# Automated Football Heatmap System

This system processes wide-angle football match videos to generate player heatmaps and tracking data.

## Features

- **Player & Ball Detection**: Uses YOLOv8 for accurate detection
- **Object Tracking**: Assigns unique IDs to players across frames
- **Field Mapping**: Converts camera view to 2D pitch coordinates using homography
- **Data Storage**: Saves tracking data in CSV format
- **Heatmap Generation**: Creates visual heatmaps for individual players and teams

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The system will automatically download YOLOv8 model on first run.

## Usage

### Basic Usage
```bash
python football_heatmap.py path/to/your/video.mp4
```

### Advanced Usage
```bash
python football_heatmap.py path/to/your/video.mp4 --output_dir results --model yolov8n.pt
```

## Output

The system generates:
- `tracking_data.csv`: Player positions with timestamps
- `heatmap_player_X.png`: Individual player heatmaps
- `heatmap_Team_1.png` / `heatmap_Team_2.png`: Team heatmaps

## CSV Data Format

| Column | Description |
|--------|-------------|
| timestamp | Time in seconds from video start |
| frame | Frame number |
| player_id | Unique player identifier |
| pixel_x, pixel_y | Original pixel coordinates |
| field_x, field_y | Mapped field coordinates (meters) |
| confidence | Detection confidence score |

## Field Calibration

The system uses automatic field calibration based on default corner points. For better accuracy with different camera angles, you can modify the `calibrate_field()` method to use manual corner selection.

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy, Pandas
- Matplotlib, Seaborn

## Notes

- Works best with wide-angle videos showing the full field
- Player tracking may need adjustment for different camera positions
- Field calibration can be improved with manual corner selection