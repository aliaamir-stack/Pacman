# Usage Guide - Football Heatmap System

This guide explains how to use the Football Heatmap System to analyze match videos and generate player heatmaps.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd football-heatmap-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Command Line Usage

#### Basic Processing
```bash
python main.py process your_match_video.mp4
```

#### With Custom Settings
```bash
python main.py process match.mp4 \
  --confidence 0.6 \
  --frame-skip 2 \
  --annotated-video \
  --output-dir results/
```

#### Field Calibration Only
```bash
python main.py calibrate match.mp4 --frame 100
```

### 3. Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Detailed Workflow

### Step 1: Video Preparation

**Requirements:**
- Video format: MP4, AVI, or MOV
- Resolution: Minimum 720p (1080p recommended)
- Field coverage: Full football pitch visible
- Camera: Fixed position, wide-angle lens preferred

**Tips:**
- Use GoPro or similar action camera for wide field of view
- Ensure good lighting conditions
- Avoid excessive camera shake
- Position camera high enough to see the entire pitch

### Step 2: Field Calibration

Field calibration maps camera coordinates to real pitch coordinates. This is crucial for accurate heatmaps.

**Interactive Calibration:**
1. The system will display a frame from your video
2. Click on the four corners of the football pitch in order:
   - Bottom-left corner
   - Bottom-right corner
   - Top-right corner
   - Top-left corner
3. Press 'c' to confirm or 'r' to reset
4. The calibration will be saved automatically

**Using Existing Calibration:**
```bash
python main.py process match.mp4 --calibration existing_calibration.json
```

### Step 3: Video Processing

The system will:
1. Detect players and ball in each frame using YOLOv8
2. Track objects across frames maintaining consistent IDs
3. Transform coordinates to pitch space (if calibrated)
4. Generate tracking data and statistics

**Processing Options:**
- `--confidence`: Detection confidence threshold (0.1-0.9)
- `--frame-skip`: Process every Nth frame for speed
- `--annotated-video`: Save video with tracking overlays
- `--pitch-length/width`: Pitch dimensions in meters

### Step 4: Results Analysis

The system generates:

**Data Files:**
- `tracking.csv`: Raw tracking data with timestamps
- `tracking.json`: Structured tracking data by frame
- `player_summary.csv`: Per-player statistics
- `metadata.json`: Processing information

**Visualizations:**
- Individual player heatmaps
- Team heatmaps
- Team comparison heatmaps
- Movement path visualizations

## Understanding the Output

### Tracking Data Format (CSV)

| Column | Description |
|--------|-------------|
| frame_number | Frame number in video |
| timestamp | Time in seconds |
| object_type | 'player' or 'ball' |
| object_id | Unique identifier |
| x, y | Position coordinates |
| bbox_x1, y1, x2, y2 | Bounding box coordinates |
| confidence | Detection confidence (0-1) |
| coordinate_type | 'pitch' or 'image' |

### Player Summary Format

| Column | Description |
|--------|-------------|
| player_id | Unique player identifier |
| total_detections | Number of detections |
| time_on_field | Total time detected (seconds) |
| avg_x_position | Average X position |
| avg_y_position | Average Y position |
| estimated_distance | Distance covered (meters) |

## Advanced Usage

### Custom Pitch Dimensions

For non-standard pitches:
```bash
python main.py process match.mp4 --pitch-length 90 --pitch-width 60
```

### Batch Processing

Process multiple videos:
```bash
for video in *.mp4; do
  python main.py process "$video" --output-dir "results/$(basename "$video" .mp4)"
done
```

### Performance Optimization

For faster processing:
- Use `--frame-skip 2` or higher
- Lower `--confidence` threshold
- Use smaller YOLO model (modify in code)
- Process shorter video segments

### Integration with Other Tools

The CSV output can be imported into:
- Excel/Google Sheets for manual analysis
- R/Python for statistical analysis
- Tableau/Power BI for advanced visualization
- Sports analytics platforms

## Troubleshooting

### Common Issues

**1. Poor Detection Quality**
- Adjust `--confidence` threshold
- Check video quality and lighting
- Ensure full field coverage

**2. Inconsistent Tracking**
- Use higher frame rate video
- Reduce `--frame-skip` value
- Improve field calibration accuracy

**3. Calibration Problems**
- Use a clear frame with visible field corners
- Ensure the entire pitch is visible
- Try different frame numbers

**4. Performance Issues**
- Increase `--frame-skip` for faster processing
- Use GPU acceleration (CUDA)
- Process shorter video segments

### Error Messages

**"Cannot open video"**
- Check file path and format
- Ensure video is not corrupted
- Try converting to MP4 format

**"Calibration failed"**
- Ensure all four corners are visible
- Click points in correct order
- Try a different frame

**"No players detected"**
- Lower confidence threshold
- Check video quality
- Verify field coverage

## Best Practices

### Video Recording
1. Use tripod for stable footage
2. Position camera at midfield line
3. Ensure 10-15 meter height minimum
4. Record entire match continuously
5. Use wide-angle lens (GoPro recommended)

### Processing
1. Always calibrate the field first
2. Test with short clips before full match
3. Use appropriate confidence thresholds
4. Save calibration files for reuse
5. Process in segments for very long videos

### Analysis
1. Verify tracking quality before analysis
2. Filter out low-confidence detections
3. Consider team formations when interpreting heatmaps
4. Account for different half orientations
5. Cross-reference with match events

## Support

For issues and questions:
1. Check this usage guide
2. Review error messages carefully
3. Test with sample videos
4. Check system requirements
5. Report bugs with video samples and error logs