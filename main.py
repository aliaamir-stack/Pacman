#!/usr/bin/env python3
"""
Football Heatmap System - Command Line Interface
Main script for processing football videos and generating heatmaps.
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.analyzer import FootballAnalyzer

def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description="Automated Football Heatmap System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with interactive calibration
  python main.py process video.mp4
  
  # Process with existing calibration
  python main.py process video.mp4 --calibration calibration.json
  
  # Process with annotated video output
  python main.py process video.mp4 --annotated-video
  
  # Calibrate field only
  python main.py calibrate video.mp4 --frame 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process video and generate heatmaps')
    process_parser.add_argument('video', help='Path to input video file')
    process_parser.add_argument('--output-dir', '-o', help='Output directory (default: video_analysis)')
    process_parser.add_argument('--calibration', '-c', help='Path to calibration file')
    process_parser.add_argument('--confidence', type=float, default=0.5, 
                               help='Detection confidence threshold (default: 0.5)')
    process_parser.add_argument('--frame-skip', type=int, default=1,
                               help='Process every Nth frame (default: 1)')
    process_parser.add_argument('--annotated-video', action='store_true',
                               help='Save annotated video with tracking overlays')
    process_parser.add_argument('--pitch-length', type=float, default=100.0,
                               help='Pitch length in meters (default: 100.0)')
    process_parser.add_argument('--pitch-width', type=float, default=64.0,
                               help='Pitch width in meters (default: 64.0)')
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser('calibrate', help='Calibrate field mapping only')
    calibrate_parser.add_argument('video', help='Path to input video file')
    calibrate_parser.add_argument('--frame', type=int, default=0,
                                 help='Frame number to use for calibration (default: 0)')
    calibrate_parser.add_argument('--output', '-o', help='Output calibration file path')
    calibrate_parser.add_argument('--pitch-length', type=float, default=100.0,
                                 help='Pitch length in meters (default: 100.0)')
    calibrate_parser.add_argument('--pitch-width', type=float, default=64.0,
                                 help='Pitch width in meters (default: 64.0)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show video information')
    info_parser.add_argument('video', help='Path to video file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'process':
            process_video(args)
        elif args.command == 'calibrate':
            calibrate_field(args)
        elif args.command == 'info':
            show_video_info(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def process_video(args):
    """Process video and generate heatmaps."""
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print("=== Football Heatmap System ===")
    print(f"Input video: {video_path}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Pitch dimensions: {args.pitch_length}m x {args.pitch_width}m")
    print()
    
    # Initialize analyzer
    analyzer = FootballAnalyzer(
        confidence_threshold=args.confidence,
        pitch_dimensions=(args.pitch_length, args.pitch_width)
    )
    
    # Load calibration or perform interactive calibration
    if args.calibration:
        calib_path = Path(args.calibration)
        if calib_path.exists():
            print(f"Loading calibration from {calib_path}")
            if not analyzer.load_calibration(str(calib_path)):
                print("Failed to load calibration. Proceeding with interactive calibration.")
                if not analyzer.calibrate_field(str(video_path)):
                    print("Calibration failed. Processing will use image coordinates.")
        else:
            print(f"Calibration file not found: {calib_path}")
            if not analyzer.calibrate_field(str(video_path)):
                print("Calibration failed. Processing will use image coordinates.")
    else:
        print("No calibration file provided. Starting interactive calibration...")
        if not analyzer.calibrate_field(str(video_path)):
            print("Calibration failed. Processing will use image coordinates.")
    
    # Process video
    print("\nStarting video processing...")
    results = analyzer.process_video(
        str(video_path),
        output_dir=args.output_dir,
        save_annotated_video=args.annotated_video,
        frame_skip=args.frame_skip
    )
    
    # Print results summary
    print("\n=== Processing Complete ===")
    print(f"Output directory: {results['output_dir']}")
    print(f"Tracking data: {results['tracking_csv']}")
    print(f"Player summary: {results['player_summary']}")
    print(f"Processing time: {results['processing_time']:.1f}s")
    print(f"Frames processed: {results['frames_processed']}")
    
    stats = results['statistics']
    print(f"\n=== Statistics ===")
    print(f"Total tracking entries: {stats.get('total_entries', 0)}")
    print(f"Unique players detected: {stats.get('unique_players', 0)}")
    print(f"Average players per frame: {stats.get('average_players_per_frame', 0):.1f}")
    print(f"Time range: {stats.get('time_range', [0, 0])[0]:.1f}s - {stats.get('time_range', [0, 0])[1]:.1f}s")

def calibrate_field(args):
    """Calibrate field mapping only."""
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print("=== Field Calibration ===")
    print(f"Video: {video_path}")
    print(f"Frame: {args.frame}")
    print(f"Pitch dimensions: {args.pitch_length}m x {args.pitch_width}m")
    print()
    
    # Initialize analyzer
    analyzer = FootballAnalyzer(
        pitch_dimensions=(args.pitch_length, args.pitch_width)
    )
    
    # Perform calibration
    success = analyzer.calibrate_field(str(video_path), args.frame)
    
    if success:
        # Save calibration
        if args.output:
            output_path = args.output
        else:
            output_path = f"{video_path.stem}_calibration.json"
        
        analyzer.field_mapper.save_calibration(output_path)
        print(f"Calibration saved to: {output_path}")
    else:
        print("Calibration failed!")

def show_video_info(args):
    """Show video file information."""
    import cv2
    
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    print("=== Video Information ===")
    print(f"File: {video_path}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    print(f"File size: {video_path.stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    sys.exit(main())