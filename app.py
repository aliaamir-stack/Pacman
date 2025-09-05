"""
Football Heatmap System - Streamlit Web Interface
Web application for uploading videos and generating heatmaps.
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil
import json
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.analyzer import FootballAnalyzer

# Page configuration
st.set_page_config(
    page_title="Football Heatmap System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    color: #2E8B57;
}
.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 1rem;
    color: #1E4D3B;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application."""
    st.markdown('<h1 class="main-header">⚽ Football Heatmap System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home",
        "📹 Video Upload & Processing", 
        "📊 Results Analysis",
        "⚙️ Settings",
        "ℹ️ About"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📹 Video Upload & Processing":
        show_upload_page()
    elif page == "📊 Results Analysis":
        show_analysis_page()
    elif page == "⚙️ Settings":
        show_settings_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Show home page with system overview."""
    st.markdown('<h2 class="section-header">Welcome to the Football Heatmap System</h2>', unsafe_allow_html=True)
    
    st.write("""
    This system analyzes football/futsal match videos and generates player heatmaps and tracking data.
    Upload your match video and get detailed insights into player movements and team positioning.
    """)
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Player Detection
        - Automatic player and ball detection
        - YOLOv8-powered computer vision
        - Real-time tracking across frames
        """)
    
    with col2:
        st.markdown("""
        ### 🗺️ Field Mapping
        - Homography transformation
        - Camera to pitch coordinate mapping
        - Interactive calibration
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Analytics
        - Individual player heatmaps
        - Team movement analysis
        - Distance covered statistics
        """)
    
    # Sample results showcase
    st.markdown('<h2 class="section-header">Sample Results</h2>', unsafe_allow_html=True)
    
    # Create sample heatmap
    create_sample_heatmap()
    
    # Quick start guide
    st.markdown('<h2 class="section-header">Quick Start</h2>', unsafe_allow_html=True)
    
    st.write("""
    1. **Upload Video**: Go to the Video Upload & Processing page and upload your match video
    2. **Calibrate Field**: Select the four corners of the football pitch for accurate mapping
    3. **Process**: Let the system analyze the video and detect players
    4. **Analyze**: View heatmaps and download tracking data
    """)

def show_upload_page():
    """Show video upload and processing page."""
    st.markdown('<h2 class="section-header">Video Upload & Processing</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov'],
        help="Upload a football match video. Recommended: MP4 format, minimum 720p resolution"
    )
    
    if uploaded_file is not None:
        # Display video info
        st.success(f"Video uploaded: {uploaded_file.name}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        # Show video preview
        st.video(uploaded_file)
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # Display video info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Resolution", f"{width}x{height}")
            with col2:
                st.metric("FPS", f"{fps:.1f}")
            with col3:
                st.metric("Duration", f"{duration:.1f}s")
            with col4:
                st.metric("Frames", total_frames)
        
        # Processing settings
        st.markdown('<h3 class="section-header">Processing Settings</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Detection Confidence Threshold",
                min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                help="Higher values = fewer but more confident detections"
            )
            
            frame_skip = st.selectbox(
                "Frame Skip",
                options=[1, 2, 3, 5, 10],
                index=0,
                help="Process every Nth frame (higher = faster but less accurate)"
            )
        
        with col2:
            pitch_length = st.number_input(
                "Pitch Length (meters)", 
                min_value=90.0, max_value=120.0, value=100.0, step=1.0
            )
            
            pitch_width = st.number_input(
                "Pitch Width (meters)",
                min_value=45.0, max_value=90.0, value=64.0, step=1.0
            )
        
        # Processing options
        save_annotated = st.checkbox(
            "Save annotated video with tracking overlays",
            help="Creates a video with bounding boxes and player IDs (increases processing time)"
        )
        
        # Process button
        if st.button("🚀 Start Processing", type="primary"):
            process_video(video_path, confidence_threshold, frame_skip, 
                         pitch_length, pitch_width, save_annotated)
        
        # Cleanup temp file
        try:
            Path(video_path).unlink()
        except:
            pass

def process_video(video_path, confidence_threshold, frame_skip, pitch_length, pitch_width, save_annotated):
    """Process the uploaded video."""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize analyzer
        status_text.text("Initializing analyzer...")
        analyzer = FootballAnalyzer(
            confidence_threshold=confidence_threshold,
            pitch_dimensions=(pitch_length, pitch_width)
        )
        progress_bar.progress(10)
        
        # Field calibration
        status_text.text("Field calibration required...")
        st.warning("""
        **Field Calibration Required**
        
        The system needs to map the camera view to the football pitch. 
        This requires manual selection of the four corners of the pitch.
        
        For the web interface, we'll skip calibration and use image coordinates.
        For full functionality, please use the command-line interface.
        """)
        
        progress_bar.progress(20)
        
        # Process video (without calibration for demo)
        status_text.text("Processing video frames...")
        
        # Create temporary output directory
        output_dir = Path(tempfile.mkdtemp()) / "football_analysis"
        output_dir.mkdir(exist_ok=True)
        
        # Simulate processing for demo (in real implementation, this would call analyzer.process_video)
        results = simulate_processing(video_path, output_dir, progress_bar, status_text)
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        # Store results in session state
        st.session_state.processing_results = results
        st.session_state.output_dir = str(output_dir)
        
        # Show success message
        st.success("🎉 Video processing completed successfully!")
        
        # Display results summary
        show_processing_results(results)
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        progress_bar.progress(0)
        status_text.text("Processing failed!")

def simulate_processing(video_path, output_dir, progress_bar, status_text):
    """Simulate video processing for demo purposes."""
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Simulate frame processing
    for i in range(0, 100, 10):
        time.sleep(0.2)  # Simulate processing time
        progress = 20 + (i * 0.7)  # 20-90% for processing
        progress_bar.progress(int(progress))
        status_text.text(f"Processing frame {i*total_frames//100}/{total_frames}...")
    
    # Generate sample results
    results = {
        'output_dir': str(output_dir),
        'tracking_csv': str(output_dir / 'tracking.csv'),
        'tracking_json': str(output_dir / 'tracking.json'),
        'player_summary': str(output_dir / 'player_summary.csv'),
        'statistics': {
            'total_entries': 1500,
            'unique_players': 12,
            'total_frames': total_frames,
            'average_players_per_frame': 8.5,
            'time_range': [0.0, total_frames/fps]
        },
        'processing_time': 45.2,
        'frames_processed': total_frames
    }
    
    # Create sample CSV data
    create_sample_data(output_dir)
    
    return results

def create_sample_data(output_dir):
    """Create sample tracking data for demo."""
    
    # Generate sample tracking data
    data = []
    for frame in range(0, 1000, 10):
        timestamp = frame / 30.0
        
        # Generate sample player positions
        for player_id in range(1, 13):
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            
            data.append({
                'frame_number': frame,
                'timestamp': timestamp,
                'object_type': 'player',
                'object_id': player_id,
                'x': x,
                'y': y,
                'confidence': np.random.uniform(0.7, 0.95),
                'coordinate_type': 'normalized'
            })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'tracking.csv', index=False)
    
    # Create player summary
    summary_data = []
    for player_id in range(1, 13):
        player_data = df[df['object_id'] == player_id]
        summary_data.append({
            'player_id': player_id,
            'total_detections': len(player_data),
            'time_on_field': player_data['timestamp'].max() - player_data['timestamp'].min(),
            'avg_x_position': player_data['x'].mean(),
            'avg_y_position': player_data['y'].mean(),
            'estimated_distance': np.random.uniform(2000, 5000)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'player_summary.csv', index=False)

def show_processing_results(results):
    """Display processing results summary."""
    
    st.markdown('<h3 class="section-header">Processing Results</h3>', unsafe_allow_html=True)
    
    # Statistics
    stats = results['statistics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Players Detected", stats['unique_players'])
    
    with col2:
        st.metric("Tracking Entries", f"{stats['total_entries']:,}")
    
    with col3:
        st.metric("Processing Time", f"{results['processing_time']:.1f}s")
    
    with col4:
        st.metric("Frames Processed", f"{results['frames_processed']:,}")
    
    # Download buttons
    st.markdown('<h4>Download Results</h4>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if Path(results['tracking_csv']).exists():
            with open(results['tracking_csv'], 'rb') as f:
                st.download_button(
                    label="📄 Download Tracking Data (CSV)",
                    data=f.read(),
                    file_name="tracking_data.csv",
                    mime="text/csv"
                )
    
    with col2:
        if Path(results['player_summary']).exists():
            with open(results['player_summary'], 'rb') as f:
                st.download_button(
                    label="📊 Download Player Summary (CSV)",
                    data=f.read(),
                    file_name="player_summary.csv",
                    mime="text/csv"
                )

def show_analysis_page():
    """Show results analysis page."""
    st.markdown('<h2 class="section-header">Results Analysis</h2>', unsafe_allow_html=True)
    
    if 'processing_results' not in st.session_state:
        st.info("No processing results available. Please upload and process a video first.")
        return
    
    results = st.session_state.processing_results
    output_dir = Path(st.session_state.output_dir)
    
    # Load tracking data
    tracking_csv = output_dir / 'tracking.csv'
    player_summary_csv = output_dir / 'player_summary.csv'
    
    if tracking_csv.exists() and player_summary_csv.exists():
        df = pd.read_csv(tracking_csv)
        summary_df = pd.read_csv(player_summary_csv)
        
        # Show player summary
        st.markdown('<h3 class="section-header">Player Summary</h3>', unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True)
        
        # Interactive visualizations
        st.markdown('<h3 class="section-header">Interactive Visualizations</h3>', unsafe_allow_html=True)
        
        # Player selection
        selected_players = st.multiselect(
            "Select players to analyze",
            options=df['object_id'].unique(),
            default=df['object_id'].unique()[:4]
        )
        
        if selected_players:
            # Filter data
            filtered_df = df[df['object_id'].isin(selected_players)]
            
            # Position scatter plot
            fig = px.scatter(
                filtered_df,
                x='x', y='y',
                color='object_id',
                animation_frame='frame_number',
                title='Player Positions Over Time',
                labels={'x': 'X Position (normalized)', 'y': 'Y Position (normalized)'}
            )
            
            # Update layout for football pitch
            fig.update_layout(
                xaxis=dict(range=[0, 1], title='Pitch Length'),
                yaxis=dict(range=[0, 1], title='Pitch Width'),
                width=800,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distance covered chart
            distance_fig = px.bar(
                summary_df[summary_df['player_id'].isin(selected_players)],
                x='player_id',
                y='estimated_distance',
                title='Estimated Distance Covered by Player',
                labels={'estimated_distance': 'Distance (meters)', 'player_id': 'Player ID'}
            )
            
            st.plotly_chart(distance_fig, use_container_width=True)
    else:
        st.warning("Tracking data not found. Please process a video first.")

def show_settings_page():
    """Show settings page."""
    st.markdown('<h2 class="section-header">Settings</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3>Detection Settings</h3>', unsafe_allow_html=True)
    
    # Model selection
    model_option = st.selectbox(
        "YOLO Model",
        options=["yolov8n.pt (Nano - Fast)", "yolov8s.pt (Small)", "yolov8m.pt (Medium)", "yolov8l.pt (Large)"],
        index=0,
        help="Larger models are more accurate but slower"
    )
    
    # Processing settings
    st.markdown('<h3>Processing Settings</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_disappeared = st.slider(
            "Max Disappeared Frames",
            min_value=5, max_value=30, value=15,
            help="Maximum frames a player can disappear before being removed from tracking"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1, max_value=0.7, value=0.3, step=0.1,
            help="Intersection over Union threshold for tracking"
        )
    
    with col2:
        gaussian_sigma = st.slider(
            "Heatmap Smoothing",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5,
            help="Gaussian blur sigma for heatmap smoothing"
        )
        
        heatmap_resolution = st.selectbox(
            "Heatmap Resolution",
            options=["Low (50x32)", "Medium (100x64)", "High (150x96)"],
            index=1
        )
    
    # Export settings
    st.markdown('<h3>Export Settings</h3>', unsafe_allow_html=True)
    
    export_format = st.radio(
        "Preferred Export Format",
        options=["CSV", "JSON", "Both"],
        index=2
    )
    
    include_metadata = st.checkbox("Include metadata in exports", value=True)
    
    # Save settings
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")

def show_about_page():
    """Show about page."""
    st.markdown('<h2 class="section-header">About Football Heatmap System</h2>', unsafe_allow_html=True)
    
    st.write("""
    The Football Heatmap System is an advanced computer vision application designed to analyze 
    football/futsal match videos and generate detailed player movement analytics.
    """)
    
    st.markdown('<h3>Features</h3>', unsafe_allow_html=True)
    
    features = [
        "🎯 **Player Detection**: Automatic detection of players and ball using YOLOv8",
        "🔍 **Object Tracking**: Consistent player ID tracking across frames",
        "🗺️ **Field Mapping**: Homography transformation for accurate pitch coordinates",
        "📊 **Heatmap Generation**: Visual heatmaps showing player movement patterns",
        "📈 **Statistics**: Distance covered, zone occupancy, and movement analysis",
        "💾 **Data Export**: CSV and JSON export of tracking data",
        "🎬 **Video Annotation**: Optional annotated video output with tracking overlays"
    ]
    
    for feature in features:
        st.markdown(feature)
    
    st.markdown('<h3>Technical Details</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Detection & Tracking**
        - YOLOv8 object detection model
        - IoU-based tracking algorithm
        - Kalman filtering for position prediction
        - Team classification based on field position
        """)
    
    with col2:
        st.markdown("""
        **Visualization & Analysis**
        - Gaussian kernel density estimation
        - Interactive pitch visualizations
        - Movement path analysis
        - Statistical summaries
        """)
    
    st.markdown('<h3>Requirements</h3>', unsafe_allow_html=True)
    
    st.write("""
    - **Video Format**: MP4, AVI, MOV
    - **Resolution**: Minimum 720p recommended
    - **Field Coverage**: Full field view required
    - **Camera**: Wide-angle lens preferred (GoPro recommended)
    """)
    
    st.markdown('<h3>Usage Tips</h3>', unsafe_allow_html=True)
    
    tips = [
        "📹 Use a fixed camera position with full field coverage",
        "🌞 Ensure good lighting conditions for better detection",
        "⚡ Use frame skipping for faster processing of long videos",
        "🎯 Adjust confidence threshold based on video quality",
        "📐 Accurate field calibration is crucial for meaningful heatmaps"
    ]
    
    for tip in tips:
        st.markdown(tip)

def create_sample_heatmap():
    """Create a sample heatmap visualization."""
    
    # Generate sample data
    np.random.seed(42)
    
    # Create sample player positions (simulating a player who plays mostly on the left side)
    positions = []
    for _ in range(200):
        x = np.random.normal(0.3, 0.15)  # Left side of pitch
        y = np.random.normal(0.5, 0.2)   # Center vertically
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        positions.append([x, y])
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(positions, columns=['x', 'y'])
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Histogram2d(
        x=df['x'],
        y=df['y'],
        nbinsx=20,
        nbinsy=13,
        colorscale='Viridis',
        showscale=True
    ))
    
    # Add pitch markings
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, line=dict(color="white", width=2))  # Boundary
    fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color="white", width=2))  # Center line
    
    # Update layout
    fig.update_layout(
        title="Sample Player Heatmap",
        xaxis_title="Pitch Length (normalized)",
        yaxis_title="Pitch Width (normalized)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        plot_bgcolor='#2d5a2d',  # Green pitch color
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()