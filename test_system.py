#!/usr/bin/env python3
"""
Test script for Football Heatmap System
Runs basic tests to verify system functionality.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.detection import FootballDetector
        print("✓ Detection module imported successfully")
    except Exception as e:
        print(f"✗ Detection module import failed: {e}")
        return False
    
    try:
        from src.tracking import PlayerTracker
        print("✓ Tracking module imported successfully")
    except Exception as e:
        print(f"✗ Tracking module import failed: {e}")
        return False
    
    try:
        from src.field_mapping import FieldMapper
        print("✓ Field mapping module imported successfully")
    except Exception as e:
        print(f"✗ Field mapping module import failed: {e}")
        return False
    
    try:
        from src.data_storage import TrackingDataManager
        print("✓ Data storage module imported successfully")
    except Exception as e:
        print(f"✗ Data storage module import failed: {e}")
        return False
    
    try:
        from src.heatmap import FootballHeatmapGenerator
        print("✓ Heatmap module imported successfully")
    except Exception as e:
        print(f"✗ Heatmap module import failed: {e}")
        return False
    
    try:
        from src.analyzer import FootballAnalyzer
        print("✓ Main analyzer imported successfully")
    except Exception as e:
        print(f"✗ Main analyzer import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of each module."""
    print("\nTesting basic functionality...")
    
    try:
        # Test detector initialization
        from src.detection import FootballDetector
        detector = FootballDetector()
        print("✓ FootballDetector initialized")
        
        # Test tracker initialization
        from src.tracking import PlayerTracker
        tracker = PlayerTracker()
        print("✓ PlayerTracker initialized")
        
        # Test field mapper initialization
        from src.field_mapping import FieldMapper
        mapper = FieldMapper()
        print("✓ FieldMapper initialized")
        
        # Test data manager initialization
        from src.data_storage import TrackingDataManager
        data_manager = TrackingDataManager()
        print("✓ TrackingDataManager initialized")
        
        # Test heatmap generator initialization
        from src.heatmap import FootballHeatmapGenerator
        heatmap_gen = FootballHeatmapGenerator()
        print("✓ FootballHeatmapGenerator initialized")
        
        # Test main analyzer initialization
        from src.analyzer import FootballAnalyzer
        analyzer = FootballAnalyzer()
        print("✓ FootballAnalyzer initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing functionality."""
    print("\nTesting data processing...")
    
    try:
        from src.tracking import PlayerTracker
        from src.data_storage import TrackingDataManager
        
        # Test tracking with sample data
        tracker = PlayerTracker()
        sample_detections = {
            'players': [
                {'bbox': [100, 200, 150, 300], 'center': [125, 250], 'confidence': 0.9},
                {'bbox': [300, 180, 350, 280], 'center': [325, 230], 'confidence': 0.8}
            ],
            'balls': [
                {'bbox': [200, 220, 220, 240], 'center': [210, 230], 'confidence': 0.7}
            ]
        }
        
        tracks = tracker.update(sample_detections)
        print(f"✓ Tracking processed: {len(tracks['players'])} players, {len(tracks['balls'])} balls")
        
        # Test data storage
        data_manager = TrackingDataManager()
        data_manager.add_frame_data(0, 0.0, tracks['players'], tracks['balls'])
        print("✓ Data storage working")
        
        # Test statistics
        stats = data_manager.get_statistics()
        print(f"✓ Statistics generated: {stats.get('total_entries', 0)} entries")
        
        return True
        
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        from src.heatmap import FootballHeatmapGenerator
        import numpy as np
        
        # Test heatmap generation
        generator = FootballHeatmapGenerator()
        
        # Generate sample positions
        np.random.seed(42)
        sample_positions = []
        for _ in range(50):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 64)
            sample_positions.append([x, y])
        
        # Test pitch figure creation
        fig, ax = generator.create_pitch_figure()
        print("✓ Pitch figure created")
        
        # Close figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print("✓ Visualization system working")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Football Heatmap System Test Suite ===")
    print()
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test data processing
    if not test_data_processing():
        all_passed = False
    
    # Test visualization
    if not test_visualization():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run command line: python main.py process your_video.mp4")
        print("3. Run web interface: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())