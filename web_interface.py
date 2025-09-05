#!/usr/bin/env python3
"""
Web interface for the Football Heatmap System
Simple Flask-based web app for uploading videos and generating heatmaps
"""

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import tempfile
from football_heatmap import FootballHeatmapProcessor
import uuid

app = Flask(__name__)
app.secret_key = 'football_heatmap_secret_key'

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process video
        try:
            processor = FootballHeatmapProcessor()
            output_dir = os.path.join(OUTPUT_FOLDER, str(uuid.uuid4()))
            os.makedirs(output_dir, exist_ok=True)
            
            # Process video
            csv_path = processor.process_video(filepath, output_dir)
            
            # Generate heatmaps
            processor.generate_heatmaps(csv_path, output_dir)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return redirect(url_for('results', output_id=os.path.basename(output_dir)))
            
        except Exception as e:
            flash(f'Error processing video: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a video file (mp4, avi, mov, mkv)')
    return redirect(url_for('index'))

@app.route('/results/<output_id>')
def results(output_id):
    output_dir = os.path.join(OUTPUT_FOLDER, output_id)
    
    if not os.path.exists(output_dir):
        flash('Results not found')
        return redirect(url_for('index'))
    
    # List generated files
    files = []
    for filename in os.listdir(output_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.csv')):
            files.append({
                'name': filename,
                'path': os.path.join(output_dir, filename),
                'type': 'image' if filename.endswith(('.png', '.jpg', '.jpeg')) else 'data'
            })
    
    return render_template('results.html', files=files, output_id=output_id)

@app.route('/download/<output_id>/<filename>')
def download_file(output_id, filename):
    file_path = os.path.join(OUTPUT_FOLDER, output_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found')
        return redirect(url_for('index'))

@app.route('/demo')
def demo():
    """Run the demo and show results"""
    try:
        from demo import run_demo
        run_demo()
        
        # Find the demo output directory
        demo_output_dir = 'demo_output'
        if os.path.exists(demo_output_dir):
            files = []
            for filename in os.listdir(demo_output_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.csv')):
                    files.append({
                        'name': filename,
                        'path': os.path.join(demo_output_dir, filename),
                        'type': 'image' if filename.endswith(('.png', '.jpg', '.jpeg')) else 'data'
                    })
            
            return render_template('results.html', files=files, output_id='demo', is_demo=True)
        else:
            flash('Demo failed to generate results')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Demo failed: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)