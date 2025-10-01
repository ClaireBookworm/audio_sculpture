from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
import json
from converter import Audio3DConverter
from stl_to_web import stl_to_threejs_json, get_mesh_info
import threading
import time

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Store conversion status
conversion_status = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page"""
    with open('audio_sculpture_web.html', 'r') as f:
        return f.read()

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle audio file upload and start conversion"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an audio file.'}), 400
        
        # Get conversion parameters
        method = request.form.get('method', 'cylindrical')
        duration = int(request.form.get('duration', 20))
        
        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{conversion_id}_{filename}")
        file.save(file_path)
        
        # Initialize conversion status
        conversion_status[conversion_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting conversion...',
            'stl_file': None,
            'error': None
        }
        
        # Start conversion in background thread
        thread = threading.Thread(target=convert_audio_background, 
                                args=(conversion_id, file_path, method, duration, file.filename))
        thread.start()
        
        return jsonify({
            'conversion_id': conversion_id,
            'message': 'Conversion started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_audio_background(conversion_id, file_path, method, duration, original_filename):
    """Background conversion process"""
    try:
        # Update status
        conversion_status[conversion_id]['message'] = 'Loading audio file...'
        conversion_status[conversion_id]['progress'] = 10
        
        # Create a better output name from the original filename
        original_filename = secure_filename(original_filename)
        # Remove file extension
        name_without_ext = os.path.splitext(original_filename)[0]
        # Get first 2 words
        words = name_without_ext.split()[:2]
        first_two_words = "_".join(words) if words else "audio"
        # Create output name with duration
        output_name = f"{first_two_words}_{duration}s_{conversion_id[:8]}"
        
        # Create converter instance
        converter = Audio3DConverter(file_path, output_name)
        
        # Update status
        conversion_status[conversion_id]['message'] = 'Processing audio spectrum...'
        conversion_status[conversion_id]['progress'] = 30
        
        # Generate the sculpture
        stl_file, spectrogram = converter.generate_sculpture(method=method, duration=duration)
        
        # Move STL file to output folder with better naming
        output_filename = f"{output_name}.stl"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        os.rename(stl_file, output_path)
        
        # Convert STL to web-compatible format for 3D preview
        json_file = stl_to_threejs_json(output_path)
        mesh_info = get_mesh_info(output_path)
        
        # Update status
        conversion_status[conversion_id]['status'] = 'completed'
        conversion_status[conversion_id]['progress'] = 100
        conversion_status[conversion_id]['message'] = 'Conversion completed successfully!'
        conversion_status[conversion_id]['stl_file'] = output_filename
        conversion_status[conversion_id]['image_file'] = None
        conversion_status[conversion_id]['json_file'] = json_file
        conversion_status[conversion_id]['mesh_info'] = mesh_info
        
        # Clean up uploaded file
        os.remove(file_path)
        
    except Exception as e:
        conversion_status[conversion_id]['status'] = 'error'
        conversion_status[conversion_id]['error'] = str(e)
        conversion_status[conversion_id]['message'] = f'Conversion failed: {str(e)}'
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/status/<conversion_id>')
def get_status(conversion_id):
    """Get conversion status"""
    if conversion_id not in conversion_status:
        return jsonify({'error': 'Invalid conversion ID'}), 404
    
    return jsonify(conversion_status[conversion_id])

@app.route('/download/<conversion_id>')
def download_file(conversion_id):
    """Download the generated STL file"""
    if conversion_id not in conversion_status:
        return jsonify({'error': 'Invalid conversion ID'}), 404
    
    status = conversion_status[conversion_id]
    if status['status'] != 'completed' or not status['stl_file']:
        return jsonify({'error': 'File not ready for download'}), 400
    
    file_path = os.path.join(OUTPUT_FOLDER, status['stl_file'])
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True, 
                    download_name=f"audio_sculpture_{conversion_id}.stl")

@app.route('/stl/<conversion_id>')
def serve_stl(conversion_id):
    """Serve STL file for 3D viewer"""
    if conversion_id not in conversion_status:
        return jsonify({'error': 'Invalid conversion ID'}), 404
    
    status = conversion_status[conversion_id]
    if status['status'] != 'completed' or not status['stl_file']:
        return jsonify({'error': 'File not ready'}), 400
    
    file_path = os.path.join(OUTPUT_FOLDER, status['stl_file'])
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, mimetype='application/octet-stream')

@app.route('/preview/<conversion_id>')
def serve_preview_data(conversion_id):
    """Serve 3D preview data (Three.js JSON format)"""
    if conversion_id not in conversion_status:
        return jsonify({'error': 'Invalid conversion ID'}), 404
    
    status = conversion_status[conversion_id]
    if status['status'] != 'completed' or not status.get('json_file'):
        return jsonify({'error': 'Preview data not ready'}), 400
    
    json_file_path = status['json_file']
    if not os.path.exists(json_file_path):
        return jsonify({'error': 'Preview file not found'}), 404
    
    try:
        with open(json_file_path, 'r') as f:
            preview_data = json.load(f)
        
        # Add mesh info to the response
        preview_data['mesh_info'] = status.get('mesh_info', {})
        
        return jsonify(preview_data)
        
    except Exception as e:
        return jsonify({'error': f'Error loading preview data: {str(e)}'}), 500

@app.route('/cleanup')
def cleanup_old_files():
    """Clean up old conversion files"""
    try:
        current_time = time.time()
        cleaned_count = 0
        
        # Clean up old status entries (older than 1 hour)
        to_remove = []
        for conv_id, status in conversion_status.items():
            if current_time - status.get('created_time', current_time) > 3600:
                to_remove.append(conv_id)
        
        for conv_id in to_remove:
            # Remove associated files
            if conversion_status[conv_id].get('stl_file'):
                file_path = os.path.join(OUTPUT_FOLDER, conversion_status[conv_id]['stl_file'])
                if os.path.exists(file_path):
                    os.remove(file_path)
            del conversion_status[conv_id]
            cleaned_count += 1
        
        return jsonify({'message': f'Cleaned up {cleaned_count} old conversions'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎵 Audio Sculpture Generator Server")
    print("=" * 40)
    print("Starting server on http://localhost:8080")
    print("Upload audio files to generate 3D sculptures!")
    print("=" * 40)
    
    app.run(debug=True, host='0.0.0.0', port=8080)
