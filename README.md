# 🎵 Audio Sculpture Generator

Transform your audio files into beautiful 3D sculptures! This web application takes audio files and converts them into 3D STL meshes that can be 3D printed or viewed in 3D modeling software.

## Features

- **Multiple Sculpture Methods**:
  - **Cylindrical**: Creates vinyl record-like spirals
  - **Rectangular**: Traditional heightfield mapping
  - **Helical**: DNA-like spiral structures
  - **Topographical**: Mountain ranges with bass as foothills, treble as peaks

- **Real-time Processing**: Upload audio and watch the conversion progress
- **3D Visualization**: Preview your generated sculptures
- **Download STL Files**: Get your sculptures ready for 3D printing
- **Modern Web Interface**: Beautiful, responsive design

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python server.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5000`

## Usage

1. **Upload Audio**: Drag and drop or select an audio file (WAV, MP3, FLAC, M4A, OGG)
2. **Choose Method**: Select your preferred sculpture generation method
3. **Set Duration**: Choose how many seconds of audio to process (5-60 seconds)
4. **Generate**: Click "Generate 3D Sculpture" and wait for processing
5. **Download**: Once complete, download your STL file

## How It Works

The application uses advanced audio processing techniques:

1. **Audio Analysis**: Loads and analyzes your audio file using librosa
2. **Spectrogram Generation**: Creates frequency-time representations
3. **3D Mapping**: Maps audio data to 3D coordinates:
   - Time → X-axis (progression)
   - Frequency → Y-axis (pitch height)  
   - Amplitude → Z-axis (volume as elevation)
4. **Mesh Generation**: Creates triangular mesh for STL export
5. **STL Export**: Generates 3D printable files

## Supported Audio Formats

- WAV
- MP3
- FLAC
- M4A
- OGG

## 3D Printing Tips

- **Scale**: STL files are generated in millimeters - scale as needed for your printer
- **Supports**: Complex geometries may require support structures
- **Layer Height**: Use 0.1-0.2mm layer height for best detail
- **Infill**: 10-20% infill is usually sufficient for decorative pieces

## Technical Details

- **Backend**: Flask server with CORS support
- **Audio Processing**: librosa for spectrogram analysis
- **3D Generation**: numpy-stl for STL file creation
- **Frontend**: Modern HTML5 with JavaScript
- **File Handling**: Secure upload and temporary file management

## Troubleshooting

- **Large Files**: Processing time increases with audio duration
- **Memory**: Very long audio files may require more RAM
- **File Size**: STL files can be large for complex audio
- **Browser**: Use modern browsers for best experience

## License

This project is open source. Feel free to modify and distribute!

---

*Transform your music into art! 🎨*
