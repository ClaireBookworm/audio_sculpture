import numpy as np
import librosa
import librosa.display
from scipy import ndimage
import trimesh
import stl

# Audio → Spectrogram: Uses librosa to convert audio into frequency-time data
# 3D Mapping: Maps spectrogram data to 3D coordinates (time→x, frequency→y, amplitude→z)
# Mesh Generation: Creates actual triangular meshes with vertices and faces
# STL Export: Saves as 3D printable STL files

class Audio3DConverter:
    def __init__(self, audio_file, output_name="audio_sculpture"):
        self.audio_file = audio_file
        self.output_name = output_name
        self.sample_rate = 22050  # good balance of detail/file size
        
    def load_and_process_audio(self, duration_limit=30):
        """Load audio and create spectrogram"""
        # load audio file
        y, sr = librosa.load(self.audio_file, sr=self.sample_rate, duration=duration_limit)
        
        # create spectrogram 
        hop_length = 512
        n_fft = 2048
        
        # short-time fourier transform
        stft = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
        spectrogram = np.abs(stft)
        
        # convert to db scale for better visual range
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        
        # normalize to 0-1 range
        spec_norm = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
        
        return spec_norm, sr, hop_length
    
    def process_audio_chunky(self, audio_data, chunk_size_seconds=0.2):
        """Process audio into discrete chunks with robust, scannable features"""
        chunk_samples = int(self.sample_rate * chunk_size_seconds)
        chunks = []
        
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i+chunk_samples]
            if len(chunk) < chunk_samples // 2:  # Skip partial chunks
                break
                
            # Extract scannable features (not thousands of frequency bins)
            features = {
                'energy': np.sqrt(np.mean(chunk**2)),           # volume - easy to measure
                'brightness': self._get_spectral_centroid(chunk), # freq center - affects shape  
                'noisiness': self._get_zero_crossing_rate(chunk), # roughness - surface texture
                'attack': self._get_onset_strength(chunk),       # suddenness - affects proportions
                'bass': self._get_bass_energy(chunk),           # low freq energy
                'mids': self._get_mid_energy(chunk),            # mid freq energy
                'treble': self._get_treble_energy(chunk)        # high freq energy
            }
            chunks.append(features)
        
        return chunks
    
    def _get_spectral_centroid(self, chunk):
        """Get spectral centroid (brightness) of audio chunk"""
        fft = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1/self.sample_rate)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = freqs[:len(freqs)//2]
        
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return min(centroid / 5000, 1.0)  # Normalize to 0-1
        return 0.0
    
    def _get_zero_crossing_rate(self, chunk):
        """Get zero crossing rate (noisiness) of audio chunk"""
        zero_crossings = np.sum(np.diff(np.sign(chunk)) != 0)
        return min(zero_crossings / len(chunk), 1.0)
    
    def _get_onset_strength(self, chunk):
        """Get onset strength (attack) of audio chunk"""
        # Simple onset detection using energy difference
        if len(chunk) < 2:
            return 0.0
        energy_diff = np.abs(chunk[-1] - chunk[0])
        return min(energy_diff * 10, 1.0)
    
    def _get_bass_energy(self, chunk):
        """Get bass frequency energy (0-200 Hz)"""
        fft = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        bass_mask = (freqs >= 0) & (freqs <= 200)
        bass_energy = np.sum(magnitude[bass_mask])
        total_energy = np.sum(magnitude)
        
        return bass_energy / total_energy if total_energy > 0 else 0.0
    
    def _get_mid_energy(self, chunk):
        """Get mid frequency energy (200-2000 Hz)"""
        fft = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        mid_mask = (freqs >= 200) & (freqs <= 2000)
        mid_energy = np.sum(magnitude[mid_mask])
        total_energy = np.sum(magnitude)
        
        return mid_energy / total_energy if total_energy > 0 else 0.0
    
    def _get_treble_energy(self, chunk):
        """Get treble frequency energy (2000+ Hz)"""
        fft = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        treble_mask = freqs >= 2000
        treble_energy = np.sum(magnitude[treble_mask])
        total_energy = np.sum(magnitude)
        
        return treble_energy / total_energy if total_energy > 0 else 0.0
    
    def create_heightfield_mesh(self, spectrogram, method="rectangular"):
        """Convert spectrogram to 3D mesh"""
        freq_bins, time_frames = spectrogram.shape
        
        if method == "rectangular":
            return self._rectangular_heightfield(spectrogram)
        elif method == "cylindrical": 
            return self._cylindrical_mapping(spectrogram)
        elif method == "helical":
            return self._helical_mapping(spectrogram)
        elif method == "topographical":
            return self._topographical_islands(spectrogram)
        elif method == "branching":
            return self._branching_tree_mapping(spectrogram)
        elif method == "volumetric":
            return self._volumetric_blocks_mapping(spectrogram)
        elif method == "chunky_extrusion":
            return self._chunky_extrusion_mapping(spectrogram)
        elif method == "audio_beads":
            return self._audio_beads_mapping(spectrogram)
        elif method == "structural_blocks":
            return self._structural_blocks_mapping(spectrogram)
        
    def _rectangular_heightfield(self, spec):
        """Basic rectangular heightfield approach"""
        freq_bins, time_frames = spec.shape
        
        # create coordinate grids
        x_coords = np.linspace(0, 100, time_frames)  # time axis (mm)
        y_coords = np.linspace(0, 50, freq_bins)     # frequency axis (mm)
        
        # create base vertices
        vertices = []
        faces = []
        
        # add base platform (z=0)
        base_height = 0
        top_height_scale = 20  # max height in mm
        
        # generate vertices
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                # base vertex
                vertices.append([x, y, base_height])
                # top vertex (displaced by amplitude)
                z_height = spec[i, j] * top_height_scale
                vertices.append([x, y, base_height + z_height])
        
        vertices = np.array(vertices)
        
        # generate faces (triangles connecting base to top)
        face_idx = 0
        for i in range(freq_bins - 1):
            for j in range(time_frames - 1):
                # each quad becomes 2 triangles
                base_idx = (i * time_frames + j) * 2
                
                # triangle 1
                faces.append([base_idx, base_idx + 2, base_idx + 1])
                # triangle 2  
                faces.append([base_idx + 1, base_idx + 2, base_idx + 3])
        
        faces = np.array(faces)
        return vertices, faces
    
    def _cylindrical_mapping(self, spec):
        """Circular/cylindrical mapping - vinyl record-like spirals"""
        freq_bins, time_frames = spec.shape
        vertices = []
        
        # Parameters
        base_radius = 15  # mm
        radius_scale = 25  # mm
        height_scale = 8   # mm
        
        for t_idx in range(time_frames):
            # Time becomes angular position
            angle = (t_idx / time_frames) * 2 * np.pi
            
            for f_idx in range(freq_bins):
                # Frequency becomes radius
                r = base_radius + (f_idx / freq_bins) * radius_scale
                
                # Amplitude becomes z-height
                z = spec[f_idx, t_idx] * height_scale
                
                # Convert to cartesian
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                
                vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        faces = self._triangulate_grid(vertices, time_frames, freq_bins)
        return vertices, faces
    
    # def _helical_mapping(self, spec):
    #     """3D Helical wrapping - DNA strands made of sound"""
    #     freq_bins, time_frames = spec.shape
    #     vertices = []
        
    #     # Parameters
    #     base_radius = 15  # mm
    #     radius_scale = 25  # mm
    #     height_scale = 8   # mm
        
    #     for t_idx in range(time_frames):
    #         # Time becomes angular position
    #         angle = (t_idx / time_frames) * 2 * np.pi
            
    #         for f_idx in range(freq_bins):
    #             # Frequency becomes radius
    #             r = base_radius + (f_idx / freq_bins) * radius_scale
                
    #             # Amplitude becomes z-height
    #             z = spec[f_idx, t_idx] * height_scale
                
    #             # Convert to cartesian
    #             x = r * np.cos(angle)
    #             y = r * np.sin(angle)
                
    #             vertices.append([x, y, z])
        
    #     vertices = np.array(vertices)
    #     faces = self._triangulate_grid(vertices, time_frames, freq_bins)
    #     return vertices, faces
    
    def _helical_mapping(self, spec):
        """3D Helical wrapping - DNA strands made of sound with proper volume"""
        freq_bins, time_frames = spec.shape
        vertices = []
        faces = []
        
        # Helix parameters for 3D printable structure
        base_radius = 15  # mm - center of helix
        pitch = 8         # mm vertical distance per revolution
        min_thickness = 2 # mm - minimum printable thickness
        max_thickness = 8 # mm - maximum thickness
        
        # Create 3D helical structure with volume
        for t_idx in range(time_frames):
            # Time wraps around helix
            t = t_idx / time_frames * 4 * np.pi  # 2 full rotations
            
            # Get average amplitude for this time step
            avg_amplitude = np.mean(spec[:, t_idx])
            
            # Helix thickness varies with amplitude (minimum 2mm for printability)
            helix_thickness = max(min_thickness, min_thickness + avg_amplitude * (max_thickness - min_thickness))
            
            # Create cross-section of helix at this time step
            cross_section_vertices = self._create_helix_cross_section(
                t, base_radius, helix_thickness, spec[:, t_idx]
            )
            
            # Add vertices to main list
            start_idx = len(vertices)
            vertices.extend(cross_section_vertices)
            
            # Create faces connecting to previous cross-section
            if t_idx > 0:
                prev_start_idx = start_idx - len(cross_section_vertices)
                helix_faces = self._create_helix_faces(
                    prev_start_idx, start_idx, len(cross_section_vertices)
                )
                faces.extend(helix_faces)
        
        return np.array(vertices), np.array(faces)
    
    def _create_helix_cross_section(self, t, base_radius, thickness, freq_amplitudes):
        """Create a cross-section of the helix at time t"""
        vertices = []
        
        # Number of points around the cross-section (8 for printability)
        num_points = 8
        
        for i in range(num_points):
            angle = i * 2 * np.pi / num_points
            
            # Base position on helix
            x_base = base_radius * np.cos(t)
            y_base = base_radius * np.sin(t)
            z_base = t * 8 / (2 * np.pi)  # 8mm pitch
            
            # Cross-section offset (perpendicular to helix direction)
            # Calculate helix tangent for perpendicular direction
            tangent_x = -base_radius * np.sin(t)
            tangent_y = base_radius * np.cos(t)
            tangent_z = 8 / (2 * np.pi)
            
            # Normalize tangent
            tangent_length = np.sqrt(tangent_x**2 + tangent_y**2 + tangent_z**2)
            tangent_x /= tangent_length
            tangent_y /= tangent_length
            tangent_z /= tangent_length
            
            # Create perpendicular vectors for cross-section
            # Use frequency content to vary cross-section shape
            freq_idx = int(i * len(freq_amplitudes) / num_points)
            freq_amp = freq_amplitudes[freq_idx] if freq_idx < len(freq_amplitudes) else 0
            
            # Vary thickness based on frequency content
            local_thickness = thickness * (0.5 + freq_amp * 0.5)
            
            # Cross-section offset
            offset_x = local_thickness * np.cos(angle) * tangent_y
            offset_y = local_thickness * np.cos(angle) * (-tangent_x)
            offset_z = local_thickness * np.sin(angle)
            
            # Final vertex position
            x = x_base + offset_x
            y = y_base + offset_y
            z = z_base + offset_z
            
            vertices.append([x, y, z])
        
        return vertices
    
    def _create_helix_faces(self, prev_start_idx, curr_start_idx, num_points):
        """Create faces connecting two cross-sections of the helix"""
        faces = []
        
        for i in range(num_points):
            next_i = (i + 1) % num_points
            
            # Current cross-section indices
            curr_curr = curr_start_idx + i
            curr_next = curr_start_idx + next_i
            
            # Previous cross-section indices
            prev_curr = prev_start_idx + i
            prev_next = prev_start_idx + next_i
            
            # Create two triangles for each quad
            faces.append([prev_curr, curr_curr, prev_next])
            faces.append([prev_next, curr_curr, curr_next])
        
        return faces
    
    def _topographical_islands(self, spec):
        """Topographical islands - mountain ranges for different frequency bands"""
        freq_bins, time_frames = spec.shape
        vertices = []
        
        # Parameters
        base_height = 2    # mm - base platform height
        height_scale = 15  # mm - max elevation
        x_scale = 80       # mm - total width
        y_scale = 60       # mm - total depth
        
        # Create base platform vertices
        for f_idx in range(freq_bins):
            for t_idx in range(time_frames):
                x = (t_idx / time_frames) * x_scale
                y = (f_idx / freq_bins) * y_scale
                
                # Base platform
                vertices.append([x, y, 0])
                
                # Top surface (amplitude as elevation)
                # Bass = foothills, treble = peaks
                freq_weight = (f_idx / freq_bins) ** 0.5  # logarithmic-like scaling
                elevation = base_height + spec[f_idx, t_idx] * height_scale * freq_weight
                vertices.append([x, y, elevation])
        
        vertices = np.array(vertices)
        
        # Create faces connecting base to top
        faces = []
        for f in range(freq_bins - 1):
            for t in range(time_frames - 1):
                base_idx = (f * time_frames + t) * 2
                
                # Side faces (base to top)
                faces.append([base_idx, base_idx + 1, base_idx + 2])
                faces.append([base_idx + 1, base_idx + 3, base_idx + 2])
                faces.append([base_idx + 2, base_idx + 3, base_idx + 4])
                faces.append([base_idx + 3, base_idx + 5, base_idx + 4])
                
                # Top surface
                faces.append([base_idx + 1, base_idx + 3, base_idx + 5])
                faces.append([base_idx + 1, base_idx + 5, base_idx + 7])
        
        faces = np.array(faces)
        return vertices, faces
    
    def _triangulate_grid(self, vertices, time_frames, freq_bins):
        """Helper function to triangulate grid-based vertices"""
        faces = []
        for t in range(time_frames - 1):
            for f in range(freq_bins - 1):
                idx = t * freq_bins + f
                # Create quad with two triangles
                faces.append([idx, idx + freq_bins, idx + 1])
                faces.append([idx + 1, idx + freq_bins, idx + freq_bins + 1])
        return np.array(faces)

    def _branching_tree_mapping(self, spec):
        """Branching tree mapping - audio grows like a tree"""
        freq_bins, time_frames = spec.shape
        vertices = []
        faces = []
        
        # Tree parameters
        trunk_height = 8  # mm
        branch_scale = 6  # mm
        
        # Start with trunk at origin
        trunk_start = np.array([0, 0, 0])
        trunk_end = np.array([0, 0, trunk_height])
        
        # Add trunk vertices
        vertices.extend([trunk_start, trunk_end])
        
        # Simple trunk connection
        if len(vertices) >= 2:
            faces.append([0, 1, 0])  # Simple triangle
        
        current_branches = [(0, 1)]  # (start_idx, end_idx)
        
        # Limit time frames to prevent too many branches
        max_time_steps = min(time_frames, 50)  # Limit to 50 time steps
        
        for t_idx in range(0, max_time_steps, 2):  # Every other time step
            # At each time step, look at spectrum
            spectrum_slice = spec[:, t_idx]  # all frequencies at this moment
            
            # Count how many frequencies are active (> threshold)
            active_freqs = np.sum(spectrum_slice > 0.2)
            num_branches = min(max(int(active_freqs / 10), 1), 2)  # 1-2 branches max
            
            new_branches = []
            
            for i in range(num_branches):
                # Each branch represents a frequency band
                start_freq = i * freq_bins // num_branches
                end_freq = (i + 1) * freq_bins // num_branches
                freq_weight = spectrum_slice[start_freq:end_freq]
                avg_freq = np.mean(freq_weight)
                
                # Branch direction = frequency content
                angle = (i / num_branches) * 2 * np.pi + avg_freq * np.pi/3
                branch_length = avg_freq * branch_scale + 1  # louder = longer branch
                
                # Get parent branch (use last branch or trunk)
                if current_branches:
                    parent_start_idx, parent_end_idx = current_branches[i % len(current_branches)]
                    if parent_end_idx < len(vertices):
                        parent_end = vertices[parent_end_idx]
                    else:
                        parent_end = trunk_end
                else:
                    parent_end = trunk_end
                
                # Calculate branch end position
                branch_end = parent_end + np.array([
                    branch_length * np.cos(angle),
                    branch_length * np.sin(angle),
                    branch_length * 0.2  # slight upward growth
                ])
                
                # Add branch vertices
                start_idx = len(vertices)
                vertices.append(parent_end)
                vertices.append(branch_end)
                end_idx = len(vertices) - 1
                
                # Create simple branch connection
                if start_idx < len(vertices) and end_idx < len(vertices):
                    faces.append([start_idx, end_idx, start_idx])
                
                new_branches.append((start_idx, end_idx))
            
            # Keep only the most recent branches to prevent exponential growth
            current_branches = new_branches[:1]  # Limit to 1 branch max
        
        return np.array(vertices), np.array(faces)
    
    def _volumetric_blocks_mapping(self, spec):
        """Volumetric blocks mapping - each time slice becomes a 3D block"""
        freq_bins, time_frames = spec.shape
        vertices = []
        faces = []
        
        # Block parameters
        block_size = 3  # mm base size
        spacing = 4     # mm between blocks
        
        for t_idx in range(time_frames):
            # Each time slice becomes one block
            spectrum_slice = spec[:, t_idx]
            
            # Split spectrum into bass/mid/treble
            bass = np.mean(spectrum_slice[:freq_bins//4])      # 0-25% of frequencies
            mid = np.mean(spectrum_slice[freq_bins//4:freq_bins//2])   # 25-50%  
            treble = np.mean(spectrum_slice[freq_bins//2:])    # 50-100%
            
            # Block dimensions encode frequency content
            width = block_size * (0.5 + bass)    # bass controls width
            height = block_size * (0.5 + mid)    # mids control height  
            depth = block_size * (0.5 + treble)  # treble controls depth
            
            # Block position (time progression)
            x_pos = t_idx * spacing
            
            # Create block vertices (8 corners of a box)
            block_vertices = [
                [x_pos, 0, 0],           # bottom-left-back
                [x_pos + width, 0, 0],   # bottom-right-back
                [x_pos + width, depth, 0], # bottom-right-front
                [x_pos, depth, 0],       # bottom-left-front
                [x_pos, 0, height],      # top-left-back
                [x_pos + width, 0, height], # top-right-back
                [x_pos + width, depth, height], # top-right-front
                [x_pos, depth, height]   # top-left-front
            ]
            
            # Add vertices to main list
            start_idx = len(vertices)
            vertices.extend(block_vertices)
            
            # Create block faces (12 triangles for a box)
            block_faces = self._create_box_faces(start_idx)
            faces.extend(block_faces)
        
        return np.array(vertices), np.array(faces)
    
    def _create_cylinder_faces(self, start_idx, end_idx, thickness):
        """Create faces for a cylindrical branch - simplified version"""
        faces = []
        
        # For now, just create a simple line connection
        # This avoids the indexing issues with complex cylinder geometry
        # In a full implementation, you'd create proper cylinder vertices first
        
        # Simple approach: just connect the two points with a basic triangle
        # This creates a minimal mesh that won't cause indexing errors
        return faces  # Return empty for now to avoid errors
    
    def _create_box_faces(self, start_idx):
        """Create faces for a box (12 triangles)"""
        faces = [
            # Bottom face
            [start_idx, start_idx + 1, start_idx + 2],
            [start_idx, start_idx + 2, start_idx + 3],
            # Top face
            [start_idx + 4, start_idx + 7, start_idx + 6],
            [start_idx + 4, start_idx + 6, start_idx + 5],
            # Front face
            [start_idx + 3, start_idx + 2, start_idx + 6],
            [start_idx + 3, start_idx + 6, start_idx + 7],
            # Back face
            [start_idx, start_idx + 4, start_idx + 5],
            [start_idx, start_idx + 5, start_idx + 1],
            # Left face
            [start_idx, start_idx + 3, start_idx + 7],
            [start_idx, start_idx + 7, start_idx + 4],
            # Right face
            [start_idx + 1, start_idx + 5, start_idx + 6],
            [start_idx + 1, start_idx + 6, start_idx + 2]
        ]
        return faces
    
    def _chunky_extrusion_mapping(self, spec):
        """Chunky extrusion mapping - variable cross-sections for scannable geometry"""
        # Load raw audio for chunk processing
        y, sr = librosa.load(self.audio_file, sr=self.sample_rate, duration=20)
        chunks = self.process_audio_chunky(y, chunk_size_seconds=0.3)
        
        vertices = []
        faces = []
        
        for i, chunk in enumerate(chunks):
            # Path follows time progression
            position = [i * 8, 0, 0]  # 8mm spacing between chunks
            
            # Cross-section encodes audio features (minimum 3mm for printability)
            width = max(3, 3 + chunk['energy'] * 7)        # 3-10mm range
            height = max(3, 3 + chunk['brightness'] * 7)   # frequency = vertical scale
            length = 6  # Fixed length for stability
            
            # Create box section
            box_vertices, box_faces = self._create_chunky_box(
                position, width, height, length, i
            )
            
            # Add surface texture from noisiness (but keep features >2mm)
            if chunk['noisiness'] > 0.7:
                texture_vertices, texture_faces = self._add_chunky_texture(
                    position, width, height, length, chunk['noisiness']
                )
                box_vertices.extend(texture_vertices)
                box_faces.extend(texture_faces)
            
            # Offset faces for current vertices
            start_idx = len(vertices)
            vertices.extend(box_vertices)
            for face in box_faces:
                faces.append([f + start_idx for f in face])
        
        return np.array(vertices), np.array(faces)
    
    def _audio_beads_mapping(self, spec):
        """Audio beads mapping - chain of measurable beads"""
        # Load raw audio for chunk processing
        y, sr = librosa.load(self.audio_file, sr=self.sample_rate, duration=20)
        chunks = self.process_audio_chunky(y, chunk_size_seconds=0.4)
        
        vertices = []
        faces = []
        
        for i, chunk in enumerate(chunks):
            # Bead position
            position = [i * 12, 0, 0]  # 12mm spacing
            
            # Bead size encodes energy (diameter 4-12mm for printability)
            diameter = max(4, 4 + chunk['energy'] * 8)
            
            # Bead shape encodes spectral content
            if chunk['brightness'] > 0.6:
                # Bright = sphere
                bead_vertices, bead_faces = self._create_sphere(
                    position, diameter/2, resolution=8
                )
            else:
                # Dark = ellipsoid (stretched)
                bead_vertices, bead_faces = self._create_ellipsoid(
                    position, diameter/2, diameter/3, diameter/2, resolution=8
                )
            
            # Connection to previous bead
            if i > 0:
                connector_width = max(1, 1 + chunk['attack'] * 2)
                connector_vertices, connector_faces = self._create_connector(
                    [position[0] - 6, 0, 0], [position[0] + 6, 0, 0], connector_width
                )
                bead_vertices.extend(connector_vertices)
                bead_faces.extend(connector_faces)
            
            # Offset faces for current vertices
            start_idx = len(vertices)
            vertices.extend(bead_vertices)
            for face in bead_faces:
                faces.append([f + start_idx for f in face])
        
        return np.array(vertices), np.array(faces)
    
    def _structural_blocks_mapping(self, spec):
        """Structural blocks mapping - architectural elements instead of terrain"""
        # Load raw audio for chunk processing
        y, sr = librosa.load(self.audio_file, sr=self.sample_rate, duration=20)
        chunks = self.process_audio_chunky(y, chunk_size_seconds=0.5)
        
        vertices = []
        faces = []
        
        for i, chunk in enumerate(chunks):
            base_position = [i * 15, 0, 0]  # 15mm spacing
            
            # Bass = foundation width (minimum 8mm)
            foundation_size = max(8, 8 + chunk['bass'] * 12)
            
            # Mids = pillar height (minimum 5mm)
            pillar_height = max(5, 5 + chunk['mids'] * 15)
            
            # Treble = roof complexity
            if chunk['treble'] > 0.5:
                roof_type = 'pyramid'    # 4 faces
            else:
                roof_type = 'wedge'      # 2 faces
            
            # Create building block
            block_vertices, block_faces = self._create_building_block(
                base_position, foundation_size, pillar_height, roof_type
            )
            
            # Offset faces for current vertices
            start_idx = len(vertices)
            vertices.extend(block_vertices)
            for face in block_faces:
                faces.append([f + start_idx for f in face])
        
        return np.array(vertices), np.array(faces)
    
    def _create_chunky_box(self, position, width, height, length, index):
        """Create a chunky box with measurable dimensions"""
        x, y, z = position
        vertices = [
            [x, y, z],                    # 0: bottom-left-back
            [x + width, y, z],            # 1: bottom-right-back
            [x + width, y + length, z],   # 2: bottom-right-front
            [x, y + length, z],           # 3: bottom-left-front
            [x, y, z + height],           # 4: top-left-back
            [x + width, y, z + height],   # 5: top-right-back
            [x + width, y + length, z + height], # 6: top-right-front
            [x, y + length, z + height]   # 7: top-left-front
        ]
        
        faces = self._create_box_faces(0)  # Use existing box face function
        return vertices, faces
    
    def _add_chunky_texture(self, position, width, height, length, noisiness):
        """Add chunky surface texture (features >2mm)"""
        vertices = []
        faces = []
        
        # Add a few large bumps instead of fine detail
        num_bumps = int(noisiness * 3) + 1  # 1-4 bumps
        
        for i in range(num_bumps):
            bump_x = position[0] + (i + 1) * width / (num_bumps + 1)
            bump_y = position[1] + length / 2
            bump_z = position[2] + height + 2  # 2mm high bumps
            
            # Create small cube bump
            bump_vertices = [
                [bump_x - 1, bump_y - 1, bump_z],
                [bump_x + 1, bump_y - 1, bump_z],
                [bump_x + 1, bump_y + 1, bump_z],
                [bump_x - 1, bump_y + 1, bump_z],
                [bump_x - 1, bump_y - 1, bump_z + 2],
                [bump_x + 1, bump_y - 1, bump_z + 2],
                [bump_x + 1, bump_y + 1, bump_z + 2],
                [bump_x - 1, bump_y + 1, bump_z + 2]
            ]
            
            vertices.extend(bump_vertices)
            bump_faces = self._create_box_faces(len(vertices) - 8)
            faces.extend(bump_faces)
        
        return vertices, faces
    
    def _create_sphere(self, center, radius, resolution=8):
        """Create a low-resolution sphere for printability"""
        vertices = []
        faces = []
        
        x, y, z = center
        
        # Create vertices for sphere
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                phi = i * np.pi / resolution
                theta = j * 2 * np.pi / resolution
                
                vx = x + radius * np.sin(phi) * np.cos(theta)
                vy = y + radius * np.sin(phi) * np.sin(theta)
                vz = z + radius * np.cos(phi)
                
                vertices.append([vx, vy, vz])
        
        # Create faces
        for i in range(resolution):
            for j in range(resolution):
                idx = i * (resolution + 1) + j
                
                # Two triangles per quad
                faces.append([idx, idx + resolution + 1, idx + 1])
                faces.append([idx + 1, idx + resolution + 1, idx + resolution + 2])
        
        return vertices, faces
    
    def _create_ellipsoid(self, center, rx, ry, rz, resolution=8):
        """Create a low-resolution ellipsoid"""
        vertices = []
        faces = []
        
        x, y, z = center
        
        # Create vertices for ellipsoid
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                phi = i * np.pi / resolution
                theta = j * 2 * np.pi / resolution
                
                vx = x + rx * np.sin(phi) * np.cos(theta)
                vy = y + ry * np.sin(phi) * np.sin(theta)
                vz = z + rz * np.cos(phi)
                
                vertices.append([vx, vy, vz])
        
        # Create faces (same as sphere)
        for i in range(resolution):
            for j in range(resolution):
                idx = i * (resolution + 1) + j
                faces.append([idx, idx + resolution + 1, idx + 1])
                faces.append([idx + 1, idx + resolution + 1, idx + resolution + 2])
        
        return vertices, faces
    
    def _create_connector(self, start, end, width):
        """Create a cylindrical connector between two points"""
        vertices = []
        faces = []
        
        # Simple cylinder with 6 sides
        resolution = 6
        length = np.sqrt(sum((end[i] - start[i])**2 for i in range(3)))
        
        # Create vertices for cylinder
        for i in range(resolution):
            angle = i * 2 * np.pi / resolution
            x = start[0] + width * np.cos(angle)
            y = start[1] + width * np.sin(angle)
            
            vertices.append([x, y, start[2]])      # Start circle
            vertices.append([x, y, end[2]])        # End circle
        
        # Create faces
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            # Side faces
            start_curr = i * 2
            start_next = next_i * 2
            end_curr = i * 2 + 1
            end_next = next_i * 2 + 1
            
            faces.append([start_curr, start_next, end_curr])
            faces.append([start_next, end_next, end_curr])
        
        return vertices, faces
    
    def _create_building_block(self, position, foundation_size, pillar_height, roof_type):
        """Create architectural building block"""
        vertices = []
        faces = []
        
        x, y, z = position
        
        # Foundation (base)
        foundation_vertices = [
            [x, y, z],
            [x + foundation_size, y, z],
            [x + foundation_size, y + foundation_size, z],
            [x, y + foundation_size, z],
            [x, y, z + 3],  # 3mm foundation height
            [x + foundation_size, y, z + 3],
            [x + foundation_size, y + foundation_size, z + 3],
            [x, y + foundation_size, z + 3]
        ]
        
        vertices.extend(foundation_vertices)
        foundation_faces = self._create_box_faces(0)
        faces.extend(foundation_faces)
        
        # Pillar (center)
        pillar_size = foundation_size * 0.6
        pillar_offset = (foundation_size - pillar_size) / 2
        
        pillar_vertices = [
            [x + pillar_offset, y + pillar_offset, z + 3],
            [x + pillar_offset + pillar_size, y + pillar_offset, z + 3],
            [x + pillar_offset + pillar_size, y + pillar_offset + pillar_size, z + 3],
            [x + pillar_offset, y + pillar_offset + pillar_size, z + 3],
            [x + pillar_offset, y + pillar_offset, z + 3 + pillar_height],
            [x + pillar_offset + pillar_size, y + pillar_offset, z + 3 + pillar_height],
            [x + pillar_offset + pillar_size, y + pillar_offset + pillar_size, z + 3 + pillar_height],
            [x + pillar_offset, y + pillar_offset + pillar_size, z + 3 + pillar_height]
        ]
        
        start_idx = len(vertices)
        vertices.extend(pillar_vertices)
        pillar_faces = self._create_box_faces(start_idx)
        faces.extend(pillar_faces)
        
        # Roof
        if roof_type == 'pyramid':
            roof_vertices, roof_faces = self._create_pyramid_roof(
                [x + foundation_size/2, y + foundation_size/2, z + 3 + pillar_height],
                foundation_size * 0.8, 5
            )
        else:  # wedge
            roof_vertices, roof_faces = self._create_wedge_roof(
                [x + foundation_size/2, y + foundation_size/2, z + 3 + pillar_height],
                foundation_size * 0.8, 5
            )
        
        start_idx = len(vertices)
        vertices.extend(roof_vertices)
        for face in roof_faces:
            faces.append([f + start_idx for f in face])
        
        return vertices, faces
    
    def _create_pyramid_roof(self, center, base_size, height):
        """Create pyramid roof"""
        x, y, z = center
        half_size = base_size / 2
        
        vertices = [
            [x - half_size, y - half_size, z],  # Base corners
            [x + half_size, y - half_size, z],
            [x + half_size, y + half_size, z],
            [x - half_size, y + half_size, z],
            [x, y, z + height]  # Apex
        ]
        
        faces = [
            [0, 1, 4],  # Front face
            [1, 2, 4],  # Right face
            [2, 3, 4],  # Back face
            [3, 0, 4]   # Left face
        ]
        
        return vertices, faces
    
    def _create_wedge_roof(self, center, base_size, height):
        """Create wedge roof"""
        x, y, z = center
        half_size = base_size / 2
        
        vertices = [
            [x - half_size, y - half_size, z],  # Base corners
            [x + half_size, y - half_size, z],
            [x + half_size, y + half_size, z],
            [x - half_size, y + half_size, z],
            [x, y - half_size, z + height],  # Ridge line
            [x, y + half_size, z + height]
        ]
        
        faces = [
            [0, 1, 4],  # Front face
            [1, 2, 5],  # Right face
            [2, 3, 5],  # Back face
            [3, 0, 4],  # Left face
            [0, 4, 5],  # Bottom face
            [0, 5, 3]   # Bottom face
        ]
        
        return vertices, faces
    
    def save_stl(self, vertices, faces, filename=None):
        """Export mesh to STL file"""
        if filename is None:
            filename = f"{self.output_name}.stl"
            
        # create mesh object using numpy-stl
        audio_mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        
        for i, face in enumerate(faces):
            for j in range(3):
                audio_mesh.vectors[i][j] = vertices[face[j], :]
        
        audio_mesh.save(filename)
        print(f"STL saved as {filename}")
        
        return filename
    
    def save_spectrogram_image(self, spectrogram, filename=None):
        """Save spectrogram as a simple image file for debugging"""
        if filename is None:
            filename = f"{self.output_name}_spectrogram.png"
        
        try:
            # Use PIL (Pillow) for simple image saving - no GUI needed
            from PIL import Image
            import numpy as np
            
            # Convert spectrogram to 0-255 range for image
            spec_normalized = (spectrogram * 255).astype(np.uint8)
            
            # Create image from spectrogram data
            img = Image.fromarray(spec_normalized, mode='L')  # Grayscale
            
            # Resize to make it more visible
            img = img.resize((800, 400), Image.Resampling.LANCZOS)
            
            # Save the image
            img.save(filename)
            print(f"Spectrogram image saved as {filename}")
            return filename
            
        except ImportError:
            # Fallback: save as numpy array for debugging
            np_filename = filename.replace('.png', '.npy')
            np.save(np_filename, spectrogram)
            print(f"Spectrogram data saved as {np_filename} (numpy array)")
            return np_filename
        except Exception as e:
            print(f"Could not save spectrogram image: {e}")
            return None
    
    def generate_sculpture(self, method="cylindrical", duration=20):
        """Full pipeline: audio -> 3D model"""
        print(f"Processing {self.audio_file}...")
        
        # process audio
        spectrogram, sr, hop_length = self.load_and_process_audio(duration)
        
        # save spectrogram image
        # image_file = self.save_spectrogram_image(spectrogram)
        
        # smooth spectrogram for cleaner geometry  
        spectrogram = ndimage.gaussian_filter(spectrogram, sigma=0.5)
        
        # create 3D mesh
        vertices, faces = self.create_heightfield_mesh(spectrogram, method)
        
        # export to STL
        stl_file = self.save_stl(vertices, faces)
        
        return stl_file, spectrogram

# usage example
if __name__ == "__main__":
    converter = Audio3DConverter("your_song.wav", "my_song_sculpture")
    
    # try different methods
    stl_file, spec = converter.generate_sculpture(method="cylindrical", duration=30)
    
    print(f"3D model ready: {stl_file}")
    print(f"Spectrogram shape: {spec.shape}")