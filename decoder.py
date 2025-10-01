import numpy as np
import trimesh
import librosa
from scipy import interpolate
from scipy.io import wavfile

class MusicalHelixDecoder:
    def __init__(self):
        self.base_radius = 15
        self.pitch = 8
        self.min_thickness = 2
        self.max_thickness = 8
        self.sample_rate = 22050
        
    def load_scanned_mesh(self, stl_filename):
        mesh = trimesh.load(stl_filename)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("invalid mesh")
        return mesh
    
    def measure_thickness_profile(self, mesh):
        """extract thickness measurements along helix height"""
        vertices = mesh.vertices
        z_values = vertices[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        
        # slice every 1mm
        num_slices = int((z_max - z_min) / 1.0) + 1
        thickness_profile = []
        roughness_profile = []
        heights = []
        
        for i in range(num_slices):
            slice_z = z_min + i * (z_max - z_min) / (num_slices - 1)
            height_mask = np.abs(z_values - slice_z) < 0.7
            
            if not np.any(height_mask):
                continue
                
            slice_verts = vertices[height_mask]
            slice_x, slice_y = slice_verts[:, 0], slice_verts[:, 1]
            
            # radial distances from origin
            radii = np.sqrt(slice_x**2 + slice_y**2)
            
            if len(radii) > 5:
                avg_radius = np.mean(radii)
                thickness = avg_radius - self.base_radius
                thickness = np.clip(thickness, self.min_thickness, self.max_thickness)
                
                # roughness = std dev of radii (surface complexity)
                roughness = np.std(radii) / avg_radius if avg_radius > 0 else 0
                
                thickness_profile.append(thickness)
                roughness_profile.append(roughness)
                heights.append(slice_z)
        
        return np.array(thickness_profile), np.array(roughness_profile), np.array(heights)
    
    def measure_thickness_profile_enhanced(self, mesh):
        vertices = mesh.vertices
        z_values = vertices[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        
        num_slices = int((z_max - z_min) / 0.5) + 1  # 0.5mm slices for more resolution
        thickness_profile = []
        roughness_profile = []
        radial_entropy = []  # new metric
        
        for i in range(num_slices):
            slice_z = z_min + i * (z_max - z_min) / (num_slices - 1)
            height_mask = np.abs(z_values - slice_z) < 0.4
            
            if not np.any(height_mask):
                continue
                
            slice_verts = vertices[height_mask]
            slice_x, slice_y = slice_verts[:, 0], slice_verts[:, 1]
            radii = np.sqrt(slice_x**2 + slice_y**2)
            
            if len(radii) > 5:
                # use max radius instead of mean (captures the actual outer edge)
                max_radius = np.percentile(radii, 90)  # 90th percentile to ignore outliers
                thickness = max_radius - self.base_radius
                thickness = np.clip(thickness, self.min_thickness, self.max_thickness)
                
                # roughness
                roughness = np.std(radii) / np.mean(radii) if np.mean(radii) > 0 else 0
                
                # radial entropy: how much variation in the cross-section
                angles = np.arctan2(slice_y, slice_x)
                sorted_idx = np.argsort(angles)
                sorted_radii = radii[sorted_idx]
                
                # measure changes as you go around the cross-section
                if len(sorted_radii) > 3:
                    radial_changes = np.abs(np.diff(sorted_radii))
                    entropy = np.mean(radial_changes)
                else:
                    entropy = 0
                
                thickness_profile.append(thickness)
                roughness_profile.append(roughness)
                radial_entropy.append(entropy)
        
        # convert entropy to additional frequency variation
        thickness_profile = np.array(thickness_profile)
        roughness_profile = np.array(roughness_profile)
        radial_entropy = np.array(radial_entropy)
        
        # amplify variations
        thickness_mean = np.mean(thickness_profile)
        thickness_profile = thickness_mean + (thickness_profile - thickness_mean) * 2.0
        thickness_profile = np.clip(thickness_profile, self.min_thickness, self.max_thickness)
        
        return thickness_profile, roughness_profile, radial_entropy

    
    def thickness_to_musical_params(self, thickness_profile, roughness_profile, entropy_profile):
        t_norm = (thickness_profile - self.min_thickness) / (self.max_thickness - self.min_thickness)
        t_norm = np.clip(t_norm, 0, 1)
        
        # use both thickness AND entropy for note selection
        e_norm = (entropy_profile - np.min(entropy_profile)) / (np.max(entropy_profile) - np.min(entropy_profile) + 1e-6)
        
        base_freq = 261.63
        scale_ratios = [1.0, 1.125, 1.25, 1.333, 1.5, 1.667, 1.875, 2.0]  # full octave
        
        musical_events = []
        
        for i in range(len(t_norm)):
            # combine thickness and entropy for note selection
            note_selector = (t_norm[i] + e_norm[i]) / 2.0
            note_idx = int(note_selector * (len(scale_ratios) - 1))
            freq = base_freq * scale_ratios[note_idx]
            
            # vary octave based on entropy
            if e_norm[i] > 0.7:
                freq *= 2  # jump up an octave for high entropy
            elif e_norm[i] < 0.3:
                freq *= 0.5  # drop an octave for low entropy
            
            amplitude = 0.3 + t_norm[i] * 0.4
            harmonics = roughness_profile[i]
            
            musical_events.append({
                'freq': freq,
                'amplitude': amplitude,
                'harmonics': harmonics,
                'index': i
            })
        
        return musical_events
    
    def synthesize_from_events(self, musical_events, duration=20):
        """generate audio from musical events"""
        num_samples = int(duration * self.sample_rate)
        audio = np.zeros(num_samples)
        
        if not musical_events:
            return audio
        
        # time per event
        event_duration = duration / len(musical_events)
        samples_per_event = int(event_duration * self.sample_rate)
        
        for evt in musical_events:
            start_sample = evt['index'] * samples_per_event
            end_sample = min(start_sample + samples_per_event, num_samples)
            
            if start_sample >= num_samples:
                break
            
            segment_length = end_sample - start_sample
            t = np.linspace(0, event_duration, segment_length)
            
            # generate tone with harmonics
            freq = evt['freq']
            amp = evt['amplitude']
            harm_weight = evt['harmonics']
            
            # fundamental
            signal = amp * np.sin(2 * np.pi * freq * t)
            
            # add harmonics based on roughness
            # more roughness = more harmonics = brighter timbre
            if harm_weight > 0.3:
                signal += amp * 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # octave
            if harm_weight > 0.5:
                signal += amp * 0.2 * np.sin(2 * np.pi * freq * 3 * t)  # 5th above octave
            if harm_weight > 0.7:
                signal += amp * 0.15 * np.sin(2 * np.pi * freq * 1.5 * t)  # perfect 5th
            
            # envelope (attack-decay) to avoid clicks
            envelope = np.ones(segment_length)
            attack_samples = min(int(0.01 * self.sample_rate), segment_length // 4)
            decay_samples = min(int(0.05 * self.sample_rate), segment_length // 4)
            
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            if decay_samples > 0:
                envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
            
            signal *= envelope
            
            # add to main audio buffer
            audio[start_sample:end_sample] += signal
        
        # normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.7
        
        # simple lowpass to remove artifacts
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 0.5, btype='low')
        audio = filtfilt(b, a, audio)
        
        return audio
    
    def add_thickness_rhythm(self, thickness_profile, roughness_profile):
        """create rhythmic variation from thickness deltas"""
        # compute rate of change
        thickness_delta = np.diff(thickness_profile, prepend=thickness_profile[0])
        
        musical_events = []
        base_freq = 261.63
        
        # minor pentatonic for more interesting sound
        scale_ratios = [1.0, 1.2, 1.333, 1.5, 1.778, 2.0]
        
        for i in range(len(thickness_profile)):
            t_norm = (thickness_profile[i] - self.min_thickness) / (self.max_thickness - self.min_thickness)
            t_norm = np.clip(t_norm, 0, 1)
            
            # rapid thickness change = note trigger
            change_magnitude = abs(thickness_delta[i])
            
            # pick note from scale based on thickness
            note_idx = int(t_norm * (len(scale_ratios) - 1))
            freq = base_freq * scale_ratios[note_idx]
            
            # amplitude from change magnitude
            amp = 0.2 + min(change_magnitude / 2.0, 0.6)
            
            # more rough = add more harmonics
            harm = roughness_profile[i] if i < len(roughness_profile) else 0.3
            
            musical_events.append({
                'freq': freq,
                'amplitude': amp,
                'harmonics': harm,
                'index': i
            })
        
        return musical_events
    
    def decode_mesh_to_audio(self, stl_filename, output_filename=None, duration=20, mode='melodic'):
        """
        modes:
        - 'melodic': smooth melody from thickness
        - 'rhythmic': percussive from thickness changes
        """
        
        mesh = self.load_scanned_mesh(stl_filename)
        thickness, roughness, heights = self.measure_thickness_profile(mesh)
        
        if mode == 'melodic':
            events = self.thickness_to_musical_params(thickness, roughness, roughness)  # Use roughness as entropy
        elif mode == 'rhythmic':
            events = self.add_thickness_rhythm(thickness, roughness)
        else:
            events = self.thickness_to_musical_params(thickness, roughness, roughness)  # Use roughness as entropy
        
        audio = self.synthesize_from_events(events, duration)
        
        if output_filename is None:
            output_filename = stl_filename.replace('.stl', '_musical.wav')
        
        # save
        audio_16bit = (audio * 32767).astype(np.int16)
        wavfile.write(output_filename, self.sample_rate, audio_16bit)
        
        print(f"musical interpretation saved: {output_filename}")
        print(f"extracted {len(thickness)} thickness measurements")
        print(f"generated {len(events)} musical events")
        
        return audio

# use it
if __name__ == "__main__":
    decoder = MusicalHelixDecoder()
    
    # try both modes
    audio_melodic = decoder.decode_mesh_to_audio(
        "generated_sculpture_test.stl", 
        "decoded_melodic.wav",
        duration=20,
        mode='melodic'
    )
    
    audio_rhythmic = decoder.decode_mesh_to_audio(
        "generated_sculpture_test.stl",
        "decoded_rhythmic.wav", 
        duration=20,
        mode='rhythmic'
    )