"""
Feature extraction from music21 scores for CNN-based page matching.
Option B: Medium feature set (15-20 features per measure)
"""

import music21
import numpy as np
from typing import List, Dict, Optional
from collections import Counter


class MeasureFeatureExtractor:
    """Extract fixed-size feature vectors from music21 measures."""
    
    def __init__(self):
        self.feature_dim = None  # Will be set after first extraction
        
    def extract_from_measure(self, measure: music21.stream.Measure) -> np.ndarray:
        """
        Extract features from a single measure.
        
        Args:
            measure: music21 Measure object
            
        Returns:
            feature_vector: np.ndarray of shape (feature_dim,)
        """
        features = []
        
        # Get all notes (flatten to handle chords)
        notes = list(measure.flatten().notes)
        
        # --- PITCH FEATURES (5 features) ---
        if len(notes) > 0:
            pitches = [n.pitch.midi for n in notes if hasattr(n, 'pitch')]
            
            if len(pitches) > 0:
                features.append(np.mean(pitches))  # Mean pitch
                features.append(np.max(pitches) - np.min(pitches))  # Pitch range
                features.append(np.std(pitches))  # Pitch variance
                features.append(np.min(pitches))  # Lowest pitch
                features.append(np.max(pitches))  # Highest pitch
            else:
                # Rest measure
                features.extend([0, 0, 0, 0, 0])
        else:
            # Empty measure
            features.extend([0, 0, 0, 0, 0])
            
        # --- PITCH CLASS HISTOGRAM (12 features) ---
        # Count occurrences of each pitch class (C, C#, D, ..., B)
        pitch_class_counts = np.zeros(12)
        if len(notes) > 0:
            for note in notes:
                if hasattr(note, 'pitch'):
                    pc = note.pitch.pitchClass  # 0-11
                    pitch_class_counts[pc] += 1
            # Normalize to sum to 1 (probability distribution)
            if pitch_class_counts.sum() > 0:
                pitch_class_counts = pitch_class_counts / pitch_class_counts.sum()
        features.extend(pitch_class_counts.tolist())
        
        # --- RHYTHM FEATURES (5 features) ---
        if len(notes) > 0:
            durations = [n.duration.quarterLength for n in notes]
            features.append(len(notes))  # Note density
            features.append(np.mean(durations))  # Average duration
            features.append(np.std(durations) if len(durations) > 1 else 0)  # Duration variance
            features.append(np.min(durations))  # Shortest note
            features.append(np.max(durations))  # Longest note
        else:
            features.extend([0, 0, 0, 0, 0])
            
        # --- STRUCTURAL FEATURES (3 features) ---
        # Key signature (encode as integer: -7 to +7 for flats to sharps)
        key_sig = measure.keySignature
        if key_sig is not None:
            features.append(key_sig.sharps)  # -7 to +7
        else:
            features.append(0)
            
        # Time signature
        time_sig = measure.timeSignature
        if time_sig is not None:
            features.append(time_sig.numerator)  # e.g., 4 in 4/4
            features.append(time_sig.denominator)  # e.g., 4 in 4/4
        else:
            features.append(4)  # Default to 4/4
            features.append(4)
            
        feature_vector = np.array(features, dtype=np.float32)
        
        # Cache feature dimension
        if self.feature_dim is None:
            self.feature_dim = len(feature_vector)
            print(f"Feature dimension: {self.feature_dim}")
            
        return feature_vector
    
    def extract_from_score(self, score: music21.stream.Score, 
                          max_measures: Optional[int] = None) -> np.ndarray:
        """
        Extract features from all measures in a score.
        
        Args:
            score: music21 Score object
            max_measures: Optional maximum number of measures to extract
            
        Returns:
            features: np.ndarray of shape (num_measures, feature_dim)
        """
        # Get first part (melody line) if multi-part score
        part = score.parts[0] if len(score.parts) > 0 else score
        
        measures = list(part.getElementsByClass(music21.stream.Measure))
        
        if max_measures is not None:
            measures = measures[:max_measures]
            
        feature_list = []
        for measure in measures:
            features = self.extract_from_measure(measure)
            feature_list.append(features)
            
        if len(feature_list) == 0:
            # Empty score, return zeros
            return np.zeros((1, self.feature_dim or 25), dtype=np.float32)
            
        return np.stack(feature_list, axis=0)
    
    def get_feature_names(self) -> List[str]:
        """Return human-readable names for each feature dimension."""
        names = [
            # Pitch features (5)
            'mean_pitch', 'pitch_range', 'pitch_std', 'min_pitch', 'max_pitch',
        ]
        # Pitch class histogram (12)
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        names.extend([f'pc_{pc}' for pc in pitch_classes])
        # Rhythm features (5)
        names.extend([
            'note_density', 'mean_duration', 'duration_std', 'min_duration', 'max_duration'
        ])
        # Structural features (3)
        names.extend(['key_signature', 'time_sig_numerator', 'time_sig_denominator'])
        
        return names


def load_and_extract_features(file_path: str, max_measures: Optional[int] = None) -> np.ndarray:
    """
    Convenience function to load a music file and extract features.
    
    Args:
        file_path: Path to music file (MusicXML, MIDI, etc.)
        max_measures: Optional limit on number of measures
        
    Returns:
        features: np.ndarray of shape (num_measures, feature_dim)
    """
    extractor = MeasureFeatureExtractor()
    score = music21.converter.parse(file_path)
    return extractor.extract_from_score(score, max_measures)


if __name__ == "__main__":
    # Test the feature extractor
    print("Testing MeasureFeatureExtractor...")
    
    # Create a simple test score
    test_score = music21.stream.Score()
    test_part = music21.stream.Part()
    
    # Add a few test measures
    for i in range(4):
        measure = music21.stream.Measure(number=i+1)
        measure.append(music21.note.Note('C4', quarterLength=1.0))
        measure.append(music21.note.Note('E4', quarterLength=1.0))
        measure.append(music21.note.Note('G4', quarterLength=2.0))
        test_part.append(measure)
    
    test_score.append(test_part)
    
    # Extract features
    extractor = MeasureFeatureExtractor()
    features = extractor.extract_from_score(test_score)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature names ({len(extractor.get_feature_names())} total):")
    for i, name in enumerate(extractor.get_feature_names()):
        print(f"  {i:2d}. {name}")
    
    print("\nFirst measure features:")
    for name, value in zip(extractor.get_feature_names(), features[0]):
        print(f"  {name:25s}: {value:.3f}")
