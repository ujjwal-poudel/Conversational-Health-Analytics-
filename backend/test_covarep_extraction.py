#!/usr/bin/env python3
"""
Test script for COVAREP feature extraction module

This script tests the COVAREP feature extraction on a sample audio file
and displays the results.
"""

from __future__ import print_function
import os
import sys
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add covarep module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from covarep.covarep_extractor import extract_and_save_features, extract_covarep_features


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print("  " + title)
    print("="*80)


def print_features(features, indent=0):
    """Pretty print features dictionary."""
    prefix = "  " * indent
    
    for key, value in features.items():
        if isinstance(value, dict):
            print("{0}{1}:".format(prefix, key))
            print_features(value, indent + 1)
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                print("{0}{1}: {2:.6f}".format(prefix, key, value))
            else:
                print("{0}{1}: {2}".format(prefix, key, value))
        elif isinstance(value, list):
            print("{0}{1}: [list with {2} items]".format(prefix, key, len(value)))
        else:
            print("{0}{1}: {2}".format(prefix, key, value))


def test_basic_extraction():
    """Test basic feature extraction."""
    print_header("Test 1: Basic Feature Extraction")
    
    audio_path = "test_audio.wav"
    
    if not os.path.exists(audio_path):
        print("X Audio file not found: {0}".format(audio_path))
        return False
    
    print("OK Audio file found: {0}".format(audio_path))
    print("   File size: {0:.2f} MB".format(os.path.getsize(audio_path) / (1024*1024)))
    
    try:
        print("\nExtracting features...")
        features = extract_covarep_features(audio_path)
        
        # Check extraction status
        status = features['metadata']['extraction_status']
        print("\nOK Extraction Status: {0}".format(status))
        
        if status == 'success':
            print("OK Audio Duration: {0:.2f} seconds".format(features['metadata']['duration']))
            print("OK Sample Rate: {0} Hz".format(features['metadata']['sample_rate']))
            print("OK Number of Samples: {0}".format(features['metadata']['num_samples']))
            
            print("\nOK Extracted Feature Types:")
            for feature_type in features['features'].keys():
                print("   - {0}".format(feature_type))
            
            return True
        else:
            print("X Extraction failed: {0}".format(features['metadata'].get('error', 'Unknown error')))
            return False
    
    except Exception as e:
        print("X Error during extraction: {0}".format(str(e)))
        logger.exception("Exception during extraction")
        return False


def test_extract_and_save():
    """Test extract and save functionality."""
    print_header("Test 2: Extract and Save Features")
    
    audio_path = "test_audio.wav"
    output_dir = "uploads/features"
    file_prefix = "test_{0}".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    if not os.path.exists(audio_path):
        print("X Audio file not found: {0}".format(audio_path))
        return False
    
    try:
        print("Extracting and saving features...")
        print("   Output directory: {0}".format(output_dir))
        print("   File prefix: {0}".format(file_prefix))
        
        features_file = extract_and_save_features(
            audio_path=audio_path,
            output_dir=output_dir,
            file_prefix=file_prefix
        )
        
        if features_file:
            print("\nOK Features saved successfully!")
            print("OK Output file: {0}".format(features_file))
            
            # Check file size
            file_size = os.path.getsize(features_file)
            print("OK File size: {0:.2f} KB".format(file_size / 1024))
            
            # Load and display summary
            with open(features_file, 'r') as f:
                saved_features = json.load(f)
            
            print("\nOK Saved Features Summary:")
            print("   Extraction timestamp: {0}".format(saved_features['extraction_timestamp']))
            print("   Extraction status: {0}".format(saved_features['metadata']['extraction_status']))
            print("   Feature types: {0}".format(', '.join(saved_features['features'].keys())))
            
            return True
        else:
            print("X Failed to extract and save features")
            return False
    
    except Exception as e:
        print("X Error: {0}".format(str(e)))
        logger.exception("Exception during extract and save")
        return False


def test_feature_values():
    """Test and display actual feature values."""
    print_header("Test 3: Feature Values")
    
    audio_path = "test_audio.wav"
    
    if not os.path.exists(audio_path):
        print("X Audio file not found: {0}".format(audio_path))
        return False
    
    try:
        print("Extracting features...")
        features = extract_covarep_features(audio_path)
        
        if features['metadata']['extraction_status'] != 'success':
            print("X Extraction failed")
            return False
        
        print("\nOK F0 (Fundamental Frequency) Features:")
        f0 = features['features'].get('f0', {})
        if 'mean' in f0:
            print("   Mean F0: {0:.2f} Hz".format(f0['mean']))
            print("   Std Dev: {0:.2f} Hz".format(f0['std']))
            print("   Range: {0:.2f} - {1:.2f} Hz".format(f0['min'], f0['max']))
            print("   Voicing Ratio: {0:.2%}".format(f0['voicing_ratio']))
        else:
            print("   {0}".format(f0.get('error', 'No F0 features extracted')))
        
        print("\nOK Spectral Features:")
        spectral = features['features'].get('spectral', {})
        if 'centroid_mean' in spectral:
            print("   Centroid: {0:.2f} Hz".format(spectral['centroid_mean']))
            print("   Rolloff: {0:.2f} Hz".format(spectral['rolloff_mean']))
            print("   Zero Crossing Rate: {0:.6f}".format(spectral['zcr_mean']))
        else:
            print("   {0}".format(spectral.get('error', 'No spectral features extracted')))
        
        print("\nOK Energy Features:")
        energy = features['features'].get('energy', {})
        if 'rms' in energy:
            print("   RMS Energy: {0:.6f}".format(energy['rms']))
            print("   Mean Energy: {0:.6f}".format(energy['mean']))
            print("   Energy Range: {0:.6f} - {1:.6f}".format(energy['min'], energy['max']))
        else:
            print("   {0}".format(energy.get('error', 'No energy features extracted')))
        
        print("\nOK MFCC Features:")
        mfcc = features['features'].get('mfcc', {})
        if 'n_mfcc' in mfcc:
            print("   Number of MFCCs: {0}".format(mfcc['n_mfcc']))
            stats = mfcc.get('statistics', {})
            if stats:
                print("   Sample MFCC statistics:")
                for i in range(min(3, mfcc['n_mfcc'])):
                    mean_key = "mfcc_{0}_mean".format(i)
                    std_key = "mfcc_{0}_std".format(i)
                    if mean_key in stats:
                        print("     MFCC {0}: mean={1:.4f}, std={2:.4f}".format(i, stats[mean_key], stats[std_key]))
        else:
            print("   {0}".format(mfcc.get('error', 'No MFCC features extracted')))
        
        return True
    
    except Exception as e:
        print("X Error: {0}".format(str(e)))
        logger.exception("Exception during feature value test")
        return False


def test_json_output():
    """Test JSON output format."""
    print_header("Test 4: JSON Output Format")
    
    audio_path = "test_audio.wav"
    output_dir = "uploads/features"
    file_prefix = "test_json_{0}".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    if not os.path.exists(audio_path):
        print("X Audio file not found: {0}".format(audio_path))
        return False
    
    try:
        features_file = extract_and_save_features(
            audio_path=audio_path,
            output_dir=output_dir,
            file_prefix=file_prefix
        )
        
        if not features_file:
            print("X Failed to extract and save features")
            return False
        
        print("OK Loading JSON file: {0}".format(features_file))
        
        with open(features_file, 'r') as f:
            data = json.load(f)
        
        print("\nOK JSON Structure:")
        print("   Top-level keys: {0}".format(', '.join(data.keys())))
        print("   Metadata keys: {0}".format(', '.join(data['metadata'].keys())))
        print("   Feature types: {0}".format(', '.join(data['features'].keys())))
        
        print("\nOK JSON is valid and properly formatted")
        
        # Display a sample of the JSON
        print("\nOK Sample JSON content (first 500 chars):")
        json_str = json.dumps(data, indent=2)
        print(json_str[:500] + "...")
        
        return True
    
    except json.JSONDecodeError as e:
        print("X JSON parsing error: {0}".format(str(e)))
        return False
    except Exception as e:
        print("X Error: {0}".format(str(e)))
        logger.exception("Exception during JSON test")
        return False


def main():
    """Run all tests."""
    print_header("COVAREP Feature Extraction - Test Suite")
    
    print("Current directory: {0}".format(os.getcwd()))
    print("Python version: {0}".format(sys.version))
    print("Test start time: {0}".format(datetime.now().isoformat()))
    
    # Run tests
    results = {
        "Basic Extraction": test_basic_extraction(),
        "Extract and Save": test_extract_and_save(),
        "Feature Values": test_feature_values(),
        "JSON Output": test_json_output()
    }
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "OK PASS" if result else "X FAIL"
        print("{0}: {1}".format(status, test_name))
    
    print("\nTotal: {0}/{1} tests passed".format(passed, total))
    
    if passed == total:
        print("\nOK All tests passed successfully!")
        return 0
    else:
        print("\nX {0} test(s) failed".format(total - passed))
        return 1


if __name__ == "__main__":
    sys.exit(main())
