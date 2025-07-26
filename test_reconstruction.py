#!/usr/bin/env python3
"""
Test script to verify the 3D reconstruction implementation works correctly.
"""

import numpy as np
import sys
import traceback
from microstructure_3d_reconstruction import MicrostructureReconstructor

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy
        import cv2
        from microstructure_3d_reconstruction import MicrostructureReconstructor
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the reconstruction class."""
    print("\nTesting basic functionality...")
    try:
        # Create simple test data
        sections_2d = [
            np.random.random((16, 16)) > 0.6,
            np.random.random((16, 16)) > 0.6
        ]
        
        # Initialize reconstructor
        target_size = (16, 16, 16)
        reconstructor = MicrostructureReconstructor(sections_2d, target_size)
        
        # Test volume fraction calculation
        vf = reconstructor.calculate_volume_fraction()
        assert 0 <= vf <= 1, "Volume fraction should be between 0 and 1"
        
        # Test initialization methods
        reconstructor.initialize_3d_volume('random')
        assert reconstructor.volume_3d is not None, "Random initialization failed"
        
        reconstructor.initialize_3d_volume('layered')
        assert reconstructor.volume_3d is not None, "Layered initialization failed"
        
        reconstructor.initialize_3d_volume('interpolated')
        assert reconstructor.volume_3d is not None, "Interpolated initialization failed"
        
        print("✓ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_correlation_functions():
    """Test statistical correlation functions."""
    print("\nTesting correlation functions...")
    try:
        # Create test section
        section = np.zeros((20, 20), dtype=bool)
        section[5:15, 5:15] = True  # Square in the middle
        
        reconstructor = MicrostructureReconstructor([section], (20, 20, 20))
        
        # Test 2-point correlation
        corr = reconstructor.calculate_two_point_correlation(section, max_distance=5)
        assert len(corr) == 5, "Correlation function length incorrect"
        assert corr[0] > 0, "Zero-distance correlation should be positive"
        
        # Test lineal path function
        lineal = reconstructor.calculate_lineal_path_function(section, max_length=5)
        assert len(lineal) == 5, "Lineal path function length incorrect"
        assert 0 <= np.max(lineal) <= 1, "Lineal path values should be probabilities"
        
        print("✓ Correlation function tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Correlation function test failed: {e}")
        traceback.print_exc()
        return False

def test_short_reconstruction():
    """Test a very short reconstruction to ensure algorithms work."""
    print("\nTesting short reconstruction...")
    try:
        # Create simple test data
        sections_2d = [
            np.random.random((12, 12)) > 0.5,
            np.random.random((12, 12)) > 0.5
        ]
        
        target_size = (12, 12, 12)
        reconstructor = MicrostructureReconstructor(sections_2d, target_size)
        
        # Test simulated annealing (very short run)
        reconstructor.initialize_3d_volume('random')
        volume_sa = reconstructor.simulated_annealing_reconstruction(
            max_iterations=10,
            initial_temp=0.1,
            cooling_rate=0.9
        )
        assert volume_sa.shape == target_size, "SA reconstruction shape incorrect"
        
        # Test phase field (very short run)
        reconstructor.initialize_3d_volume('interpolated')
        volume_pf = reconstructor.phase_field_reconstruction(
            time_steps=5,
            dt=0.1,
            mobility=1.0
        )
        assert volume_pf.shape == target_size, "PF reconstruction shape incorrect"
        
        print("✓ Short reconstruction tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Short reconstruction test failed: {e}")
        traceback.print_exc()
        return False

def test_export_functionality():
    """Test export functionality."""
    print("\nTesting export functionality...")
    try:
        # Create simple test volume
        volume = np.random.random((8, 8, 8)) > 0.5
        sections_2d = [volume[:, :, 0]]
        
        reconstructor = MicrostructureReconstructor(sections_2d, (8, 8, 8))
        reconstructor.volume_3d = volume
        
        # Test numpy export
        reconstructor.export_volume('test_volume.npy', format='npy')
        
        # Verify the file was created and can be loaded
        loaded_volume = np.load('test_volume.npy')
        assert np.array_equal(loaded_volume, volume), "Exported volume doesn't match original"
        
        print("✓ Export functionality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Export functionality test failed: {e}")
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files."""
    import os
    test_files = ['test_volume.npy']
    for file in test_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def main():
    """Run all tests."""
    print("3D Microstructure Reconstruction - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_correlation_functions,
        test_short_reconstruction,
        test_export_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    cleanup_test_files()
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
