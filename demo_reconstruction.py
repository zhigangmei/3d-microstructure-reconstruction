"""
Demonstration script for 3D microstructure reconstruction.

This script shows how to use the different reconstruction algorithms
to reconstruct 3D microstructures from 2D cross-sections.
"""

import numpy as np
import matplotlib.pyplot as plt
from microstructure_3d_reconstruction import (
    MicrostructureReconstructor, 
    create_synthetic_2d_sections
)


def demo_basic_reconstruction():
    """Demonstrate basic 3D reconstruction workflow."""
    print("=== Basic 3D Reconstruction Demo ===")
    
    # Create synthetic 2D sections for demonstration
    print("Creating synthetic 2D sections...")
    sections_2d = create_synthetic_2d_sections(
        num_sections=3,
        section_size=(32, 32),  # Smaller size for faster demo
        volume_fraction=0.4,
        correlation_length=3.0
    )
    
    # Visualize input sections
    fig, axes = plt.subplots(1, len(sections_2d), figsize=(12, 4))
    for i, section in enumerate(sections_2d):
        axes[i].imshow(section, cmap='gray')
        axes[i].set_title(f'Section {i+1}')
        axes[i].axis('off')
    plt.suptitle('Input 2D Sections')
    plt.tight_layout()
    plt.savefig('input_sections.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Initialize reconstructor
    target_size = (32, 32, 32)
    reconstructor = MicrostructureReconstructor(sections_2d, target_size)
    
    # Calculate and display volume fraction
    vf = reconstructor.calculate_volume_fraction()
    print(f"Volume fraction from 2D sections: {vf:.3f}")
    
    return reconstructor


def demo_simulated_annealing(reconstructor):
    """Demonstrate simulated annealing reconstruction."""
    print("\n=== Simulated Annealing Reconstruction ===")
    
    # Initialize with random microstructure
    reconstructor.initialize_3d_volume('random')
    
    # Run simulated annealing
    volume_sa = reconstructor.simulated_annealing_reconstruction(
        max_iterations=300,
        initial_temp=0.5,
        cooling_rate=0.99
    )
    
    print(f"SA reconstruction completed. Final volume fraction: "
          f"{np.mean(volume_sa):.3f}")
    
    # Visualize results
    reconstructor.visualize_results('sa_reconstruction.png')
    
    return volume_sa


def demo_phase_field(reconstructor):
    """Demonstrate phase field reconstruction."""
    print("\n=== Phase Field Reconstruction ===")
    
    # Initialize with interpolated microstructure
    reconstructor.initialize_3d_volume('interpolated')
    
    # Run phase field evolution
    volume_pf = reconstructor.phase_field_reconstruction(
        time_steps=100,
        dt=0.02,
        mobility=1.5
    )
    
    print(f"Phase field reconstruction completed. Final volume fraction: "
          f"{np.mean(volume_pf):.3f}")
    
    # Visualize results
    reconstructor.visualize_results('pf_reconstruction.png')
    
    return volume_pf


def demo_statistical_analysis(reconstructor):
    """Demonstrate statistical analysis of reconstructed microstructure."""
    print("\n=== Statistical Analysis ===")
    
    if reconstructor.volume_3d is None:
        print("No reconstructed volume available for analysis.")
        return
    
    # Calculate 2-point correlation for original and reconstructed
    original_section = reconstructor.sections_2d[0]
    reconstructed_slice = reconstructor.volume_3d[:, :, 16]  # Middle slice
    
    orig_corr = reconstructor.calculate_two_point_correlation(original_section)
    recon_corr = reconstructor.calculate_two_point_correlation(
        reconstructed_slice
    )
    
    # Calculate lineal path functions
    orig_lineal = reconstructor.calculate_lineal_path_function(original_section)
    recon_lineal = reconstructor.calculate_lineal_path_function(
        reconstructed_slice
    )
    
    # Plot comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2-point correlation
    min_len = min(len(orig_corr), len(recon_corr))
    ax1.plot(orig_corr[:min_len], 'b-', linewidth=2, label='Original')
    ax1.plot(recon_corr[:min_len], 'r--', linewidth=2, label='Reconstructed')
    ax1.set_xlabel('Distance (pixels)')
    ax1.set_ylabel('2-Point Correlation')
    ax1.set_title('2-Point Correlation Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lineal path function
    min_len = min(len(orig_lineal), len(recon_lineal))
    ax2.plot(range(1, min_len+1), orig_lineal[:min_len], 'b-', 
             linewidth=2, label='Original')
    ax2.plot(range(1, min_len+1), recon_lineal[:min_len], 'r--', 
             linewidth=2, label='Reconstructed')
    ax2.set_xlabel('Path Length (pixels)')
    ax2.set_ylabel('Lineal Path Probability')
    ax2.set_title('Lineal Path Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('statistical_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistical comparison
    print("\nStatistical Comparison:")
    print(f"Original volume fraction: {np.mean(original_section):.3f}")
    print(f"Reconstructed volume fraction: {np.mean(reconstructed_slice):.3f}")
    
    # Calculate RMS error for correlations
    corr_error = np.sqrt(np.mean((orig_corr[:min_len] - 
                                 recon_corr[:min_len])**2))
    print(f"2-point correlation RMS error: {corr_error:.4f}")


def demo_export_results(reconstructor):
    """Demonstrate exporting reconstruction results."""
    print("\n=== Exporting Results ===")
    
    if reconstructor.volume_3d is None:
        print("No reconstructed volume to export.")
        return
    
    # Export in different formats
    reconstructor.export_volume('reconstructed_volume.npy', format='npy')
    reconstructor.export_volume('reconstructed_volume.raw', format='raw')
    reconstructor.export_volume('reconstructed_volume.vtk', format='vtk')
    
    print("Volume exported in multiple formats:")
    print("- NumPy array: reconstructed_volume.npy")
    print("- Raw binary: reconstructed_volume.raw")
    print("- VTK format: reconstructed_volume.vtk")


def compare_reconstruction_methods():
    """Compare different reconstruction methods."""
    print("\n=== Comparing Reconstruction Methods ===")
    
    # Create test data
    sections_2d = create_synthetic_2d_sections(
        num_sections=2,
        section_size=(24, 24),  # Small for quick comparison
        volume_fraction=0.35,
        correlation_length=2.5
    )
    
    target_size = (24, 24, 24)
    methods_results = {}
    
    # Method 1: Random initialization
    print("Testing random initialization...")
    reconstructor1 = MicrostructureReconstructor(sections_2d, target_size)
    reconstructor1.initialize_3d_volume('random')
    methods_results['Random'] = reconstructor1.volume_3d.copy()
    
    # Method 2: Layered initialization
    print("Testing layered initialization...")
    reconstructor2 = MicrostructureReconstructor(sections_2d, target_size)
    reconstructor2.initialize_3d_volume('layered')
    methods_results['Layered'] = reconstructor2.volume_3d.copy()
    
    # Method 3: Interpolated initialization
    print("Testing interpolated initialization...")
    reconstructor3 = MicrostructureReconstructor(sections_2d, target_size)
    reconstructor3.initialize_3d_volume('interpolated')
    methods_results['Interpolated'] = reconstructor3.volume_3d.copy()
    
    # Method 4: Simulated annealing (short run)
    print("Testing simulated annealing...")
    reconstructor4 = MicrostructureReconstructor(sections_2d, target_size)
    reconstructor4.simulated_annealing_reconstruction(
        max_iterations=100,
        initial_temp=0.3,
        cooling_rate=0.98
    )
    methods_results['Simulated Annealing'] = reconstructor4.volume_3d.copy()
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show original section
    axes[0].imshow(sections_2d[0], cmap='gray')
    axes[0].set_title('Original 2D Section')
    axes[0].axis('off')
    
    # Show reconstruction methods
    for i, (method, volume) in enumerate(methods_results.items(), 1):
        middle_slice = volume[:, :, volume.shape[2]//2]
        axes[i].imshow(middle_slice, cmap='gray')
        axes[i].set_title(f'{method}\nVF: {np.mean(volume):.3f}')
        axes[i].axis('off')
    
    # Hide unused subplot
    if len(axes) > len(methods_results) + 1:
        axes[-1].axis('off')
    
    plt.suptitle('Comparison of Reconstruction Methods')
    plt.tight_layout()
    plt.savefig('methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main demonstration function."""
    print("3D Microstructure Reconstruction Demonstration")
    print("=" * 55)
    
    try:
        # Basic reconstruction demo
        reconstructor = demo_basic_reconstruction()
        
        # Demonstrate simulated annealing
        demo_simulated_annealing(reconstructor)
        
        # Demonstrate phase field reconstruction
        demo_phase_field(reconstructor)
        
        # Statistical analysis
        demo_statistical_analysis(reconstructor)
        
        # Export results
        demo_export_results(reconstructor)
        
        # Compare methods
        compare_reconstruction_methods()
        
        print("\n" + "=" * 55)
        print("Demonstration completed successfully!")
        print("Check the generated PNG files and exported volumes.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Make sure all required packages are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
