"""
3D Reconstruction of Two-Phase Random Heterogeneous Material from 2D Sections

This module implements various algorithms for reconstructing 3D microstructures
from 2D cross-sectional images, commonly used for materials characterization.

The implementation includes:
1. Statistical correlation-based reconstruction
2. Simulated annealing optimization
3. Multi-point statistics approach
4. Phase field reconstruction

Author: Generated based on typical 3D reconstruction approaches
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import minimize
import cv2
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class MicrostructureReconstructor:
    """
    Class for 3D reconstruction of two-phase materials from 2D sections.
    """
    
    def __init__(self, sections_2d: List[np.ndarray], 
                 target_size_3d: Tuple[int, int, int]):
        """
        Initialize the reconstructor with 2D sections.
        
        Args:
            sections_2d: List of 2D binary arrays representing cross-sections
            target_size_3d: Target dimensions for 3D reconstruction 
                           (nx, ny, nz)
        """
        self.sections_2d = [section.astype(bool) for section in sections_2d]
        self.target_size_3d = target_size_3d
        self.nx, self.ny, self.nz = target_size_3d
        
        # Validate input sections
        if not self.sections_2d:
            raise ValueError("At least one 2D section is required")
        
        # Check if all sections have the same dimensions
        base_shape = self.sections_2d[0].shape
        for i, section in enumerate(self.sections_2d):
            if section.shape != base_shape:
                raise ValueError(f"Section {i} has different dimensions than section 0")
        
        self.section_shape = base_shape
        
        # Initialize 3D reconstruction volume
        self.volume_3d = None
        
        # Statistical properties
        self.volume_fraction = None
        self.correlation_functions = {}
        
    def calculate_volume_fraction(self) -> float:
        """Calculate volume fraction from 2D sections."""
        total_pixels = sum(section.size for section in self.sections_2d)
        phase_pixels = sum(np.sum(section) for section in self.sections_2d)
        self.volume_fraction = phase_pixels / total_pixels
        return self.volume_fraction
    
    def calculate_two_point_correlation(self, section: np.ndarray, 
                                      max_distance: int = None) -> np.ndarray:
        """
        Calculate 2-point correlation function for a 2D section.
        
        Args:
            section: 2D binary array
            max_distance: Maximum distance for correlation calculation
            
        Returns:
            1D array of correlation values vs distance
        """
        if max_distance is None:
            max_distance = min(section.shape) // 4
        
        correlations = []
        section_float = section.astype(float)
        mean_val = np.mean(section_float)
        
        for r in range(max_distance):
            if r == 0:
                # Auto-correlation at zero distance
                corr = np.mean((section_float - mean_val)**2) + mean_val**2
            else:
                # Correlation at distance r
                corr_sum = 0
                count = 0
                
                # Calculate correlation in all directions
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        if dx**2 + dy**2 <= r**2:
                            shifted = np.roll(np.roll(section_float, dx, axis=0), dy, axis=1)
                            corr_sum += np.mean(section_float * shifted)
                            count += 1
                
                corr = corr_sum / count if count > 0 else 0
            
            correlations.append(corr)
        
        return np.array(correlations)
    
    def calculate_lineal_path_function(self, section: np.ndarray, 
                                     max_length: int = None) -> np.ndarray:
        """
        Calculate lineal path function for a 2D section.
        
        Args:
            section: 2D binary array
            max_length: Maximum path length
            
        Returns:
            1D array of lineal path probabilities vs length
        """
        if max_length is None:
            max_length = min(section.shape) // 4
        
        lineal_paths = []
        
        for length in range(1, max_length + 1):
            total_paths = 0
            valid_paths = 0
            
            # Horizontal paths
            for i in range(section.shape[0]):
                for j in range(section.shape[1] - length + 1):
                    path = section[i, j:j+length]
                    total_paths += 1
                    if np.all(path):
                        valid_paths += 1
            
            # Vertical paths
            for i in range(section.shape[0] - length + 1):
                for j in range(section.shape[1]):
                    path = section[i:i+length, j]
                    total_paths += 1
                    if np.all(path):
                        valid_paths += 1
            
            probability = valid_paths / total_paths if total_paths > 0 else 0
            lineal_paths.append(probability)
        
        return np.array(lineal_paths)
    
    def initialize_3d_volume(self, method: str = 'random') -> np.ndarray:
        """
        Initialize 3D volume with specified method.
        
        Args:
            method: Initialization method ('random', 'layered', 'interpolated')
            
        Returns:
            3D binary array
        """
        if self.volume_fraction is None:
            self.calculate_volume_fraction()
        
        if method == 'random':
            # Random initialization with correct volume fraction
            volume = np.random.random(self.target_size_3d) < self.volume_fraction
            
        elif method == 'layered':
            # Layer-based initialization using available sections
            volume = np.zeros(self.target_size_3d, dtype=bool)
            section_indices = np.linspace(0, len(self.sections_2d) - 1, self.nz).astype(int)
            
            for z in range(self.nz):
                section_idx = section_indices[z]
                resized_section = cv2.resize(
                    self.sections_2d[section_idx].astype(np.uint8),
                    (self.ny, self.nx),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                volume[:, :, z] = resized_section
                
        elif method == 'interpolated':
            # 3D interpolation between available sections
            volume = self._interpolate_sections()
            
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        self.volume_3d = volume
        return volume
    
    def _interpolate_sections(self) -> np.ndarray:
        """Interpolate between available 2D sections to create 3D volume."""
        # Create a 3D array by interpolating between sections
        volume = np.zeros(self.target_size_3d, dtype=float)
        
        # Place sections at regular intervals
        section_positions = np.linspace(0, self.nz - 1, len(self.sections_2d))
        
        for z in range(self.nz):
            # Find nearest sections for interpolation
            idx = np.searchsorted(section_positions, z)
            
            if idx == 0:
                # Before first section
                section = cv2.resize(
                    self.sections_2d[0].astype(np.uint8),
                    (self.ny, self.nx),
                    interpolation=cv2.INTER_NEAREST
                )
            elif idx >= len(self.sections_2d):
                # After last section
                section = cv2.resize(
                    self.sections_2d[-1].astype(np.uint8),
                    (self.ny, self.nx),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                # Interpolate between two sections
                z1, z2 = section_positions[idx-1], section_positions[idx]
                w2 = (z - z1) / (z2 - z1)
                w1 = 1 - w2
                
                section1 = cv2.resize(
                    self.sections_2d[idx-1].astype(np.uint8),
                    (self.ny, self.nx),
                    interpolation=cv2.INTER_NEAREST
                )
                section2 = cv2.resize(
                    self.sections_2d[idx].astype(np.uint8),
                    (self.ny, self.nx),
                    interpolation=cv2.INTER_NEAREST
                )
                
                section = w1 * section1 + w2 * section2
            
            volume[:, :, z] = section
        
        # Convert to binary using threshold
        return volume > 0.5
    
    def simulated_annealing_reconstruction(self, max_iterations: int = 1000,
                                         initial_temp: float = 1.0,
                                         cooling_rate: float = 0.95) -> np.ndarray:
        """
        Perform 3D reconstruction using simulated annealing.
        
        Args:
            max_iterations: Maximum number of iterations
            initial_temp: Initial temperature for annealing
            cooling_rate: Temperature cooling rate
            
        Returns:
            Optimized 3D binary array
        """
        if self.volume_3d is None:
            self.initialize_3d_volume('random')
        
        current_volume = self.volume_3d.copy()
        current_energy = self._calculate_energy(current_volume)
        best_volume = current_volume.copy()
        best_energy = current_energy
        
        temperature = initial_temp
        
        print("Starting simulated annealing reconstruction...")
        
        for iteration in range(max_iterations):
            # Generate new state by flipping random pixels
            new_volume = current_volume.copy()
            
            # Randomly select pixels to flip
            num_flips = max(1, int(0.001 * current_volume.size))
            flip_coords = [
                np.random.randint(0, dim, num_flips) 
                for dim in current_volume.shape
            ]
            
            for i in range(num_flips):
                x, y, z = flip_coords[0][i], flip_coords[1][i], flip_coords[2][i]
                new_volume[x, y, z] = not new_volume[x, y, z]
            
            # Calculate energy change
            new_energy = self._calculate_energy(new_volume)
            energy_diff = new_energy - current_energy
            
            # Accept or reject the new state
            if energy_diff < 0 or np.random.random() < np.exp(-energy_diff / temperature):
                current_volume = new_volume
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_volume = current_volume.copy()
                    best_energy = current_energy
            
            # Cool down
            temperature *= cooling_rate
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Energy: {current_energy:.4f}, "
                      f"Best: {best_energy:.4f}, Temp: {temperature:.6f}")
        
        self.volume_3d = best_volume
        return best_volume
    
    def _calculate_energy(self, volume: np.ndarray) -> float:
        """
        Calculate energy function for optimization.
        
        Args:
            volume: 3D binary array
            
        Returns:
            Energy value (lower is better)
        """
        energy = 0.0
        
        # Volume fraction constraint
        current_vf = np.mean(volume)
        if self.volume_fraction is not None:
            energy += 1000 * (current_vf - self.volume_fraction)**2
        
        # 2-point correlation matching
        for axis in range(3):
            volume_2d = np.mean(volume, axis=axis)
            target_2d = self.sections_2d[0] if self.sections_2d else np.random.random(volume_2d.shape) > 0.5
            
            # Resize target to match volume slice
            target_resized = cv2.resize(
                target_2d.astype(np.uint8),
                volume_2d.shape[::-1],
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # Calculate correlation difference
            vol_corr = self.calculate_two_point_correlation(volume_2d > 0.5, max_distance=10)
            target_corr = self.calculate_two_point_correlation(target_resized, max_distance=10)
            
            energy += np.sum((vol_corr - target_corr)**2)
        
        return energy
    
    def statistical_reconstruction(self, target_correlations: Dict[str, np.ndarray] = None,
                                 max_iterations: int = 500) -> np.ndarray:
        """
        Perform statistical reconstruction based on correlation functions.
        
        Args:
            target_correlations: Dictionary of target correlation functions
            max_iterations: Maximum optimization iterations
            
        Returns:
            Reconstructed 3D volume
        """
        if self.volume_3d is None:
            self.initialize_3d_volume('random')
        
        if target_correlations is None:
            # Calculate target correlations from input sections
            target_correlations = {}
            for i, section in enumerate(self.sections_2d):
                target_correlations[f'section_{i}_2pt'] = self.calculate_two_point_correlation(section)
                target_correlations[f'section_{i}_lineal'] = self.calculate_lineal_path_function(section)
        
        print("Starting statistical reconstruction...")
        
        def objective_function(volume_flat):
            volume = volume_flat.reshape(self.target_size_3d) > 0.5
            error = 0.0
            
            # Volume fraction error
            current_vf = np.mean(volume)
            if self.volume_fraction is not None:
                error += (current_vf - self.volume_fraction)**2
            
            # Correlation function errors
            for key, target_corr in target_correlations.items():
                if '2pt' in key:
                    # 2-point correlation
                    avg_section = np.mean(volume, axis=2)
                    current_corr = self.calculate_two_point_correlation(
                        avg_section, max_distance=len(target_corr)
                    )
                    error += np.sum((current_corr - target_corr)**2)
                
                elif 'lineal' in key:
                    # Lineal path function
                    avg_section = np.mean(volume, axis=2)
                    current_lineal = self.calculate_lineal_path_function(
                        avg_section > 0.5, max_length=len(target_corr)
                    )
                    error += np.sum((current_lineal - target_corr)**2)
            
            return error
        
        # Optimize using scipy
        result = minimize(
            objective_function,
            self.volume_3d.astype(float).flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * self.volume_3d.size,
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        # Convert result back to binary volume
        optimized_volume = result.x.reshape(self.target_size_3d) > 0.5
        self.volume_3d = optimized_volume
        
        return optimized_volume
    
    def phase_field_reconstruction(self, time_steps: int = 1000,
                                 dt: float = 0.01,
                                 mobility: float = 1.0) -> np.ndarray:
        """
        Perform phase field-based reconstruction.
        
        Args:
            time_steps: Number of evolution time steps
            dt: Time step size
            mobility: Phase field mobility parameter
            
        Returns:
            Reconstructed 3D volume
        """
        if self.volume_3d is None:
            self.initialize_3d_volume('interpolated')
        
        # Convert to phase field variable (-1 to 1)
        phi = 2 * self.volume_3d.astype(float) - 1
        
        print("Starting phase field reconstruction...")
        
        for step in range(time_steps):
            # Calculate Laplacian
            laplacian = ndimage.laplace(phi)
            
            # Calculate chemical potential
            mu = phi**3 - phi - 0.1 * laplacian
            
            # Update phase field
            phi_new = phi + dt * mobility * ndimage.laplace(mu)
            
            # Apply constraints to maintain volume fraction
            if self.volume_fraction is not None:
                current_vf = np.mean(phi_new > 0)
                if abs(current_vf - self.volume_fraction) > 0.01:
                    # Adjust threshold to maintain volume fraction
                    threshold = np.percentile(phi_new, (1 - self.volume_fraction) * 100)
                    phi_new = 2 * (phi_new > threshold).astype(float) - 1
            
            phi = phi_new
            
            if step % 100 == 0:
                current_vf = np.mean(phi > 0)
                print(f"Step {step}, Volume fraction: {current_vf:.4f}")
        
        # Convert back to binary
        self.volume_3d = phi > 0
        return self.volume_3d
    
    def visualize_results(self, save_path: str = None):
        """
        Visualize the reconstruction results.
        
        Args:
            save_path: Path to save the visualization
        """
        if self.volume_3d is None:
            print("No 3D volume to visualize. Run reconstruction first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original 2D sections
        axes[0, 0].imshow(self.sections_2d[0], cmap='gray')
        axes[0, 0].set_title('Original 2D Section')
        axes[0, 0].axis('off')
        
        # Reconstructed slices
        mid_z = self.nz // 2
        mid_y = self.ny // 2
        mid_x = self.nx // 2
        
        axes[0, 1].imshow(self.volume_3d[:, :, mid_z], cmap='gray')
        axes[0, 1].set_title(f'Reconstructed XY slice (z={mid_z})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(self.volume_3d[:, mid_y, :], cmap='gray')
        axes[0, 2].set_title(f'Reconstructed XZ slice (y={mid_y})')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(self.volume_3d[mid_x, :, :], cmap='gray')
        axes[1, 0].set_title(f'Reconstructed YZ slice (x={mid_x})')
        axes[1, 0].axis('off')
        
        # Volume fraction comparison
        original_vf = [np.mean(section) for section in self.sections_2d]
        reconstructed_vf = np.mean(self.volume_3d)
        
        axes[1, 1].bar(['Original (avg)', 'Reconstructed'], 
                      [np.mean(original_vf), reconstructed_vf])
        axes[1, 1].set_title('Volume Fraction Comparison')
        axes[1, 1].set_ylabel('Volume Fraction')
        
        # 2-point correlation comparison
        if len(self.sections_2d) > 0:
            orig_corr = self.calculate_two_point_correlation(self.sections_2d[0])
            recon_slice = self.volume_3d[:, :, mid_z]
            recon_corr = self.calculate_two_point_correlation(recon_slice)
            
            min_len = min(len(orig_corr), len(recon_corr))
            axes[1, 2].plot(orig_corr[:min_len], 'b-', label='Original', linewidth=2)
            axes[1, 2].plot(recon_corr[:min_len], 'r--', label='Reconstructed', linewidth=2)
            axes[1, 2].set_title('2-Point Correlation Function')
            axes[1, 2].set_xlabel('Distance')
            axes[1, 2].set_ylabel('Correlation')
            axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def export_volume(self, filename: str, format: str = 'npy'):
        """
        Export the reconstructed 3D volume.
        
        Args:
            filename: Output filename
            format: Export format ('npy', 'vtk', 'raw')
        """
        if self.volume_3d is None:
            print("No 3D volume to export. Run reconstruction first.")
            return
        
        if format == 'npy':
            np.save(filename, self.volume_3d)
            print(f"Volume exported as {filename}")
            
        elif format == 'raw':
            self.volume_3d.astype(np.uint8).tofile(filename)
            print(f"Volume exported as raw binary file: {filename}")
            
        elif format == 'vtk':
            self._export_vtk(filename)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_vtk(self, filename: str):
        """Export volume in VTK format for visualization."""
        nx, ny, nz = self.volume_3d.shape
        
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("3D Reconstructed Microstructure\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
            f.write("ORIGIN 0.0 0.0 0.0\n")
            f.write("SPACING 1.0 1.0 1.0\n")
            f.write(f"POINT_DATA {nx * ny * nz}\n")
            f.write("SCALARS phase unsigned_char\n")
            f.write("LOOKUP_TABLE default\n")
            
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        f.write(f"{int(self.volume_3d[x, y, z])}\n")
        
        print(f"Volume exported as VTK file: {filename}")


def create_synthetic_2d_sections(num_sections: int = 5, 
                                section_size: Tuple[int, int] = (100, 100),
                                volume_fraction: float = 0.3,
                                correlation_length: float = 10.0) -> List[np.ndarray]:
    """
    Create synthetic 2D sections for testing purposes.
    
    Args:
        num_sections: Number of 2D sections to generate
        section_size: Size of each section (height, width)
        volume_fraction: Target volume fraction
        correlation_length: Spatial correlation length
        
    Returns:
        List of 2D binary arrays
    """
    sections = []
    
    for i in range(num_sections):
        # Generate correlated random field
        noise = np.random.normal(0, 1, section_size)
        
        # Apply Gaussian filter for spatial correlation
        filtered = ndimage.gaussian_filter(noise, sigma=correlation_length)
        
        # Convert to binary based on volume fraction
        threshold = np.percentile(filtered, (1 - volume_fraction) * 100)
        binary_section = filtered > threshold
        
        sections.append(binary_section)
    
    return sections


def main():
    """Main function demonstrating the 3D reconstruction workflow."""
    print("3D Microstructure Reconstruction Demo")
    print("=" * 50)
    
    # Create synthetic 2D sections
    print("Creating synthetic 2D sections...")
    sections_2d = create_synthetic_2d_sections(
        num_sections=3,
        section_size=(64, 64),
        volume_fraction=0.35,
        correlation_length=5.0
    )
    
    # Initialize reconstructor
    target_size = (64, 64, 64)
    reconstructor = MicrostructureReconstructor(sections_2d, target_size)
    
    # Calculate volume fraction
    vf = reconstructor.calculate_volume_fraction()
    print(f"Target volume fraction: {vf:.3f}")
    
    # Method 1: Simulated Annealing
    print("\n1. Simulated Annealing Reconstruction...")
    reconstructor.initialize_3d_volume('random')
    volume_sa = reconstructor.simulated_annealing_reconstruction(
        max_iterations=500,
        initial_temp=1.0,
        cooling_rate=0.98
    )
    
    # Method 2: Phase Field Reconstruction
    print("\n2. Phase Field Reconstruction...")
    reconstructor.initialize_3d_volume('interpolated')
    volume_pf = reconstructor.phase_field_reconstruction(
        time_steps=200,
        dt=0.01,
        mobility=1.0
    )
    
    # Visualize results
    print("\n3. Visualizing results...")
    reconstructor.visualize_results('reconstruction_results.png')
    
    # Export volume
    print("\n4. Exporting volume...")
    reconstructor.export_volume('reconstructed_volume.npy')
    reconstructor.export_volume('reconstructed_volume.vtk', format='vtk')
    
    print("\nReconstruction complete!")
    
    # Print statistics
    print(f"\nFinal Statistics:")
    print(f"Original volume fraction: {vf:.3f}")
    print(f"Reconstructed volume fraction: {np.mean(reconstructor.volume_3d):.3f}")
    print(f"Reconstruction size: {reconstructor.volume_3d.shape}")


if __name__ == "__main__":
    main()
