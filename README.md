# 3D Microstructure Reconstruction from 2D Sections

This project implements algorithms for reconstructing 3D microstructures of two-phase random heterogeneous materials from 2D cross-sectional images. The implementation follows approaches commonly used in materials science and computational materials engineering.

## Overview

The reconstruction of 3D microstructures from 2D sections is a critical problem in materials characterization. This implementation provides several algorithms to address this challenge:

1. **Statistical Correlation-based Reconstruction**
2. **Simulated Annealing Optimization**
3. **Phase Field Evolution**
4. **Multi-point Statistics Approach**

## Algorithm Description

### 1. Statistical Approach
The statistical approach aims to match key statistical descriptors between the original 2D sections and the reconstructed 3D volume:

- **Volume Fraction**: The proportion of each phase
- **Two-point Correlation Function**: Spatial correlation between points at different distances
- **Lineal Path Function**: Probability that a line segment of given length lies entirely within a phase

### 2. Simulated Annealing
This optimization technique iteratively improves the 3D reconstruction by:
- Starting with a random or informed initial guess
- Proposing random changes to the microstructure
- Accepting/rejecting changes based on an energy function
- Gradually reducing the "temperature" to converge to an optimal solution

The energy function combines:
- Volume fraction matching
- Correlation function matching
- Spatial continuity constraints

### 3. Phase Field Method
This approach treats the reconstruction as a dynamic evolution problem:
- Uses a phase field variable φ that varies smoothly between phases
- Evolves the field according to thermodynamic principles
- Incorporates constraints from the 2D section data
- Results in physically realistic microstructures

### 4. Key Features

#### Statistical Descriptors
- **Two-point correlation**: `S₂(r) = ⟨I(x)I(x+r)⟩`
- **Lineal path function**: `L(r) = P(line segment of length r is entirely in phase)`
- **Volume fraction**: `φ = ⟨I(x)⟩`

#### Optimization Energy Function
```
E = w₁(φ_target - φ_current)² + w₂∑(S₂_target(r) - S₂_current(r))² + w₃∑(L_target(r) - L_current(r))²
```

#### Phase Field Evolution
```
∂φ/∂t = M∇²μ
μ = φ³ - φ - ε²∇²φ
```

## Installation

1. Clone or download the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- opencv-python >= 4.5.0
- scikit-learn >= 1.0.0

## Usage

### Basic Usage

```python
from microstructure_3d_reconstruction import MicrostructureReconstructor
import numpy as np

# Load your 2D sections (binary arrays)
sections_2d = [section1, section2, section3]  # List of 2D numpy arrays

# Initialize reconstructor
target_size = (100, 100, 100)  # Desired 3D dimensions
reconstructor = MicrostructureReconstructor(sections_2d, target_size)

# Method 1: Simulated Annealing
reconstructor.initialize_3d_volume('random')
volume_sa = reconstructor.simulated_annealing_reconstruction(
    max_iterations=1000,
    initial_temp=1.0,
    cooling_rate=0.95
)

# Method 2: Phase Field Reconstruction
reconstructor.initialize_3d_volume('interpolated')
volume_pf = reconstructor.phase_field_reconstruction(
    time_steps=500,
    dt=0.01,
    mobility=1.0
)

# Visualize results
reconstructor.visualize_results('reconstruction_results.png')

# Export volume
reconstructor.export_volume('reconstructed_volume.npy')
```

### Running the Demo

Execute the demonstration script to see all methods in action:

```bash
python demo_reconstruction.py
```

This will:
- Create synthetic 2D sections for testing
- Run different reconstruction algorithms
- Compare statistical properties
- Generate visualization plots
- Export results in multiple formats

## File Structure

```
├── microstructure_3d_reconstruction.py  # Main implementation
├── demo_reconstruction.py               # Demonstration script
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## Output Files

The scripts generate several output files:

### Visualizations
- `input_sections.png` - Original 2D sections
- `sa_reconstruction.png` - Simulated annealing results
- `pf_reconstruction.png` - Phase field results
- `statistical_comparison.png` - Statistical analysis
- `methods_comparison.png` - Method comparison

### Data Files
- `reconstructed_volume.npy` - NumPy binary format
- `reconstructed_volume.raw` - Raw binary format
- `reconstructed_volume.vtk` - VTK format for ParaView/VisIt

## Algorithm Parameters

### Simulated Annealing
- `max_iterations`: Number of optimization steps (default: 1000)
- `initial_temp`: Starting temperature (default: 1.0)
- `cooling_rate`: Temperature reduction factor (default: 0.95)

### Phase Field Method
- `time_steps`: Evolution time steps (default: 1000)
- `dt`: Time step size (default: 0.01)
- `mobility`: Phase field mobility (default: 1.0)

### Statistical Reconstruction
- `max_iterations`: Optimization iterations (default: 500)
- `target_correlations`: Dictionary of target correlation functions

## Applications

This implementation can be used for:

1. **Materials Characterization**: Reconstruct 3D microstructures from microscopy images
2. **Property Prediction**: Calculate effective properties of heterogeneous materials
3. **Process Simulation**: Model material evolution during processing
4. **Design Optimization**: Optimize microstructure for desired properties

## Theoretical Background

The reconstruction problem is mathematically formulated as an optimization problem:

```
min E(φ) = ∑ᵢ wᵢ(fᵢ_target - fᵢ(φ))²
```

Where:
- `φ` is the 3D microstructure
- `fᵢ` are statistical descriptors (volume fraction, correlations, etc.)
- `wᵢ` are weighting factors
- `fᵢ_target` are target values from 2D sections

### Common Statistical Descriptors

1. **n-point Correlation Functions**: Describe spatial correlations between n points
2. **Lineal Path Functions**: Characterize connectivity and tortuosity
3. **Chord Length Distributions**: Describe phase size distributions
4. **Surface Area and Interfacial Properties**: Geometric characteristics

## Validation

The quality of reconstruction can be assessed by comparing:

1. **Statistical descriptors** between original and reconstructed microstructures
2. **Visual similarity** in cross-sections
3. **Physical properties** (if experimental data available)
4. **Convergence metrics** during optimization

## Limitations and Considerations

1. **Computational Cost**: 3D reconstruction can be computationally intensive
2. **Non-uniqueness**: Multiple 3D structures can match the same 2D statistics
3. **Limited Information**: 2D sections contain limited 3D information
4. **Parameter Sensitivity**: Results may depend on algorithm parameters

## Future Enhancements

Potential improvements include:

1. **Deep Learning Approaches**: Use neural networks for reconstruction
2. **Multi-scale Methods**: Incorporate multiple length scales
3. **Additional Constraints**: Include physical constraints (connectivity, etc.)
4. **GPU Acceleration**: Implement CUDA/OpenCL for faster computation
5. **Advanced Statistics**: Include higher-order statistical descriptors

## References

The implementation is based on concepts from:

1. Torquato, S. (2002). Random Heterogeneous Materials: Microstructure and Macroscopic Properties
2. Yeong, C. L. Y., & Torquato, S. (1998). Reconstructing random media
3. Jiao, Y., et al. (2009). Modeling and predicting microstructure evolution in lead/tin alloy
4. Liu, Y., et al. (2016). Random heterogeneous materials via texture synthesis

## License

This implementation is provided for educational and research purposes. Please cite appropriately if used in academic work.
