# N-Body Simulation - High-Performance Computing Project build for UNI

A high-performance N-body gravitational simulation implemented in C++ with both sequential CPU and parallel OpenCL implementations, featuring real-time SFML visualization.

## Features

- **Sequential CPU Implementation**: Traditional single-threaded CPU computation for smaller body counts
- **Parallel OpenCL Implementation**: GPU-accelerated computation using OpenCL for large-scale simulations
- **Real-time Visualization**: SFML-based graphics with smooth 165 FPS target
- **Structure of Arrays (SOA)**: Optimized memory layout for parallel processing
- **Toroidal Universe**: Bodies wrap around screen edges for continuous simulation
- **Dynamic Body Count**: Supports from 5 to 1000+ bodies depending on hardware
- **Performance Monitoring**: Real-time FPS and computation time display

## Project Structure

```
NBody/
├── NBody.cpp              # Main application file
├── src/
│   └── Body.cpp          # Body class implementation
├── include/
│   └── Body.h            # Body class header
├── opencl/
│   └── NBody.cl          # OpenCL kernel implementations
├── CMakeLists.txt        # CMake build configuration
├── install_sfml.sh       # SFML installation script
├── font.ttf              # Font file for text rendering
└── README.md             # This documentation
```

## Requirements

### System Dependencies
- **C++ Compiler**: GCC 7+ or Clang 6+ with C++17 support
- **CMake**: Version 3.16 or higher
- **SFML**: Version 2.5 or higher
- **OpenCL**: Version 1.2 or higher (optional, for GPU acceleration)

### Hardware Requirements
- **CPU**: Any modern multi-core processor
- **GPU**: OpenCL-compatible GPU (NVIDIA, AMD, or Intel) for parallel implementation
- **RAM**: At least 4GB (8GB+ recommended for large simulations)

## Installation

### 1. Install Dependencies

Run the provided installation script:
```bash
chmod +x install_sfml.sh
./install_sfml.sh
```

Or install manually:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libsfml-dev opencl-headers ocl-icd-libopencl1 ocl-icd-dev build-essential cmake pkg-config

# macOS (with Homebrew)
brew install sfml opencl-headers

# Windows (with vcpkg)
vcpkg install sfml opencl
```

### 2. Build the Project

```bash
mkdir build
cd build
cmake ..
make
```

### 3. Run the Simulation

```bash
./NBodySimulation
```

## Usage

When you run the simulation, you'll be prompted to choose between:
1. **Sequential CPU**: Single-threaded implementation suitable for learning and small simulations
2. **OpenCL GPU/CPU**: Parallel implementation for high-performance computing

### Controls
- **Close Window**: ESC or click the X button
- **Performance Info**: Displayed in the top-left corner

### Configuration

Key parameters can be modified in the source code:

```cpp
// In NBody.cpp
constexpr size_t n_bodies = 1000;     // Number of bodies
constexpr float G = 1.0f;             // Gravitational constant
constexpr float dt = 0.1f;            // Time step
constexpr float eps = 1e-1f;          // Softening parameter
constexpr float TARGET_FPS = 165.0f;  // Target frame rate
```

## Implementation Details

### Physics Model

The simulation implements a simplified N-body gravitational system:

1. **Force Calculation**: For each body pair (i,j):
   ```
   F_ij = G * m_j * m_i / (r_ij^2 + ε^2)^(3/2)
   ```

2. **Integration**: Leapfrog integration scheme:
   ```
   v(t+dt) = v(t) + a(t) * dt
   x(t+dt) = x(t) + v(t+dt) * dt
   ```

3. **Boundary Conditions**: Toroidal universe with wraparound

### Performance Optimizations

- **Structure of Arrays (SOA)**: Memory layout optimized for SIMD operations
- **OpenCL Parallelization**: Each body's force calculation runs in parallel
- **Memory Coalescing**: Efficient GPU memory access patterns
- **Softening Parameter**: Prevents numerical instabilities at close distances

### Visualization Features

- **Color Coding**: Body color varies with mass (red = heavy, blue = light)
- **Size Scaling**: Visual size reflects body mass
- **Smooth Animation**: 165 FPS target with frame limiting
- **Real-time Metrics**: FPS, body count, and computation time display

## Troubleshooting

### Common Issues

1. **OpenCL Not Found**
   ```
   Error: No OpenCL platforms found
   ```
   - Install OpenCL drivers for your GPU
   - The simulation will automatically fall back to CPU mode

2. **SFML Not Found**
   ```
   Error: SFML development libraries not found
   ```
   - Run `./install_sfml.sh` or install SFML manually
   - Ensure pkg-config can find SFML

3. **Font Loading Error**
   ```
   Failed to load font
   ```
   - Ensure `font.ttf` exists in the project directory
   - The simulation will use default font if custom font fails

4. **Poor Performance**
   - Reduce `n_bodies` constant for better performance
   - Ensure GPU drivers are properly installed
   - Check if system has sufficient RAM

### Debug Build

For development and debugging:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## Educational Value

This project demonstrates several important HPC concepts:

- **Parallel Algorithm Design**: Converting sequential algorithms to parallel
- **Memory Layout Optimization**: AoS vs SoA performance implications
- **GPU Computing**: OpenCL programming model and best practices
- **Performance Analysis**: Measuring and optimizing computational bottlenecks
- **Numerical Methods**: Stable integration schemes for physical simulation

## Extensions and Modifications

Consider these enhancements for further learning:

1. **Advanced Integrators**: Implement Runge-Kutta or Verlet integration
2. **Hierarchical Methods**: Add Barnes-Hut algorithm for O(N log N) complexity
3. **Multiple Forces**: Include electromagnetic or strong nuclear forces
4. **Collision Detection**: Handle body mergers and fragmentation
5. **3D Visualization**: Extend to three-dimensional space
6. **Distributed Computing**: MPI implementation for cluster computing

## License

This project is developed for educational purposes as part of the High-Performance Computing course at the Institute for Computer Science.

## Contributors

- Prof. Dr. Ivan Kisel (Course Instructor)
- Robin Lakos (Teaching Assistant)
- Akhil Mithran (Teaching Assistant)
- Oddharak Tyagi (Teaching Assistant)

## References

- [SFML Documentation](https://www.sfml-dev.org/documentation/)
- [OpenCL Programming Guide](https://www.khronos.org/opencl/)
- [N-Body Problem - Wikipedia](https://en.wikipedia.org/wiki/N-body_problem)
- [GPU Computing Best Practices](https://developer.nvidia.com/blog/gpu-computing-best-practices/)
