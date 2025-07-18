# N-Body Simulation

This project implements a simplified N-body problem simulation with graphical visualization using SFML (Simple and Fast Multimedia Library) and an optional OpenCL parallelization.

## Project Structure

```
NBody/
├── CMakeLists.txt
├── NBody.cpp
├── font.ttf
├── include/
│   └── Body.h
├── opencl/
│   └── NBody.cl
└── src/ (no longer used for Body.cpp)
```

- `NBody.cpp`: Contains the main application logic, including the simulation loop, force calculation, integration, and SFML rendering. It now includes logic to switch between sequential and OpenCL execution.
- `include/Body.h`: Definition of the `BodiesSOA` struct for Structure of Arrays data layout.
- `opencl/NBody.cl`: OpenCL kernel file containing `compute_forces` and `integrate_bodies` functions for parallel execution on OpenCL devices.
- `font.ttf`: Placeholder for the font file used for displaying FPS. If you have a specific font, replace this file.
- `CMakeLists.txt`: CMake build script for the project, now configured to find and link OpenCL libraries.

## Prerequisites

To build and run this project, you need the following:

- C++ Compiler (g++ recommended)
- CMake (version 3.10 or higher)
- SFML 2.6.1 (Simple and Fast Multimedia Library)
- OpenCL development environment (headers, ICD loader, and a compatible runtime for your CPU/GPU).

## SFML and OpenCL Installation

SFML was installed using a custom script. The installation prefix is `$HOME/SFML`. The necessary libraries and headers are located in this directory.

OpenCL development environment (headers and ICD loader) was installed using `apt-get`. For the OpenCL runtime, an Intel OpenCL ICD was installed.

## Building the Project

1. Navigate to the `NBody` directory:
   ```bash
   cd NBody
   ```

2. Create a `build` directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```

3. Run CMake to configure the project. Ensure that `CMAKE_PREFIX_PATH` is set to the SFML installation directory:
   ```bash
   cmake .. -DCMAKE_PREFIX_PATH=$HOME/SFML
   ```

4. Build the project using `make`:
   ```bash
   make
   ```

## Running the Project

After building, the executable `NBody` will be located in the `build` directory. Before running, you need to set the `LD_LIBRARY_PATH` environment variable to include the SFML library directory.

From the `build` directory:

```bash
export LD_LIBRARY_PATH=$HOME/SFML/lib:$LD_LIBRARY_PATH
./NBody
```

## Running Modes (Sequential vs. OpenCL)

This application supports both sequential and OpenCL parallel execution modes.

### Sequential Mode

To run the simulation in sequential mode (default):

```bash
./NBody
```

### OpenCL Mode

To run the simulation using OpenCL (if an OpenCL device is available):

```bash
./NBody --opencl
```

## OpenCL Kernel Details

The `opencl/NBody.cl` file contains the OpenCL kernel functions `compute_forces` and `integrate_bodies`. These kernels perform the gravitational force calculations and body position/velocity updates in parallel on the OpenCL device.

- **`compute_forces`**: Calculates the gravitational forces exerted on each body by all other bodies.
- **`integrate_bodies`**: Updates the position and velocity of each body based on the calculated forces.

## Toroidal Universe

The `integrate_bodies` OpenCL kernel includes logic to implement a toroidal universe. This means that if a body moves beyond the window boundaries on one side, it will re-enter from the opposite side, preventing bodies from disappearing from the simulation view.

