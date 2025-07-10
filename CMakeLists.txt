cmake_minimum_required(VERSION 3.16)
project(NBodySimulation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SFML (version 2.6)
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Source files
set(SOURCES
    NBody.cpp
    src/Body.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories (add include/ directory)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # Add this line
    ${OpenCL_INCLUDE_DIRS}
)

# Set compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:DEBUG>:-g>
    $<$<CONFIG:RELEASE>:-O3>
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    ${OpenCL_LIBRARIES}
)

# Copy OpenCL kernel and create directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/opencl/NBody.cl
    ${CMAKE_CURRENT_BINARY_DIR}/opencl/NBody.cl
    COPYONLY
)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/opencl)

# Copy font file to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/font.ttf
    ${CMAKE_CURRENT_BINARY_DIR}/font.ttf
    COPYONLY
)

# Add run target
add_custom_target(run
    COMMAND ${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running N-Body Simulation"
)