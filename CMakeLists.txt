cmake_minimum_required(VERSION 3.10)

project(NBody)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SFML_DIR "${HOME}/SFML/lib/cmake/SFML")
find_package(SFML 2.6 COMPONENTS graphics window system audio network REQUIRED)

find_package(OpenCL REQUIRED)

include_directories(include)

add_executable(NBody NBody.cpp)

target_link_libraries(NBody sfml-graphics sfml-window sfml-system sfml-audio sfml-network ${OpenCL_LIBRARIES})

# Copy font to build directory
add_custom_command(
    TARGET NBody POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/font.ttf
            $<TARGET_FILE_DIR:NBody>/
)


