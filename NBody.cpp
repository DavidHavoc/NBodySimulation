#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream> // Added for std::ifstream
#include <CL/opencl.hpp>

#include "include/Body.h"

// Constants
constexpr float G = 1.f;
constexpr float dt = .1f;
constexpr float eps = 1e-1f;
constexpr size_t n_bodies = 500; // Increased for better OpenCL demonstration
constexpr float center_mass = 1000.f;

constexpr float TARGET_FPS = 165.f;
const sf::Time FRAME_DURATION = sf::seconds(1.f / TARGET_FPS);

const int WIDTH = 2560, HEIGHT = 1440;

// Function to compute forces (Sequential)
void compute_forces_sequential(BodiesSOA& bodies, float G, float eps) {
    for (size_t i = 0; i < bodies.count; ++i) {
        bodies.ax[i] = 0.0f;
        bodies.ay[i] = 0.0f;
    }

    for (size_t i = 0; i < bodies.count; ++i) {
        for (size_t j = 0; j < bodies.count; ++j) {
            if (i == j) continue;

            float dx = bodies.x[j] - bodies.x[i];
            float dy = bodies.y[j] - bodies.y[i];

            float dist_sq = dx * dx + dy * dy;
            float dist = std::sqrt(dist_sq);
            float d3_inv = 1.0f / (dist_sq * dist + eps * eps * eps);

            float f = G * bodies.m[j] * d3_inv;

            bodies.ax[i] += dx * f;
            bodies.ay[i] += dy * f;
        }
    }
}

// Function to integrate bodies (Sequential)
void integrate_bodies_sequential(BodiesSOA& bodies, float dt, int width, int height) {
    for (size_t i = 0; i < bodies.count; ++i) {
        bodies.vx[i] += bodies.ax[i] * dt;
        bodies.vy[i] += bodies.ay[i] * dt;
        bodies.x[i] += bodies.vx[i] * dt;
        bodies.y[i] += bodies.vy[i] * dt;

        // Toroidal universe
        if (bodies.x[i] > width / 2) bodies.x[i] -= width;
        if (bodies.x[i] < -width / 2) bodies.x[i] += width;
        if (bodies.y[i] > height / 2) bodies.y[i] -= height;
        if (bodies.y[i] < -height / 2) bodies.y[i] += height;
    }
}

sf::Color mass_to_color(float m) {
    float norm = std::min(1.0f, m / 10.0f);
    return sf::Color(static_cast<sf::Uint8>(255 * norm), 50, static_cast<sf::Uint8>(255 * (1 - norm)));
}

float orbital_velocity_scalar(float M, float r) {
    return std::sqrt(1.0f * M / r); // G = 1.0 assumed
}

BodiesSOA initialize_bodies() {
    BodiesSOA bodies(n_bodies);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dist(50.0f, std::min(WIDTH, HEIGHT) / 2.f - 20.f);
    std::uniform_real_distribution<float> mass_dist(0.5f, 10.f);

    // Center body with heavier mass
    bodies.x[0] = 0.0f;
    bodies.y[0] = 0.0f;
    bodies.vx[0] = 0.0f;
    bodies.vy[0] = 0.0f;
    bodies.ax[0] = 0.0f;
    bodies.ay[0] = 0.0f;
    bodies.m[0] = center_mass;

    for (size_t i = 1; i < n_bodies; ++i) {
        float angle = angle_dist(rng);
        float radius = radius_dist(rng);
        float mass = mass_dist(rng);

        bodies.x[i] = radius * std::cos(angle);
        bodies.y[i] = radius * std::sin(angle);

        float orbital_vel = orbital_velocity_scalar(center_mass, radius);
        bodies.vx[i] = -orbital_vel * std::sin(angle);
        bodies.vy[i] = orbital_vel * std::cos(angle);
        bodies.ax[i] = 0.0f;
        bodies.ay[i] = 0.0f;
        bodies.m[i] = mass;
    }
    return bodies;
}

// OpenCL related variables
cl::Context context;
cl::CommandQueue queue;
cl::Program program;
cl::Kernel compute_forces_kernel;
cl::Kernel integrate_bodies_kernel;

cl::Buffer d_x, d_y, d_vx, d_vy, d_ax, d_ay, d_m;

void init_opencl(size_t num_bodies) {
    // Get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.empty()) {
        std::cerr << "No OpenCL platforms found!\n";
        exit(1);
    }

    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    // Get default device (GPU or CPU)
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.empty()) {
        std::cerr << "No OpenCL devices found!\n";
        exit(1);
    }

    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    context = cl::Context({default_device});
    queue = cl::CommandQueue(context, default_device);

    // Load kernel source
    std::ifstream kernel_file("opencl/NBody.cl");
    std::string kernel_source(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources;
    sources.push_back({kernel_source.c_str(), kernel_source.length()});

    program = cl::Program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }

    compute_forces_kernel = cl::Kernel(program, "compute_forces");
    integrate_bodies_kernel = cl::Kernel(program, "integrate_bodies");

    // Create buffers on device
    d_x = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_bodies);
    d_y = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_bodies);
    d_vx = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_bodies);
    d_vy = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_bodies);
    d_ax = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_bodies);
    d_ay = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_bodies);
    d_m = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * num_bodies);
}

void run_opencl_simulation(BodiesSOA& bodies) {
    // Write data to device
    queue.enqueueWriteBuffer(d_x, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.x.data());
    queue.enqueueWriteBuffer(d_y, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.y.data());
    queue.enqueueWriteBuffer(d_vx, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.vx.data());
    queue.enqueueWriteBuffer(d_vy, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.vy.data());
    queue.enqueueWriteBuffer(d_m, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.m.data());

    // Set compute_forces_kernel arguments
    compute_forces_kernel.setArg(0, d_x);
    compute_forces_kernel.setArg(1, d_y);
    compute_forces_kernel.setArg(2, d_m);
    compute_forces_kernel.setArg(3, d_ax);
    compute_forces_kernel.setArg(4, d_ay);
    compute_forces_kernel.setArg(5, (int)bodies.count);
    compute_forces_kernel.setArg(6, G);
    compute_forces_kernel.setArg(7, eps);

    // Enqueue compute_forces_kernel
    queue.enqueueNDRangeKernel(compute_forces_kernel, cl::NullRange, cl::NDRange(bodies.count), cl::NullRange);

    // Set integrate_bodies_kernel arguments
    integrate_bodies_kernel.setArg(0, d_x);
    integrate_bodies_kernel.setArg(1, d_y);
    integrate_bodies_kernel.setArg(2, d_vx);
    integrate_bodies_kernel.setArg(3, d_vy);
    integrate_bodies_kernel.setArg(4, d_ax);
    integrate_bodies_kernel.setArg(5, d_ay);
    integrate_bodies_kernel.setArg(6, (int)bodies.count);
    integrate_bodies_kernel.setArg(7, dt);
    integrate_bodies_kernel.setArg(8, WIDTH);
    integrate_bodies_kernel.setArg(9, HEIGHT);

    // Enqueue integrate_bodies_kernel
    queue.enqueueNDRangeKernel(integrate_bodies_kernel, cl::NullRange, cl::NDRange(bodies.count), cl::NullRange);

    // Read results back to host
    queue.enqueueReadBuffer(d_x, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.x.data());
    queue.enqueueReadBuffer(d_y, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.y.data());
    queue.enqueueReadBuffer(d_vx, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.vx.data());
    queue.enqueueReadBuffer(d_vy, CL_TRUE, 0, sizeof(float) * bodies.count, bodies.vy.data());
}

int main(int argc, char* argv[]) {
    bool use_opencl = false;
    if (argc > 1 && std::string(argv[1]) == "--opencl") {
        use_opencl = true;
    }

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "N-Body Simulation");
    sf::Font font;
    if (!font.loadFromFile("font.ttf")) {
        std::cerr << "Failed to load font\n";
        return 1;
    }
    sf::Text fpsText("", font, 18);
    fpsText.setFillColor(sf::Color::White);
    fpsText.setPosition(10, 5);

    BodiesSOA bodies = initialize_bodies();

    if (use_opencl) {
        init_opencl(bodies.count);
    }

    sf::Clock frameClock;
    sf::Clock fpsClock;
    float lastFPS = 0.0f;

    while (window.isOpen()) {
        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed) {
                window.close();
            }
        }

        if (use_opencl) {
            run_opencl_simulation(bodies);
        } else {
            compute_forces_sequential(bodies, G, eps);
            integrate_bodies_sequential(bodies, dt, WIDTH, HEIGHT);
        }

        window.clear(sf::Color::Black);

        // Drawing of bodies
        for (size_t i = 0; i < bodies.count; ++i) {
            sf::CircleShape circle(bodies.m[i] > 50.0f ? 6 : 2);
            circle.setFillColor(mass_to_color(bodies.m[i]));
            circle.setPosition(WIDTH / 2 + bodies.x[i], HEIGHT / 2 + bodies.y[i]);
            circle.setOrigin(circle.getRadius(), circle.getRadius());
            window.draw(circle);
        }

        // FPS calculation and display
        float elapsed = fpsClock.restart().asSeconds();
        lastFPS = 1.0f / elapsed;
        fpsText.setString("FPS: " + std::to_string(static_cast<int>(lastFPS)));
        window.draw(fpsText);

        window.display();

        sf::Time elapsed_frame = frameClock.getElapsedTime();
        if (elapsed_frame < FRAME_DURATION) {
            sf::sleep(FRAME_DURATION - elapsed_frame);
        }
        frameClock.restart();
    }

    return 0;
}


