#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <memory>

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include "include/Body.h"

// Constants
constexpr float G = 1.0f;
constexpr float dt = 0.1f;
constexpr float eps = 1e-1f;
constexpr size_t n_bodies = 1000;
constexpr float center_mass = 1000.0f;
constexpr float TARGET_FPS = 165.0f;

const sf::Time FRAME_DURATION = sf::seconds(1.0f / TARGET_FPS);
const int WIDTH = 2560;
const int HEIGHT = 1440;

std::string load_kernel_source(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + filename);
    }
    
    std::string source((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    return source;
}

sf::Color mass_to_color(float m) {
    float norm = std::min(1.0f, m / 10.0f);
    return sf::Color(static_cast<sf::Uint8>(255 * norm), 
                     50, 
                     static_cast<sf::Uint8>(255 * (1 - norm)));
}

class OpenCLNBodySimulation {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel force_kernel;
    cl::Kernel integrate_kernel;
    
    cl::Buffer x_buffer, y_buffer, vx_buffer, vy_buffer;
    cl::Buffer ax_buffer, ay_buffer, m_buffer;
    
    BodySOA bodies;
    bool initialized = false;
    
public:
    OpenCLNBodySimulation() = default;
    
    void initialize() {
        try {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            
            if (platforms.empty()) {
                throw std::runtime_error("No OpenCL platforms found");
            }
            
            std::vector<cl::Device> devices;
            platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            
            if (devices.empty()) {
                throw std::runtime_error("No OpenCL devices found");
            }
            
            context = cl::Context(devices);
            queue = cl::CommandQueue(context, devices[0]);
            
            std::string kernel_source = load_kernel_source("opencl/NBody.cl");
            program = cl::Program(context, kernel_source);
            
            program.build();
            
            force_kernel = cl::Kernel(program, "compute_forces");
            integrate_kernel = cl::Kernel(program, "integrate_bodies");
            
            initialize_bodies_soa(bodies, WIDTH, HEIGHT, n_bodies);
            
            const size_t float_size = sizeof(float) * n_bodies;
            x_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, float_size);
            y_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, float_size);
            vx_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, float_size);
            vy_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, float_size);
            ax_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, float_size);
            ay_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, float_size);
            m_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, float_size);
            
            queue.enqueueWriteBuffer(x_buffer, CL_TRUE, 0, float_size, bodies.x.data());
            queue.enqueueWriteBuffer(y_buffer, CL_TRUE, 0, float_size, bodies.y.data());
            queue.enqueueWriteBuffer(vx_buffer, CL_TRUE, 0, float_size, bodies.vx.data());
            queue.enqueueWriteBuffer(vy_buffer, CL_TRUE, 0, float_size, bodies.vy.data());
            queue.enqueueWriteBuffer(m_buffer, CL_TRUE, 0, float_size, bodies.m.data());
            
            initialized = true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL initialization error: " << e.what() << std::endl;
            throw;
        }
    }
    
    void update() {
        if (!initialized) return;
        
        force_kernel.setArg(0, x_buffer);
        force_kernel.setArg(1, y_buffer);
        force_kernel.setArg(2, ax_buffer);
        force_kernel.setArg(3, ay_buffer);
        force_kernel.setArg(4, m_buffer);
        force_kernel.setArg(5, static_cast<int>(n_bodies));
        force_kernel.setArg(6, G);
        force_kernel.setArg(7, eps);
        
        queue.enqueueNDRangeKernel(force_kernel, cl::NullRange, cl::NDRange(n_bodies));
        
        integrate_kernel.setArg(0, x_buffer);
        integrate_kernel.setArg(1, y_buffer);
        integrate_kernel.setArg(2, vx_buffer);
        integrate_kernel.setArg(3, vy_buffer);
        integrate_kernel.setArg(4, ax_buffer);
        integrate_kernel.setArg(5, ay_buffer);
        integrate_kernel.setArg(6, static_cast<int>(n_bodies));
        integrate_kernel.setArg(7, dt);
        integrate_kernel.setArg(8, static_cast<float>(WIDTH));
        integrate_kernel.setArg(9, static_cast<float>(HEIGHT));
        
        queue.enqueueNDRangeKernel(integrate_kernel, cl::NullRange, cl::NDRange(n_bodies));
        queue.finish();
    }
    
    void read_positions() {
        if (!initialized) return;
        
        const size_t float_size = sizeof(float) * n_bodies;
        queue.enqueueReadBuffer(x_buffer, CL_TRUE, 0, float_size, bodies.x.data());
        queue.enqueueReadBuffer(y_buffer, CL_TRUE, 0, float_size, bodies.y.data());
    }
    
    const BodySOA& get_bodies() const { return bodies; }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "N-Body Simulation");
    
    sf::Font font;
    if (!font.loadFromFile("font.ttf")) {
        std::cerr << "Failed to load font. Using default font.\n";
    }
    
    sf::Text fpsText("", font, 18);
    fpsText.setFillColor(sf::Color::White);
    fpsText.setPosition(10, 5);
    
    sf::Text infoText("", font, 14);
    infoText.setFillColor(sf::Color::White);
    infoText.setPosition(10, 30);
    
    bool use_opencl = true;
    std::cout << "Choose simulation type:\n";
    std::cout << "1. Sequential CPU (press 1)\n";
    std::cout << "2. OpenCL GPU/CPU (press 2 or any other key)\n";
    
    char choice;
    std::cin >> choice;
    use_opencl = (choice != '1');
    
    std::vector<Body> cpu_bodies;
    std::unique_ptr<OpenCLNBodySimulation> opencl_sim;
    
    if (use_opencl) {
        try {
            opencl_sim = std::make_unique<OpenCLNBodySimulation>();
            opencl_sim->initialize();
            std::cout << "Using OpenCL simulation with " << n_bodies << " bodies\n";
        } catch (const std::exception& e) {
            std::cerr << "OpenCL initialization failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU simulation\n";
            use_opencl = false;
        }
    }
    
    if (!use_opencl) {
        initialize_bodies(cpu_bodies, WIDTH, HEIGHT);
        std::cout << "Using CPU simulation with " << cpu_bodies.size() << " bodies\n";
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
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_opencl && opencl_sim) {
            opencl_sim->update();
            opencl_sim->read_positions();
        } else {
            compute_forces(cpu_bodies, G, eps);
            integrate_bodies(cpu_bodies, dt);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        window.clear(sf::Color::Black);
        
        if (use_opencl && opencl_sim) {
            const auto& bodies = opencl_sim->get_bodies();
            for (size_t i = 0; i < bodies.size(); ++i) {
                sf::CircleShape circle(bodies.m[i] > 50.0f ? 6 : 2);
                circle.setFillColor(mass_to_color(bodies.m[i]));
                circle.setPosition(WIDTH / 2 + bodies.x[i], HEIGHT / 2 + bodies.y[i]);
                circle.setOrigin(circle.getRadius(), circle.getRadius());
                window.draw(circle);
            }
        } else {
            for (const auto& body : cpu_bodies) {
                sf::CircleShape circle(body.m > 50.0f ? 6 : 2);
                circle.setFillColor(mass_to_color(body.m));
                circle.setPosition(WIDTH / 2 + body.x, HEIGHT / 2 + body.y);
                circle.setOrigin(circle.getRadius(), circle.getRadius());
                window.draw(circle);
            }
        }
        
        float elapsed = fpsClock.restart().asSeconds();
        lastFPS = 1.0f / elapsed;
        fpsText.setString("FPS: " + std::to_string(static_cast<int>(lastFPS)));
        window.draw(fpsText);
        
        std::string info = "Bodies: " + std::to_string(use_opencl ? n_bodies : cpu_bodies.size()) + 
                          " | Mode: " + (use_opencl ? "OpenCL" : "CPU") + 
                          " | Sim Time: " + std::to_string(duration.count()) + "μs";
        infoText.setString(info);
        window.draw(infoText);
        
        window.display();
        
        sf::Time elapsed_time = frameClock.getElapsedTime();
        if (elapsed_time < FRAME_DURATION) {
            sf::sleep(FRAME_DURATION - elapsed_time);
        }
        frameClock.restart();
    }
    
    return 0;
}