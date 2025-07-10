#include "Body.h"
#include <cmath>
#include <random>
#include <algorithm>

// Constants
constexpr float G = 1.0f;
constexpr float dt = 0.1f;
constexpr float eps = 1e-1f;
constexpr size_t n_bodies = 5;
constexpr float center_mass = 1000.0f;

void compute_forces(std::vector<Body>& bodies, float G, float eps) {
    // Reset accelerations
    for (auto& body : bodies) {
        body.ax = 0.0f;
        body.ay = 0.0f;
    }
    
    // Compute forces between all pairs
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = 0; j < bodies.size(); ++j) {
            if (i == j) continue;  // Skip self-interaction
            
            // Calculate distances
            const float dx = bodies[j].x - bodies[i].x;
            const float dy = bodies[j].y - bodies[i].y;
            
            // Calculate cubed inverse distance with softening
            const float r_squared = dx * dx + dy * dy + eps * eps;
            const float r = std::sqrt(r_squared);
            const float d3_inv = 1.0f / (r * r * r);
            
            // Calculate force magnitude
            const float f = G * bodies[j].m * d3_inv;
            
            // Update acceleration components
            bodies[i].ax += dx * f;
            bodies[i].ay += dy * f;
        }
    }
}

void integrate_bodies(std::vector<Body>& bodies, float dt) {
    for (auto& body : bodies) {
        // Update velocity
        body.vx += body.ax * dt;
        body.vy += body.ay * dt;
        
        // Update position
        body.x += body.vx * dt;
        body.y += body.vy * dt;
    }
}

float orbital_velocity_scalar(float M, float r) {
    return std::sqrt(1.0f * M / r);  // G = 1.0 assumed
}

void initialize_bodies(std::vector<Body>& bodies, int width, int height) {
    bodies.clear();
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dist(50.0f, std::min(width, height) / 2.0f - 20.0f);
    std::uniform_real_distribution<float> mass_dist(0.5f, 10.0f);
    
    // Add central massive body
    bodies.emplace_back(0.0f, 0.0f, 0.0f, 0.0f, center_mass);
    
    // Add orbiting bodies
    for (size_t i = 1; i < n_bodies; ++i) {
        const float angle = angle_dist(rng);
        const float radius = radius_dist(rng);
        const float mass = mass_dist(rng);
        
        const float x = radius * std::cos(angle);
        const float y = radius * std::sin(angle);
        
        // Calculate orbital velocity for stable orbit
        const float v_orbital = orbital_velocity_scalar(center_mass, radius);
        const float vx = -v_orbital * std::sin(angle);
        const float vy = v_orbital * std::cos(angle);
        
        bodies.emplace_back(x, y, vx, vy, mass);
    }
}

void initialize_bodies_soa(BodySOA& bodies, int width, int height, size_t n_bodies) {
    bodies.clear();
    bodies.resize(n_bodies);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dist(50.0f, std::min(width, height) / 2.0f - 20.0f);
    std::uniform_real_distribution<float> mass_dist(0.5f, 10.0f);
    
    // Central massive body
    bodies.x[0] = 0.0f;
    bodies.y[0] = 0.0f;
    bodies.vx[0] = 0.0f;
    bodies.vy[0] = 0.0f;
    bodies.ax[0] = 0.0f;
    bodies.ay[0] = 0.0f;
    bodies.m[0] = center_mass;
    
    // Orbiting bodies
    for (size_t i = 1; i < n_bodies; ++i) {
        const float angle = angle_dist(rng);
        const float radius = radius_dist(rng);
        const float mass = mass_dist(rng);
        
        bodies.x[i] = radius * std::cos(angle);
        bodies.y[i] = radius * std::sin(angle);
        bodies.m[i] = mass;
        
        // Calculate orbital velocity
        const float v_orbital = orbital_velocity_scalar(center_mass, radius);
        bodies.vx[i] = -v_orbital * std::sin(angle);
        bodies.vy[i] = v_orbital * std::cos(angle);
        
        bodies.ax[i] = 0.0f;
        bodies.ay[i] = 0.0f;
    }
}