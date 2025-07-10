#pragma once

#include <vector>
#include <memory>

// Body class for scalar implementation
class Body {
public:
    float x, y;           // position
    float vx, vy;         // velocity
    float ax, ay;         // acceleration
    float m;              // mass

    Body(float x = 0.0f, float y = 0.0f, float vx = 0.0f, float vy = 0.0f, float m = 1.0f)
        : x(x), y(y), vx(vx), vy(vy), ax(0.0f), ay(0.0f), m(m) {}
};

// SOA (Structure of Arrays) for parallel implementation
struct BodySOA {
    std::vector<float> x, y;      // positions
    std::vector<float> vx, vy;    // velocities
    std::vector<float> ax, ay;    // accelerations
    std::vector<float> m;         // masses
    
    void resize(size_t n) {
        x.resize(n); y.resize(n);
        vx.resize(n); vy.resize(n);
        ax.resize(n); ay.resize(n);
        m.resize(n);
    }
    
    size_t size() const { return x.size(); }
    
    void clear() {
        x.clear(); y.clear();
        vx.clear(); vy.clear();
        ax.clear(); ay.clear();
        m.clear();
    }
};

// Function declarations
void compute_forces(std::vector<Body>& bodies, float G, float eps);
void integrate_bodies(std::vector<Body>& bodies, float dt);
void initialize_bodies(std::vector<Body>& bodies, int width, int height);
void initialize_bodies_soa(BodySOA& bodies, int width, int height, size_t n_bodies);
float orbital_velocity_scalar(float M, float r);