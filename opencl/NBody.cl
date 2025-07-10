__kernel void compute_forces(__global float* x, __global float* y,
                            __global float* ax, __global float* ay,
                            __global const float* m,
                            const int n_bodies,
                            const float G,
                            const float eps) {
    int i = get_global_id(0);
    
    if (i >= n_bodies) return;
    
    // Reset acceleration
    ax[i] = 0.0f;
    ay[i] = 0.0f;
    
    // Compute forces from all other bodies
    for (int j = 0; j < n_bodies; ++j) {
        if (i == j) continue;
        
        // Calculate distances
        float dx = x[j] - x[i];
        float dy = y[j] - y[i];
        
        // Calculate cubed inverse distance with softening
        float r_squared = dx * dx + dy * dy + eps * eps;
        float r = sqrt(r_squared);
        float d3_inv = 1.0f / (r * r * r);
        
        // Calculate force magnitude
        float f = G * m[j] * d3_inv;
        
        // Update acceleration components
        ax[i] += dx * f;
        ay[i] += dy * f;
    }
}

__kernel void integrate_bodies(__global float* x, __global float* y,
                              __global float* vx, __global float* vy,
                              __global const float* ax, __global const float* ay,
                              const int n_bodies,
                              const float dt,
                              const float width, const float height) {
    int i = get_global_id(0);
    
    if (i >= n_bodies) return;
    
    // Update velocity
    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    
    // Update position
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    
    // Toroidal boundary conditions
    float half_width = width / 2.0f;
    float half_height = height / 2.0f;
    
    if (x[i] > half_width) {
        x[i] = -half_width;
    } else if (x[i] < -half_width) {
        x[i] = half_width;
    }
    
    if (y[i] > half_height) {
        y[i] = -half_height;
    } else if (y[i] < -half_height) {
        y[i] = half_height;
    }
}