__kernel void compute_forces(__global float* x, __global float* y, __global float* m, 
                             __global float* ax, __global float* ay, 
                             int num_bodies, float G, float eps) {
    int i = get_global_id(0);

    if (i < num_bodies) {
        float current_ax = 0.0f;
        float current_ay = 0.0f;

        for (int j = 0; j < num_bodies; ++j) {
            if (i == j) continue;

            float dx = x[j] - x[i];
            float dy = y[j] - y[i];

            float dist_sq = dx * dx + dy * dy;
            float dist = sqrt(dist_sq);
            float d3_inv = 1.0f / (dist_sq * dist + eps * eps * eps);

            float f = G * m[j] * d3_inv;

            current_ax += dx * f;
            current_ay += dy * f;
        }
        ax[i] = current_ax;
        ay[i] = current_ay;
    }
}

__kernel void integrate_bodies(__global float* x, __global float* y, 
                               __global float* vx, __global float* vy, 
                               __global float* ax, __global float* ay, 
                               int num_bodies, float dt, int width, int height) {
    int i = get_global_id(0);

    if (i < num_bodies) {
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;

        // Toroidal universe
        if (x[i] > width / 2) x[i] -= width;
        if (x[i] < -width / 2) x[i] += width;
        if (y[i] > height / 2) y[i] -= height;
        if (y[i] < -height / 2) y[i] += height;
    }
}


