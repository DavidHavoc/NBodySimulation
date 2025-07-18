#ifndef BODY_H
#define BODY_H

#include <vector>

struct BodiesSOA {
    std::vector<float> x, y;         // position
    std::vector<float> vx, vy;       // velocity
    std::vector<float> ax, ay;       // acceleration
    std::vector<float> m;            // mass
    size_t count;

    BodiesSOA(size_t num_bodies) : count(num_bodies) {
        x.resize(count);
        y.resize(count);
        vx.resize(count);
        vy.resize(count);
        ax.resize(count);
        ay.resize(count);
        m.resize(count);
    }
};

#endif // BODY_H


