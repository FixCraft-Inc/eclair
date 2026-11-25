/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/formats/host_device.hpp"

#include <cmath>
#include <cstdint>

namespace eclair::render {

struct Vec3 {
    float x = 0;
    float y = 0;
    float z = 0;
};

ECLAIR_HOST_DEVICE inline Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
ECLAIR_HOST_DEVICE inline Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
ECLAIR_HOST_DEVICE inline Vec3 operator-(Vec3 a) {
    return {-a.x, -a.y, -a.z};
}
ECLAIR_HOST_DEVICE inline Vec3 operator*(Vec3 a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}
ECLAIR_HOST_DEVICE inline Vec3 operator*(Vec3 a, Vec3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
ECLAIR_HOST_DEVICE inline Vec3 operator/(Vec3 a, float b) {
    return a * (1.0F / b);
}
ECLAIR_HOST_DEVICE inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
ECLAIR_HOST_DEVICE inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
ECLAIR_HOST_DEVICE inline Vec3 normalized(Vec3 a) {
    const float length2 = dot(a, a);
    return length2 > 1.0e-30F ? a / ::sqrtf(length2) : Vec3{};
}
ECLAIR_HOST_DEVICE inline float component(Vec3 v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}
ECLAIR_HOST_DEVICE inline Vec3 axis_vector(int axis, float sign) {
    return {axis == 0 ? sign : 0, axis == 1 ? sign : 0, axis == 2 ? sign : 0};
}
ECLAIR_HOST_DEVICE inline bool finite(float x) {
    return x >= -3.402823466e38F && x <= 3.402823466e38F;
}
ECLAIR_HOST_DEVICE inline bool finite(Vec3 v) {
    return finite(v.x) && finite(v.y) && finite(v.z);
}

}  // namespace eclair::render
