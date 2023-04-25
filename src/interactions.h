#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
void lambertianBSDF(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    // �����ȡһ��������ķ��䷽��
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    // ��ֹ���ߵݹ飬��Ҫƫ��һ����
    pathSegment.ray.origin = intersect + EPSILON * normal;
}

__host__ __device__
void specularBSDF(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    // ��ȡ�������
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    // ��ֹ���ߵݹ飬��Ҫƫ��һ����
    pathSegment.ray.origin = intersect + 0.001f * normal;
    // m.specular.color�Ǿ�����ɫ
    pathSegment.color *= m.specular.color;
}
__host__ __device__
float schlick(float cos, float reflectIndex) {
    float r0 = powf((1.f - reflectIndex) / (1.f + reflectIndex), 2.f);
    return r0 + (1.f - r0) * powf((1.f - cos), 5.f);
}
__host__ __device__
void schlickBSDF(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float> u01) {
    glm::vec3 origin_direction = pathSegment.ray.direction;
    // ���isInsideΪ�棬˵�����ߴ������ڲ����
    bool isInside = glm::dot(origin_direction, normal) > 0.f;

    // indexOfRefraction ������
    float eta = isInside ? m.indexOfRefraction : (1.f / m.indexOfRefraction);
    // ����������������Ҫ��������ȡ��������Ϊ������Ĭ���ǳ�������ģ�
    glm::vec3 outwardNormal = isInside ? glm::normalize(-1.0f * normal) : glm::normalize(normal);
    // ����Snell's law��n1sin1=n2sin2��������������
    // para1������ԭ���� para2�������� para3��������
    glm::vec3 direction = glm::refract(glm::normalize(origin_direction), outwardNormal, eta);

    // ����Ƿ����ȫ���䣬���������˵������û��͸�����ʣ����Ǳ�������ȫ�����ȥ��
    if (glm::length(direction) < 0.01f) {
        pathSegment.color *= 0.f;
        direction = glm::reflect(origin_direction, normal);
    }

    // ���ݷ��������ɼ��㷴����ߵĸ��ʣ�Ȼ��sampleFloat�������仹������
    float cos = glm::dot(origin_direction, normal);
    float reflectProb = schlick(cos, m.indexOfRefraction);
    float sampleFloat = u01(rng);

    pathSegment.ray.direction = reflectProb < sampleFloat ? glm::reflect(origin_direction, normal) : direction;
    pathSegment.ray.origin = intersect + EPSILON * pathSegment.ray.direction;
    pathSegment.color *= m.specular.color;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // ���������Ƿ�ɢ�䣬ʹ��0-1�ֲ��������
    glm::vec3 originDirection = pathSegment.ray.direction;
    thrust::uniform_real_distribution<float> u01(0, 1);
    float p = u01(rng);
    if (m.hasReflective == 1.f) {
        // ������
        specularBSDF(pathSegment, intersect, normal, m, rng);
    }
    else if (m.hasRefractive == 1.f) {
        // ������
        schlickBSDF(pathSegment, intersect, normal, m, rng, u01);
    }
    else {
        // ��ɢ�䣬ʹ��Lambertian BSDF
        lambertianBSDF(pathSegment, intersect, normal, m, rng);
    }
    pathSegment.remainingBounces--;
    pathSegment.color *= m.color;
    pathSegment.color = glm::clamp(pathSegment.color, glm::vec3(0.0f), glm::vec3(1.0f));
}