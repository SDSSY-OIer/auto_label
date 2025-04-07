#include <cuda.h>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void warpaffine_kernel(
    std::uint8_t *src, int src_line_size, int src_width, int src_height,
    float *dst, int dst_width, int dst_height,
    std::uint8_t const_value_st, float *d2i, int edge);

void preprocess_kernel_img(
    std::uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height,
    float *d2i, cudaStream_t stream)
{
    int jobs = dst_height * dst_width;
    // int threads = 256;
    // int blocks = std::ceil(jobs / (float)threads);
    dim3 dimblock(256, 1, 1);
    dim3 dimgird((jobs + dimblock.x - 1) / dimblock.x, 1, 1);

    warpaffine_kernel<<<dimgird, dimblock, 0, stream>>>(
        src, src_width * 3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2i, jobs);
}

__global__ void warpaffine_kernel(
    std::uint8_t *src, int src_line_size, int src_width, int src_height,
    float *dst, int dst_width, int dst_height,
    std::uint8_t const_value_st, float *d2i, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge)
        return;

    float m_x1 = d2i[0];
    float m_y1 = d2i[1];
    float m_z1 = d2i[2];
    float m_x2 = d2i[3];
    float m_y2 = d2i[4];
    float m_z2 = d2i[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
    {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else
    {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        std::uint8_t const_value[]{const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        std::uint8_t *v1 = const_value;
        std::uint8_t *v2 = const_value;
        std::uint8_t *v3 = const_value;
        std::uint8_t *v4 = const_value;

        if (y_low >= 0)
        {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height)
        {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // bgr to rgb
    std::swap(c0, c2);

    // normalization
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dy * dst_width + dx;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}