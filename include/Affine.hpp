#ifndef AFFINE_HPP
#define AFFINE_HPP

#include <cstring>
#include <algorithm>
#include <opencv2/opencv.hpp>

struct AffineMatrix
{
    // 仿射变换逆变换
    float d2i[6];
};

/**
 * @brief 计算仿射变换的矩阵和逆矩阵
 * @todo 尝试迁移至GPU上计算
 */
void getd2i(AffineMatrix &afmt, cv::Size to, cv::Size from)
{
    float scale = std::min(1.0 * to.width / from.width, 1.0 * to.height / from.height);
    float i2d[6]{scale, 0, -scale * from.width * 0.5 + to.width * 0.5, 0, scale, -scale * from.height * 0.5 + to.height * 0.5};
    cv::Mat i2d_mat(2, 3, CV_32F, i2d), d2i_mat;
    cv::invertAffineTransform(i2d_mat, d2i_mat);
    std::memcpy(afmt.d2i, d2i_mat.ptr<float>(0), sizeof(afmt.d2i));
}

#endif