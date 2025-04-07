#ifndef ARMOR_HPP
#define ARMOR_HPP

#include <cmath>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 装甲板类型（confirm in detector.hpp，pnp解算的时候用）
enum ArmorType
{
    SMALL,
    LARGE
};

class Armor
{
public:
    // 装甲板四顶点，opencv像素坐标下的左上，左下，右下，右上
    std::vector<cv::Point2f> armorVertices_vector;
    cv::Point2f center;
    double area;
    ArmorType type;
    int number;
    int color;

public:
    Armor(const std::vector<cv::Point2f> &points)
        : armorVertices_vector(points),
          center((points[0] + points[2]) / 2),
          area(std::abs((points[0].x - points[2].x) * (points[0].y - points[2].y)))
    {
    }
};

#endif