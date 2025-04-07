#ifndef SOLVE_LIGHT_HPP
#define SOLVE_LIGHT_HPP

#include <numeric>
#include <utility>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 灯条类
class LightBar
{
public:
    float area;
    float tilt_angle;
    float width, length;
    cv::Point2f top, bottom;
    LightBar(const cv::RotatedRect &box)
    {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f &lhs, const cv::Point2f &rhs)
                  { return lhs.y < rhs.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;
        width = cv::norm(p[0] - p[1]);
        length = cv::norm(top - bottom);
        area = box.size.width * box.size.height;
        tilt_angle = std::atan2(bottom.y - top.y, std::abs(top.x - bottom.x));
    }
};

// 检测是否为灯条
bool is_light(const LightBar &light)
{
    return light.length >= 2 * light.width && light.length <= 20 * light.width && light.tilt_angle > std::numbers::pi / 4;
}

std::pair<cv::Point2f, cv::Point2f> find_light(const cv::Mat &src, cv::Point2f top_point, cv::Point2f bottom_point, float delta)
{
    // 扩大roi
    int top = std::max(0.0f, top_point.y - delta);
    int bottom = std::min(src.rows - 1.0f, bottom_point.y + delta);
    int left = std::max(0.0f, std::min(top_point.x, bottom_point.x) - delta);
    int right = std::min(src.cols - 1.0f, std::max(top_point.x, bottom_point.x) + delta);
    cv::Mat roi(src(cv::Range(top, bottom), cv::Range(left, right)));

    cv::Mat gray, binary;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 50, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary, binary, kernel);

    float max_area = 0.0f;
    cv::Point2f tl(left, top);
    std::vector<std::vector<cv::Point>> contours;
    std::pair<cv::Point2f, cv::Point2f> res(top_point, bottom_point);
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours)
    {
        LightBar light(cv::minAreaRect(contour));
        if (light.area > max_area && is_light(light))
        {
            max_area = light.area;
            res = {light.top + tl, light.bottom + tl};
        }
    }
    return res;
}

// 解决装甲板灯条的精确定位问题
void solve_light(const cv::Mat &src, std::vector<cv::Point2f> &points)
{
    float left = std::min(points[0].x, points[1].x);
    float right = std::max(points[2].x, points[3].x);
    float delta = (right - left) / 4;
    auto [tl, bl] = find_light(src, points[0], points[1], delta);
    auto [tr, br] = find_light(src, points[3], points[2], delta);
    points = {tl, bl, br, tr};
}

#endif