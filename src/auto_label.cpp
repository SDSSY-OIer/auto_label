#include "Detector.hpp"
#include "solve_light.hpp"
#include <filesystem>

namespace fs = std::filesystem;

std::unique_ptr<Detector> detector;

const cv::Scalar color_list[]{
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 255)};

std::unique_ptr<Detector> init_detector()
{
    // 初始化参数
    int num_classes = 36;
    std::string target_color = "ANY";
    float nms_thresh = 0.45;
    float bbox_conf_thresh = 0.5;
    int input_w = 448, input_h = 448;
    std::string engine_file_path = "/home/nvidia/mosas_model/RM4points_v3_5_sim_448x448.trt";

    // 装甲板限定阈值
    Detector::ArmorParams a_params;
    a_params.min_large_center_distance = 3.2;

    // 初始化识别器
    return std::make_unique<Detector>(num_classes, target_color, nms_thresh, bbox_conf_thresh, input_w, input_h, engine_file_path, a_params);
}

void process_file(fs::path path)
{
    cv::Mat src(cv::imread(path.string()));
    auto armors = detector->detect(src);
    for (auto &armor : armors)
        solve_light(src, armor.armorVertices_vector);
    path.replace_extension(".txt");
    std::ofstream ofs(path);
    for (const auto &armor : armors)
    {
        for (const auto &point : armor.armorVertices_vector)
        {
            ofs << point.x / src.cols << ' ' << point.y / src.rows << ' ';
        }
        ofs << armor.number << ' ' << armor.color << std::endl;
    }
    ofs.close();
}

void process_files(const fs::path &folder_path)
{
    // 递归检查子文件夹
    for (const auto &jpg_path : fs::recursive_directory_iterator(folder_path))
    {
        if (jpg_path.is_regular_file() && jpg_path.path().extension() == ".jpg")
        {
            process_file(jpg_path);
        }
    }
}

int main()
{
    detector = init_detector();
    process_files(fs::path("./img"));
}