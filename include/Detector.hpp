#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <cstddef>
#include <fstream>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "Armor.hpp"
#include "logging.h"
#include "Affine.hpp"
#include "CudaUtils.cuh"

using namespace nvinfer1;

// 模型参数
constexpr static int DEVICE = 0;
constexpr static int NUM_CLASSES = 36; // 类别数量
constexpr static int CKPT_NUM = 4;     // 关键点数量
constexpr static int NUM_BOX_ELEMENT = 7 + CKPT_NUM * 2;
constexpr static char *INPUT_BLOB_NAME = "input";               // 模型导出ONNX文件时设置的输入名字
constexpr static char *OUTPUT_BLOB_NAME = "output";             // 模型导出ONNX文件时设置的输出名字
constexpr static int MAX_IMAGE_INPUT_SIZE_THRESH = 5000 * 5000; // 图像输入尺寸上限
constexpr static int MAX_OBJECTS = 32;

class Detector
{
public:
    struct ArmorParams
    {
        // 两个灯条的最小长度比
        double min_large_center_distance;
    };
    int NUM_CLASSES;
    std::string TARGET_COLOUR;
    float NMS_THRESH;
    float BBOX_CONF_THRESH;
    int INPUT_W; // 目标尺寸
    int INPUT_H;
    std::string engine_file_path;
    ArmorParams a;

private:
    // 创建引擎
    IRuntime *runtime_det;
    ICudaEngine *engine_det;
    IExecutionContext *context_det;
    // CUDA与TRT相关
    Logger gLogger;
    cudaStream_t stream;
    float *buffers[2];
    int inputIndex;
    int outputIndex;
    std::uint8_t *img_host = nullptr;
    std::uint8_t *img_device = nullptr;
    float *affine_matrix_d2i_host = nullptr;
    float *affine_matrix_d2i_device = nullptr;
    float *decode_ptr_device = nullptr;
    float *decode_ptr_host = new float[1 + MAX_OBJECTS * NUM_BOX_ELEMENT];
    int OUTPUT_CANDIDATES;

public:
    Detector(int NUM_CLASSES,
             const std::string &TARGET_COLOUR,
             float NMS_THRESH,
             float BBOX_CONF_THRESH,
             int INPUT_W, int INPUT_H,
             const std::string &engine_file_path,
             const ArmorParams &a);
    void InitModelEngine();
    void AllocMem();
    std::vector<Armor> detect(const cv::Mat &frame);
    void Release();
    ~Detector();
};

Detector::Detector(int NUM_CLASSES, const std::string &TARGET_COLOUR, float NMS_THRESH, float BBOX_CONF_THRESH,
                   int INPUT_W, int INPUT_H, const std::string &engine_file_path, const ArmorParams &a)
    : NUM_CLASSES(NUM_CLASSES), TARGET_COLOUR(TARGET_COLOUR), NMS_THRESH(NMS_THRESH), BBOX_CONF_THRESH(BBOX_CONF_THRESH),
      INPUT_W(INPUT_W), INPUT_H(INPUT_H), engine_file_path(engine_file_path), a(a)
{
    Detector::InitModelEngine();
    Detector::AllocMem();
}

void Detector::InitModelEngine()
{
    cudaSetDevice(DEVICE);
    std::ifstream file{this->engine_file_path, std::ios::binary};
    std::size_t size = 0;
    char *trtModelStreamDet = nullptr;
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size];
        assert(trtModelStreamDet);
        file.read(trtModelStreamDet, size);
        file.close();
    }
    runtime_det = createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
    assert(engine_det != nullptr);
    this->context_det = engine_det->createExecutionContext();
    assert(context_det != nullptr);
    delete[] trtModelStreamDet;
    trtModelStreamDet = nullptr;
}

void Detector::AllocMem()
{
    inputIndex = engine_det->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine_det->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    auto out_dims = engine_det->getBindingDimensions(1);
    auto output_size = 1;
    OUTPUT_CANDIDATES = out_dims.d[1];
    for (int i = 0; i < out_dims.nbDims; ++i)
    {
        output_size *= out_dims.d[i];
    }

    // 尝试优化:
    CHECK(cudaMalloc(&buffers[inputIndex], sizeof(float) * (3 * INPUT_H * INPUT_W)));
    CHECK(cudaMalloc(&buffers[outputIndex], sizeof(float) * output_size));
    // CHECK(cudaMallocManaged(&buffers[inputIndex], sizeof(float) * (3 * INPUT_H * INPUT_W)));
    // CHECK(cudaMallocManaged(&buffers[outputIndex], sizeof(float) * output_size));

    CHECK(cudaMallocHost(&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CHECK(cudaMalloc(&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    //---------------------------------------------------------------------------
    // CHECK(cudaMallocManaged(&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3, cudaMemAttachHost));
    // CHECK(cudaMallocManaged(&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    CHECK(cudaMallocHost(&affine_matrix_d2i_host, sizeof(float) * 6));
    CHECK(cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6));
    //---------------------------------------------------------------------------
    // CHECK(cudaMallocManaged(&affine_matrix_d2i_device, sizeof(float) * 6));

    CHECK(cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
    // CHECK(cudaMallocManaged(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
}

std::vector<Armor> Detector::detect(const cv::Mat &frame)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 预处理
    AffineMatrix afmt;
    // CHECK(cudaMallocManaged(&(afmt.d2i), sizeof(float) * 6, cudaMemAttachHost));
    getd2i(afmt, cv::Size(INPUT_W, INPUT_H), cv::Size(frame.cols, frame.rows)); // TODO
    float *buffer_idx = (float *)buffers[inputIndex];
    std::size_t img_size = frame.cols * frame.rows * 3;

    std::memcpy(affine_matrix_d2i_host, afmt.d2i, sizeof(afmt.d2i));
    CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, affine_matrix_d2i_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));
    // CHECK(cudaStreamAttachMemAsync(stream, afmt.d2i, 0, cudaMemAttachGlobal));
    // CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, afmt.d2i, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));

    std::memcpy(img_host, frame.data, img_size);
    CHECK(cudaMemcpyAsync(img_device, img_host, img_size, cudaMemcpyHostToDevice, stream));
    /*         CHECK(cudaMallocManaged(&(this->frame.data), img_size, cudaMemAttachHost));
    CHECK(cudaStreamAttachMemAsync(stream, (this->frame.data), 0, cudaMemAttachGlobal)); */
    // CHECK(cudaMemcpyAsync(img_device, this->frame.data, img_size, cudaMemcpyHostToDevice, stream));
    preprocess_kernel_img(img_device, frame.cols, frame.rows,
                          buffer_idx, INPUT_W, INPUT_H,
                          affine_matrix_d2i_device, stream);
    // 推理
    context_det->enqueueV2((void **)buffers, stream, nullptr);
    float *predict = (float *)buffers[outputIndex];
    // 后处理
    CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream));
    decode_kernel_invoker(
        predict, NUM_BOX_ELEMENT, OUTPUT_CANDIDATES, NUM_CLASSES,
        CKPT_NUM, BBOX_CONF_THRESH, affine_matrix_d2i_device,
        decode_ptr_device, MAX_OBJECTS, stream);
    nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, stream, NUM_BOX_ELEMENT);
    CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaStreamAttachMemAsync(stream, decode_ptr_device, 0, cudaMemAttachHost));
    cudaStreamSynchronize(stream);

    // 存储结果
    std::vector<bbox> boxes;
    std::vector<Armor> armors;
    int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);
    if (!count)
        return {};
    for (int i = 0; i < count; ++i)
    {
        int basic_pos = 1 + i * NUM_BOX_ELEMENT;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1)
        {
            bbox box;
            int landmark_pos = basic_pos + 7;
            box.class_id = decode_ptr_host[basic_pos + 5];
            for (int id = 0; id < CKPT_NUM; id += 1)
            {
                box.landmarks[id << 1] = decode_ptr_host[landmark_pos + (id << 1)];
                box.landmarks[id << 1 | 1] = decode_ptr_host[landmark_pos + (id << 1 | 1)];
            }
            boxes.emplace_back(box);
        }
    }
    for (auto box : boxes)
    {
        // 左下 左上 右上 右下
        std::vector<cv::Point2f> points{cv::Point2f(box.landmarks[0], box.landmarks[1]),
                                        cv::Point2f(box.landmarks[2], box.landmarks[3]),
                                        cv::Point2f(box.landmarks[4], box.landmarks[5]),
                                        cv::Point2f(box.landmarks[6], box.landmarks[7])};
        Armor armor(points);
        float light_left_length = std::abs(points[0].y - points[1].y);
        float light_right_length = std::abs(points[2].y - points[3].y);
        float avg_light_length = (light_left_length + light_right_length) / 2;
        cv::Point2f light_left_center = (points[0] + points[1]) / 2;
        cv::Point2f light_right_center = (points[2] + points[3]) / 2;
        float center_distance = cv::norm(light_left_center - light_right_center) / avg_light_length;
        if ((this->TARGET_COLOUR == "ANY" || this->TARGET_COLOUR == "BLUE") && box.class_id >= 0 && box.class_id <= 8)
        {
            armor.type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
            armor.number = box.class_id;
            armor.color = 0;
            armors.emplace_back(armor);
        }
        else if ((this->TARGET_COLOUR == "ANY" || this->TARGET_COLOUR == "RED") && box.class_id >= 9 && box.class_id <= 17)
        {
            armor.type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
            armor.number = box.class_id - 9;
            armor.color = 1;
            armors.emplace_back(armor);
        }
    }
    cudaStreamDestroy(stream);
    return armors;
}

void Detector::Release()
{
    context_det->destroy();
    engine_det->destroy();
    runtime_det->destroy();
    cudaStreamDestroy(stream);
    CHECK(cudaFree(affine_matrix_d2i_device));
    CHECK(cudaFreeHost(affine_matrix_d2i_host));
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    decode_ptr_host = nullptr;
}

Detector::~Detector() { Detector::Release(); }

#endif