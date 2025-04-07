#include <cuda.h>
#include <algorithm>
#include <cuda_runtime.h>

static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy);

static __global__ void decode_kernel(
    float *predict, int NUM_BOX_ELEMENT, int num_bboxes, int num_classes,
    int ckpt, float confidence_threshold, float *invert_affine_matrix,
    float *parray, int max_objects);

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom);

static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT);

void decode_kernel_invoker(
    float *predict, int NUM_BOX_ELEMENT, int num_bboxes, int num_classes,
    int ckpt, float confidence_threshold, float *invert_affine_matrix,
    float *parray, int max_objects, cudaStream_t stream)
{
    // int block = 256;
    // int grid =  ceil(num_bboxes / (float)block);
    dim3 dimblock(256, 1, 1);
    dim3 dimgird((num_bboxes + dimblock.x - 1) / dimblock.x, 1, 1);

    decode_kernel<<<dimgird, dimblock, 0, stream>>>(
        predict, NUM_BOX_ELEMENT, num_bboxes, num_classes,
        ckpt, confidence_threshold, invert_affine_matrix,
        parray, max_objects);
}

void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, cudaStream_t stream, int NUM_BOX_ELEMENT)
{
    int block = std::min(256, max_objects);
    int grid = std::ceil(1.0 * max_objects / block);
    // int grid = 32;
    nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}

static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(
    float *predict, int NUM_BOX_ELEMENT, int num_bboxes, int num_classes,
    int ckpt, float confidence_threshold, float *invert_affine_matrix,
    float *parray, int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float *pitem = predict + (5 + num_classes + ckpt * 2) * position;
    // v7: cx, cy, w, h, conf, x1, y1, conf1, x2, y2, conf2, x3, y3, conf3, x4, y4, conf4, conf_c1...conf_c36
    // v5: cx, cy, w, h, conf, x1, y1, x2, y2, x3, y3, x4, y4, conf_c1...conf_c36
    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    float *class_confidence = pitem + 5 + ckpt * 2;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label = i;
        }
    }

    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float cx = pitem[0];
    float cy = pitem[1];
    float width = pitem[2];
    float height = pitem[3];

    // 4个关键点
    float *landmarks = pitem + 5;
    float x1 = landmarks[0];
    float y1 = landmarks[1];
    float x2 = landmarks[2];
    float y2 = landmarks[3];
    float x3 = landmarks[4];
    float y3 = landmarks[5];
    float x4 = landmarks[6];
    float y4 = landmarks[7];

    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    affine_project(invert_affine_matrix, x1, y1, &x1, &y1);
    affine_project(invert_affine_matrix, x2, y2, &x2, &y2);
    affine_project(invert_affine_matrix, x3, y3, &x3, &y3);
    affine_project(invert_affine_matrix, x4, y4, &x4, &y4);

    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore

    *pout_item++ = x1;
    *pout_item++ = y1;

    *pout_item++ = x2;
    *pout_item++ = y2;

    *pout_item++ = x3;
    *pout_item++ = y3;

    *pout_item++ = x4;
    *pout_item++ = y4;
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom)
{

    float cleft = std::max(aleft, bleft);
    float ctop = std::max(atop, btop);
    float cright = std::min(aright, bright);
    float cbottom = std::min(abottom, bbottom);

    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = std::max(0.0f, aright - aleft) * std::max(0.0f, abottom - atop);
    float b_area = std::max(0.0f, bright - bleft) * std::max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT)
{

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = std::min((int)*bboxes, max_objects);
    if (position >= count)
        return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i)
    {
        float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5])
            continue;

        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold)
            {
                pcurrent[6] = 0; // 1=keep, 0=ignore
                return;
            }
        }
    }
}