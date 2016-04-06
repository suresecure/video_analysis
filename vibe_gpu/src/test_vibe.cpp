#include "vibe_gpu.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace vibe_gpu;

int main()
{
    const cv::Size size(640,480);
    const int type = CV_8UC3;
    const bool useRoi = false;

    const cv::Mat fullfg(size, CV_8UC1, cv::Scalar::all(255));

    cv::Mat frame = randomMat(size, type, 0.0, 100);
    cv::gpu::GpuMat d_frame = loadMat(frame, useRoi);

    cv::gpu::VIBE_GPU vibe;
    cv::gpu::GpuMat d_fgmask = createMat(size, CV_8UC1, useRoi);
    vibe.initialize(d_frame);

    for (int i = 0; i < 20; ++i)
        vibe(d_frame, d_fgmask);

    frame = randomMat(size, type, 160, 255);
    d_frame = loadMat(frame, useRoi);
    vibe(d_frame, d_fgmask);

    return 0;
}

