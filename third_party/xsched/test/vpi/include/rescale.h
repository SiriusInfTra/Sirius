#pragma once

#include <vector>
#include <string>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <opencv2/videoio.hpp>

#include "utils.h"
#include "runner.h"

class VicRescaleRunner : public VpiRunner
{
public:
    VicRescaleRunner(const std::string &video_path_in,
                     const std::string &video_path_out,
                     int out_w, int out_h);
    virtual ~VicRescaleRunner() = default;

    virtual VPIStream CreateStream() override;
    virtual void Init(VPIStream stream) override;
    virtual void Final(VPIStream stream) override;
    virtual void Execute(VPIStream stream,
                         const size_t qlen,
                         const bool sync) override;

private:
    const std::string video_path_in_;
    const std::string video_path_out_;
    const int w_out_;
    const int h_out_;

    int w_in_;
    int h_in_;
    double fps_in_;
    size_t frames_cnt_;
    cv::VideoCapture video_cv_in_;
    cv::VideoWriter video_cv_out_;
    
    std::vector<VPIImage> frames_vpi_converted_;
    std::vector<VPIImage> frames_vpi_rescaled_;
};
