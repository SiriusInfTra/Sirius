#pragma once

#include <vector>
#include <string>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <opencv2/videoio.hpp>

#include "utils.h"
#include "runner.h"

class PvaBlurRunner : public VpiRunner
{
public:
    PvaBlurRunner(const std::string &video_path_in,
                  const std::string &video_path_out);
    virtual ~PvaBlurRunner() = default;

    virtual VPIStream CreateStream() override;
    virtual void Init(VPIStream stream) override;
    virtual void Final(VPIStream stream) override;
    virtual void Execute(VPIStream stream,
                         const size_t qlen,
                         const bool sync) override;

private:
    const std::string video_path_in_;
    const std::string video_path_out_;

    int w_;
    int h_;
    double fps_;
    size_t frames_cnt_;
    cv::VideoCapture video_cv_in_;
    cv::VideoWriter video_cv_out_;
    
    std::vector<VPIImage> frames_vpi_converted_;
    std::vector<VPIImage> frames_vpi_blurred_;
};
