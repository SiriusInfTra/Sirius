#include <opencv2/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/OpticalFlowDense.h>
#include <vpi/algo/ConvertImageFormat.h>

#include "dof.h"
#include "utils/log.h"
#include "utils/xassert.h"

OfaDofRunner::OfaDofRunner(const std::string &video_path_in,
                           const std::string &video_path_out)
    : video_path_in_(video_path_in), video_path_out_(video_path_out)
{
    XASSERT(video_cv_in_.open(video_path_in_),
            "Cannot open input video '%s'", video_path_in_.c_str());
    
    w_ = video_cv_in_.get(cv::CAP_PROP_FRAME_WIDTH);
    h_ = video_cv_in_.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps_ = video_cv_in_.get(cv::CAP_PROP_FPS);

    video_cv_out_ = cv::VideoWriter(video_path_out_,
                                    cv::VideoWriter::fourcc('M','J','P','G'),
                                    fps_, cv::Size(w_, h_), false);
    XASSERT(video_cv_out_.isOpened(),
            "Cannot create output video '%s'", video_path_out_.c_str());
}

VPIStream OfaDofRunner::CreateStream()
{
    VPIStream stream = nullptr;
    VPI_ASSERT(vpiStreamCreate(VPI_BACKEND_CUDA | VPI_BACKEND_VIC |
                               VPI_BACKEND_OFA, &stream));
    return stream;
}

void OfaDofRunner::Init(VPIStream stream)
{
    cv::Mat frame_cv_in;
    VPIImage frame_vpi_in = nullptr;

    while (video_cv_in_.read(frame_cv_in)) {
        cv::resize(frame_cv_in, frame_cv_in, cv::Size(w_, h_));
        if (frame_vpi_in == nullptr) {
            // Create a VPIImage that wraps the frame
            VPI_ASSERT(vpiImageCreateWrapperOpenCVMat(frame_cv_in, 0,
                                                      &frame_vpi_in));
        } else {
            // reuse existing VPIImage wrapper to wrap the new frame.
            VPI_ASSERT(vpiImageSetWrappedOpenCVMat(frame_vpi_in,
                                                   frame_cv_in));
        }

        VPIImage frame_vpi_y8;
        VPIImage frame_vpi_y8bl;
        VPI_ASSERT(vpiImageCreate(w_, h_,
                                  VPI_IMAGE_FORMAT_Y8_ER,
                                  VPI_EXCLUSIVE_STREAM_ACCESS,
                                  &frame_vpi_y8));
        VPI_ASSERT(vpiImageCreate(w_, h_,
                                  VPI_IMAGE_FORMAT_Y8_ER_BL,
                                  VPI_EXCLUSIVE_STREAM_ACCESS,
                                  &frame_vpi_y8bl));
        VPI_ASSERT(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA,
                                               frame_vpi_in,
                                               frame_vpi_y8,
                                               nullptr));
        VPI_ASSERT(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC,
                                               frame_vpi_y8,
                                               frame_vpi_y8bl,
                                               nullptr));
        VPI_ASSERT(vpiStreamSync(stream));
        vpiImageDestroy(frame_vpi_y8);
        frames_vpi_converted_.emplace_back(frame_vpi_y8bl);
        
        VPIImage frame_vpi_mv;
        VPI_ASSERT(vpiImageCreate(w_ / 4, h_ / 4,
                                  VPI_IMAGE_FORMAT_2S16_BL,
                                  VPI_EXCLUSIVE_STREAM_ACCESS,
                                  &frame_vpi_mv));
        frames_vpi_mv_.emplace_back(frame_vpi_mv);
    }

    XASSERT(frames_vpi_converted_.size() == frames_vpi_mv_.size(),
            "frame cnt of converted (%ld) and move (%ld) mismatch",
            frames_vpi_converted_.size(), frames_vpi_mv_.size());
    frames_cnt_ = frames_vpi_converted_.size();

    VPI_ASSERT(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, w_, h_,
                                         VPI_IMAGE_FORMAT_Y8_ER_BL,
                                         &grid_size_, 1,
                                         VPI_OPTICAL_FLOW_QUALITY_LOW,
                                         &payload_));
}

void OfaDofRunner::Final(VPIStream)
{
    for (auto frame : frames_vpi_converted_) {
        vpiImageDestroy(frame);
    }
    for (auto frame : frames_vpi_mv_) {
        vpiImageDestroy(frame);
    }
}

void OfaDofRunner::Execute(VPIStream stream,
                           const size_t qlen,
                           const bool sync)
{
    // FIXME: err msg:
    // VPI_ERROR_BUFFER_LOCKED: Can't use container with
    // VPI_EXCLUSIVE_STREAM_ACCESS flag set in multiple streams
    size_t buffered = 0;
    for (size_t i = 1; i < frames_cnt_; ++i) {
        buffered += 1;
        VPI_ASSERT(vpiSubmitOpticalFlowDense(stream, VPI_BACKEND_OFA,
                                             payload_,
                                             frames_vpi_converted_[i - 1],
                                             frames_vpi_converted_[i],
                                             frames_vpi_mv_[i - 1]));

        if (sync || (buffered >= qlen)) {
            VPI_ASSERT(vpiStreamSync(stream));
            buffered = 0;
        }
    }

    if (!sync) VPI_ASSERT(vpiStreamSync(stream));
}
