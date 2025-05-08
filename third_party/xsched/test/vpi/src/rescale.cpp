
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/ConvertImageFormat.h>

#include "rescale.h"
#include "utils/log.h"
#include "utils/xassert.h"

VicRescaleRunner::VicRescaleRunner(const std::string &video_path_in,
                                   const std::string &video_path_out,
                                   int out_w, int out_h)
    : video_path_in_(video_path_in), video_path_out_(video_path_out)
    , w_out_(out_w), h_out_(out_h)
{
    XASSERT(video_cv_in_.open(video_path_in_),
            "Cannot open input video '%s'", video_path_in_.c_str());
    
    w_in_ = video_cv_in_.get(cv::CAP_PROP_FRAME_WIDTH);
    h_in_ = video_cv_in_.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps_in_ = video_cv_in_.get(cv::CAP_PROP_FPS);

    video_cv_out_ = cv::VideoWriter(video_path_out_,
                                    cv::VideoWriter::fourcc('M','J','P','G'),
                                    fps_in_, cv::Size(w_out_, h_out_));
    XASSERT(video_cv_out_.isOpened(),
            "Cannot create output video '%s'", video_path_out_.c_str());
}

VPIStream VicRescaleRunner::CreateStream()
{
    VPIStream stream = nullptr;
    VPI_ASSERT(vpiStreamCreate(VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &stream));
    return stream;
}

void VicRescaleRunner::Init(VPIStream stream)
{
    cv::Mat frame_cv_in;
    VPIImage frame_vpi_in = nullptr;

    while (video_cv_in_.read(frame_cv_in)) {
        if (frame_vpi_in == nullptr) {
            // Create a VPIImage that wraps the frame
            VPI_ASSERT(vpiImageCreateWrapperOpenCVMat(frame_cv_in, 0,
                                                      &frame_vpi_in));
        } else {
            // reuse existing VPIImage wrapper to wrap the new frame.
            VPI_ASSERT(vpiImageSetWrappedOpenCVMat(frame_vpi_in,
                                                   frame_cv_in));
        }

        VPIImage frame_vpi_converted;
        VPI_ASSERT(vpiImageCreate(w_in_, h_in_,
                                  VPI_IMAGE_FORMAT_NV12_ER,
                                  VPI_EXCLUSIVE_STREAM_ACCESS,
                                  &frame_vpi_converted));
        VPI_ASSERT(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA,
                                               frame_vpi_in,
                                               frame_vpi_converted,
                                               nullptr));
        VPI_ASSERT(vpiStreamSync(stream));
        frames_vpi_converted_.emplace_back(frame_vpi_converted);
        
        VPIImage frame_vpi_rescaled;
        VPI_ASSERT(vpiImageCreate(w_out_, h_out_,
                                  VPI_IMAGE_FORMAT_NV12_ER,
                                  VPI_EXCLUSIVE_STREAM_ACCESS,
                                  &frame_vpi_rescaled));
        frames_vpi_rescaled_.emplace_back(frame_vpi_rescaled);
    }

    XASSERT(frames_vpi_converted_.size() == frames_vpi_rescaled_.size(),
            "frame cnt of converted (%ld) and rescaled (%ld) mismatch",
            frames_vpi_converted_.size(), frames_vpi_rescaled_.size());
    frames_cnt_ = frames_vpi_converted_.size();
}

void VicRescaleRunner::Final(VPIStream stream)
{
    for (size_t i = 0; i < frames_cnt_; ++i) {
        VPIImage frame_vpi_out;
        VPIImageData data_vpi_out;

        VPI_ASSERT(vpiImageCreate(w_out_, h_out_,
                                  VPI_IMAGE_FORMAT_BGR8,
                                  0, &frame_vpi_out));
        VPI_ASSERT(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA,
                                               frames_vpi_rescaled_[i],
                                               frame_vpi_out,
                                               nullptr));
        VPI_ASSERT(vpiStreamSync(stream));

        VPI_ASSERT(vpiImageLockData(frame_vpi_out, VPI_LOCK_READ,
                                    VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                    &data_vpi_out));

        // Returned data consists of host-accessible memory buffers
        // in pitch-linear layout.
        XASSERT(data_vpi_out.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                "Unexpected buffer type: %d", data_vpi_out.bufferType);

        VPIImageBufferPitchLinear &pitch_vpi_out = data_vpi_out.buffer.pitch;

        cv::Mat frame_cv_out(pitch_vpi_out.planes[0].height,
                             pitch_vpi_out.planes[0].width,
                             CV_8UC3,
                             pitch_vpi_out.planes[0].data,
                             pitch_vpi_out.planes[0].pitchBytes);

        // Done handling output image, don't forget to unlock it.
        VPI_ASSERT(vpiImageUnlock(frame_vpi_out));

        video_cv_out_ << frame_cv_out;
    }

    for (auto frame : frames_vpi_converted_) {
        vpiImageDestroy(frame);
    }
    for (auto frame : frames_vpi_rescaled_) {
        vpiImageDestroy(frame);
    }
}

void VicRescaleRunner::Execute(VPIStream stream,
                               const size_t qlen,
                               const bool sync) 
{
    size_t buffered = 0;
    for (size_t i = 0; i < frames_cnt_; ++i) {
        buffered += 1;
        VPI_ASSERT(vpiSubmitRescale(stream, VPI_BACKEND_VIC,
                                    frames_vpi_converted_[i],
                                    frames_vpi_rescaled_[i],
                                    VPI_INTERP_LINEAR,
                                    VPI_BORDER_CLAMP,
                                    0));

        if (sync || (buffered >= qlen)) {
            VPI_ASSERT(vpiStreamSync(stream));
            buffered = 0;
        }
    }

    if (!sync) VPI_ASSERT(vpiStreamSync(stream));
}
