#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/algo/ConvertImageFormat.h>

#include "sde.h"
#include "utils/log.h"
#include "utils/xassert.h"

OfaSdeRunner::OfaSdeRunner(const std::string &left_path_in,
                           const std::string &right_path_in)
    : left_path_in_(left_path_in), right_path_in_(right_path_in)
{
    left_cv_in_ = cv::imread(left_path_in_);
    XASSERT(!left_cv_in_.empty(),
            "Cannot open input left img '%s'",
            left_path_in_.c_str());
    right_cv_in_ = cv::imread(right_path_in_);
    XASSERT(!right_cv_in_.empty(),
            "Cannot open input right img '%s'",
            right_path_in_.c_str());
    
    frames_cnt_ = 256;
    w_ = left_cv_in_.cols * 0.75;
    h_ = left_cv_in_.rows * 0.75;
    cv::resize(left_cv_in_, left_cv_in_, cv::Size(w_, h_));
    cv::resize(right_cv_in_, right_cv_in_, cv::Size(w_, h_));
}

VPIStream OfaSdeRunner::CreateStream()
{
    VPIStream stream = nullptr;
    VPI_ASSERT(vpiStreamCreate(VPI_BACKEND_CUDA | VPI_BACKEND_VIC |
                               VPI_BACKEND_OFA, &stream));
    return stream;
}

void OfaSdeRunner::Init(VPIStream stream)
{
    VPI_ASSERT(vpiImageCreateWrapperOpenCVMat(
        left_cv_in_, 0, &left_vpi_in_));
    VPI_ASSERT(vpiImageCreateWrapperOpenCVMat(
        right_cv_in_, 0, &right_vpi_in_));
    
    VPI_ASSERT(vpiInitStereoDisparityEstimatorCreationParams(&stereo_params_));
    stereo_params_.maxDisparity = 128;

    VPI_ASSERT(vpiCreateStereoDisparityEstimator(VPI_BACKEND_OFA, w_, h_,
                                                 VPI_IMAGE_FORMAT_Y16_ER_BL,
                                                 &stereo_params_, &payload_));
    VPI_ASSERT(vpiImageCreate(
        w_, h_, VPI_IMAGE_FORMAT_S16_BL, 0, &disparity_));
    VPI_ASSERT(vpiImageCreate(
        w_, h_, VPI_IMAGE_FORMAT_Y16_ER_BL, 0, &stereo_left_));
    VPI_ASSERT(vpiImageCreate(
        w_, h_, VPI_IMAGE_FORMAT_Y16_ER_BL, 0, &stereo_right_));

    VPIImage tmp_left;
    VPIImage tmp_right;
    VPI_ASSERT(vpiImageCreate(w_, h_, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmp_left));
    VPI_ASSERT(vpiImageCreate(w_, h_, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmp_right));

    VPIConvertImageFormatParams conv_params;
    VPI_ASSERT(vpiInitConvertImageFormatParams(&conv_params));

    VPI_ASSERT(vpiSubmitConvertImageFormat(
        stream, VPI_BACKEND_CUDA, left_vpi_in_, tmp_left, &conv_params));
    VPI_ASSERT(vpiSubmitConvertImageFormat(
        stream, VPI_BACKEND_CUDA, right_vpi_in_, tmp_right, &conv_params));

    VPI_ASSERT(vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmp_left, stereo_left_,
                                VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
    VPI_ASSERT(vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmp_right, stereo_right_,
                                VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
    
    VPI_ASSERT(vpiStreamSync(stream));
    vpiImageDestroy(tmp_left);
    vpiImageDestroy(tmp_right);
}

void OfaSdeRunner::Final(VPIStream)
{
    VPIImageData disparity_data;
    VPI_ASSERT(vpiImageLockData(disparity_, VPI_LOCK_READ,
                                VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                &disparity_data));

    cv::Mat disparity_cv;
    VPI_ASSERT(vpiImageDataExportOpenCVMat(disparity_data, &disparity_cv));

    disparity_cv.convertTo(disparity_cv, CV_8UC1,
                           255.0 / (32 * stereo_params_.maxDisparity), 0);

    cv::Mat disparity_color;
    cv::applyColorMap(disparity_cv, disparity_color, cv::COLORMAP_JET);

    VPI_ASSERT(vpiImageUnlock(disparity_));

    if (confidence_map_) {
        VPIImageData confidence_data;
        VPI_ASSERT(vpiImageLockData(confidence_map_, VPI_LOCK_READ,
                                    VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                    &confidence_data));

        cv::Mat confidence_cv;
        VPI_ASSERT(vpiImageDataExportOpenCVMat(confidence_data,
                                               &confidence_cv));

        confidence_cv.convertTo(confidence_cv, CV_8UC1, 255.0 / 65535, 0);
        cv::imwrite("confidence.png", confidence_cv);

        VPI_ASSERT(vpiImageUnlock(confidence_map_));

        cv::Mat mask;
        cv::threshold(confidence_cv, mask, 1, 255, cv::THRESH_BINARY);
        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
        cv::bitwise_and(disparity_color, mask, disparity_color);
    }

    cv::imwrite("disparity.png", disparity_color);
    
    vpiImageDestroy(left_vpi_in_);
    vpiImageDestroy(right_vpi_in_);
    vpiImageDestroy(disparity_);
    vpiImageDestroy(stereo_left_);
    vpiImageDestroy(stereo_right_);
}

void OfaSdeRunner::Execute(VPIStream stream,
                           const size_t qlen,
                           const bool sync)
{
    for (size_t i = 0; i < qlen; ++i) {
        VPI_ASSERT(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_OFA,
                                                     payload_,
                                                     stereo_left_,
                                                     stereo_right_,
                                                     disparity_,
                                                     confidence_map_,
                                                     nullptr));

        if (sync) VPI_ASSERT(vpiStreamSync(stream));
    }

    if (!sync) VPI_ASSERT(vpiStreamSync(stream));
}
