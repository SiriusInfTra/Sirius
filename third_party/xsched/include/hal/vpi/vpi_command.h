#pragma once

#include "hal/vpi/driver.h"
#include "preempt/hal/def.h"
#include "preempt/hal/hal_command.h"

namespace xsched::hal::vpi
{

class VpiCommand : public preempt::HalCommand
{
public:
    VpiCommand(preempt::HalCommandType hal_type): HalCommand(hal_type) {}
    virtual ~VpiCommand();

    VPIStatus EnqueueWrapper(VPIStream stream);

    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override;
    virtual bool EnableHalSynchronization() override;

private:
    VPIEvent following_event_ = nullptr;

    virtual VPIStatus Enqueue(VPIStream stream) = 0;
};

class VpiEventRecordCommand : public VpiCommand
{
public:
    VpiEventRecordCommand(VPIEvent event);
    virtual ~VpiEventRecordCommand() = default;

    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override { return true; }
    virtual bool EnableHalSynchronization() override { return true; }

protected:
    VPIEvent event_;

private:
    virtual VPIStatus Enqueue(VPIStream stream) override;
};

class VpiAlgorithmCommand : public VpiCommand
{
public:
    VpiAlgorithmCommand(): VpiCommand(preempt::kHalCommandTypeNormal) {}
    virtual ~VpiAlgorithmCommand() = default;
};

DEFINE_HAL_COMMAND4(VpiConvertImageFormatCommand, VpiAlgorithmCommand,
                    VPIStatus, VPIStream, true,
                    Driver::SubmitConvertImageFormat,
                    uint64_t, backend, false,
                    VPIImage, input  , false,
                    VPIImage, output , false,
                    const VPIConvertImageFormatParams *, params, true);

DEFINE_HAL_COMMAND8(VpiGaussianFilterCommand, VpiAlgorithmCommand,
                    VPIStatus, VPIStream, true,
                    Driver::SubmitGaussianFilter,
                    uint64_t, backend      , false,
                    VPIImage, input        , false,
                    VPIImage, output       , false,
                    int32_t , kernel_size_x, false,
                    int32_t , kernel_size_y, false,
                    float   , sigma_x      , false,
                    float   , sigma_y      , false,
                    VPIBorderExtension, border, false);

DEFINE_HAL_COMMAND6(VpiRescaleCommand, VpiAlgorithmCommand,
                    VPIStatus, VPIStream, true,
                    Driver::SubmitRescale,
                    uint64_t, backend, false,
                    VPIImage, input  , false,
                    VPIImage, output , false,
                    VPIInterpolationType, interpolation, false,
                    VPIBorderExtension  , border       , false,
                    uint64_t, flags  , false);

DEFINE_HAL_COMMAND7(VpiStereoDisparityEstimatorCommand, VpiAlgorithmCommand,
                    VPIStatus, VPIStream, true,
                    Driver::SubmitStereoDisparityEstimator,
                    uint64_t  , backend       , false,
                    VPIPayload, payload       , false,
                    VPIImage  , left          , false,
                    VPIImage  , right         , false,
                    VPIImage  , disparity     , false,
                    VPIImage  , confidence_map, false,
                    const VPIStereoDisparityEstimatorParams *, params, true);

} // namespace xsched::hal::vpi
