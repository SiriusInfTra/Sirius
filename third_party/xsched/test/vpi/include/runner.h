#pragma once

#include <cstddef>
#include <vpi/Stream.h>

class VpiRunner
{
public:
    VpiRunner() = default;
    virtual ~VpiRunner() = default;

    virtual VPIStream CreateStream() = 0;
    virtual void Init(VPIStream stream) = 0;
    virtual void Final(VPIStream stream) = 0;
    virtual void Execute(VPIStream stream,
                         const size_t qlen,
                         const bool sync) = 0;
};
