#pragma once

#include <cstdlib>

#include "utils/log.h"
#include "utils/common.h"
#include "hal/vpi/driver.h"

#define VPI_ASSERT(cmd) \
    do { \
        VPIStatus res = cmd; \
        if (UNLIKELY(res != VPI_SUCCESS)) {       \
            char msg[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
            const char *name = xsched::hal::vpi::Driver::StatusGetName(res);  \
            xsched::hal::vpi::Driver::GetLastStatusMessage(msg, sizeof(msg)); \
            XERRO("vpi error %d(%s): %s", res, name, msg); \
        } \
    } while (0);
