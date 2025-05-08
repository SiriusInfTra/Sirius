#pragma once

#include <cstdlib>
#include <vpi/Status.h>

#include "utils/log.h"

#define VPI_ASSERT(cmd) \
    do { \
        VPIStatus res = cmd; \
        if (UNLIKELY(res != VPI_SUCCESS)) {       \
            char msg[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
            const char *name = vpiStatusGetName(res);  \
            vpiGetLastStatusMessage(msg, sizeof(msg)); \
            XERRO("vpi error %d(%s): %s", res, name, msg); \
        } \
    } while (0);
