#pragma once

#include <cstdlib>

#include "utils/log.h"
#include "utils/common.h"
#include "hal/ascend/acl.h"

#define ACL_ASSERT(cmd) \
    do { \
        aclError result = cmd; \
        if (UNLIKELY(result != ACL_SUCCESS)) { \
            XERRO("acl error %d", result); \
        } \
    } while (0);
