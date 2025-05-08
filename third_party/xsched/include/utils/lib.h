#pragma once

#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>

#include "utils/lib.h"
#include "utils/log.h"
#include "utils/common.h"

#define ENV_VPI_DLL_PATH      "X_VPI_DLL_PATH"
#define ENV_CUDA_DLL_PATH     "X_CUDA_DLL_PATH"
#define ENV_CUDLA_DLL_PATH    "X_CUDLA_DLL_PATH"
#define ENV_CUDART_DLL_PATH   "X_CUDART_DLL_PATH"
#define ENV_ASCENDCL_DLL_PATH "X_ASCENDCL_DLL_PATH"

namespace xsched::utils
{

inline bool FileExists(const std::string &path)
{
    std::ifstream file(path);
    bool exists = file.good();
    file.close();
    return exists;
}

inline std::string FindLibrary(const std::string &name,
                               const std::string &env_name,
                               const std::vector<std::string> &dirs)
{
    static const std::vector<std::string> default_dirs {
        "/lib/"          ARCH_STR "-linux-gnu",
        "/usr/lib/"      ARCH_STR "-linux-gnu",
        "/usr/local/lib" ARCH_STR "-linux-gnu",
        "/lib/",
        "/usr/lib/",
        "/usr/local/lib",
    };

    char *path = std::getenv(env_name.c_str());
    if (path != nullptr) {
        if (FileExists(path)) return std::string(path);
        XWARN("lib %s set by env %s = %s not found, fallback to path search",
              name.c_str(), env_name.c_str(), path);
    }

    for (const auto &dir : dirs) {
        std::string path = dir + "/" + name;
        if (FileExists(path)) {
            XINFO("lib %s found at %s", name.c_str(), path.c_str());
            return path;
        }
    }

    for (const auto &dir : default_dirs) {
        std::string path = dir + "/" + name;
        if (FileExists(path)) {
            XINFO("lib %s found at %s", name.c_str(), path.c_str());
            return path;
        }
    }

    XWARN("lib %s not found", name.c_str());
    return "";
}

} // namespace xsched::utils
