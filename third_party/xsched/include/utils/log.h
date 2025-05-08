#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>

#include "utils/common.h"

#define XLOG_FD stderr

#define FLUSH_XLOG() do { fflush(XLOG_FD); } while (0);

#ifdef XSCHED_DEBG
#define FLUSH_XLOG_IF_DEBG() FLUSH_XLOG()
#else
#define FLUSH_XLOG_IF_DEBG()
#endif

#define XLOG_HELPER(level, format, ...) \
    do { \
        std::stringstream ss; \
        const auto now = std::chrono::system_clock::now(); \
        const auto now_tt = std::chrono::system_clock::to_time_t(now);   \
        const auto now_lt = std::localtime(&now_tt);       \
        ss << GetThreadId()                                \
           << " @ "                                        \
           << std::put_time(now_lt, "%X");                 \
        fprintf(XLOG_FD, "[%s @ T%s] " format "\n",        \
            level, ss.str().c_str(), ##__VA_ARGS__);       \
        FLUSH_XLOG_IF_DEBG();                              \
    } while (0);

// first unfold the arguments, then unfold XLOG
#define XLOG(level, format, ...) \
    UNFOLD(XLOG_HELPER UNFOLD((level, format, ##__VA_ARGS__)))

#define XLOG_WITH_CODE(level, format, ...) \
    UNFOLD(XLOG_HELPER UNFOLD((level, format " @ %s:%d", \
           ##__VA_ARGS__, __FILE__, __LINE__)))

#ifdef XSCHED_DEBG
#define XDEBG(format, ...) XLOG_WITH_CODE("DEBG", format, ##__VA_ARGS__)
#define XINFO(format, ...) XLOG_WITH_CODE("INFO", format, ##__VA_ARGS__)
#else
#define XDEBG(format, ...)
#define XINFO(format, ...) XLOG("INFO", format, ##__VA_ARGS__)
#endif

#define XWARN(format, ...) XLOG_WITH_CODE("WARN", format, ##__VA_ARGS__)
#define XERRO(format, ...) \
    do { \
        XLOG_WITH_CODE("ERRO", format, ##__VA_ARGS__) \
        FLUSH_XLOG();       \
        /* exit(EXIT_FAILURE); */ \
        std::abort(); \
    } while (0);
