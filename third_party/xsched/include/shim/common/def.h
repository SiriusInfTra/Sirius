#pragma once

#include <memory>

#define DEFINE_SHIM_FUNC1(submit, \
                          halq_type, cmd_type, \
                          ret_type, func_name, \
                          type0, arg0)         \
    inline ret_type func_name(type0 arg0, halq_type hal_queue) \
    { \
        auto cmd = std::make_shared<cmd_type>(arg0); \
        return submit(cmd, hal_queue);               \
    }

#define DEFINE_SHIM_FUNC2(submit, \
                          halq_type, cmd_type, \
                          ret_type, func_name, \
                          type0, arg0,         \
                          type1, arg1)         \
    inline ret_type func_name(type0 arg0,      \
                              type1 arg1,      \
                              halq_type hal_queue)   \
    { \
        auto cmd = std::make_shared<cmd_type>(arg0,  \
                                              arg1); \
        return submit(cmd, hal_queue);               \
    }

#define DEFINE_SHIM_FUNC3(submit, \
                          halq_type, cmd_type, \
                          ret_type, func_name, \
                          type0, arg0,         \
                          type1, arg1,         \
                          type2, arg2)         \
    inline ret_type func_name(type0 arg0,      \
                              type1 arg1,      \
                              type2 arg2,      \
                              halq_type hal_queue)   \
    { \
        auto cmd = std::make_shared<cmd_type>(arg0,  \
                                              arg1,  \
                                              arg2); \
        return submit(cmd, hal_queue);               \
    }

#define DEFINE_SHIM_FUNC4(submit, \
                          halq_type, cmd_type, \
                          ret_type, func_name, \
                          type0, arg0,         \
                          type1, arg1,         \
                          type2, arg2,         \
                          type3, arg3)         \
    inline ret_type func_name(type0 arg0,      \
                              type1 arg1,      \
                              type2 arg2,      \
                              type3 arg3,      \
                              halq_type hal_queue)   \
    { \
        auto cmd = std::make_shared<cmd_type>(arg0,  \
                                              arg1,  \
                                              arg2,  \
                                              arg3); \
        return submit(cmd, hal_queue);               \
    }

#define DEFINE_SHIM_FUNC5(submit, \
                          halq_type, cmd_type, \
                          ret_type, func_name, \
                          type0, arg0,         \
                          type1, arg1,         \
                          type2, arg2,         \
                          type3, arg3,         \
                          type4, arg4)         \
    inline ret_type func_name(type0 arg0,      \
                              type1 arg1,      \
                              type2 arg2,      \
                              type3 arg3,      \
                              type4 arg4,      \
                              halq_type hal_queue)   \
    { \
        auto cmd = std::make_shared<cmd_type>(arg0,  \
                                              arg1,  \
                                              arg2,  \
                                              arg3,  \
                                              arg4); \
        return submit(cmd, hal_queue);               \
    }

#define DEFINE_SHIM_FUNC6(submit, \
                          halq_type, cmd_type, \
                          ret_type, func_name, \
                          type0, arg0,         \
                          type1, arg1,         \
                          type2, arg2,         \
                          type3, arg3,         \
                          type4, arg4,         \
                          type5, arg5)         \
    inline ret_type func_name(type0 arg0,      \
                              type1 arg1,      \
                              type2 arg2,      \
                              type3 arg3,      \
                              type4 arg4,      \
                              type5 arg5,      \
                              halq_type hal_queue)   \
    { \
        auto cmd = std::make_shared<cmd_type>(arg0,  \
                                              arg1,  \
                                              arg2,  \
                                              arg3,  \
                                              arg4,  \
                                              arg5); \
        return submit(cmd, hal_queue);               \
    }

#define DEFINE_SHIM_FUNC7(submit, \
                          halq_type, cmd_type, \
                          ret_type, func_name, \
                          type0, arg0,         \
                          type1, arg1,         \
                          type2, arg2,         \
                          type3, arg3,         \
                          type4, arg4,         \
                          type5, arg5,         \
                          type6, arg6)         \
    inline ret_type func_name(type0 arg0,      \
                              type1 arg1,      \
                              type2 arg2,      \
                              type3 arg3,      \
                              type4 arg4,      \
                              type5 arg5,      \
                              type6 arg6,      \
                              halq_type hal_queue)   \
    { \
        auto cmd = std::make_shared<cmd_type>(arg0,  \
                                              arg1,  \
                                              arg2,  \
                                              arg3,  \
                                              arg4,  \
                                              arg5,  \
                                              arg6); \
        return submit(cmd, hal_queue);               \
    }

#define DEFINE_SHIM_FUNC8(submit, \
                          halq_type, cmd_type, \
                          ret_type, func_name, \
                          type0, arg0,         \
                          type1, arg1,         \
                          type2, arg2,         \
                          type3, arg3,         \
                          type4, arg4,         \
                          type5, arg5,         \
                          type6, arg6,         \
                          type7, arg7)         \
    inline ret_type func_name(type0 arg0,      \
                              type1 arg1,      \
                              type2 arg2,      \
                              type3 arg3,      \
                              type4 arg4,      \
                              type5 arg5,      \
                              type6 arg6,      \
                              type7 arg7,      \
                              halq_type hal_queue)   \
    { \
        auto cmd = std::make_shared<cmd_type>(arg0,  \
                                              arg1,  \
                                              arg2,  \
                                              arg3,  \
                                              arg4,  \
                                              arg5,  \
                                              arg6,  \
                                              arg7); \
        return submit(cmd, hal_queue);               \
    }
