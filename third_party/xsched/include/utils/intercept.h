#pragma once

#include "utils/log.h"
#include "utils/common.h"

template <bool Reorder, typename R, typename F, typename Q, typename... Args>
struct RedirectCall;

template <typename R, typename F, typename A0, typename... Args>
struct RedirectCall<true, R, F, A0, Args...>
{
    static inline R Call(F func, A0 arg0, Args... args)
    {
        return func(args..., arg0);
    }
};

template <typename R, typename F, typename A0, typename... Args>
struct RedirectCall<false, R, F, A0, Args...>
{
    // The first argument type of the target function is not A0.
    // Swap the order of the arguments.
    static inline R Call(F func, A0 arg0, Args... args)
    {
        return func(arg0, args...);
    }
};

#define DEFINE_INTERCEPT_FUNC0(target_func, reorder, \
                               ret_type, func_name)  \
    EXPORT_C_FUNC ret_type func_name() \
    { \
        XDEBG("intercepted " #func_name); \
        return target_func(); \
    }

#define DEFINE_INTERCEPT_FUNC1(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0) \
    { \
        XDEBG("intercepted " #func_name); \
        return target_func(arg0); \
    }

#define DEFINE_INTERCEPT_FUNC2(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0, \
                               type1, arg1) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1  \
        >::Call(target_func, \
                arg0,  \
                arg1); \
    }

#define DEFINE_INTERCEPT_FUNC3(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0, \
                               type1, arg1, \
                               type2, arg2) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1, \
                                     type2 arg2) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1, \
            type2  \
        >::Call(target_func, \
                arg0,  \
                arg1,  \
                arg2); \
    }

#define DEFINE_INTERCEPT_FUNC4(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0, \
                               type1, arg1, \
                               type2, arg2, \
                               type3, arg3) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1, \
                                     type2 arg2, \
                                     type3 arg3) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1, \
            type2, \
            type3  \
        >::Call(target_func, \
                arg0,  \
                arg1,  \
                arg2,  \
                arg3); \
    }

#define DEFINE_INTERCEPT_FUNC5(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0, \
                               type1, arg1, \
                               type2, arg2, \
                               type3, arg3, \
                               type4, arg4) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1, \
                                     type2 arg2, \
                                     type3 arg3, \
                                     type4 arg4) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1, \
            type2, \
            type3, \
            type4  \
        >::Call(target_func, \
                arg0,  \
                arg1,  \
                arg2,  \
                arg3,  \
                arg4); \
    }

#define DEFINE_INTERCEPT_FUNC6(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0, \
                               type1, arg1, \
                               type2, arg2, \
                               type3, arg3, \
                               type4, arg4, \
                               type5, arg5) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1, \
                                     type2 arg2, \
                                     type3 arg3, \
                                     type4 arg4, \
                                     type5 arg5) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1, \
            type2, \
            type3, \
            type4, \
            type5  \
        >::Call(target_func, \
                arg0,  \
                arg1,  \
                arg2,  \
                arg3,  \
                arg4,  \
                arg5); \
    }

#define DEFINE_INTERCEPT_FUNC7(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0, \
                               type1, arg1, \
                               type2, arg2, \
                               type3, arg3, \
                               type4, arg4, \
                               type5, arg5, \
                               type6, arg6) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1, \
                                     type2 arg2, \
                                     type3 arg3, \
                                     type4 arg4, \
                                     type5 arg5, \
                                     type6 arg6) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1, \
            type2, \
            type3, \
            type4, \
            type5, \
            type6  \
        >::Call(target_func, \
                arg0,  \
                arg1,  \
                arg2,  \
                arg3,  \
                arg4,  \
                arg5,  \
                arg6); \
    }

#define DEFINE_INTERCEPT_FUNC8(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0, \
                               type1, arg1, \
                               type2, arg2, \
                               type3, arg3, \
                               type4, arg4, \
                               type5, arg5, \
                               type6, arg6, \
                               type7, arg7) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1, \
                                     type2 arg2, \
                                     type3 arg3, \
                                     type4 arg4, \
                                     type5 arg5, \
                                     type6 arg6, \
                                     type7 arg7) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1, \
            type2, \
            type3, \
            type4, \
            type5, \
            type6, \
            type7  \
        >::Call(target_func, \
                arg0,  \
                arg1,  \
                arg2,  \
                arg3,  \
                arg4,  \
                arg5,  \
                arg6,  \
                arg7); \
    }

#define DEFINE_INTERCEPT_FUNC9(target_func, reorder, \
                               ret_type, func_name,  \
                               type0, arg0, \
                               type1, arg1, \
                               type2, arg2, \
                               type3, arg3, \
                               type4, arg4, \
                               type5, arg5, \
                               type6, arg6, \
                               type7, arg7, \
                               type8, arg8) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1, \
                                     type2 arg2, \
                                     type3 arg3, \
                                     type4 arg4, \
                                     type5 arg5, \
                                     type6 arg6, \
                                     type7 arg7, \
                                     type8 arg8) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1, \
            type2, \
            type3, \
            type4, \
            type5, \
            type6, \
            type7, \
            type8  \
        >::Call(target_func, \
                arg0,  \
                arg1,  \
                arg2,  \
                arg3,  \
                arg4,  \
                arg5,  \
                arg6,  \
                arg7,  \
                arg8); \
    }

#define DEFINE_INTERCEPT_FUNC10(target_func, reorder, \
                               ret_type, func_name,  \
                                type0, arg0, \
                                type1, arg1, \
                                type2, arg2, \
                                type3, arg3, \
                                type4, arg4, \
                                type5, arg5, \
                                type6, arg6, \
                                type7, arg7, \
                                type8, arg8, \
                                type9, arg9) \
    EXPORT_C_FUNC ret_type func_name(type0 arg0, \
                                     type1 arg1, \
                                     type2 arg2, \
                                     type3 arg3, \
                                     type4 arg4, \
                                     type5 arg5, \
                                     type6 arg6, \
                                     type7 arg7, \
                                     type8 arg8, \
                                     type9 arg9) \
    { \
        XDEBG("intercepted " #func_name); \
        return RedirectCall< \
            reorder, \
            ret_type, decltype(target_func), \
            type0, \
            type1, \
            type2, \
            type3, \
            type4, \
            type5, \
            type6, \
            type7, \
            type8, \
            type9  \
        >::Call(target_func, \
                arg0,  \
                arg1,  \
                arg2,  \
                arg3,  \
                arg4,  \
                arg5,  \
                arg6,  \
                arg7,  \
                arg8,  \
                arg9); \
    }
