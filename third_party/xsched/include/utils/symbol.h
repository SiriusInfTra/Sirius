#pragma once

#include <dlfcn.h>

#include "utils/lib.h"
#include "utils/common.h"
#include "utils/xassert.h"

#define DEFINE_GET_SYMBOL_FUNC(name, dll_name, env_name, dirs) \
    static void *name(const char *symbol_name) \
    { \
        static const std::string dll_path =                               \
            xsched::utils::FindLibrary(dll_name, env_name, dirs);         \
        static void *dll_handle =                                         \
            dlopen(dll_path.c_str(), RTLD_NOW | RTLD_LOCAL);              \
        XASSERT(dll_handle != nullptr,                                    \
                "fail to dlopen %s", dll_path.c_str());                   \
        void *symbol = dlsym(dll_handle, symbol_name);                    \
        XASSERT(symbol != nullptr, "fail to get symbol %s", symbol_name); \
        return symbol; \
    }

#define DEFINE_SYMBOL0(sym_name, get_sym_func, ret_type, name) \
    ret_type name() \
    { \
        using FuncPtr = ret_type (*)(); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(); \
    }

#define DEFINE_SYMBOL1(sym_name, get_sym_func, ret_type, name, type0, arg0) \
    ret_type name(type0 arg0) \
    { \
        using FuncPtr = ret_type (*)(type0); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0); \
    }

#define DEFINE_SYMBOL2(sym_name, get_sym_func, ret_type, name, \
                       type0, arg0,  \
                       type1, arg1)  \
    ret_type name(type0 arg0, \
                         type1 arg1) \
    { \
        using FuncPtr = ret_type (*)(type0,  \
                                     type1); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
                    arg1); \
    }

#define DEFINE_SYMBOL3(sym_name, get_sym_func, ret_type, name, \
                       type0, arg0,  \
                       type1, arg1,  \
                       type2, arg2)  \
    ret_type name(type0 arg0, \
                         type1 arg1, \
                         type2 arg2) \
    { \
        using FuncPtr = ret_type (*)(type0,  \
                                     type1,  \
                                     type2); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
                    arg1,  \
                    arg2); \
    }

#define DEFINE_SYMBOL4(sym_name, get_sym_func, ret_type, name, \
                       type0, arg0,  \
                       type1, arg1,  \
                       type2, arg2,  \
                       type3, arg3)  \
    ret_type name(type0 arg0, \
                         type1 arg1, \
                         type2 arg2, \
                         type3 arg3) \
    { \
        using FuncPtr = ret_type (*)(type0,  \
                                     type1,  \
                                     type2,  \
                                     type3); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
                    arg1,  \
                    arg2,  \
                    arg3); \
    }

#define DEFINE_SYMBOL5(sym_name, get_sym_func, ret_type, name, \
                       type0, arg0,  \
                       type1, arg1,  \
                       type2, arg2,  \
                       type3, arg3,  \
                       type4, arg4)  \
    ret_type name(type0 arg0, \
                         type1 arg1, \
                         type2 arg2, \
                         type3 arg3, \
                         type4 arg4) \
    { \
        using FuncPtr = ret_type (*)(type0,  \
                                     type1,  \
                                     type2,  \
                                     type3,  \
                                     type4); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
                    arg1,  \
                    arg2,  \
                    arg3,  \
                    arg4); \
    }

#define DEFINE_SYMBOL6(sym_name, get_sym_func, ret_type, name, \
                       type0, arg0,  \
                       type1, arg1,  \
                       type2, arg2,  \
                       type3, arg3,  \
                       type4, arg4,  \
                       type5, arg5)  \
    ret_type name(type0 arg0, \
                         type1 arg1, \
                         type2 arg2, \
                         type3 arg3, \
                         type4 arg4, \
                         type5 arg5) \
    { \
        using FuncPtr = ret_type (*)(type0,  \
                                     type1,  \
                                     type2,  \
                                     type3,  \
                                     type4,  \
                                     type5); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
                    arg1,  \
                    arg2,  \
                    arg3,  \
                    arg4,  \
                    arg5); \
    }

#define DEFINE_SYMBOL7(sym_name, get_sym_func, ret_type, name, \
                       type0, arg0,  \
                       type1, arg1,  \
                       type2, arg2,  \
                       type3, arg3,  \
                       type4, arg4,  \
                       type5, arg5,  \
                       type6, arg6)  \
    ret_type name(type0 arg0, \
                         type1 arg1, \
                         type2 arg2, \
                         type3 arg3, \
                         type4 arg4, \
                         type5 arg5, \
                         type6 arg6) \
    { \
        using FuncPtr = ret_type (*)(type0,  \
                                     type1,  \
                                     type2,  \
                                     type3,  \
                                     type4,  \
                                     type5,  \
                                     type6); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
                    arg1,  \
                    arg2,  \
                    arg3,  \
                    arg4,  \
                    arg5,  \
                    arg6); \
    }

#define DEFINE_SYMBOL8(sym_name, get_sym_func, ret_type, name, \
                       type0, arg0,  \
                       type1, arg1,  \
                       type2, arg2,  \
                       type3, arg3,  \
                       type4, arg4,  \
                       type5, arg5,  \
                       type6, arg6,  \
                       type7, arg7)  \
    ret_type name(type0 arg0, \
                         type1 arg1, \
                         type2 arg2, \
                         type3 arg3, \
                         type4 arg4, \
                         type5 arg5, \
                         type6 arg6, \
                         type7 arg7) \
    { \
        using FuncPtr = ret_type (*)(type0,  \
                                     type1,  \
                                     type2,  \
                                     type3,  \
                                     type4,  \
                                     type5,  \
                                     type6,  \
                                     type7); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
                    arg1,  \
                    arg2,  \
                    arg3,  \
                    arg4,  \
                    arg5,  \
                    arg6,  \
                    arg7); \
    }

#define DEFINE_SYMBOL9(sym_name, get_sym_func, ret_type, name, \
                       type0, arg0,  \
                       type1, arg1,  \
                       type2, arg2,  \
                       type3, arg3,  \
                       type4, arg4,  \
                       type5, arg5,  \
                       type6, arg6,  \
                       type7, arg7,  \
                       type8, arg8)  \
    ret_type name(type0 arg0, \
                         type1 arg1, \
                         type2 arg2, \
                         type3 arg3, \
                         type4 arg4, \
                         type5 arg5, \
                         type6 arg6, \
                         type7 arg7, \
                         type8 arg8) \
    { \
        using FuncPtr = ret_type (*)(type0,  \
                                     type1,  \
                                     type2,  \
                                     type3,  \
                                     type4,  \
                                     type5,  \
                                     type6,  \
                                     type7,  \
                                     type8); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
                    arg1,  \
                    arg2,  \
                    arg3,  \
                    arg4,  \
                    arg5,  \
                    arg6,  \
                    arg7,  \
                    arg8); \
    }

#define DEFINE_SYMBOL10(sym_name, get_sym_func, ret_type, name, \
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
    ret_type name(type0 arg0, \
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
        using FuncPtr = ret_type (*)(type0,  \
                                     type1,  \
                                     type2,  \
                                     type3,  \
                                     type4,  \
                                     type5,  \
                                     type6,  \
                                     type7,  \
                                     type8,  \
                                     type9); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,  \
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

#define DEFINE_SYMBOL11(sym_name, get_sym_func, ret_type, name, \
                        type0, arg0,   \
                        type1, arg1,   \
                        type2, arg2,   \
                        type3, arg3,   \
                        type4, arg4,   \
                        type5, arg5,   \
                        type6, arg6,   \
                        type7, arg7,   \
                        type8, arg8,   \
                        type9, arg9,   \
                        type10, arg10) \
    ret_type name(type0 arg0,   \
                         type1 arg1,   \
                         type2 arg2,   \
                         type3 arg3,   \
                         type4 arg4,   \
                         type5 arg5,   \
                         type6 arg6,   \
                         type7 arg7,   \
                         type8 arg8,   \
                         type9 arg9,   \
                         type10 arg10) \
    { \
        using FuncPtr = ret_type (*)(type0,   \
                                     type1,   \
                                     type2,   \
                                     type3,   \
                                     type4,   \
                                     type5,   \
                                     type6,   \
                                     type7,   \
                                     type8,   \
                                     type9,   \
                                     type10); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,   \
                    arg1,   \
                    arg2,   \
                    arg3,   \
                    arg4,   \
                    arg5,   \
                    arg6,   \
                    arg7,   \
                    arg8,   \
                    arg9,   \
                    arg10); \
    }

#define DEFINE_SYMBOL12(sym_name, get_sym_func, ret_type, name, \
                        type0, arg0,   \
                        type1, arg1,   \
                        type2, arg2,   \
                        type3, arg3,   \
                        type4, arg4,   \
                        type5, arg5,   \
                        type6, arg6,   \
                        type7, arg7,   \
                        type8, arg8,   \
                        type9, arg9,   \
                        type10, arg10, \
                        type11, arg11) \
    ret_type name(type0 arg0,   \
                         type1 arg1,   \
                         type2 arg2,   \
                         type3 arg3,   \
                         type4 arg4,   \
                         type5 arg5,   \
                         type6 arg6,   \
                         type7 arg7,   \
                         type8 arg8,   \
                         type9 arg9,   \
                         type10 arg10, \
                         type11 arg11) \
    { \
        using FuncPtr = ret_type (*)(type0,   \
                                     type1,   \
                                     type2,   \
                                     type3,   \
                                     type4,   \
                                     type5,   \
                                     type6,   \
                                     type7,   \
                                     type8,   \
                                     type9,   \
                                     type10,  \
                                     type11); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,   \
                    arg1,   \
                    arg2,   \
                    arg3,   \
                    arg4,   \
                    arg5,   \
                    arg6,   \
                    arg7,   \
                    arg8,   \
                    arg9,   \
                    arg10,  \
                    arg11); \
    }
#define DEFINE_SYMBOL13(sym_name, get_sym_func, ret_type, name, \
                        type0, arg0,   \
                        type1, arg1,   \
                        type2, arg2,   \
                        type3, arg3,   \
                        type4, arg4,   \
                        type5, arg5,   \
                        type6, arg6,   \
                        type7, arg7,   \
                        type8, arg8,   \
                        type9, arg9,   \
                        type10, arg10, \
                        type11, arg11, \
                        type12, arg12) \
    ret_type name(type0 arg0,   \
                         type1 arg1,   \
                         type2 arg2,   \
                         type3 arg3,   \
                         type4 arg4,   \
                         type5 arg5,   \
                         type6 arg6,   \
                         type7 arg7,   \
                         type8 arg8,   \
                         type9 arg9,   \
                         type10 arg10, \
                         type11 arg11, \
                         type12 arg12) \
    { \
        using FuncPtr = ret_type (*)(type0,   \
                                     type1,   \
                                     type2,   \
                                     type3,   \
                                     type4,   \
                                     type5,   \
                                     type6,   \
                                     type7,   \
                                     type8,   \
                                     type9,   \
                                     type10,  \
                                     type11,  \
                                     type12); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,   \
                    arg1,   \
                    arg2,   \
                    arg3,   \
                    arg4,   \
                    arg5,   \
                    arg6,   \
                    arg7,   \
                    arg8,   \
                    arg9,   \
                    arg10,  \
                    arg11,  \
                    arg12); \
    }

#define DEFINE_SYMBOL14(sym_name, get_sym_func, ret_type, name, \
                        type0, arg0,   \
                        type1, arg1,   \
                        type2, arg2,   \
                        type3, arg3,   \
                        type4, arg4,   \
                        type5, arg5,   \
                        type6, arg6,   \
                        type7, arg7,   \
                        type8, arg8,   \
                        type9, arg9,   \
                        type10, arg10, \
                        type11, arg11, \
                        type12, arg12, \
                        type13, arg13) \
    ret_type name(type0 arg0,   \
                         type1 arg1,   \
                         type2 arg2,   \
                         type3 arg3,   \
                         type4 arg4,   \
                         type5 arg5,   \
                         type6 arg6,   \
                         type7 arg7,   \
                         type8 arg8,   \
                         type9 arg9,   \
                         type10 arg10, \
                         type11 arg11, \
                         type12 arg12, \
                         type13 arg13) \
    { \
        using FuncPtr = ret_type (*)(type0,   \
                                     type1,   \
                                     type2,   \
                                     type3,   \
                                     type4,   \
                                     type5,   \
                                     type6,   \
                                     type7,   \
                                     type8,   \
                                     type9,   \
                                     type10,  \
                                     type11,  \
                                     type12,  \
                                     type13); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,   \
                    arg1,   \
                    arg2,   \
                    arg3,   \
                    arg4,   \
                    arg5,   \
                    arg6,   \
                    arg7,   \
                    arg8,   \
                    arg9,   \
                    arg10,  \
                    arg11,  \
                    arg12,  \
                    arg13); \
    }

#define DEFINE_SYMBOL15(sym_name, get_sym_func, ret_type, name, \
                        type0, arg0,   \
                        type1, arg1,   \
                        type2, arg2,   \
                        type3, arg3,   \
                        type4, arg4,   \
                        type5, arg5,   \
                        type6, arg6,   \
                        type7, arg7,   \
                        type8, arg8,   \
                        type9, arg9,   \
                        type10, arg10, \
                        type11, arg11, \
                        type12, arg12, \
                        type13, arg13, \
                        type14, arg14) \
    ret_type name(type0 arg0,   \
                         type1 arg1,   \
                         type2 arg2,   \
                         type3 arg3,   \
                         type4 arg4,   \
                         type5 arg5,   \
                         type6 arg6,   \
                         type7 arg7,   \
                         type8 arg8,   \
                         type9 arg9,   \
                         type10 arg10, \
                         type11 arg11, \
                         type12 arg12, \
                         type13 arg13, \
                         type14 arg14) \
    { \
        using FuncPtr = ret_type (*)(type0,   \
                                     type1,   \
                                     type2,   \
                                     type3,   \
                                     type4,   \
                                     type5,   \
                                     type6,   \
                                     type7,   \
                                     type8,   \
                                     type9,   \
                                     type10,  \
                                     type11,  \
                                     type12,  \
                                     type13,  \
                                     type14); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,   \
                    arg1,   \
                    arg2,   \
                    arg3,   \
                    arg4,   \
                    arg5,   \
                    arg6,   \
                    arg7,   \
                    arg8,   \
                    arg9,   \
                    arg10,  \
                    arg11,  \
                    arg12,  \
                    arg13,  \
                    arg14); \
    }

#define DEFINE_SYMBOL16(sym_name, get_sym_func, ret_type, name, \
                        type0, arg0,   \
                        type1, arg1,   \
                        type2, arg2,   \
                        type3, arg3,   \
                        type4, arg4,   \
                        type5, arg5,   \
                        type6, arg6,   \
                        type7, arg7,   \
                        type8, arg8,   \
                        type9, arg9,   \
                        type10, arg10, \
                        type11, arg11, \
                        type12, arg12, \
                        type13, arg13, \
                        type14, arg14, \
                        type15, arg15) \
    ret_type name(type0 arg0,   \
                         type1 arg1,   \
                         type2 arg2,   \
                         type3 arg3,   \
                         type4 arg4,   \
                         type5 arg5,   \
                         type6 arg6,   \
                         type7 arg7,   \
                         type8 arg8,   \
                         type9 arg9,   \
                         type10 arg10, \
                         type11 arg11, \
                         type12 arg12, \
                         type13 arg13, \
                         type14 arg14, \
                         type15 arg15) \
    { \
        using FuncPtr = ret_type (*)(type0,   \
                                     type1,   \
                                     type2,   \
                                     type3,   \
                                     type4,   \
                                     type5,   \
                                     type6,   \
                                     type7,   \
                                     type8,   \
                                     type9,   \
                                     type10,  \
                                     type11,  \
                                     type12,  \
                                     type13,  \
                                     type14,  \
                                     type15); \
        static auto func = reinterpret_cast<FuncPtr>(get_sym_func(sym_name)); \
        return func(arg0,   \
                    arg1,   \
                    arg2,   \
                    arg3,   \
                    arg4,   \
                    arg5,   \
                    arg6,   \
                    arg7,   \
                    arg8,   \
                    arg9,   \
                    arg10,  \
                    arg11,  \
                    arg12,  \
                    arg13,  \
                    arg14,  \
                    arg15); \
    }

#define DEFINE_STATIC_SYMBOL0(...)  static DEFINE_SYMBOL0(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL1(...)  static DEFINE_SYMBOL1(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL2(...)  static DEFINE_SYMBOL2(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL3(...)  static DEFINE_SYMBOL3(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL4(...)  static DEFINE_SYMBOL4(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL5(...)  static DEFINE_SYMBOL5(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL6(...)  static DEFINE_SYMBOL6(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL7(...)  static DEFINE_SYMBOL7(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL8(...)  static DEFINE_SYMBOL8(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL9(...)  static DEFINE_SYMBOL9(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL10(...) static DEFINE_SYMBOL10(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL11(...) static DEFINE_SYMBOL11(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL12(...) static DEFINE_SYMBOL12(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL13(...) static DEFINE_SYMBOL13(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL14(...) static DEFINE_SYMBOL14(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL15(...) static DEFINE_SYMBOL15(__VA_ARGS__)
#define DEFINE_STATIC_SYMBOL16(...) static DEFINE_SYMBOL16(__VA_ARGS__)

#define DEFINE_EXPORT_SYMBOL0(...)  EXPORT_C_FUNC DEFINE_SYMBOL0(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL1(...)  EXPORT_C_FUNC DEFINE_SYMBOL1(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL2(...)  EXPORT_C_FUNC DEFINE_SYMBOL2(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL3(...)  EXPORT_C_FUNC DEFINE_SYMBOL3(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL4(...)  EXPORT_C_FUNC DEFINE_SYMBOL4(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL5(...)  EXPORT_C_FUNC DEFINE_SYMBOL5(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL6(...)  EXPORT_C_FUNC DEFINE_SYMBOL6(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL7(...)  EXPORT_C_FUNC DEFINE_SYMBOL7(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL8(...)  EXPORT_C_FUNC DEFINE_SYMBOL8(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL9(...)  EXPORT_C_FUNC DEFINE_SYMBOL9(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL10(...) EXPORT_C_FUNC DEFINE_SYMBOL10(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL11(...) EXPORT_C_FUNC DEFINE_SYMBOL11(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL12(...) EXPORT_C_FUNC DEFINE_SYMBOL12(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL13(...) EXPORT_C_FUNC DEFINE_SYMBOL13(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL14(...) EXPORT_C_FUNC DEFINE_SYMBOL14(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL15(...) EXPORT_C_FUNC DEFINE_SYMBOL15(__VA_ARGS__)
#define DEFINE_EXPORT_SYMBOL16(...) EXPORT_C_FUNC DEFINE_SYMBOL16(__VA_ARGS__)
