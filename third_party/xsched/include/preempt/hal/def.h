#pragma once

#include <type_traits>

#include "utils/common.h"

template <typename T, bool DeRef>
struct FieldCreator;

template <typename T>
struct FieldCreator<T, true>
{
    using DataType = typename std::remove_pointer<T>::type;
    const T ptr;
    const DataType field;
    FieldCreator(T arg): ptr(arg), field(arg ? *arg : DataType{}) {}
    T Get() const { return ptr ? &field : nullptr; }
};

template <typename T>
struct FieldCreator<T, false>
{
    const T field;
    FieldCreator(T arg): field(arg) {}
    T Get() const { return field; }
};

#define DEFINE_HAL_COMMAND_FIELD(type, field, deref) const FieldCreator<type, deref> field;

// Enqueue a HAL command to a HAL queue with arg type checking
template <bool HalQueueIsFirst, typename R, typename F, typename Q, typename... Args>
struct EnqueueHalCommand;

template <typename R, typename F, typename Q, typename... Args>
struct EnqueueHalCommand<true, R, F, Q, Args...>
{
    static inline R Enqueue(F enqueue_func, Q hal_queue, Args... args)
    {
        return enqueue_func(hal_queue, args...);
    }
};

template <typename R, typename F, typename Q, typename... Args>
struct EnqueueHalCommand<false, R, F, Q, Args...>
{
    static inline R Enqueue(F enqueue_func, Q hal_queue, Args... args)
    {
        return enqueue_func(args..., hal_queue);
    }
};

#define DEFINE_HAL_COMMAND1(class_type, base_type,    \
                            ret_type, halq_type,      \
                            halq_first, enqueue_func, \
                            type0, arg0, deref0) \
    class class_type : public base_type \
    { \
    public: \
        class_type(type0 arg0) \
            : arg0(arg0) {} \
        virtual ~class_type() = default; \
        virtual ret_type Enqueue(halq_type hal_queue) override \
        { \
            return EnqueueHalCommand< \
                halq_first, \
                ret_type, decltype(enqueue_func), halq_type, \
                type0  \
            >::Enqueue(enqueue_func, hal_queue, \
                       arg0.Get()); \
        } \
    private: \
        DEFINE_HAL_COMMAND_FIELD(type0, arg0, deref0); \
    };

#define DEFINE_HAL_COMMAND2(class_type, base_type,    \
                            ret_type, halq_type,      \
                            halq_first, enqueue_func, \
                            type0, arg0, deref0, \
                            type1, arg1, deref1) \
    class class_type : public base_type \
    { \
    public: \
        class_type(type0 arg0, \
                   type1 arg1) \
            : arg0(arg0) \
            , arg1(arg1) {} \
        virtual ~class_type() = default; \
        virtual ret_type Enqueue(halq_type hal_queue) override \
        { \
            return EnqueueHalCommand< \
                halq_first, \
                ret_type, decltype(enqueue_func), halq_type, \
                type0, \
                type1  \
            >::Enqueue(enqueue_func, hal_queue, \
                       arg0.Get(),  \
                       arg1.Get()); \
        } \
    private: \
        DEFINE_HAL_COMMAND_FIELD(type0, arg0, deref0); \
        DEFINE_HAL_COMMAND_FIELD(type1, arg1, deref1); \
    };

#define DEFINE_HAL_COMMAND3(class_type, base_type,    \
                            ret_type, halq_type,      \
                            halq_first, enqueue_func, \
                            type0, arg0, deref0, \
                            type1, arg1, deref1, \
                            type2, arg2, deref2) \
    class class_type : public base_type \
    { \
    public: \
        class_type(type0 arg0, \
                   type1 arg1, \
                   type2 arg2) \
            : arg0(arg0) \
            , arg1(arg1) \
            , arg2(arg2) {} \
        virtual ~class_type() = default; \
        virtual ret_type Enqueue(halq_type hal_queue) override \
        { \
            return EnqueueHalCommand< \
                halq_first, \
                ret_type, decltype(enqueue_func), halq_type, \
                type0, \
                type1, \
                type2  \
            >::Enqueue(enqueue_func, hal_queue, \
                       arg0.Get(),  \
                       arg1.Get(),  \
                       arg2.Get()); \
        } \
    private: \
        DEFINE_HAL_COMMAND_FIELD(type0, arg0, deref0); \
        DEFINE_HAL_COMMAND_FIELD(type1, arg1, deref1); \
        DEFINE_HAL_COMMAND_FIELD(type2, arg2, deref2); \
    };

#define DEFINE_HAL_COMMAND4(class_type, base_type,    \
                            ret_type, halq_type,      \
                            halq_first, enqueue_func, \
                            type0, arg0, deref0, \
                            type1, arg1, deref1, \
                            type2, arg2, deref2, \
                            type3, arg3, deref3) \
    class class_type : public base_type \
    { \
    public: \
        class_type(type0 arg0, \
                   type1 arg1, \
                   type2 arg2, \
                   type3 arg3) \
            : arg0(arg0) \
            , arg1(arg1) \
            , arg2(arg2) \
            , arg3(arg3) {} \
        virtual ~class_type() = default; \
        virtual ret_type Enqueue(halq_type hal_queue) override \
        { \
            return EnqueueHalCommand< \
                halq_first, \
                ret_type, decltype(enqueue_func), halq_type, \
                type0, \
                type1, \
                type2, \
                type3  \
            >::Enqueue(enqueue_func, hal_queue, \
                       arg0.Get(),  \
                       arg1.Get(),  \
                       arg2.Get(),  \
                       arg3.Get()); \
        } \
    private: \
        DEFINE_HAL_COMMAND_FIELD(type0, arg0, deref0); \
        DEFINE_HAL_COMMAND_FIELD(type1, arg1, deref1); \
        DEFINE_HAL_COMMAND_FIELD(type2, arg2, deref2); \
        DEFINE_HAL_COMMAND_FIELD(type3, arg3, deref3); \
    };

#define DEFINE_HAL_COMMAND5(class_type, base_type,    \
                            ret_type, halq_type,      \
                            halq_first, enqueue_func, \
                            type0, arg0, deref0, \
                            type1, arg1, deref1, \
                            type2, arg2, deref2, \
                            type3, arg3, deref3, \
                            type4, arg4, deref4) \
    class class_type : public base_type \
    { \
    public: \
        class_type(type0 arg0, \
                   type1 arg1, \
                   type2 arg2, \
                   type3 arg3, \
                   type4 arg4) \
            : arg0(arg0) \
            , arg1(arg1) \
            , arg2(arg2) \
            , arg3(arg3) \
            , arg4(arg4) {} \
        virtual ~class_type() = default; \
        virtual ret_type Enqueue(halq_type hal_queue) override \
        { \
            return EnqueueHalCommand< \
                halq_first, \
                ret_type, decltype(enqueue_func), halq_type, \
                type0, \
                type1, \
                type2, \
                type3, \
                type4  \
            >::Enqueue(enqueue_func, hal_queue, \
                       arg0.Get(),  \
                       arg1.Get(),  \
                       arg2.Get(),  \
                       arg3.Get(),  \
                       arg4.Get()); \
        } \
    private: \
        DEFINE_HAL_COMMAND_FIELD(type0, arg0, deref0); \
        DEFINE_HAL_COMMAND_FIELD(type1, arg1, deref1); \
        DEFINE_HAL_COMMAND_FIELD(type2, arg2, deref2); \
        DEFINE_HAL_COMMAND_FIELD(type3, arg3, deref3); \
        DEFINE_HAL_COMMAND_FIELD(type4, arg4, deref4); \
    };

#define DEFINE_HAL_COMMAND6(class_type, base_type,    \
                            ret_type, halq_type,      \
                            halq_first, enqueue_func, \
                            type0, arg0, deref0, \
                            type1, arg1, deref1, \
                            type2, arg2, deref2, \
                            type3, arg3, deref3, \
                            type4, arg4, deref4, \
                            type5, arg5, deref5) \
    class class_type : public base_type \
    { \
    public: \
        class_type(type0 arg0, \
                   type1 arg1, \
                   type2 arg2, \
                   type3 arg3, \
                   type4 arg4, \
                   type5 arg5) \
            : arg0(arg0) \
            , arg1(arg1) \
            , arg2(arg2) \
            , arg3(arg3) \
            , arg4(arg4) \
            , arg5(arg5) {} \
        virtual ~class_type() = default; \
        virtual ret_type Enqueue(halq_type hal_queue) override \
        { \
            return EnqueueHalCommand< \
                halq_first, \
                ret_type, decltype(enqueue_func), halq_type, \
                type0, \
                type1, \
                type2, \
                type3, \
                type4, \
                type5  \
            >::Enqueue(enqueue_func, hal_queue, \
                       arg0.Get(),  \
                       arg1.Get(),  \
                       arg2.Get(),  \
                       arg3.Get(),  \
                       arg4.Get(),  \
                       arg5.Get()); \
        } \
    private: \
        DEFINE_HAL_COMMAND_FIELD(type0, arg0, deref0); \
        DEFINE_HAL_COMMAND_FIELD(type1, arg1, deref1); \
        DEFINE_HAL_COMMAND_FIELD(type2, arg2, deref2); \
        DEFINE_HAL_COMMAND_FIELD(type3, arg3, deref3); \
        DEFINE_HAL_COMMAND_FIELD(type4, arg4, deref4); \
        DEFINE_HAL_COMMAND_FIELD(type5, arg5, deref5); \
    };

#define DEFINE_HAL_COMMAND7(class_type, base_type,    \
                            ret_type, halq_type,      \
                            halq_first, enqueue_func, \
                            type0, arg0, deref0, \
                            type1, arg1, deref1, \
                            type2, arg2, deref2, \
                            type3, arg3, deref3, \
                            type4, arg4, deref4, \
                            type5, arg5, deref5, \
                            type6, arg6, deref6) \
    class class_type : public base_type \
    { \
    public: \
        class_type(type0 arg0, \
                   type1 arg1, \
                   type2 arg2, \
                   type3 arg3, \
                   type4 arg4, \
                   type5 arg5, \
                   type6 arg6) \
            : arg0(arg0) \
            , arg1(arg1) \
            , arg2(arg2) \
            , arg3(arg3) \
            , arg4(arg4) \
            , arg5(arg5) \
            , arg6(arg6) {} \
        virtual ~class_type() = default; \
        virtual ret_type Enqueue(halq_type hal_queue) override \
        { \
            return EnqueueHalCommand< \
                halq_first, \
                ret_type, decltype(enqueue_func), halq_type, \
                type0, \
                type1, \
                type2, \
                type3, \
                type4, \
                type5, \
                type6  \
            >::Enqueue(enqueue_func, hal_queue, \
                       arg0.Get(),  \
                       arg1.Get(),  \
                       arg2.Get(),  \
                       arg3.Get(),  \
                       arg4.Get(),  \
                       arg5.Get(),  \
                       arg6.Get()); \
        } \
    private: \
        DEFINE_HAL_COMMAND_FIELD(type0, arg0, deref0); \
        DEFINE_HAL_COMMAND_FIELD(type1, arg1, deref1); \
        DEFINE_HAL_COMMAND_FIELD(type2, arg2, deref2); \
        DEFINE_HAL_COMMAND_FIELD(type3, arg3, deref3); \
        DEFINE_HAL_COMMAND_FIELD(type4, arg4, deref4); \
        DEFINE_HAL_COMMAND_FIELD(type5, arg5, deref5); \
        DEFINE_HAL_COMMAND_FIELD(type6, arg6, deref6); \
    };

#define DEFINE_HAL_COMMAND8(class_type, base_type,    \
                            ret_type, halq_type,      \
                            halq_first, enqueue_func, \
                            type0, arg0, deref0, \
                            type1, arg1, deref1, \
                            type2, arg2, deref2, \
                            type3, arg3, deref3, \
                            type4, arg4, deref4, \
                            type5, arg5, deref5, \
                            type6, arg6, deref6, \
                            type7, arg7, deref7) \
    class class_type : public base_type \
    { \
    public: \
        class_type(type0 arg0, \
                   type1 arg1, \
                   type2 arg2, \
                   type3 arg3, \
                   type4 arg4, \
                   type5 arg5, \
                   type6 arg6, \
                   type7 arg7) \
            : arg0(arg0) \
            , arg1(arg1) \
            , arg2(arg2) \
            , arg3(arg3) \
            , arg4(arg4) \
            , arg5(arg5) \
            , arg6(arg6) \
            , arg7(arg7) {} \
        virtual ~class_type() = default; \
        virtual ret_type Enqueue(halq_type hal_queue) override \
        { \
            return EnqueueHalCommand< \
                halq_first, \
                ret_type, decltype(enqueue_func), halq_type, \
                type0, \
                type1, \
                type2, \
                type3, \
                type4, \
                type5, \
                type6, \
                type7  \
            >::Enqueue(enqueue_func, hal_queue, \
                       arg0.Get(),  \
                       arg1.Get(),  \
                       arg2.Get(),  \
                       arg3.Get(),  \
                       arg4.Get(),  \
                       arg5.Get(),  \
                       arg6.Get(),  \
                       arg7.Get()); \
        } \
    private: \
        DEFINE_HAL_COMMAND_FIELD(type0, arg0, deref0); \
        DEFINE_HAL_COMMAND_FIELD(type1, arg1, deref1); \
        DEFINE_HAL_COMMAND_FIELD(type2, arg2, deref2); \
        DEFINE_HAL_COMMAND_FIELD(type3, arg3, deref3); \
        DEFINE_HAL_COMMAND_FIELD(type4, arg4, deref4); \
        DEFINE_HAL_COMMAND_FIELD(type5, arg5, deref5); \
        DEFINE_HAL_COMMAND_FIELD(type6, arg6, deref6); \
        DEFINE_HAL_COMMAND_FIELD(type7, arg7, deref7); \
    };
