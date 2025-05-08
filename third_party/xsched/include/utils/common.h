#pragma once

#include <cstdint>
#include <unistd.h>
#include <type_traits>
#include <sys/syscall.h>

#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#define STATIC_CLASS(TypeName) \
    TypeName() = default; \
    ~TypeName() = default; \
    TypeName(const TypeName &) = delete; \
    void operator=(const TypeName &) = delete;

#define UNFOLD(...) __VA_ARGS__
#define UNUSED(expr) do { (void)(expr); } while (0)

#define ROUND_UP(X, ALIGN) (((X) - 1) / (ALIGN) + 1) * (ALIGN)

#define EXPORT_C_FUNC   extern "C" __attribute__((visibility("default")))
#define EXPORT_CXX_FUNC __attribute__((visibility("default")))

#if defined(__i386__)
#define ARCH_STR "x86"
#elif defined(__x86_64__) || defined(_M_X64)
#define ARCH_STR "x86_64"
#elif defined(__arm__)
#define ARCH_STR "arm"
#elif defined(__aarch64__)
#define ARCH_STR "aarch64"
#endif

typedef int32_t TID;
typedef pid_t   PID;

inline TID GetThreadId()
{
    static thread_local TID tid = syscall(SYS_gettid);
    return tid;
}

inline PID GetProcessId()
{
    static PID pid = getpid();
    return pid;
}

inline int OpenPidFd(PID pid, unsigned int flags)
{
	return syscall(SYS_pidfd_open, (pid_t)pid, flags);
}

template <typename T1, typename T2>
struct TypeAlmostSame
{
    static constexpr bool value = std::is_same<const T1&, const T2&>::value;
};

// Get the first argument type of a function
template <typename T>
struct FirstArg;

template <typename R, typename A, typename... Args>
struct FirstArg<R(A, Args...)>
{
    using type = A;
};

template <typename T>
using FirstArgType = typename FirstArg<T>::type;

#define FIRST_ARG_TYPE_EQUAL(func, type) \
    TypeAlmostSame<FirstArgType<decltype(func)>, type>::value
