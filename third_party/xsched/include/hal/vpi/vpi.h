#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VPIStreamImpl *VPIStream;
typedef struct VPIEventImpl *VPIEvent;
typedef struct VPIImageImpl *VPIImage;
typedef struct VPIPayloadImpl *VPIPayload;

/**
 * Status codes.
 */
typedef enum
{
    VPI_SUCCESS = 0,                /**< Operation completed successfully. */
    VPI_ERROR_NOT_IMPLEMENTED,      /**< Operation isn't implemented. */
    VPI_ERROR_INVALID_ARGUMENT,     /**< Invalid argument, either wrong range or value not accepted. */
    VPI_ERROR_INVALID_IMAGE_FORMAT, /**< Image type not accepted. */
    VPI_ERROR_INVALID_ARRAY_TYPE,   /**< Array type not accepted. */
    VPI_ERROR_INVALID_PAYLOAD_TYPE, /**< Payload not created for this algorithm. */
    VPI_ERROR_INVALID_OPERATION,    /**< Operation isn't valid in this context. */
    VPI_ERROR_INVALID_CONTEXT,      /**< Context is invalid or is already destroyed. */
    VPI_ERROR_DEVICE,               /**< Device backend error. */
    VPI_ERROR_NOT_READY,            /**< Operation not completed yet, try again later. */
    VPI_ERROR_BUFFER_LOCKED,        /**< Invalid operation on a locked buffer. */
    VPI_ERROR_OUT_OF_MEMORY,        /**< Not enough free memory to allocate object. */
    VPI_ERROR_INTERNAL              /**< Internal, non specific error. */
} VPIStatus;

/** Represents how the image data is stored. */
typedef enum
{
    /** Invalid buffer type.
     *  This is commonly used to inform that no buffer type was selected. */
    VPI_IMAGE_BUFFER_INVALID,

    /** Host-accessible with planes in pitch-linear memory layout. */
    VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,

    /** CUDA-accessible with planes in pitch-linear memory layout. */
    VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR,

    /** Buffer stored in a cudaArray_t.
     * Please consult <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays">cudaArray_t</a>
     * for more information. */
    VPI_IMAGE_BUFFER_CUDA_ARRAY,

    /** EGLImage.
     * Please consult <a href="https://www.khronos.org/registry/EGL/extensions/KHR/EGL_KHR_image_base.txt">EGLImageKHR</a>
     * for more information. */
    VPI_IMAGE_BUFFER_EGLIMAGE,

    /** NvBuffer.
     * Please consult <a href="https://docs.nvidia.com/jetson/l4t-multimedia/group__ee__nvbuffering__group.html">NvBuffer</a>
     * for more information. */
    VPI_IMAGE_BUFFER_NVBUFFER,

} VPIImageBufferType;

/**
 * Pre-defined image formats.
 * An image format defines how image pixels are interpreted.
 * Each image format is defined by the following components:
 * - \ref VPIColorModel
 * - \ref VPIColorSpec
 * - \ref VPIChromaSubsampling method (when applicable)
 * - \ref VPIMemLayout
 * - \ref VPIDataType
 * - \ref VPISwizzle
 * - Number of planes
 * - Format packing of each plane.
 *
 * These pre-defined formats are guaranteed to work with algorithms that explicitly support them.
 * Image formats can also be user-defined using the vpiMakeImageFormat family of functions.
 *
 * Using user-defined image formats with algorithms can lead to undefined behavior (segfaults, etc),
 * but usually it works as expected. Result of algorithms using these image formats must be checked
 * for correctness, as it's not guaranteed that they will work.
 */
typedef uint64_t VPIImageFormat;

/**
 * Pre-defined pixel types.
 * Pixel types defines the geometry of pixels in a image plane without taking into account what the value represents.
 * For example, a \ref VPI_IMAGE_FORMAT_NV12 is composed of 2 planes, each one with the following pixel types:
 * + \ref VPI_PIXEL_TYPE_U8 representing pixels as 8-bit unsigned values.
 * + \ref VPI_PIXEL_TYPE_2U8 representing pixels as two interleaved 32-bit floating-point values.
 */
typedef uint64_t VPIPixelType;

/** Represents one image plane in pitch-linear layout. */
typedef struct VPIImagePlanePitchLinearRec
{
    /** Type of each pixel within this plane.
     *  If it is \ref VPI_PIXEL_TYPE_INVALID, it will be inferred from \ref VPIImageBufferPitchLinear::format. */
    VPIPixelType pixelType;

    /** Width of this plane in pixels.
     *  + It must be >= 1. */
    int32_t width;

    /** Height of this plane in pixels.
     *  + It must be >= 1. */
    int32_t height;

    /** Difference in bytes of beginning of one row and the beginning of the previous.
         This is used to address every row (and ultimately every pixel) in the plane.
         @code
            T *pix_addr = (T *)((uint8_t *)data + pitchBytes*height)+width;
         @endcode
         where T is the C type related to pixelType.

         + It must be at least `(width * \ref vpiPixelTypeGetBitsPerPixel(pixelType) + 7)/8`.
    */
    int32_t pitchBytes;

    /** Pointer to the first row of this plane.
         This points to the actual data represented by this plane.
         Depending on how the plane is used, the pointer might be
         addressing a GPU memory or host memory. Care is needed to
         know when it is allowed to dereference this memory. */
    void *data;

} VPIImagePlanePitchLinear;

/** Maximum number of data planes an image can have. */
#define VPI_MAX_PLANE_COUNT (6)

/** Stores the image plane contents. */
typedef struct VPIImageBufferPitchLinearRec
{
    /** Image format. */
    VPIImageFormat format;

    /** Number of planes.
     *  + Must be >= 1. */
    int32_t numPlanes;

    /** Data of all image planes in pitch-linear layout.
     *  + Only the first \ref numPlanes elements must have valid data. */
    VPIImagePlanePitchLinear planes[VPI_MAX_PLANE_COUNT];

} VPIImageBufferPitchLinear;

typedef void *EGLImageKHR;
typedef struct cudaArray *cudaArray_t;

/** Represents the available methods to access image contents.
 * The correct method depends on \ref VPIImageData::bufferType. */
typedef union VPIImageBufferRec
{
    /** Image stored in pitch-linear layout.
     * To be used when \ref VPIImageData::bufferType is:
     * - \ref VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR
     * - \ref VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR
     */
    VPIImageBufferPitchLinear pitch;

    /** Image stored in a `cudaArray_t`.
     * To be used when \ref VPIImageData::bufferType is:
     * - \ref VPI_IMAGE_BUFFER_CUDA_ARRAY
     */
    cudaArray_t cudaarray;

    /** Image stored as an EGLImageKHR.
     * To be used when \ref VPIImageData::bufferType is:
     * - \ref VPI_IMAGE_BUFFER_EGLIMAGE
     */
    EGLImageKHR egl;

    /** Image stored as an NvBuffer file descriptor.
     * To be used when \ref VPIImageData::bufferType is:
     * - \ref VPI_IMAGE_BUFFER_NVBUFFER
     */
    int fd;

} VPIImageBuffer;

/** Stores information about image characteristics and content. */
typedef struct VPIImageDataRec
{
    /** Type of image buffer.
     *  It defines which member of the \ref VPIImageBuffer tagged union that
     *  must be used to access the image contents. */
    VPIImageBufferType bufferType;

    /** Stores the image contents. */
    VPIImageBuffer buffer;

} VPIImageData;

// Utilities ================================

#define VPI_DETAIL_SET_BITFIELD(value, offset, size) (((uint64_t)(value) & ((1ULL << (size)) - 1)) << (offset))
#define VPI_DETAIL_GET_BITFIELD(value, offset, size) (((uint64_t)(value) >> (offset)) & ((1ULL << (size)) - 1))

#define VPI_DETAIL_ENCODE_BPP(bpp)             \
    ((bpp) <= 8 ? 0                            \
                : ((bpp) <= 32 ? (bpp) / 8 - 1 \
                               : ((bpp) <= 64 ? (bpp) / 16 + 1 : ((bpp) <= 128 ? (bpp) / 32 + 3 : (bpp) / 64 + 5))))

#define VPI_DETAIL_BPP_NCH(bpp, chcount)                                                                      \
    (VPI_DETAIL_SET_BITFIELD(VPI_DETAIL_ENCODE_BPP(bpp), 6, 4) | VPI_DETAIL_SET_BITFIELD((chcount)-1, 4, 2) | \
     VPI_DETAIL_SET_BITFIELD((bpp) <= 2 ? (bpp) : ((bpp) == 4 ? 3 : ((bpp) == 8 ? 4 : 0)), 0, 4))

#define VPI_DETAIL_MAKE_SWIZZLE(x, y, z, w)                                                                   \
    (VPI_DETAIL_SET_BITFIELD(x, 0, 3) | VPI_DETAIL_SET_BITFIELD(y, 3, 3) | VPI_DETAIL_SET_BITFIELD(z, 6, 3) | \
     VPI_DETAIL_SET_BITFIELD(w, 9, 3))

#define VPI_DETAIL_MAKE_SWZL(x, y, z, w) \
    VPI_DETAIL_MAKE_SWIZZLE(VPI_CHANNEL_##x, VPI_CHANNEL_##y, VPI_CHANNEL_##z, VPI_CHANNEL_##w)

#define VPI_DETAIL_DEF_SWIZZLE_ENUM(x, y, z, w) VPI_SWIZZLE_##x##y##z##w = VPI_DETAIL_MAKE_SWZL(x, y, z, w)

#define VPI_DETAIL_ADJUST_BPP_ENCODING(PACK, BPP, PACKLEN) \
    ((PACKLEN) == 0 && (BPP) == 0 && (PACK) == 4 ? (uint64_t)-1 : (BPP))

#define VPI_DETAIL_ENCODE_PACKING(P, CHLEN, PACKLEN, BPPLEN)                                                          \
    (VPI_DETAIL_SET_BITFIELD(                                                                                         \
         VPI_DETAIL_ADJUST_BPP_ENCODING(VPI_DETAIL_GET_BITFIELD(P, 0, 4), VPI_DETAIL_GET_BITFIELD(P, 6, 4), PACKLEN), \
         (PACKLEN) + (CHLEN), BPPLEN) |                                                                               \
     VPI_DETAIL_SET_BITFIELD(VPI_DETAIL_GET_BITFIELD(P, 4, 2), PACKLEN, CHLEN) |                                      \
     VPI_DETAIL_SET_BITFIELD(VPI_DETAIL_GET_BITFIELD(P, 0, 4), 0, PACKLEN))

#define VPI_DETAIL_EXTRACT_PACKING_CHANNELS(Packing) (VPI_DETAIL_GET_BITFIELD(Packing, 4, 2) + 1)

/* clang-format off */
#define VPI_DETAIL_MAKE_FMTTYPE(ColorModel, ColorSpecOrRawPattern, Subsampling, MemLayout, DataType, Swizzle, \
                               Packing0, Packing1, Packing2, Packing3)                                        \
    (                                                              \
        VPI_DETAIL_SET_BITFIELD(DataType, 61, 3) | VPI_DETAIL_SET_BITFIELD(Swizzle, 0, 4 * 3) |           \
        VPI_DETAIL_SET_BITFIELD(MemLayout, 12, 3) | \
        ((ColorModel) == VPI_COLOR_MODEL_YCbCr \
            ? VPI_DETAIL_SET_BITFIELD(ColorSpecOrRawPattern, 20, 15) | VPI_DETAIL_SET_BITFIELD(Subsampling, 17, 3) \
            : ((ColorModel) == VPI_COLOR_MODEL_UNDEFINED \
                ? VPI_DETAIL_SET_BITFIELD((1U<<19)-1, 16, 19) \
                : (VPI_DETAIL_SET_BITFIELD(1,16,1) | \
                     ((ColorModel)-2 < 0x7 \
                       ? VPI_DETAIL_SET_BITFIELD(ColorSpecOrRawPattern, 20, 15) \
                            | VPI_DETAIL_SET_BITFIELD((ColorModel)-2, 17, 3) \
                       : (VPI_DETAIL_SET_BITFIELD(0x7, 17, 3) | \
                            ((ColorModel) == VPI_COLOR_MODEL_RAW \
                              ? VPI_DETAIL_SET_BITFIELD(ColorSpecOrRawPattern, 21, 6) \
                              : (VPI_DETAIL_SET_BITFIELD(1, 20, 1) | VPI_DETAIL_SET_BITFIELD((ColorModel)-(7+2+1), 21, 6)) \
                            ) \
                         ) \
                     ) \
                  ) \
              ) \
        ) | \
        VPI_DETAIL_SET_BITFIELD(VPI_DETAIL_ENCODE_PACKING(Packing0, 2, 3, 4), 35, 9) |                           \
        VPI_DETAIL_SET_BITFIELD(VPI_DETAIL_ENCODE_PACKING(Packing1, 1, 3, 3), 44, 7) |                           \
        VPI_DETAIL_SET_BITFIELD(VPI_DETAIL_ENCODE_PACKING(Packing2, 1, 3, 3), 51, 7) |                           \
        VPI_DETAIL_SET_BITFIELD(VPI_DETAIL_ENCODE_PACKING(Packing3, 0, 0, 3), 58, 3))
/* clang-format on */

#define VPI_DETAIL_MAKE_FORMAT(ColorModel, ColorSpecOrRawPattern, Subsampling, MemLayout, DataType, Swizzle, Packing0, \
                               Packing1, Packing2, Packing3)                                                           \
    ((VPIImageFormat)VPI_DETAIL_MAKE_FMTTYPE(ColorModel, ColorSpecOrRawPattern, Subsampling, MemLayout, DataType,      \
                                             Swizzle, Packing0, Packing1, Packing2, Packing3))

#define VPI_DETAIL_MAKE_FMT(ColorModel, ColorSpec, CSS, MemLayout, DataType, Swizzle, P0, P1, P2, P3)   \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_##ColorModel, VPI_COLOR_SPEC_##ColorSpec, VPI_CSS_##CSS,     \
                           VPI_MEM_LAYOUT_##MemLayout, VPI_DATA_TYPE_##DataType, VPI_SWIZZLE_##Swizzle, \
                           VPI_PACKING_##P0, VPI_PACKING_##P1, VPI_PACKING_##P2, VPI_PACKING_##P3)

// MAKE_COLOR ================================================

// Full arg name

#define VPI_DETAIL_MAKE_COLOR_FORMAT1(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, P0) \
    VPI_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, VPI_CSS_NONE, MemLayout, DataType, Swizzle, P0, 0, 0, 0)

#define VPI_DETAIL_MAKE_COLOR_FORMAT2(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, P0, P1) \
    VPI_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, VPI_CSS_NONE, MemLayout, DataType, Swizzle, P0, P1, 0, 0)

#define VPI_DETAIL_MAKE_COLOR_FORMAT3(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, P0, P1, P2) \
    VPI_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, VPI_CSS_NONE, MemLayout, DataType, Swizzle, P0, P1, P2, 0)

#define VPI_DETAIL_MAKE_COLOR_FORMAT4(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, P0, P1, P2, P3) \
    VPI_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, VPI_CSS_NONE, MemLayout, DataType, Swizzle, P0, P1, P2, P3)

#define VPI_DETAIL_MAKE_COLOR_FORMAT(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, PlaneCount, ...) \
    VPI_DETAIL_MAKE_COLOR_FORMAT##PlaneCount(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, __VA_ARGS__)

// Abbreviated

#define VPI_DETAIL_MAKE_COLOR_FMT1(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, P0) \
    VPI_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataType, Swizzle, P0, 0, 0, 0)

#define VPI_DETAIL_MAKE_COLOR_FMT2(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, P0, P1) \
    VPI_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataType, Swizzle, P0, P1, 0, 0)

#define VPI_DETAIL_MAKE_COLOR_FMT3(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, P0, P1, P3) \
    VPI_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataType, Swizzle, P0, P1, P3, 0)

#define VPI_DETAIL_MAKE_COLOR_FMT4(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, P0, P1, P3, P4) \
    VPI_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataType, Swizzle, P0, P1, P3, P4)

#define VPI_DETAIL_MAKE_COLOR_FMT(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, PlaneCount, ...) \
    VPI_DETAIL_MAKE_COLOR_FMT##PlaneCount(ColorModel, ColorSpec, MemLayout, DataType, Swizzle, __VA_ARGS__)

// MAKE_PIXEL =========================

// Full arg name

#define VPI_DETAIL_MAKE_PIXEL_TYPE(MemLayout, DataType, Packing)                                                      \
    ((VPIPixelType)VPI_DETAIL_MAKE_FMTTYPE(                                                                           \
        VPI_COLOR_MODEL_UNDEFINED, VPI_COLOR_SPEC_UNDEFINED, VPI_CSS_NONE, MemLayout, DataType,                       \
        VPI_DETAIL_MAKE_SWIZZLE(VPI_CHANNEL_X, VPI_DETAIL_EXTRACT_PACKING_CHANNELS(Packing) >= 2 ? VPI_CHANNEL_Y : 0, \
                                VPI_DETAIL_EXTRACT_PACKING_CHANNELS(Packing) >= 3 ? VPI_CHANNEL_Z : 0,                \
                                VPI_DETAIL_EXTRACT_PACKING_CHANNELS(Packing) >= 4 ? VPI_CHANNEL_W : 0),               \
        Packing, 0, 0, 0))

// Abbreviated

#define VPI_DETAIL_MAKE_PIX_TYPE(MemLayout, DataType, Packing) \
    VPI_DETAIL_MAKE_PIXEL_TYPE(VPI_MEM_LAYOUT_##MemLayout, VPI_DATA_TYPE_##DataType, VPI_PACKING_##Packing)

// MAKE_NONCOLOR ==================================

// Full arg name

#define VPI_DETAIL_MAKE_NONCOLOR_FORMAT1(MemLayout, DataType, Swizzle, P0)                                         \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_UNDEFINED, VPI_COLOR_SPEC_UNDEFINED, VPI_CSS_NONE, MemLayout, DataType, \
                           Swizzle, P0, 0, 0, 0)

#define VPI_DETAIL_MAKE_NONCOLOR_FORMAT2(MemLayout, DataType, Swizzle, P0, P1)                                     \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_UNDEFINED, VPI_COLOR_SPEC_UNDEFINED, VPI_CSS_NONE, MemLayout, DataType, \
                           Swizzle, P0, P1, 0, 0)

#define VPI_DETAIL_MAKE_NONCOLOR_FORMAT3(MemLayout, DataType, Swizzle, P0, P1, P2)                                 \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_UNDEFINED, VPI_COLOR_SPEC_UNDEFINED, VPI_CSS_NONE, MemLayout, DataType, \
                           Swizzle, P0, P1, P2, 0)

#define VPI_DETAIL_MAKE_NONCOLOR_FORMAT4(MemLayout, DataType, Swizzle, P0, P1, P2, P3)                             \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_UNDEFINED, VPI_COLOR_SPEC_UNDEFINED, VPI_CSS_NONE, MemLayout, DataType, \
                           Swizzle, P0, P1, P2, P3)

#define VPI_DETAIL_MAKE_NONCOLOR_FORMAT(MemLayout, DataType, Swizzle, PlaneCount, ...) \
    VPI_DETAIL_MAKE_NONCOLOR_FORMAT##PlaneCount(MemLayout, DataType, Swizzle, __VA_ARGS__)

// Abbreviated

#define VPI_DETAIL_MAKE_NONCOLOR_FMT1(MemLayout, DataType, Swizzle, P0) \
    VPI_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataType, Swizzle, P0, 0, 0, 0)

#define VPI_DETAIL_MAKE_NONCOLOR_FMT2(MemLayout, DataType, Swizzle, P0, P1) \
    VPI_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataType, Swizzle, P0, P1, 0, 0)

#define VPI_DETAIL_MAKE_NONCOLOR_FMT3(MemLayout, DataType, Swizzle, P0, P1, P2) \
    VPI_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataType, Swizzle, P0, P1, P2, 0)

#define VPI_DETAIL_MAKE_NONCOLOR_FMT4(MemLayout, DataType, Swizzle, P0, P1, P2, P3) \
    VPI_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataType, Swizzle, P0, P1, P2, P3)

#define VPI_DETAIL_MAKE_NONCOLOR_FMT(MemLayout, DataType, Swizzle, PlaneCount, ...) \
    VPI_DETAIL_MAKE_NONCOLOR_FMT##PlaneCount(MemLayout, DataType, Swizzle, __VA_ARGS__)

// MAKE_RAW =============================================

// Full arg name

#define VPI_DETAIL_MAKE_RAW_FORMAT1(RawPattern, MemLayout, DataType, Swizzle, P0) \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_RAW, RawPattern, VPI_CSS_NONE, MemLayout, DataType, Swizzle, P0, 0, 0, 0)

#define VPI_DETAIL_MAKE_RAW_FORMAT2(RawPattern, MemLayout, DataType, Swizzle, P0, P1) \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_RAW, RawPattern, VPI_CSS_NONE, MemLayout, DataType, Swizzle, P0, P1, 0, 0)

#define VPI_DETAIL_MAKE_RAW_FORMAT3(RawPattern, MemLayout, DataType, Swizzle, P0, P1, P2) \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_RAW, RawPattern, VPI_CSS_NONE, MemLayout, DataType, Swizzle, P0, P1, P2, 0)

#define VPI_DETAIL_MAKE_RAW_FORMAT4(RawPattern, MemLayout, DataType, Swizzle, P0, P1, P2, P3) \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_RAW, RawPattern, VPI_CSS_NONE, MemLayout, DataType, Swizzle, P0, P1, P2, P3)

#define VPI_DETAIL_MAKE_RAW_FORMAT(RawPattern, MemLayout, DataType, Swizzle, PlaneCount, ...) \
    VPI_DETAIL_MAKE_RAW_FORMAT##PlaneCount(RawPattern, MemLayout, DataType, Swizzle, __VA_ARGS__)

// Abbreviated

#define VPI_DETAIL_MAKE_RAW_FMT1(RawPattern, MemLayout, DataType, Swizzle, P0)                              \
    VPI_DETAIL_MAKE_RAW_FORMAT1(VPI_RAW_##RawPattern, VPI_MEM_LAYOUT_##MemLayout, VPI_DATA_TYPE_##DataType, \
                                VPI_SWIZZLE_##Swizzle, VPI_PACKING_##P0)

#define VPI_DETAIL_MAKE_RAW_FMT2(RawPattern, MemLayout, DataType, Swizzle, P0, P1)                          \
    VPI_DETAIL_MAKE_RAW_FORMAT2(VPI_RAW_##RawPattern, VPI_MEM_LAYOUT_##MemLayout, VPI_DATA_TYPE_##DataType, \
                                VPI_SWIZZLE_##Swizzle, VPI_PACKING_##P0, VPI_PACKING_##P1)

#define VPI_DETAIL_MAKE_RAW_FMT3(RawPattern, MemLayout, DataType, Swizzle, P0, P1, P2)                      \
    VPI_DETAIL_MAKE_RAW_FORMAT3(VPI_RAW_##RawPattern, VPI_MEM_LAYOUT_##MemLayout, VPI_DATA_TYPE_##DataType, \
                                VPI_SWIZZLE_##Swizzle, VPI_PACKING_##P0, VPI_PACKING_##P1, VPI_PACKING_##P2)

#define VPI_DETAIL_MAKE_RAW_FMT4(RawPattern, MemLayout, DataType, Swizzle, P0, P1, P2, P3)                   \
    VPI_DETAIL_MAKE_RAW_FORMAT4(VPI_RAW_##RawPattern, VPI_MEM_LAYOUT_##MemLayout, VPI_DATA_TYPE_##DataType,  \
                                VPI_SWIZZLE_##Swizzle, VPI_PACKING_##P0, VPI_PACKING_##P1, VPI_PACKING_##P2, \
                                VPI_PACKING_##P3)

#define VPI_DETAIL_MAKE_RAW_FMT(RawPattern, MemLayout, DataType, Swizzle, PlaneCount, ...) \
    VPI_DETAIL_MAKE_RAW_FMT##PlaneCount(RawPattern, MemLayout, DataType, Swizzle, __VA_ARGS__)

// MAKE_YCbCr ===============================================

// Full arg name

#define VPI_DETAIL_MAKE_YCbCr_FORMAT1(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0) \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, 0, 0, 0)

#define VPI_DETAIL_MAKE_YCbCr_FORMAT2(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1)                  \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, 0, \
                           0)

#define VPI_DETAIL_MAKE_YCbCr_FORMAT3(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, P2)           \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, \
                           P2, 0)

#define VPI_DETAIL_MAKE_YCbCr_FORMAT4(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, P2, P3)       \
    VPI_DETAIL_MAKE_FORMAT(VPI_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, \
                           P2, P3)

#define VPI_DETAIL_MAKE_YCbCr_FORMAT(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, PlaneCount, ...) \
    VPI_DETAIL_MAKE_YCbCr_FORMAT##PlaneCount(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, __VA_ARGS__)

// Abbreviated

#define VPI_DETAIL_MAKE_YCbCr_FMT1(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0) \
    VPI_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, 0, 0, 0)

#define VPI_DETAIL_MAKE_YCbCr_FMT2(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1) \
    VPI_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, 0, 0)

#define VPI_DETAIL_MAKE_YCbCr_FMT3(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, P2) \
    VPI_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, P2, 0)

#define VPI_DETAIL_MAKE_YCbCr_FMT4(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, P2, P3) \
    VPI_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, P0, P1, P2, P3)

#define VPI_DETAIL_MAKE_YCbCr_FMT(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, PlaneCount, ...) \
    VPI_DETAIL_MAKE_YCbCr_FMT##PlaneCount(ColorSpec, ChromaSubsamp, MemLayout, DataType, Swizzle, __VA_ARGS__)

// MAKE_COLOR_SPEC --------------------------------------------

#define VPI_DETAIL_MAKE_COLOR_SPEC(CSpace, Encoding, XferFunc, Range, LocHoriz, LocVert)  \
    (VPI_DETAIL_SET_BITFIELD((CSpace), 0, 3) | VPI_DETAIL_SET_BITFIELD(XferFunc, 3, 4) |  \
     VPI_DETAIL_SET_BITFIELD(Encoding, 7, 3) | VPI_DETAIL_SET_BITFIELD(LocHoriz, 10, 2) | \
     VPI_DETAIL_SET_BITFIELD(LocVert, 12, 2) | VPI_DETAIL_SET_BITFIELD(Range, 14, 1))

#define VPI_DETAIL_MAKE_CSPC(CSpace, Encoding, XferFunc, Range, LocHoriz, LocVert)                                \
    VPI_DETAIL_MAKE_COLOR_SPEC(VPI_COLOR_##CSpace, VPI_YCbCr_##Encoding, VPI_COLOR_##XferFunc, VPI_COLOR_##Range, \
                               VPI_CHROMA_##LocHoriz, VPI_CHROMA_##LocVert)

/** Defines color models.
 * A color model gives meaning to each channel of an image format. They are specified
 * in a canonical XYZW ordering that can then be swizzled to the desired ordering. 
 */
typedef enum
{
    VPI_COLOR_MODEL_UNDEFINED = 0,     /**< Color model is undefined. */
    VPI_COLOR_MODEL_YCbCr     = 1,     /**< Luma + chroma (blue-luma, red-luma). */
    VPI_COLOR_MODEL_RGB       = 2,     /**< red, green, blue components. */
    VPI_COLOR_MODEL_RAW       = 2 + 7, /**< RAW color model, used for Bayer image formats. */
    VPI_COLOR_MODEL_XYZ,               /**< CIE XYZ tristimulus color spec. */
} VPIColorModel;

/** Defines the color primaries and the white point of a \ref VPIColorSpec. */
typedef enum
{
    VPI_COLOR_SPACE_SENSOR, /**< Color space from the sensor used to capture the image. */
    VPI_COLOR_SPACE_BT601,  /**< Color primaries from ITU-R BT.601/625 lines standard, also known as EBU 3213-E. */
    VPI_COLOR_SPACE_BT709,  /**< Color primaries from ITU-R BT.709 standard, D65 white point. */
    VPI_COLOR_SPACE_BT2020, /**< Color primaries from ITU-R BT.2020 standard, D65 white point. */
    VPI_COLOR_SPACE_DCIP3,  /**< Color primaries from DCI-P3 standard, D65 white point. */

    VPI_COLOR_SPACE_UNDEFINED = INT32_MAX, /**< Color space not defined. */
} VPIColorSpace;

/** Defines the white point associated with a \ref VPIColorSpace. */
typedef enum
{
    VPI_WHITE_POINT_D65, /**< D65 white point, K = 6504. */

    VPI_WHITE_POINT_UNDEFINED = INT32_MAX /**< White point not defined. */
} VPIWhitePoint;

/** Defines the YCbCr encoding used in a particular \ref VPIColorSpec. */
typedef enum
{
    VPI_YCbCr_ENC_UNDEFINED = 0, /**< Encoding not defined. Usually used by non-YCbCr color specs. */
    VPI_YCbCr_ENC_BT601,         /**< Encoding specified by ITU-R BT.601 standard. */
    VPI_YCbCr_ENC_BT709,         /**< Encoding specified by ITU-R BT.709 standard. */
    VPI_YCbCr_ENC_BT2020,        /**< Encoding specified by ITU-R BT.2020 standard. */
    VPI_YCbCr_ENC_BT2020c,       /**< Encoding specified by ITU-R BT.2020 with constant luminance. */
    VPI_YCbCr_ENC_SMPTE240M,     /**< Encoding specified by SMPTE 240M standard. */
} VPIYCbCrEncoding;

/** Defines the color transfer function in a particular \ref VPIColorSpec. */
typedef enum
{
    VPI_COLOR_XFER_LINEAR,    /**< Linear color transfer function. */
    VPI_COLOR_XFER_sRGB,      /**< Color transfer function specified by sRGB standard. */
    VPI_COLOR_XFER_sYCC,      /**< Color transfer function specified by sYCC standard. */
    VPI_COLOR_XFER_PQ,        /**< Perceptual quantizer color transfer function. */
    VPI_COLOR_XFER_BT709,     /**< Color transfer function specified by ITU-R BT.709 standard. */
    VPI_COLOR_XFER_BT2020,    /**< Color transfer function specified by ITU-R BT.2020 standard. */
    VPI_COLOR_XFER_SMPTE240M, /**< Color transfer function specified by SMPTE 240M standard. */
} VPIColorTransferFunction;

/** Defines the color range of a particular \ref VPIColorSpec. */
typedef enum
{
    VPI_COLOR_RANGE_FULL,   /**< Values cover the full underlying type range. */
    VPI_COLOR_RANGE_LIMITED /**< Values cover a limited range of the underlying type. */
} VPIColorRange;

/** Chroma sampling location. */
typedef enum
{
    VPI_CHROMA_LOC_EVEN   = 0, /**< Sample the chroma with even coordinate. */
    VPI_CHROMA_LOC_CENTER = 1, /**< Sample the chroma exactly between the even and odd coordinate. */
    VPI_CHROMA_LOC_ODD    = 2, /**< Sample the chroma with odd coordinate. */
    VPI_CHROMA_LOC_BOTH   = 3, /**< Sample chroma from even and odd coordinates.
                                    This is used when no sub-sampling is taking place. */
} VPIChromaLocation;

typedef enum
{
    /** Invalid color spec. This is to be used when no color spec is selected. */
    VPI_COLOR_SPEC_INVALID = INT32_MAX,

    /** Default color spec. Informs that the color spec is to be inferred. */
    VPI_COLOR_SPEC_DEFAULT          = VPI_DETAIL_MAKE_CSPC(SPACE_UNDEFINED, ENC_UNDEFINED, XFER_LINEAR, RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** No color spec defined. Used when color spec isn't relevant or is not defined.
     *  The color spec may be inferred from the context. If this isn't possible, the values for each
     *  color spec component defined below will be used. */
    VPI_COLOR_SPEC_UNDEFINED        = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_UNDEFINED, XFER_LINEAR,    RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** Color spec defining ITU-R BT.601 standard, limited range, with BT.709 chrominancies and transfer function. */
    VPI_COLOR_SPEC_BT601            = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT601,     XFER_BT709,     RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.601 standard, full range, with BT.709 chrominancies and transfer function. */
    VPI_COLOR_SPEC_BT601_ER         = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT601,     XFER_BT709,     RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.709 standard, limited range. */
    VPI_COLOR_SPEC_BT709            = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT709,     XFER_BT709,     RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.709 standard, full range. */
    VPI_COLOR_SPEC_BT709_ER         = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT709,     XFER_BT709,     RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.709 standard, limited range and linear transfer function. */
    VPI_COLOR_SPEC_BT709_LINEAR     = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT709,     XFER_LINEAR,    RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, limited range. */
    VPI_COLOR_SPEC_BT2020           = VPI_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_BT2020,    RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, full range. */
    VPI_COLOR_SPEC_BT2020_ER        = VPI_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_BT2020,    RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, limited range and linear transfer function. */
    VPI_COLOR_SPEC_BT2020_LINEAR    = VPI_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_LINEAR,    RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, limited range and perceptual quantizer transfer function. */
    VPI_COLOR_SPEC_BT2020_PQ        = VPI_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_PQ,        RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, full range and perceptual quantizer transfer function. */
    VPI_COLOR_SPEC_BT2020_PQ_ER     = VPI_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_PQ,        RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard for constant luminance, limited range. */
    VPI_COLOR_SPEC_BT2020c          = VPI_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020c,   XFER_BT2020,    RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard for constant luminance, full range. */
    VPI_COLOR_SPEC_BT2020c_ER       = VPI_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020c,   XFER_BT2020,    RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining MPEG2 standard using ITU-R BT.601 encoding. */
    VPI_COLOR_SPEC_MPEG2_BT601      = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT601,     XFER_BT709,     RANGE_FULL,    LOC_EVEN,   LOC_CENTER),

    /** Color spec defining MPEG2 standard using ITU-R BT.709 encoding. */
    VPI_COLOR_SPEC_MPEG2_BT709      = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT709,     XFER_BT709,     RANGE_FULL,    LOC_EVEN,   LOC_CENTER),

    /** Color spec defining MPEG2 standard using SMPTE 240M encoding. */
    VPI_COLOR_SPEC_MPEG2_SMPTE240M  = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_SMPTE240M, XFER_SMPTE240M, RANGE_FULL,    LOC_EVEN,   LOC_CENTER),

    /** Color spec defining sRGB standard. */
    VPI_COLOR_SPEC_sRGB             = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_UNDEFINED, XFER_sRGB,      RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** Color spec defining sYCC standard. */
    VPI_COLOR_SPEC_sYCC             = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT601,     XFER_sYCC,      RANGE_FULL,    LOC_CENTER, LOC_CENTER),

    /** Color spec defining SMPTE 240M standard, limited range. */
    VPI_COLOR_SPEC_SMPTE240M        = VPI_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_SMPTE240M, XFER_SMPTE240M, RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining Display P3 standard, with sRGB color transfer function. */
    VPI_COLOR_SPEC_DISPLAYP3        = VPI_DETAIL_MAKE_CSPC(SPACE_DCIP3,  ENC_UNDEFINED, XFER_sRGB,      RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** Color spec defining Display P3 standard, with linear color transfer function. */
    VPI_COLOR_SPEC_DISPLAYP3_LINEAR = VPI_DETAIL_MAKE_CSPC(SPACE_DCIP3,  ENC_UNDEFINED, XFER_LINEAR,    RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** Color spec used for images coming from an image sensor, right after demosaicing. */
    VPI_COLOR_SPEC_SENSOR           = VPI_DETAIL_MAKE_CSPC(SPACE_SENSOR, ENC_UNDEFINED, XFER_LINEAR,    RANGE_FULL,    LOC_BOTH,   LOC_BOTH),
} VPIColorSpec;

/** Parameters for customizing image wrapping.
 *
 * These parameters are used to customize how image wrapping will be made.
 * Make sure to call \ref vpiInitImageWrapperParams to initialize this
 * structure before updating its attributes. This guarantees that new attributes
 * added in future versions will have a suitable default value assigned.
 */
typedef struct
{
    /** Color spec to override the one defined by the \ref VPIImageData wrapper.
     * If set to \ref VPI_COLOR_SPEC_DEFAULT, infer the color spec from \ref VPIImageData,
     * i.e., no overriding will be done. */
    VPIColorSpec colorSpec;
} VPIImageWrapperParams;

/**
 * Policy used when converting between image types.
 * @ingroup VPI_ConvertImageFormat
 */
typedef enum
{
    /** Clamps input to output's type range. Overflows
        and underflows are mapped to the output type's
        maximum and minimum representable value,
        respectively. When output type is floating point,
        clamp behaves like cast. */
    VPI_CONVERSION_CLAMP = 0,

    /** Casts input to the output type. Overflows and
        underflows are handled as per C specification,
        including situations of undefined behavior. */
    VPI_CONVERSION_CAST,

    /** Invalid conversion. */
    VPI_CONVERSION_INVALID = 255,
} VPIConversionPolicy;

/**
 * Interpolation types supported by several algorithms
 * @ingroup VPI_Types
 */
typedef enum
{
    VPI_INTERP_NULL = 0,

    /** Nearest neighbor interpolation.
       \f[
          P(x,y) = \mathit{in}[\lfloor x+0.5 \rfloor, \lfloor y+0.5 \rfloor]
       \f]
     */
    VPI_INTERP_NEAREST = 1,

    /** Linear interpolation.
        Interpolation weights are defined as:
        \f{align*}{
            w_0(t)& \triangleq t-\lfloor t \rfloor \\
            w_1(t)& \triangleq 1 - w_0(t) \\
        \f}

        Bilinear-interpolated value is given by the formula below:

        \f[
            P(x,y) = \sum_{p=0}^1 \sum_{q=0}^1 \mathit{in}[\lfloor x \rfloor+p, \lfloor y \rfloor+q]w_p(x)w_q(y)
        \f]
    */
    VPI_INTERP_LINEAR = 2,

    /** Catmull-Rom cubic interpolation.
       Catmull-Rom interpolation weights with \f$A=-0.5\f$ are defined as follows:
      \f{eqnarray*}{
          w_0(t) &\triangleq& A(t+1)^3 &-& 5A(t+1)^2 &+& 8A(t+1) &-& 4A \\
          w_1(t) &\triangleq& (A+2)t^3 &-& (A+3)t^2 &\nonumber& &+& 1 \\
          w_2(t) &\triangleq& (A+2)(1-t)^3 &-& (A+3)(1-t)^2 &\nonumber& &+& 1 \\
          w_3(t) &\triangleq& \rlap{1 - w_0(t) - w_1(t) - w_2(t) }
      \f}

      Bicubic-interpolated value is given by the formula below:
      \f[
          P(x,y) = \sum_{p=-1}^2 \sum_{q=-1}^2 \mathit{in}[\lfloor x \rfloor+p, \lfloor y \rfloor+q]w_p(x)w_q(y)
      \f]
    */
    VPI_INTERP_CATMULL_ROM = 3,
} VPIInterpolationType;

/** Parameters for customizing image format conversion.
 * These parameters are used to customize how the conversion will be made.
 * Make sure to call \ref vpiInitConvertImageFormatParams to initialize this
 * structure before updating its attributes. This guarantees that new attributes
 * added in future versions will have a suitable default value assigned.
 */
typedef struct
{
    /** Conversion policy to be used.
     *  + VIC backend only supports \ref VPI_CONVERSION_CLAMP. */
    VPIConversionPolicy policy;

    /** Scaling factor.
     *  Pass 1 for no scaling. 
     *  + VIC backend doesn't support scaling, it must be 1.*/
    float scale;

    /** Offset factor. 
     *  Pass 0 for no offset.
     *  + VIC backend doesn't support offset, it must be 0. */
    float offset;

    /** Control flags.
     *  + Valid values:
     *    - 0: default, negation of all other flags.
     *    - \ref VPI_PRECISE : precise, potentially slower implementation.
     */
    uint64_t flags;

    /** Interpolation to use for chroma upsampling.
     * + Valid values:
     *   | Interpolation type          | CPU | CUDA | VIC |
     *   |-----------------------------|:---:|:----:|:---:|
     *   | \ref VPI_INTERP_NEAREST     |  *  |   *  |  1  |
     *   | \ref VPI_INTERP_LINEAR      |     |      |  *  |
     *   | \ref VPI_INTERP_CATMULL_ROM |     |      |  *  |
     *   (1) Nearest-neighbor interpolation is currently accepted,
     *   but linear is performed instead.
     */
    VPIInterpolationType chromaUpFilter;

    /** Interpolation to use for chroma downsampling.
     * + Valid values:
     *   | Interpolation type          | CPU | CUDA | VIC |
     *   |-----------------------------|:---:|:----:|:---:|
     *   | \ref VPI_INTERP_NEAREST     |  *  |   *  |  *  |
     *   | \ref VPI_INTERP_LINEAR      |     |      |     |
     *   | \ref VPI_INTERP_CATMULL_ROM |     |      |     |
     */
    VPIInterpolationType chromaDownFilter;
} VPIConvertImageFormatParams;

/**
 * Image border extension specify how pixel values outside of the image domain should be
 * constructed.
 *
 * @ingroup VPI_Types
 */
typedef enum
{
    VPI_BORDER_ZERO = 0, /**< All pixels outside the image are considered to be zero. */
    VPI_BORDER_CLAMP,    /**< Border pixels are repeated indefinitely. */
    VPI_BORDER_REFLECT,  /**< edcba|abcde|edcba */
    VPI_BORDER_MIRROR,   /**< dedcb|abcde|dcbab */
    VPI_BORDER_LIMITED,  /**< Consider image as limited to not access outside pixels. */
    VPI_BORDER_INVALID,  /**< Invalid border. */
} VPIBorderExtension;

/**
 * Defines the lock modes used by memory lock functions.
 * @ingroup VPI_Types
 */
typedef enum
{
    /** Lock memory only for reading.
     * Writing to the memory when locking for reading leads to undefined behavior. */
    VPI_LOCK_READ = 1,

    /** Lock memory only for writing.
     * Reading to the memory when locking for reading leads to undefined behavior.
     * It is expected that the whole memory is written to. If there are regions not
     * written, it might not be updated correctly during unlock. In this case, it's
     * better to use VPI_LOCK_READ_WRITE.
     *
     * It might be slightly efficient to lock only for writing, specially when
     * performing non-shared memory mapping.
     */
    VPI_LOCK_WRITE = 2,

    /** Lock memory for reading and writing. */
    VPI_LOCK_READ_WRITE = 3
} VPILockMode;

/** VPI Backend types.
 *
 * @ingroup VPI_Stream
 *
 */
typedef enum
{
    VPI_BACKEND_CPU     = (1ULL << 0), /**< CPU backend. */
    VPI_BACKEND_CUDA    = (1ULL << 1), /**< CUDA backend. */
    VPI_BACKEND_PVA     = (1ULL << 2), /**< PVA backend. */
    VPI_BACKEND_VIC     = (1ULL << 3), /**< VIC backend. */
    VPI_BACKEND_NVENC   = (1ULL << 4), /**< NVENC backend. */
    VPI_BACKEND_OFA     = (1ULL << 5), /**< OFA backend. */
    VPI_BACKEND_INVALID = (1ULL << 15) /**< Invalid backend. */
} VPIBackend;

/** @anchor event_flags @name Event-specific flags. */
/**@{*/

/** Disable time-stamping of event signaling.
 * It allows for better performance in operations involving events. */
#define VPI_EVENT_DISABLE_TIMESTAMP (1ULL << 63)
/**@}*/

/** Maximum status message length in bytes.
 *
 * This is the maximum number of bytes that will be written by \ref
 * vpiGetLastStatusMessage and \ref vpiPeekAtLastStatusMessage to the status
 * message output buffer.
 */
#define VPI_MAX_STATUS_MESSAGE_LENGTH 256

/** Defines how channels are packed into an image plane element.
 *
 * Packing encodes how many channels the plane element has, and how they
 * are arranged in memory.
 * 
 * Up to 4 channels (denoted by X, Y, Z, W) can be packed into an image
 * plane element, each one occupying a specified number of bits.
 *
 * When two channels are specified one right after the other, they are
 * ordered from most-significant bit to least-significant bit. Words are
 * separated by underscores. For example:
 *
 * X8Y8Z8W8 = a single 32-bit word containing 4 channels, 8 bits each.
 *
 * In little-endian architectures:
 * <pre>
 *      Address  0   ||  1   ||  2   ||  3
 *            WWWWWWWWZZZZZZZZYYYYYYYYXXXXXXXX
 * </pre>
 *
 * In big-endian architectures:
 * <pre>
 *      Address  0   ||  1   ||  2   ||  3
 *            XXXXXXXXYYYYYYYYZZZZZZZZWWWWWWWW
 * </pre>
 *
 * X8_Y8_Z8_W8 = four consecutive 8-bit words, corresponding to 4 channels, 8 bits each.
 *
 * In little-endian architectures:
 * <pre>
 *      Address  0   ||  1   ||  2   ||  3
 *            XXXXXXXXYYYYYYYYZZZZZZZZWWWWWWWW
 * </pre>
 *
 * In big-endian architectures:
 * <pre>
 *      Address  0   ||  1   ||  2   ||  3
 *            XXXXXXXXYYYYYYYYZZZZZZZZWWWWWWWW
 * </pre>
 *
 * In cases where a word is less than 8 bits (e.g., X1 1-bit channel), channels
 * are ordered from LSB to MSB within a word.
 *
 * @note Also note equivalences such as the following:
 * @note In little-endian: X8_Y8_Z8_W8 = W8Z8Y8X8.
 * @note In big-endian: X8_Y8_Z8_W8 = X8Y8Z8W8.
 *
 * Some formats allow different packings when pixels' horizontal coordinate is
 * even or odd. For instance, every pixel of YUV422 packed format contains an Y
 * channel, while only even pixels contain the U channel, and odd pixels contain
 * V channel. Such formats use a double-underscore to separate the even pixels from the odd
 * pixels. The packing just described might be referred to X8_Y8__X8_Z8, where X = luma, 
 * Y = U chroma, Z = V chroma.
 */
typedef enum
{
    /** No channels. */
    VPI_PACKING_0 = 0,

    /** One 1-bit channel. */
    VPI_PACKING_X1 = VPI_DETAIL_BPP_NCH(1, 1),
    /** One 2-bit channel. */
    VPI_PACKING_X2 = VPI_DETAIL_BPP_NCH(2, 1),
    /** One 4-bit channel. */
    VPI_PACKING_X4 = VPI_DETAIL_BPP_NCH(4, 1),
    /** One 8-bit channel. */
    VPI_PACKING_X8 = VPI_DETAIL_BPP_NCH(8, 1),
    /** Two 4-bit channels in one word. */
    VPI_PACKING_X4Y4 = VPI_DETAIL_BPP_NCH(8, 2),
    /** Three 3-, 3- and 2-bit channels in one 8-bit word. */
    VPI_PACKING_X3Y3Z2 = VPI_DETAIL_BPP_NCH(8, 3),

    /** One 16-bit channel. */
    VPI_PACKING_X16 = VPI_DETAIL_BPP_NCH(16, 1),
    /** One LSB 10-bit channel in one 16-bit word. */
    VPI_PACKING_b6X10,
    /** One MSB 10-bit channel in one 16-bit word. */
    VPI_PACKING_X10b6,
    /** One LSB 12-bit channel in one 16-bit word. */
    VPI_PACKING_b4X12,
    /** One MSB 12-bit channel in one 16-bit word. */
    VPI_PACKING_X12b4,
    /** One LSB 14-bit channel in one 16-bit word. */
    VPI_PACKING_b2X14,

    /** Two 8-bit channels in two 8-bit words. */
    VPI_PACKING_X8_Y8 = VPI_DETAIL_BPP_NCH(16, 2),

    /** Three 5-, 5- and 6-bit channels in one 16-bit word. */
    VPI_PACKING_X5Y5Z6 = VPI_DETAIL_BPP_NCH(16, 3),
    /** Three 5-, 6- and 5-bit channels in one 16-bit word. */
    VPI_PACKING_X5Y6Z5,
    /** Three 6-, 5- and 5-bit channels in one 16-bit word. */
    VPI_PACKING_X6Y5Z5,
    /** Three 4-bit channels in one 16-bit word. */
    VPI_PACKING_b4X4Y4Z4,
    /** Three 5-bit channels in one 16-bit word. */
    VPI_PACKING_b1X5Y5Z5,
    /** Three 5-bit channels in one 16-bit word. */
    VPI_PACKING_X5Y5b1Z5,

    /** Four 1-, 5-, 5- and 5-bit channels in one 16-bit word. */
    VPI_PACKING_X1Y5Z5W5 = VPI_DETAIL_BPP_NCH(16, 4),
    /** Four 4-bit channels in one 16-bit word. */
    VPI_PACKING_X4Y4Z4W4,
    /** Four 5-, 1-, 5- and 5-bit channels in one 16-bit word. */
    VPI_PACKING_X5Y1Z5W5,
    /** Four 5-, 5-, 1- and 5-bit channels in one 16-bit word. */
    VPI_PACKING_X5Y5Z1W5,
    /** Four 5-, 5-, 5- and 1-bit channels in one 16-bit word. */
    VPI_PACKING_X5Y5Z5W1,

    /** 2 pixels of 2 8-bit channels each, totalling 4 8-bit words. */
    VPI_PACKING_X8_Y8__X8_Z8,
    /** 2 pixels of 2 swapped 8-bit channels each, totalling 4 8-bit words. */
    VPI_PACKING_Y8_X8__Z8_X8,

    /** One 24-bit channel. */
    VPI_PACKING_X24 = VPI_DETAIL_BPP_NCH(24, 1),

    /** Three 8-bit channels in three 8-bit words. */
    VPI_PACKING_X8_Y8_Z8 = VPI_DETAIL_BPP_NCH(24, 3),

    /** One 32-bit channel. */
    VPI_PACKING_X32 = VPI_DETAIL_BPP_NCH(32, 1),
    /** One LSB 20-bit channel in one 32-bit word. */
    VPI_PACKING_b12X20,

    /** Two 16-bit channels in two 16-bit words. */
    VPI_PACKING_X16_Y16 = VPI_DETAIL_BPP_NCH(32, 2),
    /** Two MSB 10-bit channels in two 16-bit words. */
    VPI_PACKING_X10b6_Y10b6,
    /** Two MSB 12-bit channels in two 16-bit words. */
    VPI_PACKING_X12b4_Y12b4,

    /** Three 10-, 11- and 11-bit channels in one 32-bit word. */
    VPI_PACKING_X10Y11Z11 = VPI_DETAIL_BPP_NCH(32, 3),
    /** Three 11-, 11- and 10-bit channels in one 32-bit word. */
    VPI_PACKING_X11Y11Z10,

    /** Four 8-bit channels in one 32-bit word. */
    VPI_PACKING_X8_Y8_Z8_W8 = VPI_DETAIL_BPP_NCH(32, 4),
    /** Four 2-, 10-, 10- and 10-bit channels in one 32-bit word. */
    VPI_PACKING_X2Y10Z10W10,
    /** Four 10-, 10-, 10- and 2-bit channels in one 32-bit word. */
    VPI_PACKING_X10Y10Z10W2,

    /** One 48-bit channel. */
    VPI_PACKING_X48 = VPI_DETAIL_BPP_NCH(48, 1),
    /** Three 16-bit channels in three 16-bit words. */
    VPI_PACKING_X16_Y16_Z16 = VPI_DETAIL_BPP_NCH(48, 3),

    /**< One 64-bit channel. */
    VPI_PACKING_X64 = VPI_DETAIL_BPP_NCH(64, 1),
    /** Two 32-bit channels in two 32-bit words. */
    VPI_PACKING_X32_Y32 = VPI_DETAIL_BPP_NCH(64, 2),
    /** Four 16-bit channels in one 64-bit word. */
    VPI_PACKING_X16_Y16_Z16_W16 = VPI_DETAIL_BPP_NCH(64, 4),

    /** One 96-bit channel. */
    VPI_PACKING_X96 = VPI_DETAIL_BPP_NCH(96, 1),
    /** Three 32-bit channels in three 32-bit words. */
    VPI_PACKING_X32_Y32_Z32 = VPI_DETAIL_BPP_NCH(96, 3),

    /** One 128-bit channel. */
    VPI_PACKING_X128 = VPI_DETAIL_BPP_NCH(128, 1),
    /** Two 64-bit channels in two 64-bit words. */
    VPI_PACKING_X64_Y64 = VPI_DETAIL_BPP_NCH(128, 2),
    /**< Four 32-bit channels in three 32-bit words. */
    VPI_PACKING_X32_Y32_Z32_W32 = VPI_DETAIL_BPP_NCH(128, 4),

    /** One 192-bit channel. */
    VPI_PACKING_X192 = VPI_DETAIL_BPP_NCH(192, 1),
    /** Three 64-bit channels in three 64-bit words. */
    VPI_PACKING_X64_Y64_Z64 = VPI_DETAIL_BPP_NCH(192, 3),

    /** One 128-bit channel. */
    VPI_PACKING_X256 = VPI_DETAIL_BPP_NCH(256, 1),
    /** Four 64-bit channels in four 64-bit words. */
    VPI_PACKING_X64_Y64_Z64_W64 = VPI_DETAIL_BPP_NCH(256, 4),

    /** Denotes an invalid packing. */
    VPI_PACKING_INVALID = INT32_MAX
} VPIPacking;

/** Defines how chroma-subsampling is done.
 * This is only applicable to image formats whose color model is YUV.
 * Other image formats must use \ref VPI_CSS_NONE.
 * Chroma subsampling is defined by 2 parameters:
 * - Horizontal resolution relative to luma resolution.
 * - Vertical resolution relative to luma resolution.
 */
typedef enum
{
    VPI_CSS_INVALID = -1, /**< Invalid chroma subsampling. */

    /** Used when no chroma subsampling takes place, specially for color specs without chroma components. */
    VPI_CSS_NONE = 0,

    /** 4:4:4 sub-sampling. Chroma has full horizontal and vertical resolution, meaning no chroma subsampling. */
    VPI_CSS_444 = VPI_CSS_NONE,

    /** 4:2:2 BT.601 sub-sampling. Chroma has half horizontal and full vertical resolutions.*/
    VPI_CSS_422,

    /** 4:2:2R BT.601 sub-sampling. Chroma has full horizontal and half vertical resolutions.*/
    VPI_CSS_422R,

    /** 4:1:1 sub-sampling. Chroma has 1/4 horizontal and full vertical resolutions.*/
    VPI_CSS_411,

    /** 4:1:1 sub-sampling. Chroma has full horizontal and 1/4 vertical resolutions.*/
    VPI_CSS_411R,

    /** 4:2:0 sub-sampling. Chroma has half horizontal and vertical resolutions.*/
    VPI_CSS_420,
} VPIChromaSubsampling;

typedef struct
{
    /** Represents the median filter size (on PVA+NVENC+VIC or OFA+PVA+VIC backend)
        or census transform window size (other backends) used in the algorithm.
     *  + On PVA backend, it must be 5.
     *  + On CPU backend, it must be >= 1.
     *  + On CUDA backend this is ignored. A 9x7 window is used instead.
     *  + On OFA backend it is ignored.
     *  + On OFA+PVA+VIC backend, valid values are 1, 3, 5 or 7.
     *  + On PVA+NVENC+VIC backend, valid values are 1, 3, 5 or 7. */
    int32_t windowSize;

    /** Maximum disparity for matching search.
     *  + Maximum disparity must be 0 (use from payload), or positive and less or equal to what's configured in payload. */
    int32_t maxDisparity;

    /* Confidence threshold above which disparity values are considered valid.
     * Only used in CUDA, PVA+NVENC+VIC, and OFA+PVA+VIC backends.
     * + Ranges from 1 to 65280, values outside this range get clamped. */
    int32_t confidenceThreshold;

    /* Quality of disparity output.
     * It's only applicable when using PVA+NVENC+VIC backend.
     * The higher the value, better the quality and possibly slower perf.
     * + Must be a value between 1 and 8. */
    int32_t quality;

} VPIStereoDisparityEstimatorParams;

#ifdef __cplusplus
}
#endif
