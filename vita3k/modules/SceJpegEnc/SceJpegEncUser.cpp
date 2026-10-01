// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include <module/module.h>

#include <codec/state.h>
#include <codec/types.h>
#include <renderer/functions.h>
#include <renderer/state.h>

#include <limits>

#include <util/tracy.h>
TRACY_MODULE_NAME(SceJpegEncUser);

struct SceJpegEncoderContext {
    int32_t inWidth;
    int32_t inHeight;
    int32_t pixelFormat;
    Ptr<uint8_t> outBuffer;
    uint32_t outSize;
    SceJpegEncoderInitParamOption option;

    int32_t compressRatio;
    int32_t headerMode;
};

static int sceJpegEncoderInitImpl(SceJpegEncoderContext *context, int32_t inWidth, int32_t inHeight, int32_t pixelFormat, Ptr<uint8_t> outBuffer, uint32_t outSize, SceJpegEncoderInitParamOption option = SCE_JPEGENC_INIT_PARAM_OPTION_NONE) {
    context->inWidth = inWidth;
    context->inHeight = inHeight;
    context->pixelFormat = pixelFormat;
    context->outBuffer = outBuffer;
    context->outSize = outSize;
    context->option = option;

    context->compressRatio = 64;
    context->headerMode = SCE_JPEGENC_HEADER_MODE_JPEG;

    return 0;
}

EXPORT(int, sceJpegEncoderCsc, SceJpegEncoderContext *context, Ptr<uint8_t> outBuffer, Ptr<uint8_t> inBuffer, int32_t inPitch, int32_t inPixelFormat) {
    TRACY_FUNC(sceJpegEncoderCsc, context, outBuffer, inBuffer, inPitch, inPixelFormat);
    const bool sync_metal = emuenv.renderer && emuenv.renderer->current_backend == renderer::Backend::Metal;
    if (sync_metal && (!context || !inBuffer.valid(emuenv.mem) || !outBuffer.valid(emuenv.mem)))
        return RET_ERROR(SCE_JPEGENC_ERROR_INVALID_POINTER);
    auto inBufferData = inBuffer.get(emuenv.mem);
    auto outBufferData = outBuffer.get(emuenv.mem);

    DecoderColorSpace color_space = COLORSPACE_UNKNOWN;
    bool is_bgra;

    switch (context->pixelFormat & 0xF) {
    case SCE_JPEGENC_PIXEL_YCBCR422:
        color_space = COLORSPACE_YUV422P;
        break;
    case SCE_JPEGENC_PIXEL_YCBCR420:
        color_space = COLORSPACE_YUV420P;
        break;
    default:
        return SCE_JPEGENC_ERROR_INVALID_PIXELFORMAT;
    }

    switch (inPixelFormat) {
    case SCE_JPEGENC_PIXEL_RGBA8888:
        is_bgra = false;
        break;
    case SCE_JPEGENC_PIXEL_BGRA8888:
        is_bgra = true;
        break;
    default:
        return SCE_JPEGENC_ERROR_INVALID_PIXELFORMAT;
    }

    if (sync_metal) {
        if (context->inWidth <= 0 || context->inHeight <= 0 || inPitch < context->inWidth
            || inPitch > std::numeric_limits<int32_t>::max() / 4)
            return RET_ERROR(SCE_JPEGENC_ERROR_IMAGE_SIZE);
        // The converter's pitch is in RGBA pixels. Include every accessed row,
        // but do not require padding after the final row to be allocated.
        const uint64_t bytes = uint64_t(inPitch) * 4 * uint32_t(context->inHeight - 1)
            + uint64_t(context->inWidth) * 4;
        const uint64_t end = uint64_t(inBuffer.address()) + bytes;
        if (end > uint64_t(UINT32_MAX) - 4095
            || !is_valid_addr_range(emuenv.mem, inBuffer.address(), Address(end)))
            return RET_ERROR(SCE_JPEGENC_ERROR_INVALID_POINTER);
        const int synced = renderer::sync_guest_range(*emuenv.renderer, inBuffer.address(), uint32_t(bytes));
        if (synced < 0) {
            LOG_ERROR("Metal: JPEG input synchronization failed ({})", synced);
            return synced;
        }
    }
    convert_rgb_to_yuv(inBufferData, outBufferData, context->inWidth, context->inHeight, color_space, is_bgra, inPitch);

    return 0;
}

EXPORT(int, sceJpegEncoderEncode, SceJpegEncoderContext *context, Ptr<uint8_t> inBuffer) {
    TRACY_FUNC(sceJpegEncoderEncode, context, inBuffer);

    auto inBufferData = inBuffer.get(emuenv.mem);
    auto outBufferData = context->outBuffer.get(emuenv.mem);

    int width = context->inWidth;
    int height = context->inHeight;

    DecoderColorSpace color_space = COLORSPACE_UNKNOWN;

    switch (context->pixelFormat & 0xF) {
    case SCE_JPEGENC_PIXEL_YCBCR422:
        color_space = COLORSPACE_YUV422P;
        break;
    case SCE_JPEGENC_PIXEL_YCBCR420:
        color_space = COLORSPACE_YUV420P;
        break;
    default:
        return SCE_JPEGENC_ERROR_INVALID_PIXELFORMAT;
    }

    uint32_t size = convert_yuv_to_jpeg(inBufferData, outBufferData, width, height, context->outSize, color_space, context->compressRatio);

    if (size == -1) {
        return SCE_JPEGENC_ERROR_INSUFFICIENT_BUFFER;
    }

    return size;
}

EXPORT(int, sceJpegEncoderEnd, SceJpegEncoderContext *context) {
    TRACY_FUNC(sceJpegEncoderEnd, context);
    return 0;
}

EXPORT(int, sceJpegEncoderGetContextSize) {
    TRACY_FUNC(sceJpegEncoderGetContextSize);
    return sizeof(SceJpegEncoderContext);
}

EXPORT(int, sceJpegEncoderInit, SceJpegEncoderContext *context, int32_t inWidth, int32_t inHeight, int32_t pixelFormat, Ptr<uint8_t> outBuffer, uint32_t outSize) {
    TRACY_FUNC(sceJpegEncoderInit, context, inWidth, inHeight, pixelFormat, outBuffer, outSize);
    return sceJpegEncoderInitImpl(context, inWidth, inHeight, pixelFormat, outBuffer, outSize, SCE_JPEGENC_INIT_PARAM_OPTION_NONE);
}

EXPORT(int, sceJpegEncoderInitWithParam, SceJpegEncoderContext *context, SceJpegEncoderInitParam *initParam) {
    TRACY_FUNC(sceJpegEncoderInitWithParam, context, initParam);
    return sceJpegEncoderInitImpl(context, initParam->inWidth, initParam->inHeight, initParam->pixelFormat, initParam->outBuffer, initParam->outSize, initParam->option);
}

EXPORT(int, sceJpegEncoderSetCompressionRatio, SceJpegEncoderContext *context, int32_t ratio) {
    TRACY_FUNC(sceJpegEncoderSetCompressionRatio, context, ratio);
    if (ratio < 0 || ratio > 255) {
        return SCE_JPEGENC_ERROR_INVALID_COMPRATIO;
    }
    context->compressRatio = ratio;
    return 0;
}

EXPORT(int, sceJpegEncoderSetHeaderMode, SceJpegEncoderContext *context, int32_t mode) {
    TRACY_FUNC(sceJpegEncoderSetHeaderMode, context, mode);
    if (mode != SCE_JPEGENC_HEADER_MODE_JPEG && mode != SCE_JPEGENC_HEADER_MODE_MJPEG) {
        return SCE_JPEGENC_ERROR_INVALID_HEADER_MODE;
    } else if (mode == SCE_JPEGENC_HEADER_MODE_MJPEG) {
        return STUBBED("SCE_JPEGENC_HEADER_MODE_MJPEG is not supported");
    }
    context->headerMode = mode;
    return 0;
}

EXPORT(int, sceJpegEncoderSetOutputAddr, SceJpegEncoderContext *context, Ptr<uint8_t> outBuffer, uint32_t outSize) {
    TRACY_FUNC(sceJpegEncoderSetOutputAddr, context, outBuffer, outSize);
    context->outBuffer = outBuffer;
    context->outSize = outSize;
    return 0;
}

EXPORT(int, sceJpegEncoderSetValidRegion, SceJpegEncoderContext *context, int32_t inWidth, int32_t inHeight) {
    TRACY_FUNC(sceJpegEncoderSetValidRegion, context, inWidth, inHeight);
    return UNIMPLEMENTED();
}
