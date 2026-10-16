#include "SceGxm_shader_cache_metal.h"
#include <gxm/metal3/state.h>
static gxm::metal3::Bookkeeping &metal_gxm_bookkeeping(GxmState &state) {
    auto data = std::atomic_load(&state.metal_state);
    if (!data) {
        auto created = std::make_shared<gxm::metal3::Bookkeeping>();
        if (std::atomic_compare_exchange_strong(&state.metal_state, &data, created)) data = std::move(created);
    }
    return *data;
}
#include <renderer/metal3/color_surface.h>
bool MetalVertexCacheLess::operator()(const MetalVertexProgramCacheKey &a, const MetalVertexProgramCacheKey &b) const {
    if (a.metal_registration != b.metal_registration)
        return a.metal_registration < b.metal_registration;
    if (a.vertex_program < b.vertex_program) {
        return true;
    }
    if (b.vertex_program < a.vertex_program) {
        return false;
    }
    return b.hash < a.hash;
}
bool MetalFragmentCacheLess::operator()(const MetalFragmentProgramCacheKey &a, const MetalFragmentProgramCacheKey &b) const {
    if (a.metal_registration != b.metal_registration)
        return a.metal_registration < b.metal_registration;
    if (a.fragment_program < b.fragment_program) {
        return true;
    }
    if (b.fragment_program < a.fragment_program) {
        return false;
    }
    if (b.blend_info < a.blend_info) return true;
    if (a.blend_info < b.blend_info) return false;
    if (a.metal_output_format != b.metal_output_format)
        return a.metal_output_format < b.metal_output_format;
    return a.metal_multisample_mode < b.metal_multisample_mode;
}
static MetalShaderPatcherCache &metal_shader_patcher_cache(GxmState &state, SceGxmShaderPatcher &patcher) {
    auto &bookkeeping = metal_gxm_bookkeeping(state);
    const std::lock_guard lock(bookkeeping.metal_shader_caches_mutex);
    auto &cache = bookkeeping.metal_shader_caches[&patcher];
    if (!cache) cache = std::make_shared<MetalShaderPatcherCache>();
    return *std::static_pointer_cast<MetalShaderPatcherCache>(cache);
}
// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Included once by SceGxm.cpp so its opaque guest types stay private.

static void display_entry_thread_metal3(EmuEnvState &emuenv);
static Ptr<void> gxmRunDeferredMemoryCallback_metal3(KernelState &kernel, const MemState &mem, std::mutex &global_lock, std::uint32_t &return_size, Ptr<SceGxmDeferredContextCallback> callback, Ptr<void> userdata,
    const std::uint32_t size, const SceUID thread_id, bool native_metal = false);
static void destroy_pending_immediate_commands_metal3(SceGxmContext *context);
static void destroy_pending_deferred_recording_metal3(SceGxmContext *context);
static void destroy_pending_deferred_commands_metal3(SceGxmContext *context);
static void abort_metal_deferred_recording_metal3(SceGxmContext *context);
static int destroy_gxm_context_metal3(EmuEnvState &emuenv, SceGxmContext *context, const Address context_addr, const bool force_backend_destroy);
namespace gxm { void destroy_all_contexts_metal3(EmuEnvState &emuenv, const bool force_backend_destroy); }
namespace gxm { void shutdown_metal3(EmuEnvState &emuenv); }
static int destroy_gxm_render_target_metal3(EmuEnvState &emuenv, SceGxmRenderTarget *render_target, const Address render_target_addr, const bool force_backend_destroy);
namespace gxm { void destroy_all_render_targets_metal3(EmuEnvState &emuenv, const bool force_backend_destroy); }
static bool metal_program_is_live_metal3(const renderer::State &state, GxmState &gxm, Address program,
    MetalGxmProgramKind kind, const char *what);
static void metal_program_register_metal3(EmuEnvState &emuenv, Address program, MetalGxmProgramKind kind);
static void gxmSetProgram_metal3(renderer::State &state, GxmState &gxm, SceGxmContext *context, Ptr<const void> program,
    bool is_fragment, const MemState &mem);
static void gxmContextStateRestore_metal3(renderer::State &state, GxmState &gxm, SceGxmContext *context, const bool sync_viewport_and_clip, const MemState &mem);
int export_sceGxmBeginCommandList_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext);
void export_sceGxmColorSurfaceGetClip_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmColorSurface *surface, uint32_t *xMin, uint32_t *yMin, uint32_t *xMax, uint32_t *yMax);
SceGxmColorSurfaceDitherMode export_sceGxmColorSurfaceGetDitherMode_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmColorSurface *surface);
int export_sceGxmColorSurfaceInit_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, SceGxmColorFormat colorFormat, SceGxmColorSurfaceType surfaceType, SceGxmColorSurfaceScaleMode scaleMode, SceGxmOutputRegisterSize outputRegisterSize, uint32_t width, uint32_t height, uint32_t strideInPixels, Ptr<void> data);
void export_sceGxmColorSurfaceSetClip_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, uint32_t xMin, uint32_t yMin, uint32_t xMax, uint32_t yMax);
int export_sceGxmColorSurfaceSetDitherMode_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, SceGxmColorSurfaceDitherMode ditherMode);
int export_sceGxmColorSurfaceSetFormat_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, SceGxmColorFormat format);
void export_sceGxmColorSurfaceSetScaleMode_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, SceGxmColorSurfaceScaleMode scaleMode);
int export_sceGxmCreateContext_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmContextParams *params, Ptr<SceGxmContext> *context);
int export_sceGxmCreateDeferredContext_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmDeferredContextParams *params, Ptr<SceGxmContext> *deferredContext);
int export_sceGxmDisplayQueueAddEntry_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, Ptr<SceGxmSyncObject> oldBuffer, Ptr<SceGxmSyncObject> newBuffer, Ptr<const void> callbackData);
static uint32_t metal_mapped_uniform_size_metal3(const GxmState &gxm, Address address, uint32_t requested_size);
static void gxmSetUniformBuffers_metal3(renderer::State &state, GxmState &gxm, SceGxmContext *context, const SceGxmProgram &program, std::span<UniformBuffer> buffers, const UniformBufferSizes &sizes, const MemState &mem);
static bool metal_vertex_stream_sizes_metal3(const SceGxmVertexProgram &program, uint32_t max_index,
    uint32_t instances, std::array<size_t, SCE_GXM_MAX_VERTEX_STREAMS> &sizes, uint32_t &used);
static int gxmDrawElementGeneral_metal3(EmuEnvState &emuenv, const char *export_name, const SceUID thread_id, SceGxmContext *context, SceGxmPrimitiveType primType, SceGxmIndexFormat indexType, Ptr<const void> indexData, uint32_t indexCount, uint32_t instanceCount);
int export_sceGxmDrawInstanced_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmPrimitiveType primType, SceGxmIndexFormat indexType, Ptr<const void> indexData, uint32_t indexCount, uint32_t indexWrap);
int export_sceGxmDrawPrecomputed_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmPrecomputedDraw *draw);
int export_sceGxmEndCommandList_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext, SceGxmCommandList *commandList);
int export_sceGxmEndScene_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmNotification *vertexNotification, SceGxmNotification *fragmentNotification);
int export_sceGxmExecuteCommandList_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmCommandList *commandList);
int export_sceGxmFinish_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context);
int export_sceGxmGetDeferredContextFragmentBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmContext *deferredContext, Ptr<void> *mem);
int export_sceGxmGetDeferredContextVdmBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmContext *deferredContext, Ptr<void> *mem);
int export_sceGxmGetDeferredContextVertexBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmContext *deferredContext, Ptr<void> *mem);
int export_sceGxmInitialize_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmInitializeParams *params);
int export_sceGxmMapMemory_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, Ptr<void> base, uint32_t size, uint32_t attribs);
int export_sceGxmMidSceneFlush_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *immediateContext, uint32_t flags, SceGxmSyncObject *vertexSyncObject, const SceGxmNotification *vertexNotification);
int export_sceGxmNotificationWait_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmNotification *notification);
int export_sceGxmPrecomputedDrawInit_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmPrecomputedDraw *state, Ptr<const SceGxmVertexProgram> program, Ptr<void> extra_data);
int export_sceGxmPrecomputedFragmentStateInit_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmPrecomputedFragmentState *state, Ptr<const SceGxmFragmentProgram> program, Ptr<void> extra_data);
int export_sceGxmPrecomputedVertexStateInit_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmPrecomputedVertexState *state, Ptr<const SceGxmVertexProgram> program, Ptr<void> extra_data);
int export_sceGxmReserveFragmentDefaultUniformBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, Ptr<void> *uniformBuffer);
int export_sceGxmReserveVertexDefaultUniformBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, Ptr<void> *uniformBuffer);
void export_sceGxmSetBackFragmentProgramEnable_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmFragmentProgramMode enable);
void export_sceGxmSetBackVisibilityTestEnable_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmVisibilityTestMode enable);
void export_sceGxmSetBackVisibilityTestIndex_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, uint32_t index);
void export_sceGxmSetBackVisibilityTestOp_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmVisibilityTestOp op);
int export_sceGxmSetDeferredContextFragmentBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext, Ptr<void> mem, uint32_t size);
int export_sceGxmSetDeferredContextVdmBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext, Ptr<void> mem, uint32_t size);
int export_sceGxmSetDeferredContextVertexBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext, Ptr<void> mem, uint32_t size);
void export_sceGxmSetFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, Ptr<const SceGxmFragmentProgram> fragmentProgram);
void export_sceGxmSetFrontFragmentProgramEnable_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmFragmentProgramMode enable);
void export_sceGxmSetFrontVisibilityTestEnable_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmVisibilityTestMode enable);
void export_sceGxmSetFrontVisibilityTestIndex_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, uint32_t index);
void export_sceGxmSetFrontVisibilityTestOp_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmVisibilityTestOp op);
int export_sceGxmSetUniformDataF_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, void *uniformBuffer, const SceGxmProgramParameter *parameter, uint32_t componentOffset, uint32_t componentCount, const float *sourceData);
void export_sceGxmSetVertexProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, Ptr<const SceGxmVertexProgram> vertexProgram);
int export_sceGxmSetVisibilityBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *immediateContext, Ptr<void> bufferBase, uint32_t stridePerCore);
int export_sceGxmSetYuvProfile_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, uint32_t index, SceGxmYuvProfile profile);
template <typename T>
static void free_callbacked_metal3(EmuEnvState &emuenv, SceUID thread_id, SceGxmShaderPatcher *shaderPatcher, Ptr<T> data);
int export_sceGxmShaderPatcherAddRefFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, SceGxmFragmentProgram *fragmentProgram);
int export_sceGxmShaderPatcherAddRefVertexProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, SceGxmVertexProgram *vertexProgram);
int export_sceGxmShaderPatcherCreateFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, const SceGxmRegisteredProgram *programId, SceGxmOutputRegisterFormat outputFormat, SceGxmMultisampleMode multisampleMode, const SceGxmBlendInfo *blendInfo, Ptr<const SceGxmProgram> vertexProgram, Ptr<SceGxmFragmentProgram> *fragmentProgram);
int export_sceGxmShaderPatcherCreateMaskUpdateFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, Ptr<SceGxmFragmentProgram> *fragmentProgram);
int export_sceGxmShaderPatcherCreateVertexProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, const SceGxmRegisteredProgram *programId, const SceGxmVertexAttribute *attributes, uint32_t attributeCount, const SceGxmVertexStream *streams, uint32_t streamCount, Ptr<SceGxmVertexProgram> *vertexProgram);
int export_sceGxmShaderPatcherForceUnregisterProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, SceGxmShaderPatcherId programId);
int export_sceGxmShaderPatcherRegisterProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, Ptr<const SceGxmProgram> programHeader, SceGxmShaderPatcherId *programId);
int export_sceGxmShaderPatcherReleaseFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, Ptr<SceGxmFragmentProgram> fragmentProgram);
int export_sceGxmShaderPatcherReleaseVertexProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, Ptr<SceGxmVertexProgram> vertexProgram);
int export_sceGxmShaderPatcherUnregisterProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, SceGxmShaderPatcherId programId);
int export_sceGxmTerminate_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name);
uint32_t export_sceGxmTextureGetLodMin_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmTexture *texture);
static void set_linear_strided_texture_stride_metal3(SceGxmTexture &texture, uint32_t byte_stride);
int export_sceGxmTextureInitLinearStrided_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, Ptr<const void> data, SceGxmTextureFormat texFormat, uint32_t width, uint32_t height, uint32_t byteStride);
int export_sceGxmTextureSetHeight_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, uint32_t height);
int export_sceGxmTextureSetLodMin_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, uint32_t lodMin);
int export_sceGxmTextureSetStride_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, uint32_t byteStride);
int export_sceGxmTextureSetUAddrModeSafe_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, SceGxmTextureAddrMode mode);
int export_sceGxmTextureSetVAddrModeSafe_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, SceGxmTextureAddrMode mode);
int export_sceGxmTextureSetWidth_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, uint32_t width);
int export_sceGxmUnmapMemory_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, Ptr<void> base);
void gxm_context_free_command_list_metal3(SceGxmContext &self, SceGxmCommandList *command_list);
void gxm_context_insert_new_memory_range_metal3(SceGxmContext &self);
bool gxm_context_make_new_alloc_space_metal3(SceGxmContext &self, KernelState &kern, const MemState &mem, const SceUID thread_id, bool force);
renderer::Command *gxm_context_allocate_new_command_metal3(SceGxmContext &self, KernelState &kern, const MemState &mem, SceUID current_thread_id);
void gxm_context_free_new_command_metal3(SceGxmContext &self, renderer::Command *cmd);

static void display_entry_thread_metal3(EmuEnvState &emuenv) {
    auto &display_queue = emuenv.gxm.display_queue;
    const Address callback_address = emuenv.gxm.params.displayQueueCallback.address();
    const ThreadStatePtr display_thread = emuenv.kernel.get_thread(emuenv.gxm.display_queue_thread);
    if (!display_thread) {
        LOG_CRITICAL("display_thread not found. thid:{}", emuenv.gxm.display_queue_thread);
        return;
    }

    while (true) {
        auto display_callback = display_queue.top();
        if (!display_callback)
            break;

        SceGxmSyncObject *old_sync = display_callback->old_sync.get(emuenv.mem);
        SceGxmSyncObject *new_sync = display_callback->new_sync.get(emuenv.mem);

        // A missing render publication can leave the display worker waiting
        // indefinitely. Report the actual sync state without changing the
        // guest-visible queue limit or callback lifetime on Metal.
        const auto wait_sync = [&](SceGxmSyncObject *sync, uint32_t wanted, const char *which) {
            if (emuenv.renderer->current_backend != renderer::Backend::Metal)
                return renderer::wishlist(sync, wanted);
            uint32_t stalled_seconds = 0;
            while (true) {
                const auto result = renderer::wishlist(sync, wanted, 1'000'000);
                if (result != renderer::SyncWaitResult::TimedOut)
                    return result;
                if (emuenv.display.abort.load() || emuenv.renderer->render_abort.load())
                    return renderer::SyncWaitResult::Shutdown;
                ++stalled_seconds;
                if (stalled_seconds == 5 || stalled_seconds % 30 == 0)
                    LOG_ERROR("Metal display queue stalled {}s on {} sync 0x{:X}: wanted {}, current {} (ahead {})",
                        stalled_seconds, which, reinterpret_cast<uintptr_t>(sync), wanted,
                        sync->timestamp_current.load(), sync->timestamp_ahead.load());
            }
        };
        // sceGxmDisplayQueueAddEntry waits for both buffers to complete.
        if (wait_sync(old_sync, display_callback->old_sync_timestamp, "old") == renderer::SyncWaitResult::Shutdown) {
            return;
        }
        if (old_sync != new_sync) {
            if (wait_sync(new_sync, display_callback->new_sync_timestamp, "new") == renderer::SyncWaitResult::Shutdown) {
                return;
            }
        }

        // check if we're shutting down before calling run_guest_function to avoid deadlock
        if (emuenv.display.abort.load()) {
            LOG_DEBUG("Abort detected, removing display callback data and exiting");
            display_queue.pop();
            free(emuenv.mem, display_callback->data);
            break;
        }

        // specify whether the call to SceDisplaySetFrameBuf is expected to do something
        emuenv.display.predicting = display_callback->frame_predicted;
        emuenv.display.current_sync_object = display_callback->new_sync.address();

        // Now run callback
        display_thread->run_guest_function(callback_address, display_callback->data);

        // Notifies the renderer of the completion of the callback for the display_entry.
        // The last_display of the entry, when pushed into the queue, is guaranteed to be timestamp_ahead + 1 at the time of the call.
        renderer::subject_done(old_sync, display_callback->old_sync_timestamp + 1);
        if (old_sync != new_sync)
            renderer::subject_done(new_sync, display_callback->new_sync_timestamp + 1);

        free(emuenv.mem, display_callback->data);
        display_queue.pop();
    }
    }

static Ptr<void> gxmRunDeferredMemoryCallback_metal3(KernelState &kernel, const MemState &mem, std::mutex &global_lock, std::uint32_t &return_size, Ptr<SceGxmDeferredContextCallback> callback, Ptr<void> userdata,
    const std::uint32_t size, const SceUID thread_id, bool native_metal) {
    uint32_t request_size = size;
    if (native_metal) {
        // Plus leaves room for alignment and rejects an absent/short callback.
        if (!callback || size > UINT32_MAX - 4) {
            return_size = 0;
            return {};
        }
        request_size = std::max<uint32_t>(1024, size + 4);
    }
    const std::lock_guard<std::mutex> guard(global_lock);

    const ThreadStatePtr thread = kernel.get_thread(thread_id);
    const Address final_size_addr = stack_alloc(*thread->cpu, 4);
    if (native_metal)
        *Ptr<uint32_t>(final_size_addr).get(mem) = 0;

    Ptr<void> result(thread->run_callback(callback.address(), { userdata.address(), request_size, final_size_addr }));

    return_size = *Ptr<std::uint32_t>(final_size_addr).get(mem);
    stack_free(*thread->cpu, 4);

    if (native_metal && (!result || return_size < request_size
                            || uint64_t(result.address()) + return_size > UINT32_MAX)) {
        return_size = 0;
        return {};
    }
    return result;
}

static void destroy_pending_immediate_commands_metal3(SceGxmContext *context) {
    if (renderer::metal::gxm_metadata(*context->renderer).enabled) {
        auto &list = context->renderer->command_list;
        // Destroy can be cancelled while earlier submitted work still owns
        // ring slots. Discard payloads/host nodes without advancing its FIFO.
        renderer::discard_metal_commands(context->renderer.get(), list.first, list.last);
        renderer::reset_command_list(list);
        return;
    }
    renderer::Command *cmd = context->renderer->command_list.first;
    while (cmd) {
        renderer::Command *next = cmd->next;
        renderer::destroy_command_payload(*cmd);
        context->free_new_command(cmd);
        cmd = next;
    }

    renderer::reset_command_list(context->renderer->command_list);
}

static void destroy_pending_deferred_recording_metal3(SceGxmContext *context) {
    if (context->curr_command_list) {
        while (!context->curr_command_list->memory_ranges.empty()) {
            context->command_list_ranges.erase(context->curr_command_list->memory_ranges.top());
            context->curr_command_list->memory_ranges.pop();
        }

        delete context->curr_command_list;
        context->curr_command_list = nullptr;
    }

    destroy_pending_deferred_command_chain(context->renderer->command_list);
}

static void destroy_pending_deferred_commands_metal3(SceGxmContext *context) {
    destroy_pending_deferred_recording_metal3(context);
    while (!context->command_list_ranges.empty())
        context->free_command_list(context->command_list_ranges.begin()->command_list);
}

static void abort_metal_deferred_recording_metal3(SceGxmContext *context) {
    // Keep previously completed lists alive. Only the incomplete recording
    // and its owned program payloads/range markers are discarded.
    destroy_pending_deferred_recording_metal3(context);
    context->state.active = false;
    context->reset_recording();
    context->was_vert_default_uniform_reserved = false;
    context->was_frag_default_uniform_reserved = false;
    renderer::metal::gxm_metadata(*context->renderer).vertex_reserved_bytes = 0;
    renderer::metal::gxm_metadata(*context->renderer).fragment_reserved_bytes = 0;
    renderer::metal::gxm_metadata(*context->renderer).recording_failed = true;
}

static int destroy_gxm_context_metal3(EmuEnvState &emuenv, SceGxmContext *context, const Address context_addr, const bool force_backend_destroy) {
    if (!context) {
        return static_cast<int>(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (context->state.type == SCE_GXM_CONTEXT_TYPE_IMMEDIATE) {
        const auto immediate_context = emuenv.gxm.immediate_contexts.find(context);
        if (immediate_context == emuenv.gxm.immediate_contexts.end()) {
            return static_cast<int>(SCE_GXM_ERROR_INVALID_POINTER);
        }

        if (context_addr != 0 && immediate_context->second != context_addr) {
            return static_cast<int>(SCE_GXM_ERROR_INVALID_POINTER);
        }

        if (context->state.active && !force_backend_destroy) {
            return static_cast<int>(SCE_GXM_ERROR_WITHIN_SCENE);
        }

        destroy_pending_immediate_commands_metal3(context);

        if (force_backend_destroy) {
            renderer::destroy_context_during_shutdown(*emuenv.renderer, context->renderer);
        } else {
            renderer::destroy_context(*emuenv.renderer, context->renderer);
            // P64 guarantees the submitted handler has stopped accessing the
            // output owner. Non-null means destruction was not performed;
            // retain the wrapper/registry for teardown after the render join.
            if (emuenv.renderer->current_backend == renderer::Backend::Metal && context->renderer)
                return static_cast<int>(SCE_GXM_ERROR_DRIVER);
        }

        if (emuenv.gxm.last_immediate_context == immediate_context->second) {
            emuenv.gxm.last_immediate_context = 0;
        }
        emuenv.gxm.immediate_contexts.erase(immediate_context);
    } else if (context->state.type == SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        const auto deferred_context = emuenv.gxm.deferred_contexts.find(context);
        if (deferred_context == emuenv.gxm.deferred_contexts.end()) {
            return static_cast<int>(SCE_GXM_ERROR_INVALID_POINTER);
        }

        if (context_addr != 0 && deferred_context->second != context_addr) {
            return static_cast<int>(SCE_GXM_ERROR_INVALID_POINTER);
        }

        if (context->state.active && !force_backend_destroy) {
            return static_cast<int>(SCE_GXM_ERROR_WITHIN_COMMAND_LIST);
        }

        destroy_pending_deferred_commands_metal3(context);
        emuenv.gxm.deferred_contexts.erase(deferred_context);
    } else {
        return static_cast<int>(SCE_GXM_ERROR_INVALID_VALUE);
    }

    context->~SceGxmContext();
    return 0;
    }

namespace gxm {
void destroy_all_contexts_metal3(EmuEnvState &emuenv, const bool force_backend_destroy) {
    for (auto immediate_context = emuenv.gxm.immediate_contexts.begin(); immediate_context != emuenv.gxm.immediate_contexts.end();) {
        // destroy_gxm_context erases this entry on success, so advance the iterator first.
        const auto current_context = immediate_context++;
        const auto [context, context_addr] = *current_context;
        const int result = destroy_gxm_context_metal3(emuenv, context, context_addr, force_backend_destroy);
        if (result < 0) {
            LOG_WARN("Failed to destroy immediate GXM context during cleanup: {}", log_hex(result));
        }
    }

    while (!emuenv.gxm.deferred_contexts.empty()) {
        const auto [context, context_addr] = *emuenv.gxm.deferred_contexts.begin();
        const int result = destroy_gxm_context_metal3(emuenv, context, context_addr, force_backend_destroy);
        if (result < 0) {
            LOG_WARN("Failed to destroy deferred GXM context during cleanup: {}", log_hex(result));
            if (emuenv.renderer->current_backend == renderer::Backend::Metal && !force_backend_destroy)
                return;
            emuenv.gxm.deferred_contexts.erase(context);
        }
    }
    }
}

namespace gxm {
void shutdown_metal3(EmuEnvState &emuenv) {
    emuenv.display.abort = true;
    emuenv.renderer->notification_ready.notify_all();
    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        emuenv.gxm.display_queue.abort_synchronized();
    else
        emuenv.gxm.display_queue.abort();
    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        renderer::request_metal_render_abort(*emuenv.renderer);
    else
        emuenv.renderer->render_abort = true;
    invalidate_sync_objects(emuenv.gxm);
    emuenv.renderer->command_finish_one.notify_all();

    // wait for any deferred GXM callback to finish before continuing shutdown
    const std::lock_guard<std::mutex> callback_guard(emuenv.gxm.callback_lock);
    }
}

static int destroy_gxm_render_target_metal3(EmuEnvState &emuenv, SceGxmRenderTarget *render_target, const Address render_target_addr, const bool force_backend_destroy) {
    if (!render_target) {
        return static_cast<int>(SCE_GXM_ERROR_INVALID_POINTER);
    }

    const auto tracked_render_target = emuenv.gxm.render_targets.find(render_target);
    if (tracked_render_target == emuenv.gxm.render_targets.end() || tracked_render_target->second != render_target_addr) {
        return static_cast<int>(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (force_backend_destroy) {
        renderer::destroy_render_target_during_shutdown(*emuenv.renderer, render_target->renderer);
    } else {
        renderer::destroy_render_target(*emuenv.renderer, render_target->renderer);
        if (emuenv.renderer->current_backend == renderer::Backend::Metal && render_target->renderer)
            return static_cast<int>(SCE_GXM_ERROR_DRIVER);
    }

    emuenv.gxm.render_targets.erase(tracked_render_target);
    free(emuenv.mem, Ptr<SceGxmRenderTarget>(render_target_addr));
    return 0;
    }

namespace gxm {
void destroy_all_render_targets_metal3(EmuEnvState &emuenv, const bool force_backend_destroy) {
    while (!emuenv.gxm.render_targets.empty()) {
        const auto [render_target, render_target_addr] = *emuenv.gxm.render_targets.begin();
        const int result = destroy_gxm_render_target_metal3(emuenv, render_target, render_target_addr, force_backend_destroy);
        if (result < 0) {
            LOG_WARN("Failed to destroy render target during cleanup: {}", log_hex(result));
            if (emuenv.renderer->current_backend == renderer::Backend::Metal && !force_backend_destroy)
                return;
            emuenv.gxm.render_targets.erase(render_target);
        }
    }
    }
}

static bool metal_program_is_live_metal3(const renderer::State &state, GxmState &gxm, Address program,
    MetalGxmProgramKind kind, const char *what) {
    if (state.current_backend != renderer::Backend::Metal)
        return true;

    // Like Plus, this rejects freed and wrong-kind addresses. It does not pin
    // a wrapper against concurrent release or distinguish same-kind reuse.
    const char *reason;
    uint64_t warning_count;
    {
        const std::lock_guard<std::mutex> lock(metal_gxm_bookkeeping(gxm).metal_programs_mutex);
        const auto live = metal_gxm_bookkeeping(gxm).metal_live_programs.find(program);
        if (live != metal_gxm_bookkeeping(gxm).metal_live_programs.end() && live->second == kind)
            return true;
        reason = live == metal_gxm_bookkeeping(gxm).metal_live_programs.end() ? "freed or never created" : "address holds the other program kind";
        warning_count = ++metal_gxm_bookkeeping(gxm).metal_stale_program_warnings;
    }
    if (warning_count <= 8 || (warning_count & 1023) == 0)
        LOG_WARN("Metal {}: program 0x{:X} is not a live {} program ({}) - ignored (#{})",
            what, program, kind == MetalGxmProgramKind::Vertex ? "vertex" : "fragment", reason, warning_count);
    return false;
}

static void metal_program_register_metal3(EmuEnvState &emuenv, Address program, MetalGxmProgramKind kind) {
    if (emuenv.renderer->current_backend != renderer::Backend::Metal)
        return;
    const std::lock_guard<std::mutex> lock(metal_gxm_bookkeeping(emuenv.gxm).metal_programs_mutex);
    metal_gxm_bookkeeping(emuenv.gxm).metal_live_programs[program] = kind;
}

static void gxmSetProgram_metal3(renderer::State &state, GxmState &gxm, SceGxmContext *context, Ptr<const void> program,
    bool is_fragment, const MemState &mem) {
    if (!metal_program_is_live_metal3(state, gxm, program.address(),
            is_fragment ? MetalGxmProgramKind::Fragment : MetalGxmProgramKind::Vertex, "gxmSetProgram"))
        return;
#ifdef __APPLE__
    if (state.current_backend == renderer::Backend::Metal) {
        const auto binding = is_fragment ? program.cast<const SceGxmFragmentProgram>().get(mem)->renderer_data->metal_binding
                                         : program.cast<const SceGxmVertexProgram>().get(mem)->renderer_data->metal_binding;
        renderer::set_program_metal3(state, context->renderer.get(), program, is_fragment, binding);
        return;
    }
#endif
    renderer::set_program(state, context->renderer.get(), program, is_fragment);
}

static void gxmContextStateRestore_metal3(renderer::State &state, GxmState &gxm, SceGxmContext *context, const bool sync_viewport_and_clip, const MemState &mem) {
    if (sync_viewport_and_clip) {
        renderer::set_region_clip(state, context->renderer.get(), context->state.region_clip_mode,
            context->state.region_clip_min.x, context->state.region_clip_max.x, context->state.region_clip_min.y,
            context->state.region_clip_max.y);

        update_viewport(state, context);
    }

    for (uint32_t index = 0; index < static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.yuv_profiles.size(); ++index)
        renderer::set_yuv_profile(state, context->renderer.get(), index, static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.yuv_profiles[index]);

    renderer::set_cull_mode(state, context->renderer.get(), context->state.cull_mode);
    renderer::set_depth_bias(state, context->renderer.get(), true, context->state.front_depth_bias_factor, context->state.front_depth_bias_units);
    renderer::set_depth_bias(state, context->renderer.get(), false, context->state.back_depth_bias_factor, context->state.back_depth_bias_units);
    renderer::set_depth_func(state, context->renderer.get(), true, context->state.front_depth_func);
    renderer::set_depth_func(state, context->renderer.get(), false, context->state.back_depth_func);
    renderer::set_depth_write_enable_mode(state, context->renderer.get(), true, context->state.front_depth_write_enable);
    renderer::set_depth_write_enable_mode(state, context->renderer.get(), false, context->state.back_depth_write_enable);
    renderer::set_point_line_width(state, context->renderer.get(), true, context->state.front_point_line_width);
    renderer::set_point_line_width(state, context->renderer.get(), false, context->state.back_point_line_width);
    renderer::set_polygon_mode(state, context->renderer.get(), true, context->state.front_polygon_mode);
    renderer::set_polygon_mode(state, context->renderer.get(), false, context->state.back_polygon_mode);
    renderer::set_two_sided_enable(state, context->renderer.get(), context->state.two_sided);
    renderer::set_side_fragment_program_enable(state, context->renderer.get(), true,
        context->state.front_side_fragment_program_mode);
    renderer::set_side_fragment_program_enable(state, context->renderer.get(), false,
        context->state.back_side_fragment_program_mode);
    renderer::set_stencil_func(state, context->renderer.get(), true, context->state.front_stencil.func, context->state.front_stencil.stencil_fail,
        context->state.front_stencil.depth_fail, context->state.front_stencil.depth_pass, context->state.front_stencil.compare_mask,
        context->state.front_stencil.write_mask);
    renderer::set_stencil_func(state, context->renderer.get(), false, context->state.back_stencil.func, context->state.back_stencil.stencil_fail,
        context->state.back_stencil.depth_fail, context->state.back_stencil.depth_pass, context->state.back_stencil.compare_mask,
        context->state.back_stencil.write_mask);
    renderer::set_stencil_ref(state, context->renderer.get(), true, context->state.front_stencil.ref);
    renderer::set_stencil_ref(state, context->renderer.get(), false, context->state.back_stencil.ref);

    if (state.current_backend == renderer::Backend::Metal) {
        renderer::set_visibility_index(state, context->renderer.get(), context->state.visibility_enable,
            context->state.visibility_index, context->state.visibility_is_increment);
        renderer::set_back_visibility_index(state, context->renderer.get(), static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_enable,
            static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_index, static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_is_increment);
    } else if (state.features.enable_memory_mapping) {
        context->state.visibility_enable = false;
        context->state.visibility_index = 0;
        context->state.visibility_is_increment = true;
        renderer::set_visibility_index(state, context->renderer.get(), false, 0, true);
    }

    if (context->state.vertex_program) {
        gxmSetProgram_metal3(state, gxm, context, context->state.vertex_program, false, mem);

        context->is_vert_texture_dirty.set();
    }

    // The uniform buffer, vertex stream will be uploaded later, for now only need to resync de textures
    if (context->state.fragment_program) {
        gxmSetProgram_metal3(state, gxm, context, context->state.fragment_program, true, mem);

        context->is_frag_texture_dirty.set();
    }
    }

int export_sceGxmBeginCommandList_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext) {
    TRACY_FUNC(sceGxmBeginCommandList, deferredContext);
    if (!deferredContext) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (deferredContext->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (deferredContext->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_WITHIN_COMMAND_LIST);
    }

    if (!renderer::metal::gxm_metadata(*deferredContext->renderer).enabled) {
        deferredContext->state.fragment_ring_buffer_used = 0;
        deferredContext->state.vertex_ring_buffer_used = 0;
    } else {
        renderer::metal::gxm_metadata(*deferredContext->renderer).recording_failed = false;
    }

    if (!deferredContext->make_new_alloc_space(emuenv.kernel, emuenv.mem, thread_id)) {
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
    }

    // in case the same vdm buffer was used for two consecutive command lists
    deferredContext->alloc_space_start = deferredContext->alloc_space;

    if (!renderer::metal::gxm_metadata(*deferredContext->renderer).enabled && !deferredContext->state.vertex_ring_buffer) {
        deferredContext->state.vertex_ring_buffer = gxmRunDeferredMemoryCallback_metal3(emuenv.kernel, emuenv.mem, emuenv.gxm.callback_lock, deferredContext->state.vertex_ring_buffer_size,
            deferredContext->state.vertex_memory_callback, deferredContext->state.memory_callback_userdata, DEFAULT_RING_SIZE, thread_id);

        if (!deferredContext->state.vertex_ring_buffer) {
            return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
        }
    }

    if (!renderer::metal::gxm_metadata(*deferredContext->renderer).enabled && !deferredContext->state.fragment_ring_buffer) {
        deferredContext->state.fragment_ring_buffer = gxmRunDeferredMemoryCallback_metal3(emuenv.kernel, emuenv.mem, emuenv.gxm.callback_lock, deferredContext->state.fragment_ring_buffer_size,
            deferredContext->state.fragment_memory_callback, deferredContext->state.memory_callback_userdata, DEFAULT_RING_SIZE, thread_id);

        if (!deferredContext->state.fragment_ring_buffer) {
            return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
        }
    }

    // Set command allocate functions
    KernelState *kernel = &emuenv.kernel;
    MemState *mem = &emuenv.mem;

    deferredContext->renderer->alloc_func = [deferredContext, kernel, mem, thread_id]() {
        return deferredContext->allocate_new_command(*kernel, *mem, thread_id);
    };

    deferredContext->renderer->free_func = [](renderer::Command *cmd) {
        // do not delete here, commands will be deleted when they are overwritten
    };

    // Keep the pending list only after the required callback allocations.
    // Metal obtains uniform memory lazily; a failed Begin must not leave an
    // orphan that a later Begin overwrites.
    deferredContext->curr_command_list = new SceGxmCommandList();

    // Begin the command list by white washing previous command list, and restoring deferred state
    renderer::reset_command_list(deferredContext->renderer->command_list);
    gxmContextStateRestore_metal3(*emuenv.renderer, emuenv.gxm, deferredContext, false, emuenv.mem);
    if (renderer::metal::gxm_metadata(*deferredContext->renderer).enabled && renderer::metal::gxm_metadata(*deferredContext->renderer).recording_failed) {
        abort_metal_deferred_recording_metal3(deferredContext);
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
    }

    deferredContext->state.active = true;

    return 0;
    }

void export_sceGxmColorSurfaceGetClip_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmColorSurface *surface, uint32_t *xMin, uint32_t *yMin, uint32_t *xMax, uint32_t *yMax) {
    TRACY_FUNC(sceGxmColorSurfaceGetClip, surface, xMin, yMin, xMax, yMax);
    assert(surface);
    if (xMin)
        *xMin = renderer::metal::color_surface_bits(*surface).clip_enabled ? renderer::metal::color_surface_bits(*surface).clip_x_min : 0;
    if (yMin)
        *yMin = renderer::metal::color_surface_bits(*surface).clip_enabled ? renderer::metal::color_surface_bits(*surface).clip_y_min : 0;
    if (xMax)
        *xMax = renderer::metal::color_surface_bits(*surface).clip_enabled ? renderer::metal::color_surface_bits(*surface).clip_x_max : (surface->width ? surface->width - 1 : 0);
    if (yMax)
        *yMax = renderer::metal::color_surface_bits(*surface).clip_enabled ? renderer::metal::color_surface_bits(*surface).clip_y_max : (surface->height ? surface->height - 1 : 0);
    }

SceGxmColorSurfaceDitherMode export_sceGxmColorSurfaceGetDitherMode_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmColorSurface *surface) {
    TRACY_FUNC(sceGxmColorSurfaceGetDitherMode, surface);
    assert(surface);
    return renderer::metal::color_surface_bits(*surface).dither_enabled ? SCE_GXM_COLOR_SURFACE_DITHER_ENABLED : SCE_GXM_COLOR_SURFACE_DITHER_DISABLED;
    }

int export_sceGxmColorSurfaceInit_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, SceGxmColorFormat colorFormat, SceGxmColorSurfaceType surfaceType, SceGxmColorSurfaceScaleMode scaleMode, SceGxmOutputRegisterSize outputRegisterSize, uint32_t width, uint32_t height, uint32_t strideInPixels, Ptr<void> data) {
    TRACY_FUNC(sceGxmColorSurfaceInit, surface, colorFormat, surfaceType, scaleMode, outputRegisterSize, width, height, strideInPixels, data);
    if (!surface || !data)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (width == 0 || width > 4096 || height == 0 || height > 4096)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);

    if (strideInPixels & 1)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_ALIGNMENT);

    if ((strideInPixels < width) || ((data.address() & 3) != 0))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);

    // if the surface is swizzled, width and height must be power of 2
    if (surfaceType == SCE_GXM_COLOR_SURFACE_SWIZZLED && ((width & (width - 1)) || (height & (height - 1))))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_ALIGNMENT);

    SceGxmTextureFormat tex_format;
    if (!gxm::convert_color_format_to_texture_format_metal3(colorFormat, tex_format)) {
        LOG_WARN("Unable to convert color surface type 0x{:X} to texture format enum for background texture of color surface!", static_cast<std::uint32_t>(colorFormat));
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    memset(surface, 0, sizeof(SceGxmColorSurface));
    surface->disabled = 0;
    surface->downscale = scaleMode == SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE;
    surface->width = width;
    surface->height = height;
    surface->strideInPixels = strideInPixels;
    surface->data = data;
    surface->colorFormat = colorFormat;
    surface->surfaceType = surfaceType;
    renderer::metal::color_surface_bits(*surface).outputRegisterSize = outputRegisterSize;

    // initialize the background texture
    switch (surfaceType) {
    case SCE_GXM_COLOR_SURFACE_SWIZZLED:
        return CALL_EXPORT(sceGxmTextureInitSwizzled, &surface->backgroundTex, surface->data, tex_format, width, height, 1);
    case SCE_GXM_COLOR_SURFACE_TILED:
        return CALL_EXPORT(sceGxmTextureInitTiled, &surface->backgroundTex, surface->data, tex_format, width, height, 1);
    default:
        // linear
        if (align(width, 8) == strideInPixels) {
            return CALL_EXPORT(sceGxmTextureInitLinear, &surface->backgroundTex, surface->data, tex_format, width, height, 1);
        } else {
            uint32_t stride_in_bytes = static_cast<uint32_t>(gxm::bits_per_pixel(gxm::get_base_format(colorFormat)) * strideInPixels / 8);
            return CALL_EXPORT(sceGxmTextureInitLinearStrided, &surface->backgroundTex, surface->data, tex_format, width, height, stride_in_bytes);
        }
    }
    }

void export_sceGxmColorSurfaceSetClip_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, uint32_t xMin, uint32_t yMin, uint32_t xMax, uint32_t yMax) {
    TRACY_FUNC(sceGxmColorSurfaceSetClip, surface, xMin, yMin, xMax, yMax);
    assert(surface);
    renderer::metal::color_surface_bits(*surface).clip_x_min = std::min(xMin, 4095u);
    renderer::metal::color_surface_bits(*surface).clip_y_min = std::min(yMin, 4095u);
    renderer::metal::color_surface_bits(*surface).clip_x_max = std::min(xMax, 4095u);
    renderer::metal::color_surface_bits(*surface).clip_y_max = std::min(yMax, 4095u);
    renderer::metal::color_surface_bits(*surface).clip_enabled = 1;
    }

int export_sceGxmColorSurfaceSetDitherMode_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, SceGxmColorSurfaceDitherMode ditherMode) {
    TRACY_FUNC(sceGxmColorSurfaceSetDitherMode, surface, ditherMode);
    if (!surface) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (ditherMode != SCE_GXM_COLOR_SURFACE_DITHER_DISABLED && ditherMode != SCE_GXM_COLOR_SURFACE_DITHER_ENABLED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    renderer::metal::color_surface_bits(*surface).dither_enabled = ditherMode == SCE_GXM_COLOR_SURFACE_DITHER_ENABLED;
    return 0;
    }

int export_sceGxmColorSurfaceSetFormat_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, SceGxmColorFormat format) {
    TRACY_FUNC(sceGxmColorSurfaceSetFormat, surface, format);
    if (!surface) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    SceGxmTextureFormat tex_format;
    if (!gxm::convert_color_format_to_texture_format_metal3(format, tex_format)) {
        LOG_WARN("Unable to convert color surface type 0x{:X} to texture format enum for background texture of color surface!", static_cast<std::uint32_t>(format));
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    surface->colorFormat = format;
    return CALL_EXPORT(sceGxmTextureSetFormat, &surface->backgroundTex, tex_format);
    }

void export_sceGxmColorSurfaceSetScaleMode_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmColorSurface *surface, SceGxmColorSurfaceScaleMode scaleMode) {
    TRACY_FUNC(sceGxmColorSurfaceSetScaleMode, surface, scaleMode);
    if (!surface && emuenv.renderer->current_backend == renderer::Backend::Metal)
        return;
    assert(surface);
    surface->downscale = scaleMode == SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE;
    }

int export_sceGxmCreateContext_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmContextParams *params, Ptr<SceGxmContext> *context) {
    TRACY_FUNC(sceGxmCreateContext, params, context);
    if (!params || !context)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    // Plus permits only one immediate GXM context. Keep that guest-visible
    // limit in the native Metal path without changing the Vulkan backend.
    if (emuenv.renderer->current_backend == renderer::Backend::Metal
        && !emuenv.gxm.immediate_contexts.empty())
        return RET_ERROR(SCE_GXM_ERROR_ALREADY_INITIALIZED);

    if (params->hostMemSize < sizeof(SceGxmContext)) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    // This structure needs 8-byte alignment
    *context = Ptr<SceGxmContext>(align(params->hostMem.address(), 8));

    SceGxmContext *ctx = context->get(emuenv.mem);
    new (ctx) SceGxmContext(emuenv.gxm.callback_lock);

    ctx->state.fragment_ring_buffer = params->fragmentRingBufferMem;
    ctx->state.vertex_ring_buffer = params->vertexRingBufferMem;
    ctx->state.fragment_ring_buffer_size = params->fragmentRingBufferMemSize;
    ctx->state.vertex_ring_buffer_size = params->vertexRingBufferMemSize;

    ctx->state.type = SCE_GXM_CONTEXT_TYPE_IMMEDIATE;
    ctx->state.visibility_index = 0;

    if (!renderer::create_context(*emuenv.renderer, ctx->renderer)) {
        ctx->~SceGxmContext();
        context->reset();
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    }

    renderer::metal::gxm_metadata(*ctx->renderer).enabled = emuenv.renderer->current_backend == renderer::Backend::Metal;
    if (renderer::metal::gxm_metadata(*ctx->renderer).enabled)
        renderer::metal::gxm_metadata(*ctx->renderer).owner = &emuenv.gxm;

    // Set VDM buffer space
    ctx->state.vdm_buffer = params->vdmRingBufferMem;
    ctx->state.vdm_buffer_size = params->vdmRingBufferMemSize;

    ctx->make_new_alloc_space(emuenv.kernel, emuenv.mem, thread_id);

    // Set command allocate functions
    // The command buffer will not be reallocated, so this is fine to use this thread ID
    KernelState *kernel = &emuenv.kernel;
    MemState *mem = &emuenv.mem;

    ctx->renderer->alloc_func = [ctx, kernel, mem, thread_id]() {
        return ctx->allocate_new_command(*kernel, *mem, thread_id);
    };

    ctx->renderer->free_func = [ctx](renderer::Command *cmd) {
        return ctx->free_new_command(cmd);
    };

    emuenv.gxm.immediate_contexts.emplace(ctx, context->address());
    if (emuenv.gxm.last_immediate_context == 0)
        emuenv.gxm.last_immediate_context = context->address();
    return 0;
    }

int export_sceGxmCreateDeferredContext_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmDeferredContextParams *params, Ptr<SceGxmContext> *deferredContext) {
    TRACY_FUNC(sceGxmCreateDeferredContext, params, deferredContext);
    if (!params || !deferredContext)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (params->hostMemSize < sizeof(SceGxmContext)) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    *deferredContext = Ptr<SceGxmContext>(align(params->hostMem.address(), 8));
    SceGxmContext *ctx = deferredContext->get(emuenv.mem);
    new (ctx) SceGxmContext(emuenv.gxm.callback_lock);

    ctx->state.vertex_memory_callback = params->vertexCallback;
    ctx->state.fragment_memory_callback = params->fragmentCallback;
    ctx->state.vdm_memory_callback = params->vdmCallback;
    ctx->state.memory_callback_userdata = params->userData;

    if (emuenv.renderer->current_backend == renderer::Backend::Metal) {
        // Plus accepts buffers supplied when the deferred context is created.
        // A game need not repeat them through the SetDeferredContext exports.
        const bool has_vdm = params->vdmBufferMem && params->vdmBufferMemSize;
        const bool has_vertex = params->vertexBufferMem && params->vertexBufferMemSize;
        const bool has_fragment = params->fragmentBufferMem && params->fragmentBufferMemSize;
        ctx->state.vdm_buffer = has_vdm ? params->vdmBufferMem : Ptr<void>();
        ctx->state.vdm_buffer_size = has_vdm ? params->vdmBufferMemSize : 0;
        ctx->state.vertex_ring_buffer = has_vertex ? params->vertexBufferMem : Ptr<void>();
        ctx->state.vertex_ring_buffer_size = has_vertex ? params->vertexBufferMemSize : 0;
        ctx->state.fragment_ring_buffer = has_fragment ? params->fragmentBufferMem : Ptr<void>();
        ctx->state.fragment_ring_buffer_size = has_fragment ? params->fragmentBufferMemSize : 0;
    }

    ctx->state.type = SCE_GXM_CONTEXT_TYPE_DEFERRED;
    ctx->state.visibility_index = 0;

    // Create a generic context. This is only used for storing command list
    ctx->renderer = std::make_unique<renderer::metal::MetalGuestContext>();
    renderer::metal::gxm_metadata(*ctx->renderer).enabled = emuenv.renderer->current_backend == renderer::Backend::Metal;
    if (renderer::metal::gxm_metadata(*ctx->renderer).enabled)
        renderer::metal::gxm_metadata(*ctx->renderer).owner = &emuenv.gxm;
    emuenv.gxm.deferred_contexts.emplace(ctx, deferredContext->address());

    return 0;
    }

int export_sceGxmDisplayQueueAddEntry_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, Ptr<SceGxmSyncObject> oldBuffer, Ptr<SceGxmSyncObject> newBuffer, Ptr<const void> callbackData) {
    TRACY_FUNC(sceGxmDisplayQueueAddEntry, oldBuffer, newBuffer, callbackData);
    if (!oldBuffer || !newBuffer)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    const Address address = alloc(emuenv.mem, emuenv.gxm.params.displayQueueCallbackDataSize, __FUNCTION__);
    const Ptr<void> ptr(address);
    memcpy(ptr.get(emuenv.mem), callbackData.get(emuenv.mem), emuenv.gxm.params.displayQueueCallbackDataSize);

    DisplayFrameInfo *frame = predict_next_image(emuenv, newBuffer.address());

    // Block future rendering by setting values of sync object
    SceGxmSyncObject *oldBufferSync = oldBuffer.get(emuenv.mem);
    SceGxmSyncObject *newBufferSync = newBuffer.get(emuenv.mem);

    DisplayCallback display_callback{
        .data = address,
        .old_sync = oldBuffer,
        .new_sync = newBuffer,
        .old_sync_timestamp = oldBufferSync->timestamp_ahead,
        .new_sync_timestamp = newBufferSync->timestamp_ahead,
        .frame_predicted = frame != nullptr
    };

    oldBufferSync->last_display = ++oldBufferSync->timestamp_ahead;
    if (oldBufferSync != newBufferSync)
        newBufferSync->last_display = ++newBufferSync->timestamp_ahead;
    emuenv.gxm.last_display_global = emuenv.gxm.global_timestamp.fetch_add(1, std::memory_order_relaxed);

    // function may be blocking here (expected behavior)
    emuenv.gxm.display_queue.push(display_callback);

    // TODO: I do this because the sync function does not have access to the display state, but this is not great
    renderer::Context *active_renderer_context = nullptr;
    if (emuenv.gxm.last_immediate_context != 0) {
        const auto immediate_context = std::ranges::find_if(emuenv.gxm.immediate_contexts, [&](const auto &entry) {
            return entry.second == emuenv.gxm.last_immediate_context;
        });
        if (immediate_context != emuenv.gxm.immediate_contexts.end())
            active_renderer_context = immediate_context->first->renderer.get();
    }

    renderer::send_single_command(*emuenv.renderer, nullptr, renderer::CommandOpcode::NewFrame, false, frame, &emuenv.display, active_renderer_context);

    if (emuenv.renderer->current_backend != renderer::Backend::Metal
        && emuenv.gxm.params.displayQueueMaxPendingCount == 1)
        // Metal keeps the callback queued until it returns, so the next push
        // already enforces a one-entry limit without draining this entry here.
        emuenv.gxm.display_queue.wait_empty();

    return 0;
    }

static uint32_t metal_mapped_uniform_size_metal3(const GxmState &gxm, Address address, uint32_t requested_size) {
    const uint64_t requested_end = uint64_t(address) + requested_size;
    const auto next = gxm.memory_mapped_regions.upper_bound(address);
    uint64_t mapped_end = address;

    // A later overlapping mapping may end before the binding while an older
    // mapping still contains it. Find the full mapped prefix from this address.
    for (auto it = next; it != gxm.memory_mapped_regions.begin();) {
        --it;
        mapped_end = std::max(mapped_end, uint64_t(it->first) + it->second.size);
        if (mapped_end >= requested_end)
            return requested_size;
    }
    if (mapped_end == address)
        return requested_size; // No GXM mapping at the binding address.

    // Adjacent or overlapping mappings extend the readable prefix.
    for (auto it = next; it != gxm.memory_mapped_regions.end() && it->first <= mapped_end; ++it) {
        mapped_end = std::max(mapped_end, uint64_t(it->first) + it->second.size);
        if (mapped_end >= requested_end)
            return requested_size;
    }
    return static_cast<uint32_t>(mapped_end - address);
}

static void gxmSetUniformBuffers_metal3(renderer::State &state, GxmState &gxm, SceGxmContext *context, const SceGxmProgram &program, std::span<UniformBuffer> buffers, const UniformBufferSizes &sizes, const MemState &mem) {
    for (size_t i = 0; i < buffers.size(); i++) {
        if (!buffers[i] || sizes.at(i) == 0) {
            continue;
        }

        uint32_t bytes_to_copy = sizes.at(i) * 4;
        if (state.current_backend == renderer::Backend::Metal)
            bytes_to_copy = metal_mapped_uniform_size_metal3(gxm, buffers[i].address(), bytes_to_copy);
        if (sizes.at(i) == SCE_GXM_MAX_UB_IN_FLOAT_UNIT) {
            if (state.current_backend != renderer::Backend::Metal) {
                // The region containing the address is the last one starting at or before it.
                auto ite = gxm.memory_mapped_regions.upper_bound(buffers[i].address());
                if (ite != gxm.memory_mapped_regions.begin()) {
                    --ite;
                    if (uint64_t(ite->first) + ite->second.size > buffers[i].address())
                        bytes_to_copy = static_cast<uint32_t>(std::min<uint64_t>(
                            uint64_t(ite->first) + ite->second.size - buffers[i].address(), bytes_to_copy));
                }
            }

            // Check other UB friends and bound the size
            for (size_t j = 0; j < buffers.size(); j++) {
                if (i == j) {
                    continue;
                }

                if (buffers[j].address() > buffers[i].address()) {
                    bytes_to_copy = std::min<std::uint32_t>(buffers[j].address() - buffers[i].address(), bytes_to_copy);
                }
            }
        }
        renderer::set_uniform_buffer_metal3(state, context->renderer.get(), !program.is_fragment(), i, bytes_to_copy, buffers[i]);
    }
    }

static bool metal_vertex_stream_sizes_metal3(const SceGxmVertexProgram &program, uint32_t max_index,
    uint32_t instances, std::array<size_t, SCE_GXM_MAX_VERTEX_STREAMS> &sizes, uint32_t &used) {
    if (!instances || program.streams.size() > SCE_GXM_MAX_VERTEX_STREAMS)
        return false;
    if (!program.renderer_data || !program.renderer_data->metal_binding)
        return false;
    const auto &attribute_infos = program.renderer_data->metal_binding->attribute_infos;
    for (const auto &attribute : program.attributes) {
        const auto found = attribute_infos.find(attribute.regIndex);
        if (found == attribute_infos.end())
            continue;
        if (attribute.streamIndex >= program.streams.size())
            return false;
        const auto shape = renderer::metal::vertex_attribute_shape(attribute, found->second);
        if (!shape)
            return false;
        const auto &stream = program.streams[attribute.streamIndex];
        const uint64_t last = gxm::is_stream_instancing(static_cast<SceGxmIndexSource>(stream.indexSource))
            ? uint64_t(instances) - 1 : uint64_t(max_index);
        const uint64_t length = last * stream.stride + attribute.offset + shape->byte_size();
        if (length > UINT32_MAX)
            return false;
        sizes[attribute.streamIndex] = std::max(sizes[attribute.streamIndex], size_t(length));
        used |= 1u << attribute.streamIndex;
    }
    return true;
}

static int gxmDrawElementGeneral_metal3(EmuEnvState &emuenv, const char *export_name, const SceUID thread_id, SceGxmContext *context, SceGxmPrimitiveType primType, SceGxmIndexFormat indexType, Ptr<const void> indexData, uint32_t indexCount, uint32_t instanceCount) {
    if (!context || !indexData)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (!context->state.active) {
        if (context->state.type == SCE_GXM_CONTEXT_TYPE_DEFERRED) {
            return RET_ERROR(SCE_GXM_ERROR_NOT_WITHIN_COMMAND_LIST);
        } else {
            return RET_ERROR(SCE_GXM_ERROR_NOT_WITHIN_SCENE);
        }
    }

    if (!context->state.fragment_program || !context->state.vertex_program) {
        return RET_ERROR(SCE_GXM_ERROR_NULL_PROGRAM);
    }

    // Plus applies a bound precomputed state to ordinary draws too. Keep the
    // other backends on their existing context-only path.
    const bool native_metal = emuenv.renderer->current_backend == renderer::Backend::Metal;
    const bool empty_metal_draw = native_metal && (!indexCount || !instanceCount);
    const auto *pre_vert = native_metal ? context->state.precomputed_vertex_state.get(emuenv.mem) : nullptr;
    const auto *pre_frag = native_metal ? context->state.precomputed_fragment_state.get(emuenv.mem) : nullptr;
    const auto vert_program_ptr = pre_vert && pre_vert->program ? pre_vert->program : context->state.vertex_program;
    const auto frag_program_ptr = pre_frag && pre_frag->program ? pre_frag->program : context->state.fragment_program;

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, frag_program_ptr.address(), MetalGxmProgramKind::Fragment, export_name)
        || !metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, vert_program_ptr.address(), MetalGxmProgramKind::Vertex, export_name))
        return SCE_GXM_ERROR_NULL_PROGRAM;

    const SceGxmFragmentProgram &gxm_fragment_program = *frag_program_ptr.get(emuenv.mem);
    const SceGxmVertexProgram &gxm_vertex_program = *vert_program_ptr.get(emuenv.mem);

    // Set uniforms
    const SceGxmProgram &vertex_program_gxp = *gxm_vertex_program.program.get(emuenv.mem);
    const SceGxmProgram &fragment_program_gxp = *gxm_fragment_program.program.get(emuenv.mem);

    const void *indices_ptr = indexData.get(emuenv.mem);
    if (native_metal && !empty_metal_draw && !indices_ptr)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (native_metal && (context->last_precomputed || pre_vert || pre_frag)) {
        // Uniform commands resolve their layout from the currently bound
        // program, so the override (or its removal) must be queued first.
        gxmSetProgram_metal3(*emuenv.renderer, emuenv.gxm, context, vert_program_ptr, false, emuenv.mem);
        gxmSetProgram_metal3(*emuenv.renderer, emuenv.gxm, context, frag_program_ptr, true, emuenv.mem);
        context->last_precomputed = false;
    }

    const auto &vert_sizes = gxm_vertex_program.renderer_data->uniform_buffer_sizes;
    const auto &frag_sizes = gxm_fragment_program.renderer_data->uniform_buffer_sizes;
    std::span<UniformBuffer> vert_buffers = pre_vert && pre_vert->uniform_buffers
        ? std::span(pre_vert->uniform_buffers.get(emuenv.mem), std::min<size_t>(pre_vert->buffer_count, vert_sizes.size()))
        : std::span(context->state.vertex_uniform_buffers);
    std::span<UniformBuffer> frag_buffers = pre_frag && pre_frag->uniform_buffers
        ? std::span(pre_frag->uniform_buffers.get(emuenv.mem), std::min<size_t>(pre_frag->buffer_count, frag_sizes.size()))
        : std::span(context->state.fragment_uniform_buffers);
    gxmSetUniformBuffers_metal3(*emuenv.renderer, emuenv.gxm, context, vertex_program_gxp, vert_buffers, vert_sizes, emuenv.mem);
    gxmSetUniformBuffers_metal3(*emuenv.renderer, emuenv.gxm, context, fragment_program_gxp, frag_buffers, frag_sizes, emuenv.mem);

    if (!native_metal && context->last_precomputed) {
        // Need to re-set the data

        gxmSetProgram_metal3(*emuenv.renderer, emuenv.gxm, context, vert_program_ptr, false, emuenv.mem);
        gxmSetProgram_metal3(*emuenv.renderer, emuenv.gxm, context, frag_program_ptr, true, emuenv.mem);

        context->last_precomputed = false;
    }

    // Metal follows Plus and synchronizes every sampled descriptor. Rebinding
    // dirty flags alone miss edits to a precomputed array and override removal.
    const auto vert_used = gxm_vertex_program.renderer_data->textures_used;
    const auto frag_used = gxm_fragment_program.renderer_data->textures_used;
    const gxp::TextureInfo vert_textures_sync = native_metal ? vert_used : vert_used & context->is_vert_texture_dirty;
    const gxp::TextureInfo frag_textures_sync = native_metal ? frag_used : frag_used & context->is_frag_texture_dirty;
    context->is_vert_texture_dirty &= ~vert_textures_sync;
    context->is_frag_texture_dirty &= ~frag_textures_sync;
    const auto &textures = context->state.textures;
    const SceGxmTexture *frag_textures = pre_frag && pre_frag->textures ? pre_frag->textures.get(emuenv.mem) : textures.data();
    const SceGxmTexture *vert_textures = pre_vert && pre_vert->textures ? pre_vert->textures.get(emuenv.mem) : textures.data() + SCE_GXM_MAX_TEXTURE_UNITS;
    const uint16_t frag_texture_count = pre_frag && pre_frag->textures ? pre_frag->texture_count : SCE_GXM_MAX_TEXTURE_UNITS;
    const uint16_t vert_texture_count = pre_vert && pre_vert->textures ? pre_vert->texture_count : SCE_GXM_MAX_TEXTURE_UNITS;
    for (uint16_t texture_index = 0; texture_index < SCE_GXM_MAX_TEXTURE_UNITS; texture_index++) {
        if (vert_textures_sync[texture_index]) {
            const uint16_t index_position = SCE_GXM_MAX_TEXTURE_UNITS + texture_index;
            renderer::set_texture(*emuenv.renderer, context->renderer.get(), index_position,
                texture_index < vert_texture_count ? vert_textures[texture_index] : textures[index_position]);
        }

        if (frag_textures_sync[texture_index])
            renderer::set_texture(*emuenv.renderer, context->renderer.get(), texture_index,
                texture_index < frag_texture_count ? frag_textures[texture_index] : textures[texture_index]);
    }

    // Update vertex data. We should stores a copy of the data to pass it to GPU later, since another scene
    // may start to overwrite stuff when this scene is being processed in our queue (in case of OpenGL).
    size_t max_index = 0;
    if ((native_metal || !emuenv.renderer->features.enable_memory_mapping) && !empty_metal_draw) {
        // Metal snapshots need explicit bounds even if mapping is enabled.
        if (indexType == SCE_GXM_INDEX_FORMAT_U16) {
            const uint16_t *const data = static_cast<const uint16_t *>(indices_ptr);
            max_index = *std::max_element(&data[0], &data[indexCount]);
        } else {
            const uint32_t *const data = static_cast<const uint32_t *>(indices_ptr);
            max_index = *std::max_element(&data[0], &data[indexCount]);
        }
    }

    std::array<size_t, SCE_GXM_MAX_VERTEX_STREAMS> max_data_length{};
    std::uint32_t stream_used = 0;
    if (native_metal) {
        if (!empty_metal_draw && !metal_vertex_stream_sizes_metal3(gxm_vertex_program, max_index, instanceCount, max_data_length, stream_used))
            return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    } else {
        for (const SceGxmVertexAttribute &attribute : gxm_vertex_program.attributes) {
            if (!emuenv.renderer->features.enable_memory_mapping) {
                const size_t attribute_size = gxm::attribute_format_size(attribute.format) * attribute.componentCount;
                const SceGxmVertexStream &stream = gxm_vertex_program.streams[attribute.streamIndex];
                const SceGxmIndexSource index_source = static_cast<SceGxmIndexSource>(stream.indexSource);
                const size_t data_passed_length = gxm::is_stream_instancing(index_source) ? ((instanceCount - 1) * stream.stride) : (max_index * stream.stride);
                const size_t data_length = attribute.offset + data_passed_length + attribute_size;
                max_data_length[attribute.streamIndex] = std::max<size_t>(max_data_length[attribute.streamIndex], data_length);
            }

            stream_used |= (1 << attribute.streamIndex);
        }
    }

    // Copy and queue upload
    for (size_t stream_index = 0; stream_index < SCE_GXM_MAX_VERTEX_STREAMS; ++stream_index) {
        // Upload it
        if (stream_used & (1 << static_cast<std::uint16_t>(stream_index))) {
            const size_t data_length = max_data_length[stream_index];
            const Ptr<const void> data = context->state.stream_data[stream_index];
            if (native_metal && (!data || uint64_t(data.address()) + data_length > UINT32_MAX))
                return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

            renderer::set_vertex_stream(*emuenv.renderer, context->renderer.get(), stream_index,
                data_length, data);
        }
    }

    renderer::draw(*emuenv.renderer, context->renderer.get(), primType, indexType, indexData, indexCount, instanceCount);
    if (native_metal && renderer::metal::gxm_metadata(*context->renderer).recording_failed)
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);

    // increase the ringbuffer position if a default vertex or fragment buffer was reserved, we know the new position will fit in the ringbuffer
    if (context->was_vert_default_uniform_reserved) {
        const size_t size = native_metal ? renderer::metal::gxm_metadata(*context->renderer).vertex_reserved_bytes
                                       : (size_t)vertex_program_gxp.default_uniform_buffer_count * 4;
        context->state.vertex_ring_buffer_used += size;
        context->was_vert_default_uniform_reserved = false;
    }

    if (context->was_frag_default_uniform_reserved) {
        const size_t size = native_metal ? renderer::metal::gxm_metadata(*context->renderer).fragment_reserved_bytes
                                       : (size_t)fragment_program_gxp.default_uniform_buffer_count * 4;
        context->state.fragment_ring_buffer_used += size;
        context->was_frag_default_uniform_reserved = false;
    }

    if (native_metal)
        context->last_precomputed = pre_vert || pre_frag;
    return 0;
    }

int export_sceGxmDrawInstanced_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmPrimitiveType primType, SceGxmIndexFormat indexType, Ptr<const void> indexData, uint32_t indexCount, uint32_t indexWrap) {
    TRACY_FUNC(sceGxmDrawInstanced, context, primType, indexType, indexData, indexCount, indexWrap);
    if (emuenv.renderer->current_backend == renderer::Backend::Metal && !indexWrap)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    if (indexCount % indexWrap != 0) {
        LOG_WARN("Extra vertexes are requested to be drawn (ignored)");
    }

    return gxmDrawElementGeneral_metal3(emuenv, export_name, thread_id, context, primType, indexType, indexData, indexWrap, indexCount / indexWrap);
    }

int export_sceGxmDrawPrecomputed_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmPrecomputedDraw *draw) {
    TRACY_FUNC(sceGxmDrawPrecomputed, context, draw);
    if (!context) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (!context->state.active) {
        if (context->state.type == SCE_GXM_CONTEXT_TYPE_DEFERRED) {
            return RET_ERROR(SCE_GXM_ERROR_NOT_WITHIN_COMMAND_LIST);
        } else {
            return RET_ERROR(SCE_GXM_ERROR_NOT_WITHIN_SCENE);
        }
    }

    if (!draw) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    const bool native_metal = emuenv.renderer->current_backend == renderer::Backend::Metal;
    const bool empty_metal_draw = native_metal && (!draw->vertex_count || !draw->instance_count);
    if (native_metal && !empty_metal_draw && !draw->index_data.get(emuenv.mem))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    SceGxmPrecomputedVertexState *vertex_state = context->state.precomputed_vertex_state.cast<SceGxmPrecomputedVertexState>().get(emuenv.mem);
    SceGxmPrecomputedFragmentState *fragment_state = context->state.precomputed_fragment_state.cast<SceGxmPrecomputedFragmentState>().get(emuenv.mem);

    // not sure if precomputed uses current program... maybe it does?
    // anyway states have to be made on a program to program basis so this should be safe
    const Ptr<const SceGxmFragmentProgram> fragment_program_gptr = fragment_state && (!native_metal || fragment_state->program) ? fragment_state->program : context->state.fragment_program;
    const Ptr<const SceGxmVertexProgram> vertex_program_gptr = vertex_state && (!native_metal || vertex_state->program) ? vertex_state->program : context->state.vertex_program;

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, fragment_program_gptr.address(), MetalGxmProgramKind::Fragment, export_name)
        || !metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, vertex_program_gptr.address(), MetalGxmProgramKind::Vertex, export_name))
        return SCE_GXM_ERROR_NULL_PROGRAM;

    const SceGxmFragmentProgram *fragment_program = fragment_program_gptr.get(emuenv.mem);
    const SceGxmVertexProgram *vertex_program = vertex_program_gptr.get(emuenv.mem);

    if (!vertex_program || !fragment_program) {
        return RET_ERROR(SCE_GXM_ERROR_NULL_PROGRAM);
    }

    gxmSetProgram_metal3(*emuenv.renderer, emuenv.gxm, context, fragment_program_gptr, true, emuenv.mem);
    gxmSetProgram_metal3(*emuenv.renderer, emuenv.gxm, context, vertex_program_gptr, false, emuenv.mem);

    // Set uniforms
    const SceGxmProgram &vertex_program_gxp = *vertex_program->program.get(emuenv.mem);
    const SceGxmProgram &fragment_program_gxp = *fragment_program->program.get(emuenv.mem);

    std::span<UniformBuffer> vertex_buffers = vertex_state && (!native_metal || vertex_state->uniform_buffers)
        ? std::span(vertex_state->uniform_buffers.get(emuenv.mem), native_metal
                ? std::min<size_t>(vertex_state->buffer_count, vertex_program->renderer_data->uniform_buffer_sizes.size())
                : vertex_state->buffer_count)
        : std::span(context->state.vertex_uniform_buffers);
    std::span<UniformBuffer> fragment_buffers = fragment_state && (!native_metal || fragment_state->uniform_buffers)
        ? std::span(fragment_state->uniform_buffers.get(emuenv.mem), native_metal
                ? std::min<size_t>(fragment_state->buffer_count, fragment_program->renderer_data->uniform_buffer_sizes.size())
                : fragment_state->buffer_count)
        : std::span(context->state.fragment_uniform_buffers);

    gxmSetUniformBuffers_metal3(*emuenv.renderer, emuenv.gxm, context, vertex_program_gxp, vertex_buffers, vertex_program->renderer_data->uniform_buffer_sizes,
        emuenv.mem);

    gxmSetUniformBuffers_metal3(*emuenv.renderer, emuenv.gxm, context, fragment_program_gxp, fragment_buffers, fragment_program->renderer_data->uniform_buffer_sizes,
        emuenv.mem);

    // Update vertex data. We should stores a copy of the data to pass it to GPU later, since another scene
    // may start to overwrite stuff when this scene is being processed in our queue (in case of OpenGL).
    uint32_t max_index = 0;
    if ((native_metal || !emuenv.renderer->features.enable_memory_mapping) && !empty_metal_draw) {
        // Metal snapshots need explicit bounds even if mapping is enabled.
        if (draw->index_format == SCE_GXM_INDEX_FORMAT_U16) {
            const uint16_t *const data = draw->index_data.cast<const uint16_t>().get(emuenv.mem);
            max_index = *std::max_element(&data[0], &data[draw->vertex_count]);
        } else {
            const uint32_t *const data = draw->index_data.cast<const uint32_t>().get(emuenv.mem);
            max_index = *std::max_element(&data[0], &data[draw->vertex_count]);
        }
    }

    // set all textures that are used and mark them as dirty
    const gxp::TextureInfo vert_textures_sync = vertex_program->renderer_data->textures_used;
    context->is_vert_texture_dirty |= vert_textures_sync;
    const gxp::TextureInfo frag_textures_sync = fragment_program->renderer_data->textures_used;
    context->is_frag_texture_dirty |= frag_textures_sync;
    const SceGxmTexture *frag_textures = fragment_state && (!native_metal || fragment_state->textures) ? fragment_state->textures.get(emuenv.mem) : context->state.textures.data();
    const SceGxmTexture *vert_textures = vertex_state && (!native_metal || vertex_state->textures) ? vertex_state->textures.get(emuenv.mem) : (context->state.textures.data() + SCE_GXM_MAX_TEXTURE_UNITS);
    const uint16_t frag_texture_count = native_metal && fragment_state && fragment_state->textures ? fragment_state->texture_count : SCE_GXM_MAX_TEXTURE_UNITS;
    const uint16_t vert_texture_count = native_metal && vertex_state && vertex_state->textures ? vertex_state->texture_count : SCE_GXM_MAX_TEXTURE_UNITS;
    for (uint16_t texture_index = 0; texture_index < SCE_GXM_MAX_TEXTURE_UNITS; texture_index++) {
        if (vert_textures_sync[texture_index]) {
            const uint16_t index_position = SCE_GXM_MAX_TEXTURE_UNITS + texture_index;
            renderer::set_texture(*emuenv.renderer, context->renderer.get(), index_position,
                texture_index < vert_texture_count ? vert_textures[texture_index] : context->state.textures[index_position]);
        }

        if (frag_textures_sync[texture_index])
            renderer::set_texture(*emuenv.renderer, context->renderer.get(), texture_index,
                texture_index < frag_texture_count ? frag_textures[texture_index] : context->state.textures[texture_index]);
    }

    std::array<size_t, SCE_GXM_MAX_VERTEX_STREAMS> max_data_length{};
    std::uint32_t stream_used = 0;
    if (native_metal) {
        if (!empty_metal_draw && !metal_vertex_stream_sizes_metal3(*vertex_program, max_index, draw->instance_count, max_data_length, stream_used))
            return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    } else {
        for (const SceGxmVertexAttribute &attribute : vertex_program->attributes) {
            if (!emuenv.renderer->features.enable_memory_mapping) {
                const size_t attribute_size = gxm::attribute_format_size(attribute.format) * attribute.componentCount;
                const SceGxmVertexStream &stream = vertex_program->streams[attribute.streamIndex];
                const SceGxmIndexSource index_source = static_cast<SceGxmIndexSource>(stream.indexSource);
                const size_t data_passed_length = gxm::is_stream_instancing(index_source) ? ((draw->instance_count - 1) * stream.stride) : (max_index * stream.stride);
                const size_t data_length = attribute.offset + data_passed_length + attribute_size;
                max_data_length[attribute.streamIndex] = std::max<size_t>(max_data_length[attribute.streamIndex], data_length);
            }

            stream_used |= (1 << attribute.streamIndex);
        }
    }

    auto stream_data = draw->stream_data.get(emuenv.mem);
    if (native_metal && stream_used && !stream_data)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    // Copy and queue upload
    for (size_t stream_index = 0; stream_index < SCE_GXM_MAX_VERTEX_STREAMS; ++stream_index) {
        // Upload it
        if (stream_used & (1 << static_cast<std::uint16_t>(stream_index))) {
            const size_t data_length = max_data_length[stream_index];
            const Ptr<const void> data = stream_data[stream_index];
            if (native_metal && (!data || uint64_t(data.address()) + data_length > UINT32_MAX))
                return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

            renderer::set_vertex_stream(*emuenv.renderer, context->renderer.get(), stream_index,
                data_length, data);
        }
    }

    renderer::draw(*emuenv.renderer, context->renderer.get(), draw->type, draw->index_format, draw->index_data, draw->vertex_count, draw->instance_count);
    if (native_metal && renderer::metal::gxm_metadata(*context->renderer).recording_failed)
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);

    // increase the ringbuffer position if a default vertex or fragment buffer was reserved, we know the new position will fit in the ringbuffer
    // also even in a precomputed draw, this is needed as some parts of the pipeline can be not precomputed
    if (context->was_vert_default_uniform_reserved) {
        const size_t size = native_metal ? renderer::metal::gxm_metadata(*context->renderer).vertex_reserved_bytes
                                       : (size_t)vertex_program_gxp.default_uniform_buffer_count * 4;
        context->state.vertex_ring_buffer_used += size;
        context->was_vert_default_uniform_reserved = false;
    }

    if (context->was_frag_default_uniform_reserved) {
        const size_t size = native_metal ? renderer::metal::gxm_metadata(*context->renderer).fragment_reserved_bytes
                                       : (size_t)fragment_program_gxp.default_uniform_buffer_count * 4;
        context->state.fragment_ring_buffer_used += size;
        context->was_frag_default_uniform_reserved = false;
    }

    context->last_precomputed = true;
    return 0;
    }

int export_sceGxmEndCommandList_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext, SceGxmCommandList *commandList) {
    TRACY_FUNC(sceGxmEndCommandList, deferredContext, commandList);
    if (!deferredContext || !commandList)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    if (deferredContext->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (!deferredContext->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_NOT_WITHIN_COMMAND_LIST);
    }

    if (renderer::metal::gxm_metadata(*deferredContext->renderer).enabled && renderer::metal::gxm_metadata(*deferredContext->renderer).recording_failed) {
        commandList->list = nullptr;
        abort_metal_deferred_recording_metal3(deferredContext);
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
    }

    // only set the first two fields for commandList (its size is assumed to be 32 bytes by the game)
    renderer::CommandList *list = deferredContext->linearly_allocate<renderer::CommandList>(emuenv.kernel, emuenv.mem,
        thread_id);
    if (!list && renderer::metal::gxm_metadata(*deferredContext->renderer).enabled) {
        // Plus allows the completed host command chain to be finalized even
        // if its last VDM marker could not obtain another callback buffer.
        list = static_cast<renderer::CommandList *>(malloc(sizeof(renderer::CommandList)));
    }
    if (!list) {
        if (renderer::metal::gxm_metadata(*deferredContext->renderer).enabled) {
            commandList->list = nullptr;
            abort_metal_deferred_recording_metal3(deferredContext);
        }
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
    }
    commandList->list = list;

    // also update our own command list
    deferredContext->curr_command_list->list = commandList->list;

    *commandList->list = deferredContext->renderer->command_list;

    // insert last memory range
    deferredContext->insert_new_memory_range();
    if (renderer::metal::gxm_metadata(*deferredContext->renderer).enabled) {
        const std::lock_guard<std::mutex> lock(metal_gxm_bookkeeping(emuenv.gxm).metal_command_lists_mutex);
        metal_gxm_bookkeeping(emuenv.gxm).metal_command_lists.insert(list);
    }
    deferredContext->curr_command_list = nullptr;

    // Reset active state
    deferredContext->state.active = false;
    deferredContext->reset_recording();

    renderer::reset_command_list(deferredContext->renderer->command_list);

    return 0;
    }

int export_sceGxmEndScene_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmNotification *vertexNotification, SceGxmNotification *fragmentNotification) {
    TRACY_FUNC(sceGxmEndScene, context, vertexNotification, fragmentNotification);
    const MemState &mem = emuenv.mem;

    if (!context) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (context->state.type != SCE_GXM_CONTEXT_TYPE_IMMEDIATE) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (!context->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_WITHIN_SCENE);
    }

    SceGxmNotification empty_notification = { Ptr<uint32_t>(0), 0 };

    // Add command to end the scene
    renderer::sync_surface_data(*emuenv.renderer, context->renderer.get(), vertexNotification ? *vertexNotification : empty_notification, fragmentNotification ? *fragmentNotification : empty_notification);

    if (context->state.fragment_sync_object) {
        SceGxmSyncObject *sync = context->state.fragment_sync_object.get(mem);
        uint32_t cmd_timestamp = ++sync->timestamp_ahead;
        sync->last_operation_global = emuenv.gxm.global_timestamp.fetch_add(1, std::memory_order_relaxed);

        renderer::add_command(context->renderer.get(), renderer::CommandOpcode::SignalSyncObject,
            nullptr, context->state.fragment_sync_object, cmd_timestamp);
    }

    // Vertex processing starts at the scene kick; preserve small streams
    // before the guest can recycle them while Metal drains the render queue.
#ifdef __APPLE__
    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        static_cast<renderer::metal::MetalState &>(*emuenv.renderer).snapshot_vertex_streams(emuenv.mem, context->renderer->command_list);
#endif
    // Submit our command list
    renderer::submit_command_list(*emuenv.renderer, context->renderer.get(), context->renderer->command_list);
    renderer::reset_command_list(context->renderer->command_list);

    const auto immediate_context = emuenv.gxm.immediate_contexts.find(context);
    if (immediate_context != emuenv.gxm.immediate_contexts.end())
        emuenv.gxm.last_immediate_context = immediate_context->second;

    context->state.active = false;
    return 0;
    }

int export_sceGxmExecuteCommandList_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmCommandList *commandList) {
    TRACY_FUNC(sceGxmExecuteCommandList, context, commandList);
    if (!context) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (context->state.type != SCE_GXM_CONTEXT_TYPE_IMMEDIATE) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (!context->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_NOT_WITHIN_SCENE);
    }

    if (!commandList || !commandList->list)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (emuenv.renderer->current_backend == renderer::Backend::Metal) {
        const std::lock_guard<std::mutex> lock(metal_gxm_bookkeeping(emuenv.gxm).metal_command_lists_mutex);
        const auto *list = commandList->list;
        if (!metal_gxm_bookkeeping(emuenv.gxm).metal_command_lists.contains(list))
            return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
        if (!renderer::append_deferred_command_list(*context->renderer, *list))
            return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
    } else if (!renderer::append_deferred_command_list(*context->renderer, *commandList->list))
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);

    // Restore back our GXM state
    gxmContextStateRestore_metal3(*emuenv.renderer, emuenv.gxm, context, true, emuenv.mem);

    return 0;
    }

int export_sceGxmFinish_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context) {
    TRACY_FUNC(sceGxmFinish, context);

    if (!context)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    const Address context_addr = Ptr<SceGxmContext>(context, emuenv.mem).address();
    if (emuenv.renderer->current_backend == renderer::Backend::Metal
        && context->state.type == SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        const auto deferred_context = emuenv.gxm.deferred_contexts.find(context);
        if (deferred_context == emuenv.gxm.deferred_contexts.end()
            || deferred_context->second != context_addr || !context->renderer)
            return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
        // A deferred allocator may be unavailable outside recording. Plus
        // waits through the generic command allocator in this case.
        renderer::finish(*emuenv.renderer, nullptr);
        return 0;
    }
    const auto immediate_context = emuenv.gxm.immediate_contexts.find(context);
    if (context->state.type != SCE_GXM_CONTEXT_TYPE_IMMEDIATE
        || immediate_context == emuenv.gxm.immediate_contexts.end()
        || immediate_context->second != context_addr)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    renderer::Context *renderer_context = context->renderer.get();
    if (!renderer_context)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    // Wait on this context's rendering finish code.
    renderer::finish(*emuenv.renderer, renderer_context);

    return 0;
    }

int export_sceGxmGetDeferredContextFragmentBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmContext *deferredContext, Ptr<void> *mem) {
    TRACY_FUNC(sceGxmGetDeferredContextFragmentBuffer, deferredContext, mem);
    if (!deferredContext || !mem) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (deferredContext->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (deferredContext->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_WITHIN_COMMAND_LIST);
    }

    const auto &state = deferredContext->state;
    *mem = renderer::metal::gxm_metadata(*deferredContext->renderer).enabled && state.fragment_ring_buffer
        ? Ptr<void>(state.fragment_ring_buffer.address() + static_cast<Address>(state.fragment_ring_buffer_used))
        : state.fragment_ring_buffer;
    return 0;
    }

int export_sceGxmGetDeferredContextVdmBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmContext *deferredContext, Ptr<void> *mem) {
    TRACY_FUNC(sceGxmGetDeferredContextVdmBuffer, deferredContext, mem);
    if (!deferredContext || !mem) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (deferredContext->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (deferredContext->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_WITHIN_COMMAND_LIST);
    }

    *mem = renderer::metal::gxm_metadata(*deferredContext->renderer).enabled && (!deferredContext->state.vdm_buffer || !deferredContext->state.vdm_buffer_size)
        ? deferredContext->alloc_space.cast<void>()
        : deferredContext->state.vdm_buffer;
    return 0;
    }

int export_sceGxmGetDeferredContextVertexBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmContext *deferredContext, Ptr<void> *mem) {
    TRACY_FUNC(sceGxmGetDeferredContextVertexBuffer, deferredContext, mem);
    if (!deferredContext || !mem) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (deferredContext->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (deferredContext->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_WITHIN_COMMAND_LIST);
    }

    const auto &state = deferredContext->state;
    *mem = renderer::metal::gxm_metadata(*deferredContext->renderer).enabled && state.vertex_ring_buffer
        ? Ptr<void>(state.vertex_ring_buffer.address() + static_cast<Address>(state.vertex_ring_buffer_used))
        : state.vertex_ring_buffer;
    return 0;
    }

int export_sceGxmInitialize_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmInitializeParams *params) {
    TRACY_FUNC(sceGxmInitialize, params);
    if (!params) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (((params->flags != 0x00000000U) && (params->flags != 0x00010000U) && (params->flags != 0x00020000U)) || (params->parameterBufferSize & 0x3FFFF)) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if ((params->displayQueueMaxPendingCount * params->displayQueueCallbackDataSize) > 0x200) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

#ifdef __APPLE__
    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        static_cast<renderer::metal::MetalState &>(*emuenv.renderer).reset_vertex_stream_snapshots();
#endif
    emuenv.gxm.params = *params;
    // Plus subtracts a slot only when the worker releases an entry before its
    // callback. Metal retains that entry through the callback, including Finish.
    // Keep the existing limit of three pending frames; other backends retain
    // their previous slot subtraction and the AddEntry wait for a limit of one.
    const uint32_t max_queue_size = emuenv.renderer->current_backend == renderer::Backend::Metal
        ? std::clamp(params->displayQueueMaxPendingCount, 1U, 3U)
        : std::max(std::min(params->displayQueueMaxPendingCount, 3U) - 1, 1U);
    emuenv.gxm.display_queue.maxPendingCount_ = max_queue_size;

    const ThreadStatePtr main_thread = emuenv.kernel.get_thread(thread_id);
    const ThreadStatePtr display_queue_thread = emuenv.kernel.create_thread(emuenv.mem, "SceGxmDisplayQueue", Ptr<void>(0), SCE_KERNEL_HIGHEST_PRIORITY_USER, SCE_KERNEL_THREAD_CPU_AFFINITY_MASK_DEFAULT, SCE_KERNEL_STACK_SIZE_USER_DEFAULT, nullptr);
    if (!display_queue_thread) {
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    }
    emuenv.gxm.display_queue_thread = display_queue_thread->id;

    // Reset the queue in case sceGxmTerminate was called earlier
    emuenv.gxm.display_queue.reset();
    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        metal_gxm_bookkeeping(emuenv.gxm).clear_metal_programs();
    emuenv.gxm.display_host_thread = std::thread(display_entry_thread_metal3, std::ref(emuenv));
    emuenv.gxm.notification_region = Ptr<uint32_t>(alloc(emuenv.mem, MiB(1), "SceGxmNotificationRegion"));
    memset(emuenv.gxm.notification_region.get(emuenv.mem), 0, MiB(1));
    return 0;
    }

int export_sceGxmMapMemory_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, Ptr<void> base, uint32_t size, uint32_t attribs) {
    TRACY_FUNC(sceGxmMapMemory, base, size, attribs);
    if (!base) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if ((base.address() % KiB(4) != 0) || (size % KiB(4) != 0))
        LOG_WARN_ONCE("Mapping unaligned GPU memory");

    // Make sure the base address and size are 4KiB-aligned
    Address aligned_base = align_down(base.address(), KiB(4));
    const uint64_t aligned_end = align(uint64_t(base.address()) + size, uint64_t(KiB(4)));
    if (aligned_end > UINT32_MAX)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    size = uint32_t(aligned_end - aligned_base);

    // Check if it has already been mapped
    // Some games intentionally overlapping mapped region. Nothing we can do. Allow it, bear your own consequences.
    GxmState &gxm = emuenv.gxm;
    auto ite = gxm.memory_mapped_regions.lower_bound(aligned_base);
    if ((ite == gxm.memory_mapped_regions.end()) || (ite->first != aligned_base)) {
        if ((ite != gxm.memory_mapped_regions.end()) && ((aligned_base + size) > ite->first)) {
            LOG_ERROR("Overlapping mapped memory detected");

            if (emuenv.renderer->features.enable_memory_mapping) {
                // overlapping memory mapping is not supported
                return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
            }
        }
        // little big planet maps regions of size 0
        if ((emuenv.renderer->features.enable_memory_mapping
                || emuenv.renderer->current_backend == renderer::Backend::Metal) && size > 0
            && renderer::send_single_command(*emuenv.renderer, nullptr,
                renderer::CommandOpcode::MemoryMap, true, aligned_base, size) != renderer::CommandErrorCodeNone)
            return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

        gxm.memory_mapped_regions.emplace(aligned_base, MemoryMapInfo{ aligned_base, size, attribs });

        return 0;
    }

    return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

int export_sceGxmMidSceneFlush_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *immediateContext, uint32_t flags, SceGxmSyncObject *vertexSyncObject, const SceGxmNotification *vertexNotification) {
    TRACY_FUNC(sceGxmMidSceneFlush, immediateContext, flags, vertexSyncObject, vertexNotification);
    if (flags != 0)
        STUBBED("Flags ignored");

    if (!immediateContext)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if ((flags & 0xFFFFFFFE) || (immediateContext->state.type != SCE_GXM_CONTEXT_TYPE_IMMEDIATE))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);

    if (vertexSyncObject != nullptr)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (!immediateContext->state.active)
        return RET_ERROR(SCE_GXM_ERROR_NOT_WITHIN_SCENE);

#ifdef __APPLE__
    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        static_cast<renderer::metal::MetalState &>(*emuenv.renderer).snapshot_vertex_streams(emuenv.mem, immediateContext->renderer->command_list);
#endif
    SceGxmNotification notification = vertexNotification ? *vertexNotification : SceGxmNotification{ Ptr<uint32_t>(0), 0 };
    renderer::add_command(immediateContext->renderer.get(), renderer::CommandOpcode::MidSceneFlush, nullptr, notification);

    if (vertexNotification) {
        // this is necessary only if the program could wait for a notification
        renderer::submit_command_list(*emuenv.renderer, immediateContext->renderer.get(), immediateContext->renderer->command_list);
        renderer::reset_command_list(immediateContext->renderer->command_list);
    }

    return 0;
    }

int export_sceGxmNotificationWait_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmNotification *notification) {
    TRACY_FUNC(sceGxmNotificationWait, notification);
    if (!notification) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    std::uint32_t volatile *value = notification->address.get(emuenv.mem);
    const std::uint32_t target_value = notification->value;

    std::unique_lock<std::mutex> lock(emuenv.renderer->notification_mutex);
    if (*value != target_value) {
        emuenv.renderer->notification_ready.wait(lock, [&]() {
            return *value == target_value || emuenv.display.abort.load()
                || (emuenv.renderer->current_backend == renderer::Backend::Metal
                    && emuenv.renderer->render_abort.load(std::memory_order_relaxed));
        });
    }

    return 0;
    }

int export_sceGxmPrecomputedDrawInit_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmPrecomputedDraw *state, Ptr<const SceGxmVertexProgram> program, Ptr<void> extra_data) {
    TRACY_FUNC(sceGxmPrecomputedDrawInit, state, program, extra_data);
    if (!state || !program || !extra_data) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (extra_data.address() & 0xF) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_ALIGNMENT);
    }

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, program.address(), MetalGxmProgramKind::Vertex, export_name))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    SceGxmPrecomputedDraw new_draw;

    new_draw.program = program;

    uint16_t max_stream_index = 0;
    const auto &gxm_vertex_program = *program.get(emuenv.mem);
    for (const SceGxmVertexAttribute &attribute : gxm_vertex_program.attributes) {
        max_stream_index = std::max(attribute.streamIndex, max_stream_index);
    }

    new_draw.stream_count = max_stream_index + 1;
    new_draw.stream_data = extra_data.cast<StreamData>();

    *state = new_draw;

    return 0;
    }

int export_sceGxmPrecomputedFragmentStateInit_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmPrecomputedFragmentState *state, Ptr<const SceGxmFragmentProgram> program, Ptr<void> extra_data) {
    TRACY_FUNC(sceGxmPrecomputedFragmentStateInit, state, program, extra_data);
    if (!state || !program || !extra_data) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (extra_data.address() & 0xF) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_ALIGNMENT);
    }

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, program.address(), MetalGxmProgramKind::Fragment, export_name))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    SceGxmPrecomputedFragmentState new_state;
    new_state.program = program;

    auto &renderer_data = program.get(emuenv.mem)->renderer_data;
    new_state.texture_count = renderer_data->texture_count;
    new_state.buffer_count = renderer_data->buffer_count;

    new_state.textures = extra_data.cast<TextureData>();
    new_state.uniform_buffers = (extra_data.cast<TextureData>() + new_state.texture_count).cast<UniformBuffer>();

    *state = new_state;

    return 0;
    }

int export_sceGxmPrecomputedVertexStateInit_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmPrecomputedVertexState *state, Ptr<const SceGxmVertexProgram> program, Ptr<void> extra_data) {
    TRACY_FUNC(sceGxmPrecomputedVertexStateInit, state, program, extra_data);
    if (!state || !program || !extra_data) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (extra_data.address() & 0xF) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_ALIGNMENT);
    }

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, program.address(), MetalGxmProgramKind::Vertex, export_name))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    SceGxmPrecomputedVertexState new_state;
    new_state.program = program;

    auto &renderer_data = program.get(emuenv.mem)->renderer_data;
    new_state.texture_count = renderer_data->texture_count;
    new_state.buffer_count = renderer_data->buffer_count;

    new_state.textures = extra_data.cast<TextureData>();
    new_state.uniform_buffers = (extra_data.cast<TextureData>() + new_state.texture_count).cast<UniformBuffer>();

    *state = new_state;

    return 0;
    }

int export_sceGxmReserveFragmentDefaultUniformBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, Ptr<void> *uniformBuffer) {
    TRACY_FUNC(sceGxmReserveFragmentDefaultUniformBuffer, context, uniformBuffer);
    if (!context || !uniformBuffer)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, context->state.fragment_program.address(), MetalGxmProgramKind::Fragment, export_name))
        return RET_ERROR(SCE_GXM_ERROR_NULL_PROGRAM);

    const auto fragment_program = context->state.fragment_program.get(emuenv.mem);
    const auto program = fragment_program->program.get(emuenv.mem);

    const size_t size = (size_t)program->default_uniform_buffer_count * 4;
    // data for the ring buffer must be 4 bytes aligned
    const bool native_metal = renderer::metal::gxm_metadata(*context->renderer).enabled;
    context->state.fragment_ring_buffer_used = native_metal
        ? align(uint64_t(context->state.fragment_ring_buffer.address()) + context->state.fragment_ring_buffer_used, uint64_t(4)) - context->state.fragment_ring_buffer.address()
        : align(context->state.fragment_ring_buffer_used, 4);
    const size_t next_used = context->state.fragment_ring_buffer_used + size;

    if (size == 0) {
        if (native_metal) {
            context->was_frag_default_uniform_reserved = false;
            renderer::metal::gxm_metadata(*context->renderer).fragment_reserved_bytes = 0;
        }
        *uniformBuffer = Ptr<void>();
        context->state.fragment_uniform_buffers[SCE_GXM_DEFAULT_UNIFORM_BUFFER_CONTAINER_INDEX] = *uniformBuffer;

        return 0;
    }

    if (next_used > context->state.fragment_ring_buffer_size) {
        if (native_metal) {
            context->state.fragment_ring_buffer_used = 0;
            context->was_frag_default_uniform_reserved = false;
            renderer::metal::gxm_metadata(*context->renderer).fragment_reserved_bytes = 0;
            if (size > UINT32_MAX - 4)
                return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
        }
        if (context->state.type != SCE_GXM_CONTEXT_TYPE_IMMEDIATE) {
            context->state.fragment_ring_buffer = gxmRunDeferredMemoryCallback_metal3(emuenv.kernel, emuenv.mem, emuenv.gxm.callback_lock, context->state.fragment_ring_buffer_size,
                context->state.fragment_memory_callback, context->state.memory_callback_userdata, native_metal ? static_cast<uint32_t>(size) : DEFAULT_RING_SIZE, thread_id, native_metal);

            if (!context->state.fragment_ring_buffer) {
                return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
            }
        }

        context->state.fragment_ring_buffer_used = native_metal
            ? ((4 - (context->state.fragment_ring_buffer.address() & 3)) & 3)
            : 0;
    }

    if (native_metal && (!context->state.fragment_ring_buffer
                            || context->state.fragment_ring_buffer_used + size > context->state.fragment_ring_buffer_size
                            || uint64_t(context->state.fragment_ring_buffer.address()) + context->state.fragment_ring_buffer_used + size > UINT32_MAX))
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);

    *uniformBuffer = context->state.fragment_ring_buffer.cast<uint8_t>() + static_cast<int32_t>(context->state.fragment_ring_buffer_used);
    context->was_frag_default_uniform_reserved = true;
    if (native_metal)
        renderer::metal::gxm_metadata(*context->renderer).fragment_reserved_bytes = size;
    context->state.fragment_uniform_buffers[SCE_GXM_DEFAULT_UNIFORM_BUFFER_CONTAINER_INDEX] = *uniformBuffer;

    return 0;
    }

int export_sceGxmReserveVertexDefaultUniformBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, Ptr<void> *uniformBuffer) {
    TRACY_FUNC(sceGxmReserveVertexDefaultUniformBuffer, context, uniformBuffer);
    if (!context || !uniformBuffer)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, context->state.vertex_program.address(), MetalGxmProgramKind::Vertex, export_name))
        return RET_ERROR(SCE_GXM_ERROR_NULL_PROGRAM);

    const auto vertex_program = context->state.vertex_program.get(emuenv.mem);
    const auto program = vertex_program->program.get(emuenv.mem);

    const size_t size = (size_t)program->default_uniform_buffer_count * 4;
    // data for the ring buffer must be 4 bytes aligned
    const bool native_metal = renderer::metal::gxm_metadata(*context->renderer).enabled;
    context->state.vertex_ring_buffer_used = native_metal
        ? align(uint64_t(context->state.vertex_ring_buffer.address()) + context->state.vertex_ring_buffer_used, uint64_t(4)) - context->state.vertex_ring_buffer.address()
        : align(context->state.vertex_ring_buffer_used, 4);
    const size_t next_used = context->state.vertex_ring_buffer_used + size;

    if (size == 0) {
        if (native_metal) {
            context->was_vert_default_uniform_reserved = false;
            renderer::metal::gxm_metadata(*context->renderer).vertex_reserved_bytes = 0;
        }
        *uniformBuffer = Ptr<void>();
        context->state.vertex_uniform_buffers[SCE_GXM_DEFAULT_UNIFORM_BUFFER_CONTAINER_INDEX] = *uniformBuffer;

        return 0;
    }

    if (next_used > context->state.vertex_ring_buffer_size) {
        if (native_metal) {
            context->state.vertex_ring_buffer_used = 0;
            context->was_vert_default_uniform_reserved = false;
            renderer::metal::gxm_metadata(*context->renderer).vertex_reserved_bytes = 0;
            if (size > UINT32_MAX - 4)
                return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
        }
        if (context->state.type != SCE_GXM_CONTEXT_TYPE_IMMEDIATE) {
            context->state.vertex_ring_buffer = gxmRunDeferredMemoryCallback_metal3(emuenv.kernel, emuenv.mem, emuenv.gxm.callback_lock, context->state.vertex_ring_buffer_size,
                context->state.vertex_memory_callback, context->state.memory_callback_userdata, native_metal ? static_cast<uint32_t>(size) : DEFAULT_RING_SIZE, thread_id, native_metal);

            if (!context->state.vertex_ring_buffer) {
                return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);
            }
        }

        context->state.vertex_ring_buffer_used = native_metal
            ? ((4 - (context->state.vertex_ring_buffer.address() & 3)) & 3)
            : 0;
    }

    if (native_metal && (!context->state.vertex_ring_buffer
                            || context->state.vertex_ring_buffer_used + size > context->state.vertex_ring_buffer_size
                            || uint64_t(context->state.vertex_ring_buffer.address()) + context->state.vertex_ring_buffer_used + size > UINT32_MAX))
        return RET_ERROR(SCE_GXM_ERROR_RESERVE_FAILED);

    *uniformBuffer = context->state.vertex_ring_buffer.cast<uint8_t>() + static_cast<int32_t>(context->state.vertex_ring_buffer_used);
    context->was_vert_default_uniform_reserved = true;
    if (native_metal)
        renderer::metal::gxm_metadata(*context->renderer).vertex_reserved_bytes = size;
    context->state.vertex_uniform_buffers[SCE_GXM_DEFAULT_UNIFORM_BUFFER_CONTAINER_INDEX] = *uniformBuffer;

    return 0;
    }

void export_sceGxmSetBackFragmentProgramEnable_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmFragmentProgramMode enable) {
    TRACY_FUNC(sceGxmSetBackFragmentProgramEnable, context, enable);
    if (context->state.back_side_fragment_program_mode != enable) {
        context->state.back_side_fragment_program_mode = enable;
        if (context->alloc_space)
            renderer::set_side_fragment_program_enable(*emuenv.renderer, context->renderer.get(), false, enable);
    }
    }

void export_sceGxmSetBackVisibilityTestEnable_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmVisibilityTestMode enable) {
    TRACY_FUNC(sceGxmSetBackVisibilityTestEnable, context, enable);
    if (!emuenv.renderer || emuenv.renderer->current_backend != renderer::Backend::Metal) {
        UNIMPLEMENTED();
        return;
    }
    static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_enable = enable != SCE_GXM_VISIBILITY_TEST_DISABLED;
    renderer::set_back_visibility_index(*emuenv.renderer, context->renderer.get(), static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_enable,
        static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_index, static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_is_increment);
    }

void export_sceGxmSetBackVisibilityTestIndex_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, uint32_t index) {
    TRACY_FUNC(sceGxmSetBackVisibilityTestIndex, context, index);
    if (!emuenv.renderer || emuenv.renderer->current_backend != renderer::Backend::Metal) {
        UNIMPLEMENTED();
        return;
    }
    static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_index = index;
    renderer::set_back_visibility_index(*emuenv.renderer, context->renderer.get(), static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_enable,
        static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_index, static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_is_increment);
    }

void export_sceGxmSetBackVisibilityTestOp_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmVisibilityTestOp op) {
    TRACY_FUNC(sceGxmSetBackVisibilityTestOp, context, op);
    if (!emuenv.renderer || emuenv.renderer->current_backend != renderer::Backend::Metal) {
        UNIMPLEMENTED();
        return;
    }
    static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_is_increment = op == SCE_GXM_VISIBILITY_TEST_OP_INCREMENT;
    renderer::set_back_visibility_index(*emuenv.renderer, context->renderer.get(), static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_enable,
        static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_index, static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.back_visibility_is_increment);
    }

int export_sceGxmSetDeferredContextFragmentBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext, Ptr<void> mem, uint32_t size) {
    TRACY_FUNC(sceGxmSetDeferredContextFragmentBuffer, deferredContext, mem, size);
    if (!deferredContext) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (deferredContext->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (deferredContext->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_WITHIN_COMMAND_LIST);
    }

    if ((size != 0) && (size < SCE_GXM_DEFERRED_CONTEXT_MINIMUM_BUFFER_SIZE)) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (mem && !size) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    // Use the one specified
    deferredContext->state.fragment_ring_buffer = mem;
    deferredContext->state.fragment_ring_buffer_size = size;
    if (renderer::metal::gxm_metadata(*deferredContext->renderer).enabled) {
        deferredContext->state.fragment_ring_buffer_used = 0;
        deferredContext->was_frag_default_uniform_reserved = false;
        renderer::metal::gxm_metadata(*deferredContext->renderer).fragment_reserved_bytes = 0;
    }

    return 0;
    }

int export_sceGxmSetDeferredContextVdmBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext, Ptr<void> mem, uint32_t size) {
    TRACY_FUNC(sceGxmSetDeferredContextVdmBuffer, deferredContext, mem, size);
    if (!deferredContext) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (deferredContext->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (deferredContext->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_WITHIN_COMMAND_LIST);
    }

    // Use the one specified
    deferredContext->state.vdm_buffer = mem;
    deferredContext->state.vdm_buffer_size = size;

    // make sure the next call will use the new vdm buffer
    if (renderer::metal::gxm_metadata(*deferredContext->renderer).enabled && !mem)
        deferredContext->alloc_space = deferredContext->alloc_space_end = Ptr<uint8_t>();
    else
        deferredContext->alloc_space = deferredContext->alloc_space_end;

    return 0;
    }

int export_sceGxmSetDeferredContextVertexBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *deferredContext, Ptr<void> mem, uint32_t size) {
    TRACY_FUNC(sceGxmSetDeferredContextVertexBuffer, deferredContext, mem, size);
    if (!deferredContext) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (deferredContext->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (deferredContext->state.active) {
        return RET_ERROR(SCE_GXM_ERROR_WITHIN_COMMAND_LIST);
    }

    if ((size != 0) && (size < SCE_GXM_DEFERRED_CONTEXT_MINIMUM_BUFFER_SIZE)) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (mem && !size) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    // Use the one specified
    deferredContext->state.vertex_ring_buffer = mem;
    deferredContext->state.vertex_ring_buffer_size = size;
    if (renderer::metal::gxm_metadata(*deferredContext->renderer).enabled) {
        deferredContext->state.vertex_ring_buffer_used = 0;
        deferredContext->was_vert_default_uniform_reserved = false;
        renderer::metal::gxm_metadata(*deferredContext->renderer).vertex_reserved_bytes = 0;
    }

    return 0;
    }

void export_sceGxmSetFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, Ptr<const SceGxmFragmentProgram> fragmentProgram) {
    TRACY_FUNC(sceGxmSetFragmentProgram, context, fragmentProgram);
    if (!context || !fragmentProgram)
        return;

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, fragmentProgram.address(), MetalGxmProgramKind::Fragment, export_name))
        return;

    context->state.fragment_program = fragmentProgram;
    gxmSetProgram_metal3(*emuenv.renderer, emuenv.gxm, context, fragmentProgram, true, emuenv.mem);
    }

void export_sceGxmSetFrontFragmentProgramEnable_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmFragmentProgramMode enable) {
    TRACY_FUNC(sceGxmSetFrontFragmentProgramEnable, context, enable);
    if (context->state.front_side_fragment_program_mode != enable) {
        context->state.front_side_fragment_program_mode = enable;
        if (context->alloc_space)
            renderer::set_side_fragment_program_enable(*emuenv.renderer, context->renderer.get(), true, enable);
    }
    }

void export_sceGxmSetFrontVisibilityTestEnable_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmVisibilityTestMode enable) {
    TRACY_FUNC(sceGxmSetFrontVisibilityTestEnable, context, enable);

    if (!emuenv.renderer->features.enable_memory_mapping
        && emuenv.renderer->current_backend != renderer::Backend::Metal) {
        UNIMPLEMENTED();
        return;
    }

    context->state.visibility_enable = enable != SCE_GXM_VISIBILITY_TEST_DISABLED;
    renderer::set_visibility_index(*emuenv.renderer, context->renderer.get(), context->state.visibility_enable, context->state.visibility_index, context->state.visibility_is_increment);
    }

void export_sceGxmSetFrontVisibilityTestIndex_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, uint32_t index) {
    TRACY_FUNC(sceGxmSetFrontVisibilityTestIndex, context, index);

    if (!emuenv.renderer->features.enable_memory_mapping
        && emuenv.renderer->current_backend != renderer::Backend::Metal) {
        UNIMPLEMENTED();
        return;
    }

    context->state.visibility_index = index;
    renderer::set_visibility_index(*emuenv.renderer, context->renderer.get(), context->state.visibility_enable, context->state.visibility_index, context->state.visibility_is_increment);
    }

void export_sceGxmSetFrontVisibilityTestOp_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, SceGxmVisibilityTestOp op) {
    TRACY_FUNC(sceGxmSetFrontVisibilityTestOp, context, op);

    if (!emuenv.renderer->features.enable_memory_mapping
        && emuenv.renderer->current_backend != renderer::Backend::Metal) {
        UNIMPLEMENTED();
        return;
    }

    context->state.visibility_is_increment = (op == SCE_GXM_VISIBILITY_TEST_OP_INCREMENT);
    renderer::set_visibility_index(*emuenv.renderer, context->renderer.get(), context->state.visibility_enable, context->state.visibility_index, context->state.visibility_is_increment);
    }

int export_sceGxmSetUniformDataF_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, void *uniformBuffer, const SceGxmProgramParameter *parameter, uint32_t componentOffset, uint32_t componentCount, const float *sourceData) {
    TRACY_FUNC(sceGxmSetUniformDataF, uniformBuffer, parameter, componentOffset, componentCount, sourceData);
    assert(parameter);

    if (!uniformBuffer || !parameter || !sourceData)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (parameter->category != SceGxmParameterCategory::SCE_GXM_PARAMETER_CATEGORY_UNIFORM)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);

    const bool native_metal = emuenv.renderer->current_backend == renderer::Backend::Metal;
    if (native_metal) {
        const uint64_t parameter_components = uint64_t(parameter->component_count) * std::max<uint32_t>(parameter->array_size, 1);
        if (uint64_t(componentOffset) + componentCount > parameter_components)
            return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    size_t size = 0;
    size_t offset = 0;
    bool is_float = false;

    const std::uint16_t param_type = parameter->type;

    // Component size is in bytes
    int comp_size = gxp::get_parameter_type_size(static_cast<SceGxmParameterType>(param_type));
    const std::uint8_t *source = reinterpret_cast<const std::uint8_t *>(sourceData);
    std::vector<std::uint8_t> converted_data;

    switch (parameter->type) {
    case SCE_GXM_PARAMETER_TYPE_S8: {
        convert_uniform_data<int8_t>(converted_data, sourceData, componentCount);
        source = converted_data.data();
        break;
    }

    case SCE_GXM_PARAMETER_TYPE_U8: {
        convert_uniform_data<uint8_t>(converted_data, sourceData, componentCount);
        source = converted_data.data();
        break;
    }

    case SCE_GXM_PARAMETER_TYPE_U16: {
        convert_uniform_data<uint16_t>(converted_data, sourceData, componentCount);
        source = converted_data.data();
        break;
    }

    case SCE_GXM_PARAMETER_TYPE_S16: {
        convert_uniform_data<int16_t>(converted_data, sourceData, componentCount);
        source = converted_data.data();
        break;
    }

    case SCE_GXM_PARAMETER_TYPE_U32: {
        convert_uniform_data<uint32_t>(converted_data, sourceData, componentCount);
        source = converted_data.data();
        break;
    }

    case SCE_GXM_PARAMETER_TYPE_S32: {
        convert_uniform_data<int32_t>(converted_data, sourceData, componentCount);
        source = converted_data.data();
        break;
    }

    case SCE_GXM_PARAMETER_TYPE_F16: {
        converted_data.resize(((componentCount + 7) / 8) * 8 * 2);
        float_to_half(sourceData, reinterpret_cast<std::uint16_t *>(converted_data.data()), componentCount);

        source = converted_data.data();
        is_float = true;
        break;
    }

    case SCE_GXM_PARAMETER_TYPE_F32: {
        is_float = true;
        break;
    }
    default:
        assert(false);
    }

    if (parameter->array_size == 1 || parameter->component_count == 1) {
        // Case 1: No array. Only a single vector. Don't apply any alignment
        // Case 2: Array but component count equals to 1. This case, a scalar array, align it to 32-bit bound
        if (parameter->component_count == 1) {
            // Apply 32 bit alignment, by making each component has 4 bytes
            comp_size = 4;
        }

        size = componentCount * comp_size;
        offset = parameter->resource_index * sizeof(float) + componentOffset * comp_size;

        memcpy(static_cast<uint8_t *>(uniformBuffer) + offset, source, size);
    } else {
        // This is the size of each element.
        size = parameter->component_count * comp_size;
        int align_bytes = 0;

        if (is_float) {
            // Align it to 64-bit boundary (8 bytes)
            if ((size & 7) != 0) {
                align_bytes = 8 - (size & 7);
            }
        } else {
            // Align it to 32-bit boundary (4 bytes)
            if ((size & 3) != 0) {
                align_bytes = 4 - (size & 3);
            }
        }

        // wtf
        // wtf
        const int vec_to_start_write = componentOffset / parameter->component_count;
        int component_cursor_inside_vector = (componentOffset % parameter->component_count);
        std::uint8_t *dest = static_cast<uint8_t *>(uniformBuffer) + parameter->resource_index * sizeof(float)
            + vec_to_start_write * (size + align_bytes) + component_cursor_inside_vector * comp_size;

        int component_to_copy_remain_per_elem = parameter->component_count - component_cursor_inside_vector;
        int component_left_to_copy = componentCount;

        while (component_left_to_copy > 0) {
            if (native_metal) {
                const int copy_count = std::min(component_to_copy_remain_per_elem, component_left_to_copy);
                memcpy(dest, source, copy_count * comp_size);
                dest += comp_size * copy_count + (copy_count == component_to_copy_remain_per_elem ? align_bytes : 0);
                source += copy_count * comp_size;
                component_left_to_copy -= copy_count;
                component_to_copy_remain_per_elem = parameter->component_count;
            } else {
                memcpy(dest, source, component_to_copy_remain_per_elem * comp_size);

                // Add and align destination
                dest += comp_size * component_to_copy_remain_per_elem + align_bytes;
                source += component_to_copy_remain_per_elem * comp_size;

                component_left_to_copy -= component_to_copy_remain_per_elem;
                component_to_copy_remain_per_elem = std::min<int>(4, component_to_copy_remain_per_elem);
            }
        }
    }

    return 0;
    }

void export_sceGxmSetVertexProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, Ptr<const SceGxmVertexProgram> vertexProgram) {
    TRACY_FUNC(sceGxmSetVertexProgram, context, vertexProgram);
    if (!context || !vertexProgram)
        return;

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, vertexProgram.address(), MetalGxmProgramKind::Vertex, export_name))
        return;

    context->state.vertex_program = vertexProgram;
    gxmSetProgram_metal3(*emuenv.renderer, emuenv.gxm, context, vertexProgram, false, emuenv.mem);
    }

int export_sceGxmSetVisibilityBuffer_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *immediateContext, Ptr<void> bufferBase, uint32_t stridePerCore) {
    TRACY_FUNC(sceGxmSetVisibilityBuffer, immediateContext, bufferBase, stridePerCore);
    if (!immediateContext) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (immediateContext->state.type != SCE_GXM_CONTEXT_TYPE_IMMEDIATE)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);

    if (bufferBase.address() & (SCE_GXM_VISIBILITY_ALIGNMENT - 1))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_ALIGNMENT);

    if (emuenv.renderer->features.enable_memory_mapping
        || emuenv.renderer->current_backend == renderer::Backend::Metal) {
        renderer::set_visibility_buffer(*emuenv.renderer, immediateContext->renderer.get(), bufferBase.cast<uint32_t>(), stridePerCore);
    } else {
        STUBBED("Set all visible");
        memset(bufferBase.get(emuenv.mem), 0xFF, SCE_GXM_GPU_CORE_COUNT * stridePerCore);
    }

    return 0;
    }

int export_sceGxmSetYuvProfile_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmContext *context, uint32_t index, SceGxmYuvProfile profile) {
    TRACY_FUNC(sceGxmSetYuvProfile, context, index, profile);
    if (!context)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    if (index >= static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.yuv_profiles.size() || profile > SCE_GXM_YUV_PROFILE_BT709_FULL_RANGE)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);

    static_cast<renderer::metal::MetalGuestContext &>(*context->renderer).guest_state.yuv_profiles[index] = profile;
    // Deferred contexts acquire their command allocator at BeginCommandList.
    // Between lists, only the saved state changes; Begin restores it.
    if (context->state.type != SCE_GXM_CONTEXT_TYPE_DEFERRED || context->state.active)
        renderer::set_yuv_profile(*emuenv.renderer, context->renderer.get(), index, profile);
    context->is_vert_texture_dirty.set();
    context->is_frag_texture_dirty.set();
    return 0;
    }

template <typename T>
static void free_callbacked_metal3(EmuEnvState &emuenv, SceUID thread_id, SceGxmShaderPatcher *shaderPatcher, Ptr<T> data) {
    if (emuenv.renderer->current_backend == renderer::Backend::Metal) {
        // Placement-constructed guest wrappers own host vectors/maps and the
        // frozen Metal bindings. The patcher itself can be the freed object.
        const SceGxmShaderPatcherParams params = shaderPatcher->params;
        {
            const std::lock_guard<std::mutex> lock(metal_gxm_bookkeeping(emuenv.gxm).metal_programs_mutex);
            metal_gxm_bookkeeping(emuenv.gxm).metal_live_programs.erase(data.address());
        }
        if constexpr (std::is_same_v<T, SceGxmShaderPatcher>) {
            auto &bookkeeping = metal_gxm_bookkeeping(emuenv.gxm);
            const std::lock_guard lock(bookkeeping.metal_shader_caches_mutex);
            bookkeeping.metal_shader_caches.erase(data.get(emuenv.mem));
        }
        data.get(emuenv.mem)->~T();
        if (!params.hostFreeCallback)
            LOG_ERROR("Empty hostFreeCallback");
        const auto thread = emuenv.kernel.get_thread(thread_id);
        thread->run_callback(params.hostFreeCallback.address(), { params.userData.address(), data.address() });
        return;
    }
    free_callbacked(emuenv, thread_id, shaderPatcher, data.address());
    }

int export_sceGxmShaderPatcherAddRefFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, SceGxmFragmentProgram *fragmentProgram) {
    TRACY_FUNC(sceGxmShaderPatcherAddRefFragmentProgram, shaderPatcher, fragmentProgram);
    if (!shaderPatcher || !fragmentProgram)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (emuenv.renderer->current_backend == renderer::Backend::Metal
        && !metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, Ptr<const SceGxmFragmentProgram>(fragmentProgram, emuenv.mem).address(), MetalGxmProgramKind::Fragment, export_name))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    ++fragmentProgram->reference_count;

    return 0;
    }

int export_sceGxmShaderPatcherAddRefVertexProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, SceGxmVertexProgram *vertexProgram) {
    TRACY_FUNC(sceGxmShaderPatcherAddRefVertexProgram, shaderPatcher, vertexProgram);
    if (!shaderPatcher || !vertexProgram)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (emuenv.renderer->current_backend == renderer::Backend::Metal
        && !metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, Ptr<const SceGxmVertexProgram>(vertexProgram, emuenv.mem).address(), MetalGxmProgramKind::Vertex, export_name))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    ++vertexProgram->reference_count;

    return 0;
    }

int export_sceGxmShaderPatcherCreateFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, const SceGxmRegisteredProgram *programId, SceGxmOutputRegisterFormat outputFormat, SceGxmMultisampleMode multisampleMode, const SceGxmBlendInfo *blendInfo, Ptr<const SceGxmProgram> vertexProgram, Ptr<SceGxmFragmentProgram> *fragmentProgram) {
    TRACY_FUNC(sceGxmShaderPatcherCreateFragmentProgram, shaderPatcher, programId, outputFormat, multisampleMode, blendInfo, vertexProgram, fragmentProgram);
    MemState &mem = emuenv.mem;

    if (!shaderPatcher || !programId || !fragmentProgram)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    static const SceGxmBlendInfo default_blend_info = {
        SCE_GXM_COLOR_MASK_ALL,
        SCE_GXM_BLEND_FUNC_NONE,
        SCE_GXM_BLEND_FUNC_NONE,
        SCE_GXM_BLEND_FACTOR_ONE,
        SCE_GXM_BLEND_FACTOR_ZERO,
        SCE_GXM_BLEND_FACTOR_ONE,
        SCE_GXM_BLEND_FACTOR_ZERO
    };
    const MetalFragmentProgramCacheKey key = {
        *programId,
        (blendInfo != nullptr) ? *blendInfo : default_blend_info,
        emuenv.renderer->current_backend == renderer::Backend::Metal ? outputFormat : SCE_GXM_OUTPUT_REGISTER_FORMAT_DECLARED,
        emuenv.renderer->current_backend == renderer::Backend::Metal ? multisampleMode : SCE_GXM_MULTISAMPLE_NONE,
        emuenv.renderer->current_backend == renderer::Backend::Metal ? Ptr<const SceGxmRegisteredProgram>(programId, mem).address() : 0
    };
    MetalFragmentProgramCache::const_iterator cached = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.find(key);
    if (cached != metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.end()) {
        ++cached->second.get(mem)->reference_count;
        *fragmentProgram = cached->second;
        return 0;
    }

    *fragmentProgram = alloc_callbacked<SceGxmFragmentProgram>(emuenv, thread_id, shaderPatcher);
    assert(*fragmentProgram);
    if (!*fragmentProgram) {
        return RET_ERROR(SCE_GXM_ERROR_OUT_OF_MEMORY);
    }

    SceGxmFragmentProgram *const fp = fragmentProgram->get(mem);
    fp->is_maskupdate = false;
    fp->program = programId->program;

    if (!renderer::create(fp->renderer_data, *emuenv.renderer, *programId->program.get(mem), blendInfo, emuenv.renderer->gxp_ptr_map)) {
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    }

#ifdef __APPLE__
    if (emuenv.renderer->current_backend == renderer::Backend::Metal) {
        static_cast<renderer::metal::MetalFragmentProgram &>(*fp->renderer_data).output_register_format = outputFormat;
        renderer::metal::freeze_program(*fp, mem);
    }
#endif

    metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.emplace(key, *fragmentProgram);
    metal_program_register_metal3(emuenv, fragmentProgram->address(), MetalGxmProgramKind::Fragment);

    return 0;
    }

int export_sceGxmShaderPatcherCreateMaskUpdateFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, Ptr<SceGxmFragmentProgram> *fragmentProgram) {
    TRACY_FUNC(sceGxmShaderPatcherCreateMaskUpdateFragmentProgram, shaderPatcher, fragmentProgram);
    MemState &mem = emuenv.mem;

    if (!shaderPatcher || !fragmentProgram)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    *fragmentProgram = alloc_callbacked<SceGxmFragmentProgram>(emuenv, thread_id, shaderPatcher);
    assert(*fragmentProgram);
    if (!*fragmentProgram) {
        return RET_ERROR(SCE_GXM_ERROR_OUT_OF_MEMORY);
    }

    SceGxmFragmentProgram *const fp = fragmentProgram->get(mem);
    fp->is_maskupdate = true;
    fp->program = Ptr<const SceGxmProgram>(alloc_callbacked(emuenv, thread_id, shaderPatcher->params, size_mask_gxp));
    memcpy(const_cast<SceGxmProgram *>(fp->program.get(mem)), mask_gxp, size_mask_gxp);

    if (!renderer::create(fp->renderer_data, *emuenv.renderer, *fp->program.get(mem), nullptr, emuenv.renderer->gxp_ptr_map)) {
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    }
#ifdef __APPLE__
    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        renderer::metal::freeze_program(*fp, mem);
#endif

    metal_program_register_metal3(emuenv, fragmentProgram->address(), MetalGxmProgramKind::Fragment);
    return 0;
    }

int export_sceGxmShaderPatcherCreateVertexProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, const SceGxmRegisteredProgram *programId, const SceGxmVertexAttribute *attributes, uint32_t attributeCount, const SceGxmVertexStream *streams, uint32_t streamCount, Ptr<SceGxmVertexProgram> *vertexProgram) {
    TRACY_FUNC(sceGxmShaderPatcherCreateVertexProgram, shaderPatcher, programId, attributes, attributeCount, streams, streamCount, vertexProgram);
    MemState &mem = emuenv.mem;

    if (!shaderPatcher || !programId || !vertexProgram)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    MetalVertexProgramCacheKey key = {
        *programId,
        0,
        emuenv.renderer->current_backend == renderer::Backend::Metal ? Ptr<const SceGxmRegisteredProgram>(programId, mem).address() : 0
    };

    if (attributes) {
        key.hash = hash_data(attributes, sizeof(SceGxmVertexAttribute) * attributeCount);
    }

    if (streams) {
        key.hash ^= hash_data(streams, sizeof(SceGxmVertexStream) * streamCount);
    }

    MetalVertexProgramCache::const_iterator cached = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.find(key);
    if (cached != metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.end()) {
        ++cached->second.get(mem)->reference_count;
        *vertexProgram = cached->second;
        return 0;
    }

    *vertexProgram = alloc_callbacked<SceGxmVertexProgram>(emuenv, thread_id, shaderPatcher);
    assert(*vertexProgram);
    if (!*vertexProgram) {
        return RET_ERROR(SCE_GXM_ERROR_OUT_OF_MEMORY);
    }

    SceGxmVertexProgram *const vp = vertexProgram->get(mem);
    vp->program = programId->program;
    vp->key_hash = key.hash;

    if (streams && streamCount > 0) {
        vp->streams.insert(vp->streams.end(), &streams[0], &streams[streamCount]);
    }

    if (attributes && attributeCount > 0) {
        vp->attributes.insert(vp->attributes.end(), &attributes[0], &attributes[attributeCount]);
    }

    if (!renderer::create(vp->renderer_data, *emuenv.renderer, *programId->program.get(mem), emuenv.renderer->gxp_ptr_map, vp->attributes)) {
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    }

#ifdef __APPLE__
    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        renderer::metal::freeze_program(*vp, mem);
#endif

    metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.emplace(key, *vertexProgram);
    metal_program_register_metal3(emuenv, vertexProgram->address(), MetalGxmProgramKind::Vertex);

    return 0;
    }

int export_sceGxmShaderPatcherForceUnregisterProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, SceGxmShaderPatcherId programId) {
    TRACY_FUNC(sceGxmShaderPatcherForceUnregisterProgram, shaderPatcher, programId);
    if (!shaderPatcher || !programId) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    SceGxmRegisteredProgram *rp = programId.get(emuenv.mem);

    const bool native_metal = emuenv.renderer->current_backend == renderer::Backend::Metal;
    // Drain submitted commands before recycling the guest wrappers. Metal's
    // queued bindings and compiler descriptors retain their own owned data.
    if (native_metal)
        renderer::finish(*emuenv.renderer, nullptr);

    // look for existing programs and free them
    const bool vertex = native_metal || rp->program.get(emuenv.mem)->is_vertex();
    if (vertex) {
        for (auto it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.begin(); it != metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.end();) {
            if (native_metal ? it->first.metal_registration == programId.address()
                             : it->first.vertex_program.program == rp->program) {
                SceGxmVertexProgram *vertex_program = it->second.get(emuenv.mem);
                while (vertex_program->compile_threads_on.load(std::memory_order_acquire) > 0)
                    std::this_thread::yield();

                free_callbacked_metal3(emuenv, thread_id, shaderPatcher, it->second);
                it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.erase(it);
            } else {
                ++it;
            }
        }
    }
    if (native_metal || !vertex) {
        for (auto it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.begin(); it != metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.end();) {
            if (native_metal ? it->first.metal_registration == programId.address()
                             : it->first.fragment_program.program == rp->program) {
                SceGxmFragmentProgram *frag_program = it->second.get(emuenv.mem);
                while (frag_program->compile_threads_on.load(std::memory_order_acquire) > 0)
                    std::this_thread::yield();

                free_callbacked_metal3(emuenv, thread_id, shaderPatcher, it->second);
                it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.erase(it);
            } else {
                ++it;
            }
        }
    }

    rp->program.reset();
    free_callbacked_metal3(emuenv, thread_id, shaderPatcher, programId);

    return 0;
    }

int export_sceGxmShaderPatcherRegisterProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, Ptr<const SceGxmProgram> programHeader, SceGxmShaderPatcherId *programId) {
    TRACY_FUNC(sceGxmShaderPatcherRegisterProgram, shaderPatcher, programHeader, programId);
    if (!shaderPatcher || !programHeader || !programId)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    *programId = alloc_callbacked<SceGxmRegisteredProgram>(emuenv, thread_id, shaderPatcher);
    assert(*programId);
    if (!*programId) {
        return RET_ERROR(SCE_GXM_ERROR_OUT_OF_MEMORY);
    }

    SceGxmRegisteredProgram *const rp = programId->get(emuenv.mem);
    rp->program = programHeader;
#ifdef __APPLE__
    if (emuenv.renderer->current_backend == renderer::Backend::Metal
        && (programHeader.get(emuenv.mem)->program_flags & SCE_GXM_PROGRAM_FLAG_BUFFER_STORE))
        static_cast<renderer::metal::MetalState &>(*emuenv.renderer).disable_vertex_stream_snapshots();
#endif

    return 0;
    }

int export_sceGxmShaderPatcherReleaseFragmentProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, Ptr<SceGxmFragmentProgram> fragmentProgram) {
    TRACY_FUNC(sceGxmShaderPatcherReleaseFragmentProgram, shaderPatcher, fragmentProgram);
    if (!shaderPatcher || !fragmentProgram)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, fragmentProgram.address(), MetalGxmProgramKind::Fragment, export_name))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    SceGxmFragmentProgram *const fp = fragmentProgram.get(emuenv.mem);
    --fp->reference_count;
    if (fp->reference_count == 0) {
        while (fp->compile_threads_on.load(std::memory_order_acquire) > 0)
            std::this_thread::yield();

        if (emuenv.renderer->current_backend == renderer::Backend::Metal)
            renderer::finish(*emuenv.renderer, nullptr);

        for (MetalFragmentProgramCache::const_iterator it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.begin(); it != metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.end(); ++it) {
            if (it->second == fragmentProgram) {
                metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.erase(it);
                break;
            }
        }
        free_callbacked_metal3(emuenv, thread_id, shaderPatcher, fragmentProgram);
    }

    return 0;
    }

int export_sceGxmShaderPatcherReleaseVertexProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, Ptr<SceGxmVertexProgram> vertexProgram) {
    TRACY_FUNC(sceGxmShaderPatcherReleaseVertexProgram, shaderPatcher, vertexProgram);
    if (!shaderPatcher || !vertexProgram)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (!metal_program_is_live_metal3(*emuenv.renderer, emuenv.gxm, vertexProgram.address(), MetalGxmProgramKind::Vertex, export_name))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    SceGxmVertexProgram *const vp = vertexProgram.get(emuenv.mem);
    --vp->reference_count;
    if (vp->reference_count == 0) {
        while (vp->compile_threads_on.load(std::memory_order_acquire) > 0)
            std::this_thread::yield();

        if (emuenv.renderer->current_backend == renderer::Backend::Metal)
            renderer::finish(*emuenv.renderer, nullptr);

        for (MetalVertexProgramCache::const_iterator it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.begin(); it != metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.end(); ++it) {
            if (it->second == vertexProgram) {
                metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.erase(it);
                break;
            }
        }
        free_callbacked_metal3(emuenv, thread_id, shaderPatcher, vertexProgram);
    }

    return 0;
    }

int export_sceGxmShaderPatcherUnregisterProgram_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmShaderPatcher *shaderPatcher, SceGxmShaderPatcherId programId) {
    TRACY_FUNC(sceGxmShaderPatcherUnregisterProgram, shaderPatcher, programId);
    if (!shaderPatcher || !programId)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    SceGxmRegisteredProgram *const rp = programId.get(emuenv.mem);
    if (emuenv.renderer->current_backend == renderer::Backend::Metal) {
        // Normal unregister removes lookup ownership, not the game's live
        // program handles. Later Release calls still destroy those wrappers.
        // Reusing this registration/GXP address must not return an old handle.
        for (auto it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.begin(); it != metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.end();) {
            if (it->first.metal_registration == programId.address())
                it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).vertex_program_cache.erase(it);
            else ++it;
        }
        for (auto it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.begin(); it != metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.end();) {
            if (it->first.metal_registration == programId.address())
                it = metal_shader_patcher_cache(emuenv.gxm, *shaderPatcher).fragment_program_cache.erase(it);
            else ++it;
        }
    }
    rp->program.reset();

    free_callbacked_metal3(emuenv, thread_id, shaderPatcher, programId);

    return 0;
    }

int export_sceGxmTerminate_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name) {
    TRACY_FUNC(sceGxmTerminate);
    const bool native_metal = emuenv.renderer->current_backend == renderer::Backend::Metal;
    if (native_metal && emuenv.renderer->render_abort.load(std::memory_order_relaxed))
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    // Make sure everything is done in SDL side before killing Vita thread
    emuenv.gxm.display_queue.wait_empty();
    if (native_metal && emuenv.renderer->render_abort.load(std::memory_order_relaxed))
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    gxm::destroy_all_contexts_metal3(emuenv, false);
    if (native_metal && (!emuenv.gxm.immediate_contexts.empty() || !emuenv.gxm.deferred_contexts.empty()))
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    gxm::destroy_all_render_targets_metal3(emuenv, false);
    if (native_metal && !emuenv.gxm.render_targets.empty())
        return RET_ERROR(SCE_GXM_ERROR_DRIVER);
    if (native_metal)
        emuenv.gxm.display_queue.abort_synchronized();
    else
        emuenv.gxm.display_queue.abort();
    // a later sceGxmInitialize reassigns this std::thread, which terminates if still joinable
    if (emuenv.gxm.display_host_thread.joinable())
        emuenv.gxm.display_host_thread.join();
    // The joined host has finished its callbacks; remove the dormant guest thread.
    emuenv.kernel.get_thread(emuenv.gxm.display_queue_thread)->remove();
    return 0;
    }

uint32_t export_sceGxmTextureGetLodMin_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, const SceGxmTexture *texture) {
    TRACY_FUNC(sceGxmTextureGetLodMin, texture);
    if (!texture)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if ((texture->type << 29) == SCE_GXM_TEXTURE_LINEAR_STRIDED) {
        return 0;
    }

    if (emuenv.renderer->current_backend == renderer::Backend::Metal)
        return (texture->lod_min0 << 2) | texture->lod_min1;
    return texture->lod_min0 | (texture->lod_min1 << 2);
    }

static void set_linear_strided_texture_stride_metal3(SceGxmTexture &texture, uint32_t byte_stride) {
    const uint32_t stride_compressed = (byte_stride >> 2) - 1;
    texture.mip_filter = stride_compressed & 1;
    texture.min_filter = (stride_compressed >> 1) & 3;
    texture.mip_count = (stride_compressed >> 3) & 15;
    texture.lod_bias = (stride_compressed >> 7) & 63;
}

int export_sceGxmTextureInitLinearStrided_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, Ptr<const void> data, SceGxmTextureFormat texFormat, uint32_t width, uint32_t height, uint32_t byteStride) {
    TRACY_FUNC(sceGxmTextureInitLinearStrided, texture, data, texFormat, width, height, byteStride);
    if (!texture) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if ((width > 4096) || (height > 4096)) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (byteStride & 3) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_ALIGNMENT);
    }

    if ((byteStride < 4) || (byteStride > 131072)) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    memset(texture, 0, sizeof(SceGxmTexture));

    set_linear_strided_texture_stride_metal3(*texture, byteStride);
    texture->base_format = (texFormat & 0x1F000000) >> 24;
    texture->type = SCE_GXM_TEXTURE_LINEAR_STRIDED >> 29;
    texture->data_addr = data.address() >> 2;
    texture->swizzle_format = (texFormat & 0x7000) >> 12;
    texture->normalize_mode = 1;
    texture->format0 = (texFormat & 0x80000000) >> 31;
    texture->uaddr_mode = texture->vaddr_mode = SCE_GXM_TEXTURE_ADDR_CLAMP;
    texture->height = height - 1;
    texture->width = width - 1;
    texture->mag_filter = SCE_GXM_TEXTURE_FILTER_POINT;
    texture->gamma_mode = 0;
    texture->lod_min0 = 0;
    texture->lod_min1 = 0;

    return 0;
    }

int export_sceGxmTextureSetHeight_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, uint32_t height) {
    TRACY_FUNC(sceGxmTextureSetHeight, texture, height);
    if (!texture)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    if (!height || height > 4096)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);

    // SWIZZLED and CUBE encode dimensions as log2, unlike the other layouts.
    if (texture->texture_type() == SCE_GXM_TEXTURE_SWIZZLED
        || texture->texture_type() == SCE_GXM_TEXTURE_CUBE) {
        if (!std::has_single_bit(height))
            return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
        texture->height_base2 = std::countr_zero(height);
        return 0;
    }

    if ((texture->type << 29) == SCE_GXM_TEXTURE_TILED) {
        if (texture->mip_count > 1) {
            if (height >> (texture->mip_count - 1) >> 0x1F) {
                goto METAL_LINEAR;
            }
        }
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if ((texture->type << 29) != SCE_GXM_TEXTURE_TILED) {
    METAL_LINEAR:
        texture->height = height - 1;
        return 0;
    }
    return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

int export_sceGxmTextureSetLodMin_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, uint32_t lodMin) {
    TRACY_FUNC(sceGxmTextureSetLodMin, texture, lodMin);
    if (!texture) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if ((texture->type << 29) == SCE_GXM_TEXTURE_LINEAR_STRIDED) {
        return RET_ERROR(SCE_GXM_ERROR_UNSUPPORTED);
    }

    if (emuenv.renderer->current_backend == renderer::Backend::Metal) {
        texture->lod_min0 = lodMin >> 2;
        texture->lod_min1 = lodMin & 3;
    } else {
        texture->lod_min0 = lodMin & 3;
        texture->lod_min1 = lodMin >> 2;
    }

    return 0;
    }

int export_sceGxmTextureSetStride_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, uint32_t byteStride) {
    TRACY_FUNC(sceGxmTextureSetStride, texture, byteStride);
    if (!texture)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    if ((texture->type << 29) != SCE_GXM_TEXTURE_LINEAR_STRIDED)
        return RET_ERROR(SCE_GXM_ERROR_UNSUPPORTED);
    if (byteStride & 3)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_ALIGNMENT);
    if ((byteStride < 4) || (byteStride > 131072))
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);

    set_linear_strided_texture_stride_metal3(*texture, byteStride);
    return 0;
    }

int export_sceGxmTextureSetUAddrModeSafe_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, SceGxmTextureAddrMode mode) {
    TRACY_FUNC(sceGxmTextureSetUAddrModeSafe, texture, mode);
    if (!texture)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (!verify_texture_mode(texture, mode)) {
        if (emuenv.renderer->current_backend == renderer::Backend::Metal
            && (texture->texture_type() == SCE_GXM_TEXTURE_CUBE
                || texture->texture_type() == SCE_GXM_TEXTURE_CUBE_ARBITRARY)) {
            texture->uaddr_mode = SCE_GXM_TEXTURE_ADDR_CLAMP;
            return 0;
        }
        return RET_ERROR(SCE_GXM_ERROR_UNSUPPORTED);
    }

    texture->uaddr_mode = mode;
    return 0;
    }

int export_sceGxmTextureSetVAddrModeSafe_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, SceGxmTextureAddrMode mode) {
    TRACY_FUNC(sceGxmTextureSetVAddrModeSafe, texture, mode);
    if (!texture)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    if (!verify_texture_mode(texture, mode)) {
        if (emuenv.renderer->current_backend == renderer::Backend::Metal
            && (texture->texture_type() == SCE_GXM_TEXTURE_CUBE
                || texture->texture_type() == SCE_GXM_TEXTURE_CUBE_ARBITRARY)) {
            texture->vaddr_mode = SCE_GXM_TEXTURE_ADDR_CLAMP;
            return 0;
        }
        return RET_ERROR(SCE_GXM_ERROR_UNSUPPORTED);
    }

    texture->vaddr_mode = mode;
    return 0;
    }

int export_sceGxmTextureSetWidth_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, SceGxmTexture *texture, uint32_t width) {
    TRACY_FUNC(sceGxmTextureSetWidth, texture, width);
    if (!texture) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    } else if (!width || width > 4096) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if (texture->texture_type() == SCE_GXM_TEXTURE_SWIZZLED
        || texture->texture_type() == SCE_GXM_TEXTURE_CUBE) {
        if (!std::has_single_bit(width))
            return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
        texture->width_base2 = std::countr_zero(width);
        return 0;
    }

    if ((texture->type << 29) == SCE_GXM_TEXTURE_TILED) {
        if (texture->mip_count > 1) {
            if (width >> (texture->mip_count - 1) >> 0x1F) {
                goto METAL_LINEAR;
            }
        }
        return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

    if ((texture->type << 29) != SCE_GXM_TEXTURE_TILED) {
    METAL_LINEAR:
        texture->width = width - 1;
        return 0;
    }
    return RET_ERROR(SCE_GXM_ERROR_INVALID_VALUE);
    }

int export_sceGxmUnmapMemory_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, Ptr<void> base) {
    TRACY_FUNC(sceGxmUnmapMemory, base);
    if (!base) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    if (base.address() % KiB(4) != 0)
        LOG_WARN_ONCE("Unmapping unaligned GPU memory");

    // Make sure the base address are 4KiB-aligned
    Address aligned_base = align_down(base.address(), KiB(4));

    auto ite = emuenv.gxm.memory_mapped_regions.find(aligned_base);
    if (ite == emuenv.gxm.memory_mapped_regions.end()) {
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);
    }

    // this memory range may contain trapped region, so untrap everything to make sure
    // we don't run into issues later
    // TODO: call a mem function to invalidate the range instead
    uint8_t *addr_start = base.cast<uint8_t>().get(emuenv.mem);
    for (volatile uint8_t *addr = addr_start; addr < addr_start + ite->second.size; addr += emuenv.mem.host_page_size) {
        // this should cause a read and a write
        *addr = *addr;
    }

    if ((emuenv.renderer->features.enable_memory_mapping
            || emuenv.renderer->current_backend == renderer::Backend::Metal) && ite->second.size > 0)
        renderer::send_single_command(*emuenv.renderer, nullptr, renderer::CommandOpcode::MemoryUnmap, true, aligned_base);

    emuenv.gxm.memory_mapped_regions.erase(ite);
    return 0;
    }

void gxm_context_free_command_list_metal3(SceGxmContext &self, SceGxmCommandList *command_list) {
        assert(command_list->list);

        if (renderer::metal::gxm_metadata(*self.renderer).enabled) {
            auto &gxm = *renderer::metal::gxm_metadata(*self.renderer).owner;
            // Execute holds the same mutex until it owns a complete copy.
            // Remove membership before freeing any original node/payload.
            const std::lock_guard<std::mutex> lock(metal_gxm_bookkeeping(gxm).metal_command_lists_mutex);
            metal_gxm_bookkeeping(gxm).metal_command_lists.erase(command_list->list);
        }

        // command list has been overwritten, free the memory
        // everything except the command_list except was allocated using malloc
        renderer::Command *cmd = command_list->list->first;
        while (cmd != command_list->list->last) {
            renderer::Command *next = cmd->next;
            renderer::destroy_command_payload(*cmd);
            free(cmd);
            cmd = next;
        }
        renderer::destroy_command_payload(*cmd);
        free(cmd);
        free(command_list->list);

        // we also need to delete all ranges occupied by this list
        while (!command_list->memory_ranges.empty()) {
            self.command_list_ranges.erase(command_list->memory_ranges.top());
            command_list->memory_ranges.pop();
        }

        delete command_list;
    }

void gxm_context_insert_new_memory_range_metal3(SceGxmContext &self) {
        // An empty deferred list has not occupied guest VDM memory. In
        // particular, do not attach a zero-length range after a failed refill.
        if (renderer::metal::gxm_metadata(*self.renderer).enabled && self.alloc_space.address() == self.alloc_space_start.address())
            return;

        CommandListRange range = {
            self.alloc_space_start.address(),
            self.alloc_space.address(),
            self.curr_command_list
        };
        if (renderer::metal::gxm_metadata(*self.renderer).enabled) {
            // A failed VDM refill already sealed this range. End's host-only
            // descriptor fallback must not push the same iterator twice.
            const auto existing = self.command_list_ranges.find(range);
            if (existing != self.command_list_ranges.end() && existing->end == range.end
                && existing->command_list == self.curr_command_list)
                return;
        }
        self.deferred_check_for_free(range);
        RangeIterator it = self.command_list_ranges.emplace(std::move(range)).first;
        self.curr_command_list->memory_ranges.push(it);
    }

bool gxm_context_make_new_alloc_space_metal3(SceGxmContext &self, KernelState &kern, const MemState &mem, const SceUID thread_id, bool force) {
        if (self.alloc_space && (self.state.type == SCE_GXM_CONTEXT_TYPE_IMMEDIATE)) {
            return false;
        }

        if (!force && self.alloc_space && self.alloc_space < self.alloc_space_end) {
            // we already have spare space
            return true;
        }

        if (self.state.active && self.state.type == SCE_GXM_CONTEXT_TYPE_DEFERRED) {
            // update memory ranges
            self.insert_new_memory_range();
            // A failed callback must not register the sealed span again when
            // the command list is ended or another refill is attempted.
            if (renderer::metal::gxm_metadata(*self.renderer).enabled)
                self.alloc_space_start = self.alloc_space;
        }

        std::uint32_t actual_size = 0;

        if (self.state.vdm_buffer && self.state.vdm_buffer_size > 0) {
            self.alloc_space = self.state.vdm_buffer.cast<uint8_t>();
            actual_size = self.state.vdm_buffer_size;

            if (self.state.type == SCE_GXM_CONTEXT_TYPE_IMMEDIATE) {
                self.command_allocator_size = self.state.vdm_buffer_size / sizeof(renderer::Command);
                self.command_next_free_pos = 0;
                self.command_last_free_pos = self.command_allocator_size - 1;
            } else {
                // setting the vdm buffer size to 0 means we are using it
                self.state.vdm_buffer_size = 0;
            }
        } else {
            constexpr uint32_t DEFAULT_SIZE = 1024;

            Ptr<void> space = gxmRunDeferredMemoryCallback_metal3(kern, mem, self.callback_lock, actual_size, self.state.vdm_memory_callback,
                self.state.memory_callback_userdata, DEFAULT_SIZE, thread_id, renderer::metal::gxm_metadata(*self.renderer).enabled);

            if (!space) {
                LOG_ERROR("VDM callback runs out of memory!");
                return false;
            }

            self.alloc_space = space.cast<uint8_t>();
        }

        self.alloc_space_start = self.alloc_space;
        self.alloc_space_end = self.alloc_space + actual_size;
        return true;
    }

renderer::Command *gxm_context_allocate_new_command_metal3(SceGxmContext &self, KernelState &kern, const MemState &mem, SceUID current_thread_id) {
        renderer::Command *new_command = nullptr;

        if (self.state.type == SCE_GXM_CONTEXT_TYPE_IMMEDIATE) {
            if (self.command_allocator_size > 0 && self.command_next_free_pos <= self.command_last_free_pos.load(std::memory_order_acquire)) {
                size_t offset = self.command_next_free_pos % self.command_allocator_size;
                self.command_next_free_pos++;
                new_command = self.alloc_space.cast<renderer::Command>().get(mem) + offset;
                new (new_command) renderer::Command;
            } else {
                new_command = new renderer::Command;
                new_command->flags |= renderer::Command::FLAG_FROM_HOST;
            }
        } else {
            if (renderer::metal::gxm_metadata(*self.renderer).enabled && renderer::metal::gxm_metadata(*self.renderer).recording_failed)
                return nullptr;
            new_command = self.linearly_allocate<renderer::Command>(kern, mem, current_thread_id);
            if (renderer::metal::gxm_metadata(*self.renderer).enabled && !new_command) {
                renderer::metal::gxm_metadata(*self.renderer).recording_failed = true;
                return nullptr;
            }

            new (new_command) renderer::Command;
            new_command->flags |= renderer::Command::FLAG_NO_FREE;
        }

        return new_command;
    }

void gxm_context_free_new_command_metal3(SceGxmContext &self, renderer::Command *cmd) {
        if (!(cmd->flags & renderer::Command::FLAG_NO_FREE)) {
#ifdef __APPLE__
            if (cmd->flags & renderer::Command::FLAG_METAL_PROGRAM_BINDING)
                renderer::destroy_command_payload(*cmd);
#endif
            if (cmd->flags & renderer::Command::FLAG_FROM_HOST) {
                delete cmd;
            } else {
                self.command_last_free_pos.fetch_add(1, std::memory_order_release);
            }
        }
    }

int export_sceGxmShaderPatcherDestroy_metal3(EmuEnvState &emuenv, SceUID thread_id, const char *export_name, Ptr<SceGxmShaderPatcher> shaderPatcher) {
    TRACY_FUNC(sceGxmShaderPatcherDestroy, shaderPatcher);
    if (!shaderPatcher)
        return RET_ERROR(SCE_GXM_ERROR_INVALID_POINTER);

    free_callbacked_metal3(emuenv, thread_id, shaderPatcher.get(emuenv.mem), shaderPatcher);

    return 0;
}

void import_sceGxmSetYuvProfile_dispatch_metal3(EmuEnvState &emuenv, CPUState &cpu, SceUID thread_id) {
    if (emuenv.renderer && emuenv.renderer->current_backend == renderer::Backend::Metal) {
        const auto bridge = make_bridge<&export_sceGxmSetYuvProfile_metal3>("sceGxmSetYuvProfile", &export_sceGxmSetYuvProfile_metal3);
        bridge(emuenv, cpu, thread_id);
    } else {
        const auto bridge = make_bridge<&export_sceGxmSetYuvProfile>("sceGxmSetYuvProfile", &export_sceGxmSetYuvProfile);
        bridge(emuenv, cpu, thread_id);
    }
}
