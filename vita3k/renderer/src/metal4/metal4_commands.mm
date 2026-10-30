// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/metal4_commands.h>
#include <renderer/metal4/command_dependencies.h>
#include <renderer/metal4/residency_cache.h>
#include <algorithm>
#include <memory>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <bitset>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

namespace {
constexpr MTLStages graphics_stages = MTLStageVertex | MTLStageFragment;
constexpr MTLStages compute_stages = MTLStageDispatch | MTLStageBlit;
constexpr MTLStages all_stages = graphics_stages | compute_stages;
using renderer::metal::CommandAccess;
using renderer::metal::CommandDependencies;
uintptr_t resource_identity(id resource) {
    return reinterpret_cast<uintptr_t>((__bridge void *)resource);
}
// Remote peer resources are not used on Apple Silicon. Treat any such object
// conservatively as unknown rather than assuming distinct objects cannot alias.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
CommandAccess buffer_access(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger length, bool write) {
    CommandAccess result{};
    result.write = write;
    if (!buffer || buffer.remoteStorageBuffer || offset > buffer.length
        || !length || length > buffer.length - offset) return result;
    const auto gpu = buffer.gpuAddress;
    const auto cpu = reinterpret_cast<uintptr_t>(buffer.contents);
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    if (!gpu || offset > max - gpu || length > max - gpu - offset
        || (cpu && (offset > max - cpu || length > max - cpu - offset))) return result;
    result.gpu_begin = gpu + offset; result.gpu_end = result.gpu_begin + length;
    if (cpu) { result.cpu_begin = cpu + offset; result.cpu_end = result.cpu_begin + length; }
    result.heap = resource_identity(buffer.heap);
    result.opaque = false;
    return result;
}
CommandAccess buffer_access(id<MTLBuffer> buffer, bool write) {
    return buffer_access(buffer, 0, buffer.length, write);
}
CommandAccess texture_access(id<MTLTexture> texture, bool write) {
    CommandAccess result{};
    result.write = write;
    if (!texture) return result;
    while (texture.parentTexture) texture = texture.parentTexture;
    if (texture.iosurface || texture.shareable || texture.remoteStorageTexture) return result;
    if (texture.buffer) return buffer_access(texture.buffer, write);
    result.texture = resource_identity(texture);
    result.heap = resource_identity(texture.heap);
    result.opaque = false;
    return result;
}
#pragma clang diagnostic pop
std::mutex submission_mutex;
struct InlineBytesSnapshot {
    const void *contents = nullptr;
    NSUInteger length = 0;
    MTLGPUAddress address = 0;
};
id<MTL4ArgumentTable> make_table(id<MTLDevice> device) {
    auto desc = [MTL4ArgumentTableDescriptor new];
    desc.maxBufferBindCount = 31;
    desc.maxTextureBindCount = 128;
    desc.maxSamplerStateBindCount = 16;
    desc.initializeBindings = YES;
    NSError *error = nil;
    auto table = [device newArgumentTableWithDescriptor:desc error:&error];
    if (!table) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal 4 argument table allocation failed");
    return table;
}
void check_index(NSUInteger index, NSUInteger count) {
    if (index >= count) throw std::out_of_range("Metal 4 argument binding index");
}
}

// Keep dirty slots with the table lease; never clear or recycle a live lease.
@interface V3KMetal4Arguments : NSObject
@property(nonatomic, readonly) id<MTL4ArgumentTable> native;
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)setAddress:(MTLGPUAddress)address atIndex:(NSUInteger)index;
- (void)setTexture:(MTLResourceID)resource atIndex:(NSUInteger)index;
- (void)setSamplerState:(MTLResourceID)resource atIndex:(NSUInteger)index;
- (void)resetBindings;
@end
@implementation V3KMetal4Arguments {
    std::bitset<31> _buffers;
    std::bitset<128> _textures;
    std::bitset<16> _samplers;
}
- (instancetype)initWithDevice:(id<MTLDevice>)device {
    if ((self = [super init])) _native = make_table(device);
    return self;
}
- (void)setAddress:(MTLGPUAddress)address atIndex:(NSUInteger)index {
    _buffers.set(index); [_native setAddress:address atIndex:index];
}
- (void)setTexture:(MTLResourceID)resource atIndex:(NSUInteger)index {
    _textures.set(index); [_native setTexture:resource atIndex:index];
}
- (void)setSamplerState:(MTLResourceID)resource atIndex:(NSUInteger)index {
    _samplers.set(index); [_native setSamplerState:resource atIndex:index];
}
- (void)resetBindings {
    for (size_t i=0;i<_buffers.size();++i) if (_buffers[i]) [_native setAddress:0 atIndex:i];
    for (size_t i=0;i<_textures.size();++i) if (_textures[i]) [_native setTexture:MTLResourceID{} atIndex:i];
    for (size_t i=0;i<_samplers.size();++i) if (_samplers[i]) [_native setSamplerState:MTLResourceID{} atIndex:i];
    _buffers.reset(); _textures.reset(); _samplers.reset();
}
@end

@interface V3KMetal4Queue ()
- (id<MTL4CommandAllocator>)acquireAllocator;
- (V3KMetal4Arguments *)acquireArguments;
- (void)recycleArguments:(V3KMetal4Arguments *)arguments;
- (id<MTLResidencySet>)acquireResidency;
- (void)recycleResidency:(id<MTLResidencySet>)residency;
- (void)recycleAllocator:(id<MTL4CommandAllocator>)allocator;
- (bool)pinTexture:(id<MTLTexture>)texture;
- (void)unpinTexture:(uintptr_t)identity;
- (void)commitTextureResidency;
@end

@implementation V3KMetal4Queue {
    NSLock *_allocatorLock;
    NSMutableArray<id<MTL4CommandAllocator>> *_allocatorPool;
    NSMutableArray<V3KMetal4Arguments *> *_argumentPool;
    NSMutableArray<id<MTLResidencySet>> *_residencyPool;
    // Producers serialize native residency mutations before taking the short
    // policy lock. Feedback only needs the latter to retire texture leases.
    std::mutex _textureDriverMutex;
    std::mutex _textureMutex;
    id<MTLResidencySet> _textureResidency;
    std::unique_ptr<renderer::metal::ResidencyCachePolicy> _texturePolicy;
    std::map<uintptr_t,id<MTLTexture>> _textureOwners;
    bool _textureResidencyDirty;
    bool _textureResidencyAttached;
}
- (instancetype)initWithDevice:(id<MTLDevice>)device {
    if ((self = [super init])) {
        _device = device;
        _allocatorLock = [NSLock new];
        _allocatorPool = [NSMutableArray new];
        _argumentPool = [NSMutableArray new];
        _residencyPool = [NSMutableArray new];
        auto descriptor = [MTL4CommandQueueDescriptor new];
        descriptor.label = @"Vita3K Metal 4 graphics queue";
        NSError *error = nil;
        _native = [device newMTL4CommandQueueWithDescriptor:descriptor error:&error];
        if (!_native) return nil;
        auto texture_desc=[MTLResidencySetDescriptor new];
        texture_desc.label=@"Vita3K cached native textures";
        texture_desc.initialCapacity=256;
        _textureResidency=[device newResidencySetWithDescriptor:texture_desc error:&error];
        if (_textureResidency) {
            const uint64_t budget=std::min<uint64_t>(512ull*1024*1024,device.recommendedMaxWorkingSetSize/16);
            _texturePolicy=std::make_unique<renderer::metal::ResidencyCachePolicy>(budget,256);
        }
    }
    return self;
}
- (V3KMetal4Commands *)commandBuffer { return [[V3KMetal4Commands alloc] initWithQueue:self]; }
- (bool)pinTexture:(id<MTLTexture>)texture {
    if (!_texturePolicy) return false;
    const auto identity=resource_identity(texture);
    // Extract owners without allocating/copying them. Their last releases occur
    // after both locks are gone, including the failed-insertion path.
    decltype(_textureOwners) retired;
    decltype(_textureOwners)::node_type rejected;
    std::lock_guard driver(_textureDriverMutex);
    const auto bytes=texture.allocatedSize;
    bool present=false, pinned=false;
    std::exception_ptr failure;
    {
        std::lock_guard lock(_textureMutex);
        present=_textureOwners.contains(identity);
        try {
            // Allocate ownership before admitting a policy lease, so a failed
            // map insertion cannot leave a lease without its resource owner.
            if (!present) _textureOwners.emplace(identity,texture);
            pinned=_texturePolicy->pin(identity,bytes,[&](uintptr_t victim) {
                retired.insert(_textureOwners.extract(victim));
            });
        } catch (...) { failure=std::current_exception(); }
        if (!pinned && !present) rejected=_textureOwners.extract(identity);
    }
    // Even if policy allocation failed after evictions, reconcile all removed
    // memberships before propagating that failure. Never call the driver while
    // holding the lock required by feedback-side unpinTexture.
    for (const auto &[key, owner]:retired) {
        [_textureResidency removeAllocation:owner];_textureResidencyDirty=true;
    }
    if (pinned && !present) {
        [_textureResidency addAllocation:texture];_textureResidencyDirty=true;
    }
    if (failure) std::rethrow_exception(failure);
    return pinned;
}
- (void)unpinTexture:(uintptr_t)identity {
    std::lock_guard lock(_textureMutex);_texturePolicy->unpin(identity);
}
- (void)commitTextureResidency {
    std::lock_guard driver(_textureDriverMutex);
    if (_textureResidencyDirty) {
        [_textureResidency commit];_textureResidencyDirty=false;
        if (!_textureResidencyAttached) {
            [_native addResidencySet:_textureResidency];_textureResidencyAttached=true;
        }
    }
}
- (id<MTL4CommandAllocator>)acquireAllocator {
    [_allocatorLock lock];
    auto allocator = _allocatorPool.lastObject;
    if (allocator) [_allocatorPool removeLastObject];
    [_allocatorLock unlock];
    return allocator ?: [_device newCommandAllocator];
}
- (void)recycleAllocator:(id<MTL4CommandAllocator>)allocator {
    // Only called after GPU feedback: no in-flight command can reference this
    // storage. Bound the idle pool so burst submissions do not pin memory.
    [allocator reset];
    [_allocatorLock lock];
    if (_allocatorPool.count < 8) [_allocatorPool addObject:allocator];
    [_allocatorLock unlock];
}
- (V3KMetal4Arguments *)acquireArguments {
    [_allocatorLock lock];
    auto arguments = _argumentPool.lastObject;
    if (arguments) [_argumentPool removeLastObject];
    [_allocatorLock unlock];
    return arguments ?: [[V3KMetal4Arguments alloc] initWithDevice:_device];
}
- (void)recycleArguments:(V3KMetal4Arguments *)arguments {
    // Feedback and CPU completion handlers have finished using these bindings.
    // Driver calls occur outside the pool lock; a racing return may fill it.
    [_allocatorLock lock];
    const bool full = _argumentPool.count >= 64;
    [_allocatorLock unlock];
    if (full) return;
    [arguments resetBindings];
    [_allocatorLock lock];
    if (_argumentPool.count < 64) [_argumentPool addObject:arguments];
    [_allocatorLock unlock];
}
- (id<MTLResidencySet>)acquireResidency {
    [_allocatorLock lock];
    auto residency = _residencyPool.lastObject;
    if (residency) [_residencyPool removeLastObject];
    [_allocatorLock unlock];
    if (residency) return residency;
    auto desc = [MTLResidencySetDescriptor new];
    desc.label = @"Vita3K Metal 4 submission resources";
    desc.initialCapacity = 128;
    NSError *error = nil;
    return [_device newResidencySetWithDescriptor:desc error:&error];
}
- (void)recycleResidency:(id<MTLResidencySet>)residency {
    // The retiring submission has already removed and committed allocations.
    [_allocatorLock lock];
    if (_residencyPool.count < 8) [_residencyPool addObject:residency];
    [_allocatorLock unlock];
}

@end

@interface V3KMetal4Commands ()
- (V3KMetal4Arguments *)acquireArguments;
- (MTLGPUAddress)copyBytes:(const void *)bytes length:(NSUInteger)length cached:(InlineBytesSnapshot *)snapshot;
- (void)releaseTexturePins;
@end

@implementation V3KMetal4Commands {
    V3KMetal4Queue *_queue;
    id<MTL4CommandAllocator> _allocator;
    id<MTLResidencySet> _residency;
    NSMutableSet *_resources;
    std::unordered_set<const void *> _residentResources;
    std::unordered_set<uintptr_t> _cachedTextureRoots;
    NSMutableArray *_handlers;
    NSMutableArray<V3KMetal4Arguments *> *_argumentTables;
    NSCondition *_completion;
    MTLCommandBufferStatus _status;
    NSError *_error;
    CFTimeInterval _GPUStartTime, _GPUEndTime;
    id<MTLDrawable> _drawable;
    id<MTLBuffer> _inlineBuffer;
    NSUInteger _inlineUsed;
    bool _finished;
}
- (instancetype)initWithQueue:(V3KMetal4Queue *)queue {
    if ((self = [super init])) {
        _queue = queue;
        _native = [queue.device newCommandBuffer];
        _allocator = [queue acquireAllocator];
        _residency = [queue acquireResidency];
        if (!_native || !_allocator || !_residency) return nil;
        _resources = [NSMutableSet new];
        _handlers = [NSMutableArray new];
        _argumentTables = [NSMutableArray new];
        _completion = [NSCondition new];
        _status = MTLCommandBufferStatusNotEnqueued;
        [_native beginCommandBufferWithAllocator:_allocator];
    }
    return self;
}
- (V3KMetal4Arguments *)acquireArguments {
    auto arguments = [_queue acquireArguments];
    [_argumentTables addObject:arguments];
    return arguments;
}
- (NSString *)label { return _native.label; }
- (void)setLabel:(NSString *)label { _native.label = label; }
- (id<MTLDevice>)device { return _queue.device; }
- (MTLCommandBufferStatus)status { [_completion lock]; auto value = _status; [_completion unlock]; return value; }
- (NSError *)error { [_completion lock]; auto value = _error; [_completion unlock]; return value; }
- (CFTimeInterval)GPUStartTime { [_completion lock]; auto value = _GPUStartTime; [_completion unlock]; return value; }
- (CFTimeInterval)GPUEndTime { [_completion lock]; auto value = _GPUEndTime; [_completion unlock]; return value; }
- (void)retainObject:(id)object { if (object) [_resources addObject:object]; }
- (void)retainResource:(id<MTLResource>)resource residencySet:(id<MTLResidencySet>)residency {
    if (!residency) { [self retainResource:resource]; return; }
    if (!resource || !_residentResources.insert((__bridge const void *)resource).second) return;
    // Cached guest buffers own a committed, immutable one-allocation set.
    // Retain its lease alongside the buffer, never mutate/recycle that set at
    // feedback, and leave ordinary transient resources on the submission set.
    [self retainObject:resource];
    [self retainObject:residency];
    [_native useResidencySet:residency];
}
- (void)retainResource:(id<MTLResource>)resource {
    // Retention alone does not imply residency. Track registration separately,
    // by identity, for this submission and visit each view's backing only once.
    if (!resource || !_residentResources.insert((__bridge const void *)resource).second) return;
    [self retainObject:resource];
    if ([resource conformsToProtocol:@protocol(MTLTexture)]) {
        id<MTLTexture> root=(id<MTLTexture>)resource;
        while (root.parentTexture) root=root.parentTexture;
        // Only root allocations with native owned storage may outlive a
        // submission. Keep borrowed pages, heaps and external images local.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        if (!root.buffer && !root.heap && !root.iosurface && !root.shareable && !root.remoteStorageTexture) {
            const auto identity=resource_identity(root);
            if (_cachedTextureRoots.contains(identity) || [_queue pinTexture:root]) {
                _cachedTextureRoots.insert(identity);[self retainObject:root];return;
            }
        }
#pragma clang diagnostic pop
    }
    [_residency addAllocation:resource];
    // Views can refer to a backing texture or buffer not directly bound.
    if ([resource conformsToProtocol:@protocol(MTLTexture)]) {
        id<MTLTexture> texture = (id<MTLTexture>)resource;
        if (texture.parentTexture) [self retainResource:texture.parentTexture];
        if (texture.buffer) [self retainResource:texture.buffer];
    }
}
- (MTLGPUAddress)copyBytes:(const void *)bytes length:(NSUInteger)length {
    return [self copyBytes:bytes length:length cached:nullptr];
}
- (MTLGPUAddress)copyBytes:(const void *)bytes length:(NSUInteger)length cached:(InlineBytesSnapshot *)snapshot {
    if (!length) return 0;
    if (!bytes) throw std::invalid_argument("Metal 4 null inline data");
    // Compare immutable submission-owned bytes, not the caller's reusable
    // serialization buffer. Old blocks remain retained until GPU completion.
    if (snapshot && snapshot->length == length && std::memcmp(snapshot->contents, bytes, length) == 0)
        return snapshot->address;
    NSUInteger offset = (_inlineUsed + 255) & ~NSUInteger(255);
    if (!_inlineBuffer || offset > _inlineBuffer.length || length > _inlineBuffer.length - offset) {
        _inlineBuffer = [_queue.device newBufferWithLength:MAX(NSUInteger(65536), length) options:MTLResourceStorageModeShared];
        if (!_inlineBuffer) throw std::runtime_error("Metal 4 inline upload allocation failed");
        [self retainResource:_inlineBuffer];
        offset = 0;
    }
    std::memcpy(static_cast<char *>(_inlineBuffer.contents) + offset, bytes, length);
    _inlineUsed = offset + length;
    const auto address = _inlineBuffer.gpuAddress + offset;
    if (snapshot) *snapshot = {static_cast<const char *>(_inlineBuffer.contents) + offset, length, address};
    return address;
}
- (V3KMetal4RenderEncoder *)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)source {
    if (self.status != MTLCommandBufferStatusNotEnqueued) return nil;
    auto desc = [MTL4RenderPassDescriptor new];
    for (NSUInteger i = 0; i < 8; ++i) {
        desc.colorAttachments[i] = source.colorAttachments[i];
        [self retainResource:source.colorAttachments[i].texture];
        [self retainResource:source.colorAttachments[i].resolveTexture];
    }
    desc.depthAttachment = source.depthAttachment;
    desc.stencilAttachment = source.stencilAttachment;
    for (id<MTLResource> texture in @[source.depthAttachment.texture ?: (id)[NSNull null], source.depthAttachment.resolveTexture ?: (id)[NSNull null], source.stencilAttachment.texture ?: (id)[NSNull null], source.stencilAttachment.resolveTexture ?: (id)[NSNull null]])
        if ((id)texture != [NSNull null]) [self retainResource:texture];
    desc.renderTargetWidth = source.renderTargetWidth;
    desc.renderTargetHeight = source.renderTargetHeight;
    desc.renderTargetArrayLength = source.renderTargetArrayLength;
    desc.imageblockSampleLength = source.imageblockSampleLength;
    desc.threadgroupMemoryLength = source.threadgroupMemoryLength;
    desc.tileWidth = source.tileWidth;
    desc.tileHeight = source.tileHeight;
    desc.rasterizationRateMap = source.rasterizationRateMap;
    [self retainObject:source.rasterizationRateMap];
    desc.defaultRasterSampleCount = source.defaultRasterSampleCount;
    desc.visibilityResultBuffer = source.visibilityResultBuffer;
    desc.visibilityResultType = source.visibilityResultType;
    [self retainResource:source.visibilityResultBuffer];
    MTLSamplePosition positions[32];
    const auto count = [source getSamplePositions:nullptr count:0];
    if (count > 32) throw std::runtime_error("Metal 4 unsupported MSAA sample position count");
    if (count) {
        [source getSamplePositions:positions count:count];
        [desc setSamplePositions:positions count:count];
    }
    auto native = [_native renderCommandEncoderWithDescriptor:desc];
    if (!native) return nil;
    [native barrierAfterQueueStages:all_stages beforeStages:graphics_stages visibilityOptions:MTL4VisibilityOptionDevice];
    return [[V3KMetal4RenderEncoder alloc] initWithNative:native commands:self];
}
- (V3KMetal4ComputeEncoder *)computeCommandEncoder {
    if (self.status != MTLCommandBufferStatusNotEnqueued) return nil;
    auto native = [_native computeCommandEncoder];
    if (!native) return nil;
    [native barrierAfterQueueStages:all_stages beforeStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    return [[V3KMetal4ComputeEncoder alloc] initWithNative:native commands:self];
}
- (V3KMetal4ComputeEncoder *)blitCommandEncoder { return [self computeCommandEncoder]; }
- (void)presentDrawable:(id<MTLDrawable>)drawable { _drawable = drawable; [self retainObject:drawable]; }
- (void)addCompletedHandler:(void (^)(V3KMetal4Commands *))handler {
    [_completion lock];
    if (_status >= MTLCommandBufferStatusCompleted) { [_completion unlock]; handler(self); }
    else { [_handlers addObject:[handler copy]]; [_completion unlock]; }
}
- (void)commit {
    [_completion lock];
    if (_status != MTLCommandBufferStatusNotEnqueued) { [_completion unlock]; throw std::logic_error("Metal 4 submission committed twice"); }
    _status = MTLCommandBufferStatusCommitted;
    [_completion unlock];
    [_residency commit];
    [_queue commitTextureResidency];
    [_native useResidencySet:_residency];
    [_native endCommandBuffer];
    auto options = [MTL4CommitOptions new];
    // The feedback block owns this submission until GPU execution is complete.
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        [_completion lock];
        _error = feedback.error;
        _GPUStartTime = feedback.GPUStartTime;
        _GPUEndTime = feedback.GPUEndTime;
        _status = feedback.error ? MTLCommandBufferStatusError : MTLCommandBufferStatusCompleted;
        NSArray *handlers = [_handlers copy];
        [_handlers removeAllObjects];
        [_completion unlock];
        for (void (^handler)(V3KMetal4Commands *) in handlers) handler(self);
        // Feedback owns these submission-only resources. Driver recycling and
        // owner destruction must not hold up terminal metadata/late handlers.
        // The full waiter still observes _finished only after all cleanup.
        [_queue recycleAllocator:_allocator];
        _allocator = nil;
        if (!feedback.error) for (V3KMetal4Arguments *arguments in _argumentTables)
            [_queue recycleArguments:arguments];
        [_argumentTables removeAllObjects];
        [self releaseTexturePins];
        [_resources removeAllObjects];
        _residentResources.clear();
        _inlineBuffer = nil;
        [_residency removeAllAllocations];
        [_residency commit];
        if (!feedback.error) [_queue recycleResidency:_residency];
        _residency = nil;
        [_completion lock];
        _finished = true;
        [_completion broadcast];
        [_completion unlock];
    }];
    std::lock_guard lock(submission_mutex);
    if (_drawable) [_queue.native waitForDrawable:_drawable];
    id<MTL4CommandBuffer> buffer = _native;
    [_queue.native commit:&buffer count:1 options:options];
    if (_drawable) { [_queue.native signalDrawable:_drawable]; [_drawable present]; }
}
- (void)waitUntilCompleted {
    [_completion lock];
    while (!_finished) [_completion wait];
    [_completion unlock];
}
- (void)releaseTexturePins {
    for (const auto identity:_cachedTextureRoots) [_queue unpinTexture:identity];
    _cachedTextureRoots.clear();
}
- (void)dealloc { [self releaseTexturePins]; }
@end

@implementation V3KMetal4RenderEncoder {
    id<MTL4RenderCommandEncoder> _native;
    V3KMetal4Commands *_commands;
    V3KMetal4Arguments *_vertex, *_fragment;
    id<MTLRenderPipelineState> _pipeline;
    InlineBytesSnapshot _vertexInline[3], _fragmentInline[3];
    bool _argumentsBound;
}
- (instancetype)initWithNative:(id<MTL4RenderCommandEncoder>)encoder commands:(V3KMetal4Commands *)commands {
    if ((self = [super init])) {
        _native = encoder; _commands = commands;
        _vertex = [commands acquireArguments]; _fragment = [commands acquireArguments];
    }
    return self;
}
- (NSString *)label { return _native.label; }
- (void)setLabel:(NSString *)label { _native.label = label; }
- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    check_index(index, 31); [_commands retainResource:buffer]; [_vertex setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
}
- (void)setFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    check_index(index, 31); [_commands retainResource:buffer]; [_fragment setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
}
- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    check_index(index, 31);
    // Slots 0/2 hold render and texture metadata. Vertex streams and large
    // transient payloads keep the ordinary upload path without comparisons.
    auto *snapshot = index == 0 || index == 2 ? &_vertexInline[index] : nullptr;
    [_vertex setAddress:[_commands copyBytes:bytes length:length cached:snapshot] atIndex:index];
}
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    check_index(index, 31);
    auto *snapshot = index == 0 || index == 2 ? &_fragmentInline[index] : nullptr;
    [_fragment setAddress:[_commands copyBytes:bytes length:length cached:snapshot] atIndex:index];
}
- (void)setVertexAndFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    check_index(index, 31);
    auto *snapshot = index == 0 || index == 2 ? &_vertexInline[index] : nullptr;
    const auto address = [_commands copyBytes:bytes length:length cached:snapshot];
    if (snapshot) _fragmentInline[index] = *snapshot;
    [_vertex setAddress:address atIndex:index];
    [_fragment setAddress:address atIndex:index];
}
- (void)setVertexTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index {
    check_index(index, 128); [_commands retainResource:texture]; [_vertex setTexture:texture ? texture.gpuResourceID : MTLResourceID{} atIndex:index];
}
- (void)setFragmentTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index {
    check_index(index, 128); [_commands retainResource:texture]; [_fragment setTexture:texture ? texture.gpuResourceID : MTLResourceID{} atIndex:index];
}
- (void)setVertexSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index {
    check_index(index, 16); [_commands retainObject:sampler]; [_vertex setSamplerState:sampler ? sampler.gpuResourceID : MTLResourceID{} atIndex:index];
}
- (void)setFragmentSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index {
    check_index(index, 16); [_commands retainObject:sampler]; [_fragment setSamplerState:sampler ? sampler.gpuResourceID : MTLResourceID{} atIndex:index];
}
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage stages:(MTLRenderStages)stages { [_commands retainResource:resource]; }
- (void)useResource:(id<MTLResource>)resource residencySet:(id<MTLResidencySet>)residency { [_commands retainResource:resource residencySet:residency]; }
- (void)bindArguments {
    // Draws snapshot table contents; association is needed only on the first
    // draw. Clear-only passes need no bindings, and PSO changes preserve them.
    [_native setArgumentTable:_vertex.native atStages:MTLRenderStageVertex];
    [_native setArgumentTable:_fragment.native atStages:MTLRenderStageFragment];
    _argumentsBound = true;
}
- (void)orderVertexMemoryWrites {
    [_native barrierAfterEncoderStages:graphics_stages beforeEncoderStages:MTLStageVertex
        visibilityOptions:MTL4VisibilityOptionDevice];
}
- (void)endEncoding {
    [_native barrierAfterStages:graphics_stages beforeQueueStages:all_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native endEncoding];
}
- (void)setRenderPipelineState:(id <MTLRenderPipelineState>)pipelineState {
    if (_pipeline == pipelineState) return;
    [_commands retainObject:pipelineState];
    [_native setRenderPipelineState:pipelineState];
    _pipeline = pipelineState;
}
- (void)setViewport:(MTLViewport)viewport {
    [_native setViewport:viewport];
}
- (void)setFrontFacingWinding:(MTLWinding)frontFacingWinding {
    [_native setFrontFacingWinding:frontFacingWinding];
}
- (void)setCullMode:(MTLCullMode)cullMode {
    [_native setCullMode:cullMode];
}
- (void)setDepthClipMode:(MTLDepthClipMode)depthClipMode {
    [_native setDepthClipMode:depthClipMode];
}
- (void)setDepthBias:(float)depthBias slopeScale:(float)slopeScale clamp:(float)clamp {
    [_native setDepthBias:depthBias slopeScale:slopeScale clamp:clamp];
}
- (void)setScissorRect:(MTLScissorRect)rect {
    [_native setScissorRect:rect];
}
- (void)setTriangleFillMode:(MTLTriangleFillMode)fillMode {
    [_native setTriangleFillMode:fillMode];
}
- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha {
    [_native setBlendColorRed:red green:green blue:blue alpha:alpha];
}
- (void)setDepthStencilState:(nullable id <MTLDepthStencilState>)depthStencilState {
    [_commands retainObject:depthStencilState];
    [_native setDepthStencilState:depthStencilState];
}
- (void)setStencilReferenceValue:(uint32_t)referenceValue {
    [_native setStencilReferenceValue:referenceValue];
}
- (void)setStencilFrontReferenceValue:(uint32_t)frontReferenceValue backReferenceValue:(uint32_t)backReferenceValue {
    [_native setStencilFrontReferenceValue:frontReferenceValue backReferenceValue:backReferenceValue];
}
- (void)setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:(NSUInteger)offset {
    [_native setVisibilityResultMode:mode offset:offset];
}
- (void)drawPrimitives:(MTLPrimitiveType)primitiveType vertexStart:(NSUInteger)vertexStart vertexCount:(NSUInteger)vertexCount {
    if (!_argumentsBound) [self bindArguments];
    [_native drawPrimitives:primitiveType vertexStart:vertexStart vertexCount:vertexCount];
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType indexCount:(NSUInteger)indexCount indexType:(MTLIndexType)indexType indexBuffer:(id <MTLBuffer>)indexBuffer indexBufferOffset:(NSUInteger)indexBufferOffset instanceCount:(NSUInteger)instanceCount {
    if (!indexBuffer || indexBufferOffset > indexBuffer.length) throw std::out_of_range("Metal 4 index buffer offset");
    [_commands retainResource:indexBuffer];
    if (!_argumentsBound) [self bindArguments];
    [_native drawIndexedPrimitives:primitiveType indexCount:indexCount indexType:indexType indexBuffer:indexBuffer.gpuAddress + indexBufferOffset indexBufferLength:indexBuffer.length - indexBufferOffset instanceCount:instanceCount];
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType indexCount:(NSUInteger)indexCount indexType:(MTLIndexType)indexType indexBuffer:(id <MTLBuffer>)indexBuffer indexBufferOffset:(NSUInteger)indexBufferOffset instanceCount:(NSUInteger)instanceCount baseVertex:(NSInteger)baseVertex baseInstance:(NSUInteger)baseInstance {
    if (!indexBuffer || indexBufferOffset > indexBuffer.length) throw std::out_of_range("Metal 4 index buffer offset");
    [_commands retainResource:indexBuffer];
    if (!_argumentsBound) [self bindArguments];
    [_native drawIndexedPrimitives:primitiveType indexCount:indexCount indexType:indexType indexBuffer:indexBuffer.gpuAddress + indexBufferOffset indexBufferLength:indexBuffer.length - indexBufferOffset instanceCount:instanceCount baseVertex:baseVertex baseInstance:baseInstance];
}
@end

@implementation V3KMetal4ComputeEncoder {
    id<MTL4ComputeCommandEncoder> _native;
    V3KMetal4Commands *_commands;
    V3KMetal4Arguments *_arguments;
    CommandDependencies _dependencies;
    bool _argumentsBound;
}
- (instancetype)initWithNative:(id<MTL4ComputeCommandEncoder>)encoder commands:(V3KMetal4Commands *)commands {
    if ((self = [super init])) {
        _native = encoder; _commands = commands;
    }
    return self;
}
// Blit-only encoders need no argument table. Allocate initialized bindings
// only when a compute binding or dispatch first needs them.
- (void)ensureArguments {
    if (!_arguments) {
        _arguments = [_commands acquireArguments];
    }
}
- (NSString *)label { return _native.label; }
- (void)setLabel:(NSString *)label { _native.label = label; }
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    [self ensureArguments];
    check_index(index, 31); [_commands retainResource:buffer]; [_arguments setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
}
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    [self ensureArguments];
    check_index(index, 31); [_arguments setAddress:[_commands copyBytes:bytes length:length] atIndex:index];
}
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index {
    [self ensureArguments];
    check_index(index, 128); [_commands retainResource:texture]; [_arguments setTexture:texture ? texture.gpuResourceID : MTLResourceID{} atIndex:index];
}
- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index {
    [self ensureArguments];
    check_index(index, 16); [_commands retainObject:sampler]; [_arguments setSamplerState:sampler ? sampler.gpuResourceID : MTLResourceID{} atIndex:index];
}
- (void)prepareAccesses:(std::span<const CommandAccess>)accesses stage:(uint8_t)stage {
    const auto source = _dependencies.prepare(stage, accesses);
    MTLStages after = 0;
    if (source & CommandDependencies::dispatch) after |= MTLStageDispatch;
    if (source & CommandDependencies::blit) after |= MTLStageBlit;
    if (after)
        [_native barrierAfterEncoderStages:after beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
}
- (void)prepareCommand:(BOOL)usesArguments {
    // Dispatch is opaque: arbitrary kernels may dereference GPU addresses in
    // bound buffers. Retain that dependency across intervening independent blits.
    const CommandAccess unknown{};
    [self prepareAccesses:std::span<const CommandAccess>(&unknown, 1) stage:CommandDependencies::dispatch];
    if (usesArguments && !_argumentsBound) {
        [self ensureArguments];
        [_native setArgumentTable:_arguments.native];
        _argumentsBound = true;
    }
}
- (void)endEncoding {
    [_native barrierAfterStages:compute_stages beforeQueueStages:all_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native endEncoding];
}
- (void)setComputePipelineState:(id <MTLComputePipelineState>)state {
    [_commands retainObject:state];
    [_native setComputePipelineState:state];
}
- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup {
    [self prepareCommand:YES];
    [_native dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}
- (void)dispatchThreads:(MTLSize)threadsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup {
    [self prepareCommand:YES];
    [_native dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationTexture];
    const CommandAccess accesses[]{texture_access(sourceTexture, false), texture_access(destinationTexture, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native copyFromTexture:sourceTexture sourceSlice:sourceSlice sourceLevel:sourceLevel sourceOrigin:sourceOrigin sourceSize:sourceSize toTexture:destinationTexture destinationSlice:destinationSlice destinationLevel:destinationLevel destinationOrigin:destinationOrigin];
}
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset sourceBytesPerRow:(NSUInteger)sourceBytesPerRow sourceBytesPerImage:(NSUInteger)sourceBytesPerImage sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin {
    [_commands retainResource:sourceBuffer];
    [_commands retainResource:destinationTexture];
    const CommandAccess accesses[]{buffer_access(sourceBuffer, false), texture_access(destinationTexture, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native copyFromBuffer:sourceBuffer sourceOffset:sourceOffset sourceBytesPerRow:sourceBytesPerRow sourceBytesPerImage:sourceBytesPerImage sourceSize:sourceSize toTexture:destinationTexture destinationSlice:destinationSlice destinationLevel:destinationLevel destinationOrigin:destinationOrigin];
}
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset sourceBytesPerRow:(NSUInteger)sourceBytesPerRow sourceBytesPerImage:(NSUInteger)sourceBytesPerImage sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin options:(MTLBlitOption)options {
    [_commands retainResource:sourceBuffer];
    [_commands retainResource:destinationTexture];
    const CommandAccess accesses[]{buffer_access(sourceBuffer, false), texture_access(destinationTexture, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native copyFromBuffer:sourceBuffer sourceOffset:sourceOffset sourceBytesPerRow:sourceBytesPerRow sourceBytesPerImage:sourceBytesPerImage sourceSize:sourceSize toTexture:destinationTexture destinationSlice:destinationSlice destinationLevel:destinationLevel destinationOrigin:destinationOrigin options:options];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toBuffer:(id<MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset destinationBytesPerRow:(NSUInteger)destinationBytesPerRow destinationBytesPerImage:(NSUInteger)destinationBytesPerImage {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationBuffer];
    const CommandAccess accesses[]{texture_access(sourceTexture, false), buffer_access(destinationBuffer, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native copyFromTexture:sourceTexture sourceSlice:sourceSlice sourceLevel:sourceLevel sourceOrigin:sourceOrigin sourceSize:sourceSize toBuffer:destinationBuffer destinationOffset:destinationOffset destinationBytesPerRow:destinationBytesPerRow destinationBytesPerImage:destinationBytesPerImage];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toBuffer:(id<MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset destinationBytesPerRow:(NSUInteger)destinationBytesPerRow destinationBytesPerImage:(NSUInteger)destinationBytesPerImage options:(MTLBlitOption)options {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationBuffer];
    const CommandAccess accesses[]{texture_access(sourceTexture, false), buffer_access(destinationBuffer, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native copyFromTexture:sourceTexture sourceSlice:sourceSlice sourceLevel:sourceLevel sourceOrigin:sourceOrigin sourceSize:sourceSize toBuffer:destinationBuffer destinationOffset:destinationOffset destinationBytesPerRow:destinationBytesPerRow destinationBytesPerImage:destinationBytesPerImage options:options];
}
- (void)generateMipmapsForTexture:(id<MTLTexture>)texture {
    [_commands retainResource:texture];
    const CommandAccess accesses[]{texture_access(texture, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native generateMipmapsForTexture:texture];
}
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value {
    [_commands retainResource:buffer];
    const CommandAccess accesses[]{buffer_access(buffer, range.location, range.length, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native fillBuffer:buffer range:range value:value];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel sliceCount:(NSUInteger)sliceCount levelCount:(NSUInteger)levelCount {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationTexture];
    const CommandAccess accesses[]{texture_access(sourceTexture, false), texture_access(destinationTexture, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native copyFromTexture:sourceTexture sourceSlice:sourceSlice sourceLevel:sourceLevel toTexture:destinationTexture destinationSlice:destinationSlice destinationLevel:destinationLevel sliceCount:sliceCount levelCount:levelCount];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture toTexture:(id<MTLTexture>)destinationTexture {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationTexture];
    const CommandAccess accesses[]{texture_access(sourceTexture, false), texture_access(destinationTexture, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native copyFromTexture:sourceTexture toTexture:destinationTexture];
}
- (void)copyFromBuffer:(id <MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset toBuffer:(id <MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset size:(NSUInteger)size {
    [_commands retainResource:sourceBuffer];
    [_commands retainResource:destinationBuffer];
    const CommandAccess accesses[]{buffer_access(sourceBuffer, sourceOffset, size, false), buffer_access(destinationBuffer, destinationOffset, size, true)};
    [self prepareAccesses:accesses stage:CommandDependencies::blit];
    [_native copyFromBuffer:sourceBuffer sourceOffset:sourceOffset toBuffer:destinationBuffer destinationOffset:destinationOffset size:size];
}
@end
