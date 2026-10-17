// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/metal4_commands.h>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace {
constexpr MTLStages graphics_stages = MTLStageVertex | MTLStageFragment;
constexpr MTLStages compute_stages = MTLStageDispatch | MTLStageBlit;
constexpr MTLStages all_stages = graphics_stages | compute_stages;
std::mutex submission_mutex;
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

@interface V3KMetal4Queue ()
- (id<MTL4CommandAllocator>)acquireAllocator;
- (void)recycleAllocator:(id<MTL4CommandAllocator>)allocator;
@end

@implementation V3KMetal4Queue {
    NSLock *_allocatorLock;
    NSMutableArray<id<MTL4CommandAllocator>> *_allocatorPool;
}
- (instancetype)initWithDevice:(id<MTLDevice>)device {
    if ((self = [super init])) {
        _device = device;
        _allocatorLock = [NSLock new];
        _allocatorPool = [NSMutableArray new];
        auto descriptor = [MTL4CommandQueueDescriptor new];
        descriptor.label = @"Vita3K Metal 4 graphics queue";
        NSError *error = nil;
        _native = [device newMTL4CommandQueueWithDescriptor:descriptor error:&error];
        if (!_native) return nil;
    }
    return self;
}
- (V3KMetal4Commands *)commandBuffer { return [[V3KMetal4Commands alloc] initWithQueue:self]; }
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
@end

@implementation V3KMetal4Commands {
    V3KMetal4Queue *_queue;
    id<MTL4CommandAllocator> _allocator;
    id<MTLResidencySet> _residency;
    NSMutableSet *_resources;
    NSMutableArray *_handlers;
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
        auto desc = [MTLResidencySetDescriptor new];
        desc.label = @"Vita3K Metal 4 submission resources";
        desc.initialCapacity = 128;
        NSError *error = nil;
        _residency = [queue.device newResidencySetWithDescriptor:desc error:&error];
        if (!_native || !_allocator || !_residency) return nil;
        _resources = [NSMutableSet new];
        _handlers = [NSMutableArray new];
        _completion = [NSCondition new];
        _status = MTLCommandBufferStatusNotEnqueued;
        [_native beginCommandBufferWithAllocator:_allocator];
    }
    return self;
}
- (NSString *)label { return _native.label; }
- (void)setLabel:(NSString *)label { _native.label = label; }
- (id<MTLDevice>)device { return _queue.device; }
- (MTLCommandBufferStatus)status { [_completion lock]; auto value = _status; [_completion unlock]; return value; }
- (NSError *)error { [_completion lock]; auto value = _error; [_completion unlock]; return value; }
- (CFTimeInterval)GPUStartTime { [_completion lock]; auto value = _GPUStartTime; [_completion unlock]; return value; }
- (CFTimeInterval)GPUEndTime { [_completion lock]; auto value = _GPUEndTime; [_completion unlock]; return value; }
- (void)retainObject:(id)object { if (object) [_resources addObject:object]; }
- (void)retainResource:(id<MTLResource>)resource {
    if (!resource) return;
    [self retainObject:resource];
    [_residency addAllocation:resource];
    // Views can refer to a backing texture or buffer not directly bound.
    if ([resource conformsToProtocol:@protocol(MTLTexture)]) {
        id<MTLTexture> texture = (id<MTLTexture>)resource;
        if (texture.parentTexture) [self retainResource:texture.parentTexture];
        if (texture.buffer) [self retainResource:texture.buffer];
    }
}
- (MTLGPUAddress)copyBytes:(const void *)bytes length:(NSUInteger)length {
    if (!length) return 0;
    if (!bytes) throw std::invalid_argument("Metal 4 null inline data");
    NSUInteger offset = (_inlineUsed + 255) & ~NSUInteger(255);
    if (!_inlineBuffer || offset > _inlineBuffer.length || length > _inlineBuffer.length - offset) {
        _inlineBuffer = [_queue.device newBufferWithLength:MAX(NSUInteger(65536), length) options:MTLResourceStorageModeShared];
        if (!_inlineBuffer) throw std::runtime_error("Metal 4 inline upload allocation failed");
        [self retainResource:_inlineBuffer];
        offset = 0;
    }
    std::memcpy(static_cast<char *>(_inlineBuffer.contents) + offset, bytes, length);
    _inlineUsed = offset + length;
    return _inlineBuffer.gpuAddress + offset;
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
        [_completion lock];
        [_queue recycleAllocator:_allocator];
        _allocator = nil;
        [_resources removeAllObjects];
        _inlineBuffer = nil;
        [_residency removeAllAllocations];
        [_residency commit];
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
@end

@implementation V3KMetal4RenderEncoder {
    id<MTL4RenderCommandEncoder> _native;
    V3KMetal4Commands *_commands;
    id<MTL4ArgumentTable> _vertex, _fragment;
}
- (instancetype)initWithNative:(id<MTL4RenderCommandEncoder>)encoder commands:(V3KMetal4Commands *)commands {
    if ((self = [super init])) {
        _native = encoder; _commands = commands;
        _vertex = make_table(commands.native.device); _fragment = make_table(commands.native.device);
        [commands retainObject:_vertex]; [commands retainObject:_fragment];
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
    check_index(index, 31); [_vertex setAddress:[_commands copyBytes:bytes length:length] atIndex:index];
}
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    check_index(index, 31); [_fragment setAddress:[_commands copyBytes:bytes length:length] atIndex:index];
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
- (void)bindArguments {
    [_native setArgumentTable:_vertex atStages:MTLRenderStageVertex];
    [_native setArgumentTable:_fragment atStages:MTLRenderStageFragment];
}
- (void)endEncoding {
    [_native barrierAfterStages:graphics_stages beforeQueueStages:all_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native endEncoding];
}
- (void)setRenderPipelineState:(id <MTLRenderPipelineState>)pipelineState {
    [_commands retainObject:pipelineState];
    [_native setRenderPipelineState:pipelineState];
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
    [self bindArguments];
    [_native drawPrimitives:primitiveType vertexStart:vertexStart vertexCount:vertexCount];
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType indexCount:(NSUInteger)indexCount indexType:(MTLIndexType)indexType indexBuffer:(id <MTLBuffer>)indexBuffer indexBufferOffset:(NSUInteger)indexBufferOffset instanceCount:(NSUInteger)instanceCount {
    if (!indexBuffer || indexBufferOffset > indexBuffer.length) throw std::out_of_range("Metal 4 index buffer offset");
    [_commands retainResource:indexBuffer];
    [self bindArguments];
    [_native drawIndexedPrimitives:primitiveType indexCount:indexCount indexType:indexType indexBuffer:indexBuffer.gpuAddress + indexBufferOffset indexBufferLength:indexBuffer.length - indexBufferOffset instanceCount:instanceCount];
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType indexCount:(NSUInteger)indexCount indexType:(MTLIndexType)indexType indexBuffer:(id <MTLBuffer>)indexBuffer indexBufferOffset:(NSUInteger)indexBufferOffset instanceCount:(NSUInteger)instanceCount baseVertex:(NSInteger)baseVertex baseInstance:(NSUInteger)baseInstance {
    if (!indexBuffer || indexBufferOffset > indexBuffer.length) throw std::out_of_range("Metal 4 index buffer offset");
    [_commands retainResource:indexBuffer];
    [self bindArguments];
    [_native drawIndexedPrimitives:primitiveType indexCount:indexCount indexType:indexType indexBuffer:indexBuffer.gpuAddress + indexBufferOffset indexBufferLength:indexBuffer.length - indexBufferOffset instanceCount:instanceCount baseVertex:baseVertex baseInstance:baseInstance];
}
@end

@implementation V3KMetal4ComputeEncoder {
    id<MTL4ComputeCommandEncoder> _native;
    V3KMetal4Commands *_commands;
    id<MTL4ArgumentTable> _arguments;
}
- (instancetype)initWithNative:(id<MTL4ComputeCommandEncoder>)encoder commands:(V3KMetal4Commands *)commands {
    if ((self = [super init])) { _native = encoder; _commands = commands; _arguments = make_table(commands.native.device); [commands retainObject:_arguments]; }
    return self;
}
- (NSString *)label { return _native.label; }
- (void)setLabel:(NSString *)label { _native.label = label; }
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    check_index(index, 31); [_commands retainResource:buffer]; [_arguments setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
}
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    check_index(index, 31); [_arguments setAddress:[_commands copyBytes:bytes length:length] atIndex:index];
}
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index {
    check_index(index, 128); [_commands retainResource:texture]; [_arguments setTexture:texture ? texture.gpuResourceID : MTLResourceID{} atIndex:index];
}
- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index {
    check_index(index, 16); [_commands retainObject:sampler]; [_arguments setSamplerState:sampler ? sampler.gpuResourceID : MTLResourceID{} atIndex:index];
}
- (void)bindArguments {
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native setArgumentTable:_arguments];
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
    [self bindArguments];
    [_native dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}
- (void)dispatchThreads:(MTLSize)threadsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup {
    [self bindArguments];
    [_native dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationTexture];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native copyFromTexture:sourceTexture sourceSlice:sourceSlice sourceLevel:sourceLevel sourceOrigin:sourceOrigin sourceSize:sourceSize toTexture:destinationTexture destinationSlice:destinationSlice destinationLevel:destinationLevel destinationOrigin:destinationOrigin];
}
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset sourceBytesPerRow:(NSUInteger)sourceBytesPerRow sourceBytesPerImage:(NSUInteger)sourceBytesPerImage sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin {
    [_commands retainResource:sourceBuffer];
    [_commands retainResource:destinationTexture];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native copyFromBuffer:sourceBuffer sourceOffset:sourceOffset sourceBytesPerRow:sourceBytesPerRow sourceBytesPerImage:sourceBytesPerImage sourceSize:sourceSize toTexture:destinationTexture destinationSlice:destinationSlice destinationLevel:destinationLevel destinationOrigin:destinationOrigin];
}
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset sourceBytesPerRow:(NSUInteger)sourceBytesPerRow sourceBytesPerImage:(NSUInteger)sourceBytesPerImage sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin options:(MTLBlitOption)options {
    [_commands retainResource:sourceBuffer];
    [_commands retainResource:destinationTexture];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native copyFromBuffer:sourceBuffer sourceOffset:sourceOffset sourceBytesPerRow:sourceBytesPerRow sourceBytesPerImage:sourceBytesPerImage sourceSize:sourceSize toTexture:destinationTexture destinationSlice:destinationSlice destinationLevel:destinationLevel destinationOrigin:destinationOrigin options:options];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toBuffer:(id<MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset destinationBytesPerRow:(NSUInteger)destinationBytesPerRow destinationBytesPerImage:(NSUInteger)destinationBytesPerImage {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationBuffer];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native copyFromTexture:sourceTexture sourceSlice:sourceSlice sourceLevel:sourceLevel sourceOrigin:sourceOrigin sourceSize:sourceSize toBuffer:destinationBuffer destinationOffset:destinationOffset destinationBytesPerRow:destinationBytesPerRow destinationBytesPerImage:destinationBytesPerImage];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toBuffer:(id<MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset destinationBytesPerRow:(NSUInteger)destinationBytesPerRow destinationBytesPerImage:(NSUInteger)destinationBytesPerImage options:(MTLBlitOption)options {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationBuffer];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native copyFromTexture:sourceTexture sourceSlice:sourceSlice sourceLevel:sourceLevel sourceOrigin:sourceOrigin sourceSize:sourceSize toBuffer:destinationBuffer destinationOffset:destinationOffset destinationBytesPerRow:destinationBytesPerRow destinationBytesPerImage:destinationBytesPerImage options:options];
}
- (void)generateMipmapsForTexture:(id<MTLTexture>)texture {
    [_commands retainResource:texture];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native generateMipmapsForTexture:texture];
}
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value {
    [_commands retainResource:buffer];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native fillBuffer:buffer range:range value:value];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel sliceCount:(NSUInteger)sliceCount levelCount:(NSUInteger)levelCount {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationTexture];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native copyFromTexture:sourceTexture sourceSlice:sourceSlice sourceLevel:sourceLevel toTexture:destinationTexture destinationSlice:destinationSlice destinationLevel:destinationLevel sliceCount:sliceCount levelCount:levelCount];
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture toTexture:(id<MTLTexture>)destinationTexture {
    [_commands retainResource:sourceTexture];
    [_commands retainResource:destinationTexture];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native copyFromTexture:sourceTexture toTexture:destinationTexture];
}
- (void)copyFromBuffer:(id <MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset toBuffer:(id <MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset size:(NSUInteger)size {
    [_commands retainResource:sourceBuffer];
    [_commands retainResource:destinationBuffer];
    [_native barrierAfterEncoderStages:compute_stages beforeEncoderStages:compute_stages visibilityOptions:MTL4VisibilityOptionDevice];
    [_native copyFromBuffer:sourceBuffer sourceOffset:sourceOffset toBuffer:destinationBuffer destinationOffset:destinationOffset size:size];
}
@end
