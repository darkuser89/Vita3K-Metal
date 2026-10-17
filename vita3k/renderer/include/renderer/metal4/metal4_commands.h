// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal4/apple.h>

// These owners preserve the renderer's encoding operations while executing
// exclusively on Metal 4. They own allocator memory, argument tables and all
// referenced resources until commit feedback confirms GPU completion.
@class V3KMetal4Commands, V3KMetal4RenderEncoder, V3KMetal4ComputeEncoder;

@interface V3KMetal4Queue : NSObject
@property(nonatomic, readonly) id<MTL4CommandQueue> native;
@property(nonatomic, readonly) id<MTLDevice> device;
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (V3KMetal4Commands *)commandBuffer;
@end

@interface V3KMetal4Commands : NSObject
@property(nonatomic, readonly) id<MTL4CommandBuffer> native;
@property(nonatomic, readonly) id<MTLDevice> device;
@property(nonatomic, copy) NSString *label;
@property(nonatomic, readonly) MTLCommandBufferStatus status;
@property(nonatomic, readonly) NSError *error;
@property(nonatomic, readonly) CFTimeInterval GPUStartTime;
@property(nonatomic, readonly) CFTimeInterval GPUEndTime;
- (instancetype)initWithQueue:(V3KMetal4Queue *)queue;
- (V3KMetal4RenderEncoder *)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor;
- (V3KMetal4ComputeEncoder *)computeCommandEncoder;
- (V3KMetal4ComputeEncoder *)blitCommandEncoder;
- (void)retainObject:(id)object;
- (void)retainResource:(id<MTLResource>)resource;
- (MTLGPUAddress)copyBytes:(const void *)bytes length:(NSUInteger)length;
- (void)presentDrawable:(id<MTLDrawable>)drawable;
- (void)addCompletedHandler:(void (^)(V3KMetal4Commands *))handler;
- (void)commit;
- (void)waitUntilCompleted;
@end

@interface V3KMetal4RenderEncoder : NSObject
@property(nonatomic, copy) NSString *label;
- (instancetype)initWithNative:(id<MTL4RenderCommandEncoder>)encoder commands:(V3KMetal4Commands *)commands;
- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)setVertexAndFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)setVertexTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setFragmentTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setVertexSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index;
- (void)setFragmentSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index;
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage stages:(MTLRenderStages)stages;
- (void)endEncoding;
- (void)setRenderPipelineState:(id <MTLRenderPipelineState>)pipelineState;
- (void)setViewport:(MTLViewport)viewport;
- (void)setFrontFacingWinding:(MTLWinding)frontFacingWinding;
- (void)setCullMode:(MTLCullMode)cullMode;
- (void)setDepthClipMode:(MTLDepthClipMode)depthClipMode;
- (void)setDepthBias:(float)depthBias slopeScale:(float)slopeScale clamp:(float)clamp;
- (void)setScissorRect:(MTLScissorRect)rect;
- (void)setTriangleFillMode:(MTLTriangleFillMode)fillMode;
- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha;
- (void)setDepthStencilState:(nullable id <MTLDepthStencilState>)depthStencilState;
- (void)setStencilReferenceValue:(uint32_t)referenceValue;
- (void)setStencilFrontReferenceValue:(uint32_t)frontReferenceValue backReferenceValue:(uint32_t)backReferenceValue;
- (void)setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:(NSUInteger)offset;
- (void)drawPrimitives:(MTLPrimitiveType)primitiveType vertexStart:(NSUInteger)vertexStart vertexCount:(NSUInteger)vertexCount;
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType indexCount:(NSUInteger)indexCount indexType:(MTLIndexType)indexType indexBuffer:(id <MTLBuffer>)indexBuffer indexBufferOffset:(NSUInteger)indexBufferOffset instanceCount:(NSUInteger)instanceCount;
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType indexCount:(NSUInteger)indexCount indexType:(MTLIndexType)indexType indexBuffer:(id <MTLBuffer>)indexBuffer indexBufferOffset:(NSUInteger)indexBufferOffset instanceCount:(NSUInteger)instanceCount baseVertex:(NSInteger)baseVertex baseInstance:(NSUInteger)baseInstance;
@end

@interface V3KMetal4ComputeEncoder : NSObject
@property(nonatomic, copy) NSString *label;
- (instancetype)initWithNative:(id<MTL4ComputeCommandEncoder>)encoder commands:(V3KMetal4Commands *)commands;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index;
- (void)endEncoding;
- (void)setComputePipelineState:(id <MTLComputePipelineState>)state;
- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;
- (void)dispatchThreads:(MTLSize)threadsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin;
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset sourceBytesPerRow:(NSUInteger)sourceBytesPerRow sourceBytesPerImage:(NSUInteger)sourceBytesPerImage sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin;
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset sourceBytesPerRow:(NSUInteger)sourceBytesPerRow sourceBytesPerImage:(NSUInteger)sourceBytesPerImage sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin options:(MTLBlitOption)options;
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toBuffer:(id<MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset destinationBytesPerRow:(NSUInteger)destinationBytesPerRow destinationBytesPerImage:(NSUInteger)destinationBytesPerImage;
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toBuffer:(id<MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset destinationBytesPerRow:(NSUInteger)destinationBytesPerRow destinationBytesPerImage:(NSUInteger)destinationBytesPerImage options:(MTLBlitOption)options;
- (void)generateMipmapsForTexture:(id<MTLTexture>)texture;
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value;
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel sliceCount:(NSUInteger)sliceCount levelCount:(NSUInteger)levelCount;
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture toTexture:(id<MTLTexture>)destinationTexture;
- (void)copyFromBuffer:(id <MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset toBuffer:(id <MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset size:(NSUInteger)size;
@end
