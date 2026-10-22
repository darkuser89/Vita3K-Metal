// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// CPU-only boundary double: no Metal device, queue, window or emulator.
#include <renderer/metal4/buffers.h>
#include <renderer/metal4/texture_views.h>
#include <renderer/metal4/visibility_buffers.h>
#include <renderer/metal4/residency_cache.h>
#include <cassert>
#include <objc/runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

@interface TestBuffer : NSObject {
    void *_bytes;
    NSUInteger _length;
    bool _owns;
}
- (instancetype)initWithLength:(NSUInteger)length;
- (instancetype)initWithBytes:(void *)bytes length:(NSUInteger)length;
- (NSUInteger)length;
- (void *)contents;
- (uint64_t)gpuAddress;
@end
@implementation TestBuffer
- (instancetype)initWithLength:(NSUInteger)length {
    if ((self = [super init])) { _length = length; _bytes = std::malloc(length); _owns = true; }
    return self;
}
- (instancetype)initWithBytes:(void *)bytes length:(NSUInteger)length {
    if ((self = [super init])) { _length = length; _bytes = bytes; _owns = false; }
    return self;
}
- (NSUInteger)length { return _length; }
- (void *)contents { return _bytes; }
- (uint64_t)gpuAddress { return reinterpret_cast<uintptr_t>(_bytes); }
- (id<MTLHeap>)heap { return nil; }
- (id<MTLBuffer>)remoteStorageBuffer { return nil; }
- (void)dealloc { if (_owns) std::free(_bytes); }
@end
@interface TestDevice : NSObject
- (NSUInteger)maxBufferLength;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)bytes length:(NSUInteger)length options:(MTLResourceOptions)options deallocator:(void (^)(void *, NSUInteger))deallocator;
@end
@implementation TestDevice
- (NSUInteger)maxBufferLength { return 64 * 1024 * 1024; }
- (NSUInteger)recommendedMaxWorkingSetSize { return 16 * 1024 * 1024; }
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {
    return (id<MTLBuffer>)[[TestBuffer alloc] initWithLength:length];
}
- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)bytes length:(NSUInteger)length options:(MTLResourceOptions)options deallocator:(void (^)(void *, NSUInteger))deallocator {
    assert(!deallocator);
    return (id<MTLBuffer>)[[TestBuffer alloc] initWithBytes:bytes length:length];
}
@end
namespace renderer::metal {
struct PersistentCache::Impl {};
PersistentCache::~PersistentCache() = default;
struct Device::LibraryCache {};
Device::Device() = default;
Device::~Device() = default;
std::unique_ptr<Device> Device::create(std::string &error) {
    auto result = std::unique_ptr<Device>(new Device);
    result->device = (id<MTLDevice>)[TestDevice new];
    return result;
}
}

// Metal 4 API boundary doubles. The real wrappers own and allocate all bytes;
// these objects replace only driver calls and record their observable inputs.
@interface TestTable : NSObject {
@public
    MTLGPUAddress addresses[31];
    MTLResourceID textures[128], samplers[16];
    NSUInteger clears;
}
- (void)setAddress:(MTLGPUAddress)address atIndex:(NSUInteger)index;
- (void)setTexture:(MTLResourceID)resource atIndex:(NSUInteger)index;
- (void)setSamplerState:(MTLResourceID)resource atIndex:(NSUInteger)index;
@end
@implementation TestTable
- (void)setAddress:(MTLGPUAddress)address atIndex:(NSUInteger)index { addresses[index] = address; clears += address == 0; }
- (void)setTexture:(MTLResourceID)resource atIndex:(NSUInteger)index { textures[index] = resource; clears += resource._impl == 0; }
- (void)setSamplerState:(MTLResourceID)resource atIndex:(NSUInteger)index { samplers[index] = resource; clears += resource._impl == 0; }
@end
@interface TestResidency : NSObject
@property(strong) NSMutableSet *allocations;
@property NSUInteger commits;
- (void)addAllocation:(id)allocation;
- (void)removeAllocation:(id)allocation;
- (void)removeAllAllocations;
- (void)commit;
@end
@implementation TestResidency
- (instancetype)init { if ((self=[super init])) _allocations=[NSMutableSet new]; return self; }
- (void)addAllocation:(id)allocation { [_allocations addObject:allocation]; }
- (void)removeAllocation:(id)allocation { [_allocations removeObject:allocation]; }
- (void)removeAllAllocations { [_allocations removeAllObjects]; }
- (void)commit { ++_commits; }
@end
@interface TestNativeCommands : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) TestResidency *residency;
@property(strong) NSHashTable *residencyUses;
- (void)beginCommandBufferWithAllocator:(id)allocator;
- (void)useResidencySet:(id)residency;
- (void)endCommandBuffer;
@end
@implementation TestNativeCommands
- (void)beginCommandBufferWithAllocator:(id)allocator {}
- (void)useResidencySet:(id)residency {
    if (!_residencyUses) _residencyUses=[NSHashTable weakObjectsHashTable];
    [_residencyUses addObject:residency];
    _residency=residency;
}
- (void)endCommandBuffer {}
@end
@interface TestAllocator : NSObject
- (void)reset;
@end
@implementation TestAllocator
- (void)reset {}
@end
@interface TestFeedback : NSObject
@property(strong) NSError *error;
- (CFTimeInterval)GPUStartTime;
- (CFTimeInterval)GPUEndTime;
@end
@implementation TestFeedback
- (CFTimeInterval)GPUStartTime {return 1;}
- (CFTimeInterval)GPUEndTime {return 2;}
@end
// Intercept only the CPU feedback-registration boundary in this test process.
// The production commit/retirement code and its completion handler run intact.
static char feedback_key;
static void capture_feedback(id options, SEL, MTL4CommitFeedbackHandler handler) {
    NSMutableArray *list=objc_getAssociatedObject(options,&feedback_key);
    if(!list) {list=[NSMutableArray new];objc_setAssociatedObject(options,&feedback_key,list,OBJC_ASSOCIATION_RETAIN_NONATOMIC);}
    [list addObject:[handler copy]];
}
@interface TestNativeQueue : NSObject
@property(strong) NSMutableArray<NSArray *> *pending;
@property(strong) TestResidency *textureResidency;
- (void)addResidencySet:(id)residency;
- (void)commit:(const id<MTL4CommandBuffer> *)buffers count:(NSUInteger)count options:(MTL4CommitOptions *)options;
- (void)completeNext:(BOOL)failed;
@end
@implementation TestNativeQueue
- (void)addResidencySet:(id)residency { _textureResidency=residency; }
- (instancetype)init {if((self=[super init]))_pending=[NSMutableArray new];return self;}
- (void)commit:(const id<MTL4CommandBuffer> *)buffers count:(NSUInteger)count options:(MTL4CommitOptions *)options {
    assert(count==1 && buffers[0]);
    NSArray *handlers=objc_getAssociatedObject(options,&feedback_key);assert(handlers.count==1);
    [_pending addObject:[handlers copy]];
}
- (void)completeNext:(BOOL)failed {
    assert(_pending.count);NSArray *handlers=_pending[0];[_pending removeObjectAtIndex:0];
    auto feedback=[TestFeedback new];
    if(failed)feedback.error=[NSError errorWithDomain:@"CPU test" code:1 userInfo:nil];
    for(MTL4CommitFeedbackHandler handler in handlers)handler((id<MTL4CommitFeedback>)feedback);
}
@end
static NSMutableArray<TestTable *> *tables;
static size_t residency_allocations;
@interface TestDevice (Commands)
- (id)newArgumentTableWithDescriptor:(id)descriptor error:(NSError **)error;
- (id)newCommandBuffer;
- (id)newResidencySetWithDescriptor:(id)descriptor error:(NSError **)error;
- (id)newMTL4CommandQueueWithDescriptor:(id)descriptor error:(NSError **)error;
- (id)newCommandAllocator;
@end
@implementation TestDevice (Commands)
- (id)newArgumentTableWithDescriptor:(MTL4ArgumentTableDescriptor *)descriptor error:(NSError **)error {
    assert(descriptor.initializeBindings && descriptor.maxBufferBindCount==31 && descriptor.maxTextureBindCount==128 && descriptor.maxSamplerStateBindCount==16);
    auto table=[TestTable new];[tables addObject:table];return table;
}
- (id)newCommandBuffer {auto commands=[TestNativeCommands new];commands.device=(id<MTLDevice>)self;return commands;}
- (id)newResidencySetWithDescriptor:(id)descriptor error:(NSError **)error {++residency_allocations;return [TestResidency new];}
- (id)newMTL4CommandQueueWithDescriptor:(id)descriptor error:(NSError **)error {return [TestNativeQueue new];}
- (id)newCommandAllocator {return [TestAllocator new];}
@end
@interface TestBindingResource : NSObject
- (MTLResourceID)gpuResourceID;
@end
@implementation TestBindingResource
- (MTLResourceID)gpuResourceID {return MTLResourceID{uint64_t(reinterpret_cast<uintptr_t>((__bridge void *)self))};}
@end
@interface TestResidentTexture : NSObject
@property(strong) TestResidentTexture *parentTexture;
@property NSUInteger allocatedSize;
@end
@implementation TestResidentTexture
- (BOOL)conformsToProtocol:(Protocol *)p { return p==@protocol(MTLTexture) || [super conformsToProtocol:p]; }
- (id)buffer { return nil; }
- (id)heap { return nil; }
- (IOSurfaceRef)iosurface { return nullptr; }
- (BOOL)isShareable { return NO; }
- (id)remoteStorageTexture { return nil; }
@end
@interface TestNativeEncoder : NSObject
@property NSUInteger pipeline_binds, vertex_write_barriers, blit_barriers, dispatch_barriers;
@property(strong) TestTable *vertex_table,*fragment_table,*compute_table;
- (void)setArgumentTable:(id)table atStages:(MTLRenderStages)stages;
- (void)setArgumentTable:(id)table;
- (void)drawPrimitives:(MTLPrimitiveType)primitive vertexStart:(NSUInteger)start vertexCount:(NSUInteger)count;
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group;
- (void)barrierAfterEncoderStages:(MTLStages)after beforeEncoderStages:(MTLStages)before visibilityOptions:(MTL4VisibilityOptions)options;
- (void)barrierAfterStages:(MTLStages)after beforeQueueStages:(MTLStages)before visibilityOptions:(MTL4VisibilityOptions)options;
- (void)endEncoding;
- (void)setRenderPipelineState:(id)pipeline;
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value;
@end
@implementation TestNativeEncoder
- (void)setArgumentTable:(id)table atStages:(MTLRenderStages)stages {if(stages==MTLRenderStageVertex)_vertex_table=table;else _fragment_table=table;}
- (void)setArgumentTable:(id)table {_compute_table=table;}
- (void)drawPrimitives:(MTLPrimitiveType)primitive vertexStart:(NSUInteger)start vertexCount:(NSUInteger)count {}
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group {}
- (void)barrierAfterEncoderStages:(MTLStages)after beforeEncoderStages:(MTLStages)before visibilityOptions:(MTL4VisibilityOptions)options {
    assert(options==MTL4VisibilityOptionDevice);
    if (before == MTLStageVertex) {
        assert(after==(MTLStageVertex|MTLStageFragment)); ++_vertex_write_barriers;
    } else {
        assert(before == (MTLStageDispatch|MTLStageBlit));
        assert(after && !(after & ~(MTLStageDispatch|MTLStageBlit)));
        _blit_barriers += bool(after & MTLStageBlit);
        _dispatch_barriers += bool(after & MTLStageDispatch);
    }
}
- (void)barrierAfterStages:(MTLStages)after beforeQueueStages:(MTLStages)before visibilityOptions:(MTL4VisibilityOptions)options {}
- (void)endEncoding {}
- (void)setRenderPipelineState:(id)pipeline { ++_pipeline_binds; }
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value {
    std::memset(static_cast<uint8_t *>(buffer.contents) + range.location, value, range.length);
}
@end
// COMMAND_DEPENDENCY_HOOK
static void check_commands(id<MTLDevice> device) {
    tables = [NSMutableArray new];
    auto queue = [[V3KMetal4Queue alloc] initWithDevice:device];
    auto commands = [[V3KMetal4Commands alloc] initWithQueue:(V3KMetal4Queue *)queue];
    auto native = [TestNativeEncoder new];
    auto render = [[V3KMetal4RenderEncoder alloc] initWithNative:(id<MTL4RenderCommandEncoder>)native commands:commands];
    assert(tables.count == 2);
    auto vertex = tables[0], fragment = tables[1];
    std::array<uint32_t, 128> data{};
    data.fill(0x12345678);
    [render setVertexBytes:data.data() length:sizeof(data) atIndex:0];
    const auto first = vertex->addresses[0];
    [render setVertexBytes:data.data() length:sizeof(data) atIndex:0];
    assert(vertex->addresses[0] == first);
    data[3] = 0x87654321;
    [render setVertexBytes:data.data() length:sizeof(data) atIndex:0];
    assert(vertex->addresses[0] != first);
    assert(reinterpret_cast<const uint32_t *>(first)[3] == 0x12345678);
    const auto changed = vertex->addresses[0];
    auto other = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    [render setVertexBuffer:other offset:0 atIndex:0];
    assert(vertex->addresses[0] == other.gpuAddress);
    [render setVertexBytes:data.data() length:sizeof(data) atIndex:0];
    assert(vertex->addresses[0] == changed); // identical bytes still rebind after a buffer override
    [render setVertexAndFragmentBytes:data.data() length:sizeof(data) atIndex:2];
    assert(vertex->addresses[2] == fragment->addresses[2]);
    const auto joint = fragment->addresses[2];
    data[7] = 42;
    [render setFragmentBytes:data.data() length:sizeof(data) atIndex:2];
    assert(fragment->addresses[2] != joint && vertex->addresses[2] == joint);
    assert(reinterpret_cast<const uint32_t *>(joint)[7] == 0x12345678);
    [render setVertexAndFragmentBytes:data.data() length:sizeof(data) atIndex:2];
    assert(vertex->addresses[2] == fragment->addresses[2]);
    std::vector<uint8_t> large(70000, 0x5a);
    [render setVertexBytes:large.data() length:large.size() atIndex:1]; // advances to a fresh upload block
    assert(std::memcmp(reinterpret_cast<const void *>(vertex->addresses[1]), large.data(), large.size()) == 0);
    [render setVertexBytes:data.data() length:sizeof(data) atIndex:0];
    assert(reinterpret_cast<const uint32_t *>(first)[3] == 0x12345678); // old block is still alive
    auto pipeline = [NSObject new], second_pipeline = [NSObject new];
    [render setRenderPipelineState:(id<MTLRenderPipelineState>)pipeline];
    [render setRenderPipelineState:(id<MTLRenderPipelineState>)pipeline];
    [render setRenderPipelineState:(id<MTLRenderPipelineState>)second_pipeline];
    [render setRenderPipelineState:(id<MTLRenderPipelineState>)pipeline];
    assert(native.pipeline_binds == 3);
    [render orderVertexMemoryWrites]; [render orderVertexMemoryWrites];
    assert(native.vertex_write_barriers == 2);
    auto compute = [[V3KMetal4ComputeEncoder alloc] initWithNative:(id<MTL4ComputeCommandEncoder>)native commands:commands];
    assert(tables.count == 2);
    [compute fillBuffer:other range:NSMakeRange(0, 1024) value:0x7b];
    assert(tables.count == 2 && static_cast<uint8_t *>(other.contents)[1023] == 0x7b);
    [compute setBytes:data.data() length:sizeof(data) atIndex:0];
    [compute setBytes:data.data() length:sizeof(data) atIndex:0];
    assert(tables.count == 3);
    // Actual production mapper and planner: disjoint fill intervals need no
    // barrier, overlapping no-copy aliases and opaque dispatches do.
    [compute fillBuffer:other range:NSMakeRange(1024, 16) value:1];
    assert(native.blit_barriers == 0);
    [compute fillBuffer:other range:NSMakeRange(1000, 32) value:2];
    assert(native.blit_barriers == 1);
    auto alias = (id<MTLBuffer>)[[TestBuffer alloc] initWithBytes:static_cast<uint8_t *>(other.contents)+1000 length:32];
    [compute fillBuffer:alias range:NSMakeRange(0, 32) value:3];
    assert(native.blit_barriers == 2);
    [compute dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    assert(native.blit_barriers == 3 && native.dispatch_barriers == 0);
    [compute fillBuffer:other range:NSMakeRange(2048, 16) value:4];
    assert(native.blit_barriers == 3 && native.dispatch_barriers == 1);
    [compute fillBuffer:other range:NSMakeRange(2064, 16) value:4];
    assert(native.blit_barriers == 3 && native.dispatch_barriers == 1);
    check_blit_dependencies(commands);
    tables = nil;
    std::cout << "PASS Metal 4 wrappers: metadata reuse/mutation/rebind/rollover, 70KB upload, PSO bind counts, writing-vertex barrier stages/visibility and lazy blit tables\n";
}


static void check_direct_cache(renderer::metal::Device &device) {
    using namespace renderer::metal;
    std::mt19937 random(0x43ac9);
    size_t queries = 0;
    for (size_t round = 0; round < 100; ++round) {
        DirectGuestBufferCache cache;
        std::map<std::pair<uintptr_t, uintptr_t>, id<MTLBuffer>> reference;
        const auto query = [&](uintptr_t begin, uintptr_t end) {
            const auto key = std::pair{begin, end};
            id<MTLBuffer> expected = nil;
            uintptr_t origin = begin;
            if (auto exact = reference.find(key); exact != reference.end()) expected = exact->second;
            else for (const auto &[range, buffer] : reference) {
                if (range.first > begin) break;
                if (range.second >= end) { expected = buffer; origin = range.first; break; }
            }
            const auto result = cache.get(device, begin, end);
            if (expected) assert(result.buffer == expected && result.offset == begin - origin);
            else {
                assert(result.offset == 0 && result.buffer.length == end - begin);
                for (auto it = reference.lower_bound({begin, 0}); it != reference.end();) {
                    if (it->first.first >= end) break;
                    if (it->first.second <= end) it = reference.erase(it); else ++it;
                }
                reference.emplace(key, result.buffer);
            }
            assert(cache.size() == reference.size());
            ++queries;
        };
        // Cross-overlap, equal endpoints, containing insertion and exact keys.
        for (auto [begin, end] : {std::pair{100u, 200u}, {150u, 250u}, {160u, 180u},
                 {100u, 250u}, {50u, 175u}, {60u, 150u}, {50u, 250u}, {50u, 250u}})
            query(begin, end);
        for (size_t i = 0; i < 1000; ++i) {
            const uintptr_t begin = 4096 + random() % 100000;
            query(begin, begin + 1 + random() % 2000);
        }
        cache.clear(); reference.clear();
        for (size_t i = 0; i < 512; ++i) query(4096 + i * 64, 4096 + i * 64 + 32);
        for (size_t i = 0; i < 512; ++i) query(4096 + i * 64 + 1, 4096 + i * 64 + 17);
        cache.clear(); reference.clear(); assert(cache.size() == 0);
        query(4096, 8192); // both indexes must have been cleared
    }
    std::cout << "PASS direct buffer index: " << queries << " queries preserve original linear owner/offset, overlaps and clear\n";
}

static size_t view_allocations;
@interface TestTexture : NSObject
@property(strong) id parent;
@property renderer::metal::TextureViewDescription viewDescription;
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format;
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format textureType:(MTLTextureType)type levels:(NSRange)levels slices:(NSRange)slices swizzle:(MTLTextureSwizzleChannels)swizzle;
@end
@implementation TestTexture
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format {
    auto view = [TestTexture new]; view.parent = self; ++view_allocations;
    return (id<MTLTexture>)view;
}
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format textureType:(MTLTextureType)type levels:(NSRange)levels slices:(NSRange)slices swizzle:(MTLTextureSwizzleChannels)swizzle {
    assert(levels.location == 0 && slices.location == 0);
    auto view = [TestTexture new]; view.parent = self; ++view_allocations;
    view.viewDescription = {format, type, levels.length, slices.length,
        {swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha}, false};
    return (id<MTLTexture>)view;
}
@end
static void check_texture_views() {
    using namespace renderer::metal;
    const TextureViewDescription description{MTLPixelFormatRGBA8Unorm, MTLTextureType2D, 4, 1,
        {MTLTextureSwizzleBlue, MTLTextureSwizzleGreen, MTLTextureSwizzleRed, MTLTextureSwizzleOne}};
    TextureViewCache cache;
    auto source = [TestTexture new];
    auto view = cache.get((id<MTLTexture>)source, description);
    assert(((TestTexture *)view).viewDescription == description);
    const auto before = view_allocations;
    for (size_t i = 0; i < 10000; ++i) assert(cache.get((id<MTLTexture>)source, description) == view);
    assert(view_allocations == before);
    auto retained = [NSMutableArray new];
    for (size_t field = 0; field < 6; ++field) {
        auto other = description;
        switch (field) {
        case 0: other.format = MTLPixelFormatRGBA8Unorm_sRGB; break;
        case 1: other.type = MTLTextureTypeCube; break;
        case 2: other.levels = 3; break;
        case 3: other.slices = 6; break;
        case 4: other.swizzle[3] = MTLTextureSwizzleZero; break;
        case 5: other.format_only = true; break;
        }
        auto changed = cache.get((id<MTLTexture>)source, other);
        assert(changed != view);
        assert(cache.get((id<MTLTexture>)source, other) == changed);
        [retained addObject:changed];
    }
    assert(view_allocations == before + 6);
    auto other_source = [TestTexture new];
    auto other_view = cache.get((id<MTLTexture>)other_source, description);
    assert(other_view != view);
    __weak id weak_source, weak_view;
    @autoreleasepool {
        auto temporary = [TestTexture new];
        auto temporary_view = cache.get((id<MTLTexture>)temporary, description);
        weak_source = temporary; weak_view = temporary_view;
    }
    assert(!weak_source && !weak_view); // cache adds no resource ownership/cycle
    // Keep views alive to exercise the bound independently of weak expiry.
    for (size_t i = 0; i < TextureViewCache::capacity + 1; ++i) {
        auto key = description; key.levels = i + 10;
        [retained addObject:cache.get((id<MTLTexture>)source, key)];
        assert(cache.size() <= TextureViewCache::capacity);
    }
    const auto before_evicted = view_allocations;
    auto replacement = cache.get((id<MTLTexture>)source, description);
    assert(replacement != view && view_allocations == before_evicted + 1);
    assert(((TestTexture *)view).parent == source); // evicted in-flight view stays valid
    assert(cache.get((id<MTLTexture>)source, description) == replacement);
    std::cout << "PASS texture view cache: 10000 repeats, all key fields/source identity, weak lifetime and bounded eviction\n";
}

static void visibility_pool_test(id<MTLDevice> device) {
    using renderer::metal::VisibilityBufferPool;
    VisibilityBufferPool pool;
    auto first = pool.acquire(device, 4096);
    auto second = pool.acquire(device, 4096);
    assert(first != second);
    static_cast<uint64_t *>(first.contents)[0] = 37;
    static_cast<uint64_t *>(second.contents)[0] = 11;
    // Merely ending a pass does not return its buffer. Model publication only
    // after the caller's completion wait; both results still have their bytes.
    assert(static_cast<uint64_t *>(first.contents)[0] == 37);
    pool.recycle_after_publication(first);
    pool.recycle_after_publication(first); // Many query records share a pass.
    auto reused = pool.acquire(device, 4096);
    assert(reused == first);
    for (size_t i = 0; i < reused.length; ++i) assert(static_cast<uint8_t *>(reused.contents)[i] == 0);
    auto concurrent = pool.acquire(device, 4096);
    assert(concurrent != reused && concurrent != second);
    assert(static_cast<uint64_t *>(second.contents)[0] == 11);
    for (int i = 0; i < 10000; ++i) {
        static_cast<uint64_t *>(reused.contents)[i % 512] = i + 1;
        pool.recycle_after_publication(reused);
        reused = pool.acquire(device, 4096);
        assert(reused == first);
        assert(static_cast<uint64_t *>(reused.contents)[i % 512] == 0);
    }
    VisibilityBufferPool sizes;
    auto small = sizes.acquire(device, 4096), large = sizes.acquire(device, 8192);
    sizes.recycle_after_publication(large);
    sizes.recycle_after_publication(small);
    assert(sizes.acquire(device, 4096) == small);
    assert(sizes.acquire(device, 8192) == large);
    sizes.recycle_after_publication(small);
    assert(sizes.acquire(device, 16384).length >= 16384);
    sizes.recycle_after_publication(nil);
    // Entry cap and byte cap bound idle retention, not active query capacity.
    std::array<id<MTLBuffer>, 10> leased{};
    VisibilityBufferPool count;
    for (auto &buffer : leased) buffer = count.acquire(device, 4096);
    for (auto buffer : leased) count.recycle_after_publication(buffer);
    for (size_t i = 0; i < 8; ++i) assert(count.acquire(device, 4096) == leased[i]);
    auto excess = count.acquire(device, 4096);
    for (auto buffer : leased) assert(buffer != excess);
    VisibilityBufferPool bytes;
    auto half = bytes.acquire(device, 512 * 1024), full = bytes.acquire(device, 1024 * 1024);
    bytes.recycle_after_publication(half);
    bytes.recycle_after_publication(full);
    assert(bytes.acquire(device, 512 * 1024) == half);
    assert(bytes.acquire(device, 1024 * 1024) != full);
    auto oversized = bytes.acquire(device, 1024 * 1024 + 8);
    bytes.recycle_after_publication(oversized);
    assert(bytes.acquire(device, oversized.length) != oversized);
    std::cout << "PASS visibility pool: 10000 zeroed reuses, distinct active passes, duplicate returns, capacity growth and 8-buffer/1-MiB idle bounds\n";
}

static void check_submission_pools(id<MTLDevice> device) {
    const auto method=class_getInstanceMethod([MTL4CommitOptions class],@selector(addFeedbackHandler:));
    const auto original=method_setImplementation(method,reinterpret_cast<IMP>(capture_feedback));
    tables=[NSMutableArray new];
    auto queue=[[V3KMetal4Queue alloc] initWithDevice:device];
    auto native_queue=(TestNativeQueue *)queue.native;
    auto buffer=[device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    auto resource=[TestBindingResource new];
    const auto zero=[](TestTable *table) {
        for(auto address:table->addresses)assert(address==0);
        for(auto resource:table->textures)assert(resource._impl==0);
        for(auto resource:table->samplers)assert(resource._impl==0);
    };
    const auto render=[&](V3KMetal4Commands *commands) {
        auto native=[TestNativeEncoder new];
        auto encoder=[[V3KMetal4RenderEncoder alloc] initWithNative:(id<MTL4RenderCommandEncoder>)native commands:commands];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        zero(native.vertex_table);zero(native.fragment_table);
        [encoder setVertexBuffer:buffer offset:0 atIndex:30];
        [encoder setVertexTexture:(id<MTLTexture>)resource atIndex:127];
        [encoder setVertexSamplerState:(id<MTLSamplerState>)resource atIndex:15];
        [encoder setFragmentBuffer:buffer offset:0 atIndex:30];
        [encoder setFragmentTexture:(id<MTLTexture>)resource atIndex:127];
        [encoder setFragmentSamplerState:(id<MTLSamplerState>)resource atIndex:15];
        [encoder endEncoding];
        return native;
    };
    auto first=[queue commandBuffer];auto first_encoder=render(first);
    auto second=[queue commandBuffer];auto second_encoder=render(second);
    assert(first_encoder.vertex_table!=second_encoder.vertex_table && tables.count==4);
    __block bool handler_ran=false;
    [first addCompletedHandler:^(V3KMetal4Commands *) {
        assert(first_encoder.vertex_table->addresses[30]==buffer.gpuAddress);
        assert(first_encoder.fragment_table->textures[127]._impl && first_encoder.fragment_table->samplers[15]._impl);
        handler_ran=true;
    }];
    [first commit];[second commit];
    assert(first.status==MTLCommandBufferStatusCommitted && native_queue.pending.count==2);
    assert(((TestNativeCommands *)first.native).residency!=((TestNativeCommands *)second.native).residency);
    const auto cleared=first_encoder.vertex_table->clears+first_encoder.fragment_table->clears;
    [native_queue completeNext:NO];[first waitUntilCompleted];
    assert(handler_ran);zero(first_encoder.vertex_table);zero(first_encoder.fragment_table);
    assert(first_encoder.vertex_table->clears+first_encoder.fragment_table->clears==cleared+6);
    assert(((TestNativeCommands *)first.native).residency.allocations.count==0);
    assert(second_encoder.vertex_table->addresses[30]==buffer.gpuAddress);
    const auto before=residency_allocations;
    auto third=[queue commandBuffer];auto third_encoder=render(third);
    assert(tables.count==4 && residency_allocations==before);
    [third commit];
    [native_queue completeNext:NO];[second waitUntilCompleted];
    [native_queue completeNext:YES];[third waitUntilCompleted];
    assert(third.status==MTLCommandBufferStatusError);
    // The next two checkouts may reuse second's tables, never the failed pair.
    auto after_error=[queue commandBuffer];auto error_encoder=render(after_error);
    assert(error_encoder.vertex_table!=third_encoder.vertex_table && error_encoder.vertex_table!=third_encoder.fragment_table);
    assert(error_encoder.fragment_table!=third_encoder.vertex_table && error_encoder.fragment_table!=third_encoder.fragment_table);
    assert(((TestNativeCommands *)third.native).residency.allocations.count==0);
    [after_error commit];
    assert(((TestNativeCommands *)after_error.native).residency!=((TestNativeCommands *)third.native).residency);
    [native_queue completeNext:NO];[after_error waitUntilCompleted];
    // Reuse all 175 binding slots across repeated compute submissions.
    auto compute_queue=[[V3KMetal4Queue alloc] initWithDevice:device];
    auto compute_native=(TestNativeQueue *)compute_queue.native;
    const auto old_tables=tables.count,old_residency=residency_allocations;
    for(size_t round=0;round<1000;++round) @autoreleasepool {
        auto commands=[compute_queue commandBuffer];
        auto native=[TestNativeEncoder new];
        auto encoder=[[V3KMetal4ComputeEncoder alloc] initWithNative:(id<MTL4ComputeCommandEncoder>)native commands:commands];
        [encoder setBuffer:nil offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        zero(native.compute_table);
        for(NSUInteger i=0;i<31;++i)[encoder setBuffer:buffer offset:0 atIndex:i];
        for(NSUInteger i=0;i<128;++i)[encoder setTexture:(id<MTLTexture>)resource atIndex:i];
        for(NSUInteger i=0;i<16;++i)[encoder setSamplerState:(id<MTLSamplerState>)resource atIndex:i];
        [encoder endEncoding];const auto clears=native.compute_table->clears;
        [commands commit];[compute_native completeNext:NO];[commands waitUntilCompleted];
        zero(native.compute_table);assert(native.compute_table->clears==clears+175);
        assert(((TestNativeCommands *)commands.native).residency.allocations.count==0);
    }
    assert(tables.count==old_tables+1 && residency_allocations==old_residency+1);
    // Bounded idle pools: a burst of 80 tables retains at most 64; a burst
    // of ten independent residency sets retains at most eight.
    auto burst_queue=[[V3KMetal4Queue alloc] initWithDevice:device];
    auto burst_native=(TestNativeQueue *)burst_queue.native;
    auto burst=[burst_queue commandBuffer];
    for(size_t i=0;i<40;++i)render(burst);
    [burst commit];[burst_native completeNext:NO];[burst waitUntilCompleted];
    const auto burst_tables=tables.count;
    auto next_burst=[burst_queue commandBuffer];
    for(size_t i=0;i<40;++i)render(next_burst);
    assert(tables.count==burst_tables+16);
    auto residency_queue=[[V3KMetal4Queue alloc] initWithDevice:device];
    auto residency_native=(TestNativeQueue *)residency_queue.native;
    auto held=[NSMutableArray new];
    for(size_t i=0;i<10;++i){auto c=[residency_queue commandBuffer];[held addObject:c];[c commit];}
    for(size_t i=0;i<10;++i)[residency_native completeNext:NO];
    const auto residency_before=residency_allocations;
    for(size_t i=0;i<10;++i)[held addObject:[residency_queue commandBuffer]];
    assert(residency_allocations==residency_before+2);
    // The production cache's immutable membership survives repeated commands,
    // while clearing/replacing the cache cannot release a submitted GPU user.
    using namespace renderer::metal;
    std::string error;
    auto guest_device=Device::create(error);
    auto cached_queue=[[V3KMetal4Queue alloc] initWithDevice:device];
    auto cached_native=(TestNativeQueue *)cached_queue.native;
    DirectGuestBufferCache cached;
    auto slice=cached.get(*guest_device,4096,8192);
    auto lease=(TestResidency *)slice.residency;
    assert(lease.commits==1 && lease.allocations.count==1);
    for(size_t i=0;i<1000;++i) @autoreleasepool {
        auto c=[cached_queue commandBuffer];
        for(size_t draw=0;draw<8;++draw) [c retainResource:slice.buffer residencySet:slice.residency];
        auto native=(TestNativeCommands *)c.native;
        assert([native.residencyUses containsObject:lease]);
        [c commit];
        assert(!native.residency.allocations.count);
        [cached_native completeNext:NO];[c waitUntilCompleted];
        assert(lease.commits==1 && lease.allocations.count==1);
        auto sub=cached.get(*guest_device,4112,4144);
        assert(sub.residency==slice.residency && sub.buffer==slice.buffer && sub.offset==16);
    }
    lease=nil;slice={};cached.clear();
    for(bool failure:{false,true}) {
        __weak id weak_buffer=nil,weak_set=nil;
        auto c=[cached_queue commandBuffer];
        @autoreleasepool {
            auto active=cached.get(*guest_device,4096,8192);
            weak_buffer=active.buffer;weak_set=active.residency;
            [c retainResource:active.buffer residencySet:active.residency];
            [c commit];
            // A covering insertion drops only cache ownership of the old pair.
            cached.get(*guest_device,2048,16384);
            cached.clear();
        }
        assert(weak_buffer && weak_set);
        [cached_native completeNext:failure];[c waitUntilCompleted];
        assert(!weak_buffer && !weak_set);
    }
    auto texture_queue=[[V3KMetal4Queue alloc] initWithDevice:device];
    auto texture_native=(TestNativeQueue *)texture_queue.native;
    auto texture=[TestResidentTexture new];texture.allocatedSize=512*1024;
    auto view=[TestResidentTexture new];view.parentTexture=texture;
    for(size_t i=0;i<1000;++i) @autoreleasepool {
        auto c=[texture_queue commandBuffer];
        [c retainResource:(id<MTLResource>)texture];
        [c retainResource:(id<MTLResource>)view];
        [c commit];
        assert(((TestNativeCommands *)c.native).residency.allocations.count==0);
        assert(texture_native.textureResidency.commits==1);
        assert(texture_native.textureResidency.allocations.count==1);
        assert([texture_native.textureResidency.allocations containsObject:texture]);
        [texture_native completeNext:NO];[c waitUntilCompleted];
    }
    __weak id weak_view=view;view=nil;assert(!weak_view);
    // A cancelled unsubmitted encoder releases its pin too.
    @autoreleasepool {auto abandoned=[texture_queue commandBuffer];[abandoned retainResource:(id<MTLResource>)texture];}
    for(bool failure:{false,true}) {
        auto delayed=[texture_queue commandBuffer];
        [delayed retainResource:(id<MTLResource>)texture];[delayed commit];
        // Budget is 1 MiB in this device double. Repeated idle eviction must
        // retain the half-MiB root still used by delayed GPU work.
        for(size_t i=0;i<10;++i) @autoreleasepool {
            auto replacement=[TestResidentTexture new];replacement.allocatedSize=512*1024;
            auto c=[texture_queue commandBuffer];[c retainResource:(id<MTLResource>)replacement];
            // Cancel this command, so replacement becomes idle immediately.
        }
        assert([texture_native.textureResidency.allocations containsObject:texture]);
        [texture_native completeNext:failure];[delayed waitUntilCompleted];
        for(size_t i=0;i<3;++i) @autoreleasepool {
            auto replacement=[TestResidentTexture new];replacement.allocatedSize=512*1024;
            auto c=[texture_queue commandBuffer];[c retainResource:(id<MTLResource>)replacement];
        }
        assert(![texture_native.textureResidency.allocations containsObject:texture]);
    }
    method_setImplementation(method,original);
    tables=nil;
    std::cout << "PASS submission pools: actual feedback retirement, delayed distinct leases, handlers before reset, all 175 bindings cleared, 1000 reuses, GPU-error discard and 64-table/8-residency idle bounds\n";
    std::cout << "PASS cached guest residency: 1000 submissions reuse one committed set, subranges preserve lease, cache replacement/clear retain pending users through success/error feedback\n";
    std::cout << "PASS native texture residency: 1000 unchanged submissions commit membership once, views share roots, bounded idle eviction preserves active GPU users, cancellation and success/error retirement release pins\n";
}

// COLLISION_ARENA_HOOK

static void check_readonly_uploads(renderer::metal::Device &device) {
    using namespace renderer::metal;
    const auto bytes = [](const UploadBufferSlice &s) {
        return static_cast<const uint8_t *>(s.buffer.contents) + s.offset;
    };
    std::vector<uint8_t> input(8193, 0x17);
    UploadBufferArena arena(16384);
    const auto first = arena.copy_readonly(device, input);
    assert(first.copied_bytes == input.size() && first.slice.offset % 256 == 0);
    for (size_t i = 0; i < 10000; ++i) {
        const auto hit = arena.copy_readonly(device, input);
        assert(!hit.copied_bytes && hit.slice.buffer == first.slice.buffer && hit.slice.offset == first.slice.offset);
    }
    input[4096] ^= 1;
    const auto changed = arena.copy_readonly(device, input);
    assert(changed.copied_bytes == input.size());
    assert(bytes(first.slice)[4096] == 0x17 && bytes(changed.slice)[4096] == 0x16);
    input[4096] ^= 1;
    assert(!arena.copy_readonly(device, input).copied_bytes); // Write-restore is byte-identical input.
    assert(arena.copy_readonly(device, std::span(input).first(8192)).copied_bytes == 8192);
    // Immutable identities avoid the hash/compare/copy on repeat draws.
    auto owner = std::make_shared<const std::vector<uint8_t>>(input);
    std::weak_ptr<const std::vector<uint8_t>> weak = owner;
    const auto immutable = arena.copy_snapshot(device, owner);
    assert(immutable.copied_bytes == owner->size());
    for (size_t i = 0; i < 10000; ++i) {
        const auto hit = arena.copy_snapshot(device, owner);
        assert(!hit.copied_bytes && hit.slice.buffer == immutable.slice.buffer && hit.slice.offset == immutable.slice.offset);
    }
    // Simulate a deferred batch: move both ownership and lookup tables with
    // its arena. An independent next batch must not borrow pending slices.
    UploadBufferArena pending(std::move(arena));
    assert(!pending.copy_snapshot(device, owner).copied_bytes);
    assert(!pending.copy_readonly(device, input).copied_bytes);
    const auto fresh = arena.copy_snapshot(device, owner);
    assert(fresh.copied_bytes == input.size() && fresh.slice.buffer != immutable.slice.buffer);
    std::fill(input.begin(), input.end(), 0x63);
    assert(std::memcmp(bytes(immutable.slice), owner->data(), owner->size()) == 0);
    UploadBufferArena recycled;
    recycled = std::move(pending);
    assert(!recycled.copy_snapshot(device, owner).copied_bytes);
    assert(pending.copy_snapshot(device, owner).copied_bytes == owner->size());
    arena.reset_after_completion(); pending.reset_after_completion();
    owner.reset(); assert(!weak.expired());
    recycled.reset_after_completion(); assert(weak.expired());
    const auto overwritten = recycled.allocate(device, input.size());
    std::memset(static_cast<uint8_t *>(overwritten.buffer.contents) + overwritten.offset, 0x45, input.size());
    assert(recycled.copy_readonly(device, input).copied_bytes == input.size());
    // Metadata is bounded; evicted payloads still belong to their arena and
    // remain untouched for commands that are using them asynchronously.
    UploadBufferArena bounded(4096);
    std::vector<UploadBufferSlice> held;
    std::vector<std::weak_ptr<const std::vector<uint8_t>>> owners;
    for (unsigned i = 0; i < 65; ++i) {
        auto current = std::make_shared<const std::vector<uint8_t>>(256, uint8_t(i));
        owners.push_back(current); held.push_back(bounded.copy_snapshot(device, current).slice);
        assert(bounded.copy_readonly(device, *current).copied_bytes == current->size());
    }
    assert(owners.front().expired() && !owners.back().expired());
    for (size_t i = 0; i < held.size(); ++i)
        for (size_t b = 0; b < 256; ++b) assert(bytes(held[i])[b] == i);
    bounded.reset_after_completion(); for (const auto &w : owners) assert(w.expired());
    bool invalid = false;
    try { arena.copy_readonly(device, {}); } catch (const std::invalid_argument &) { invalid = true; }
    assert(invalid); invalid = false;
    try { arena.copy_snapshot(device, {}); } catch (const std::invalid_argument &) { invalid = true; }
    assert(invalid);
    // Same production arena/algorithm with a constant hash supplied by the
    // test: all equal-size inputs collide and must still compare every byte.
    CollisionUploadArena collisions(4096);
    input.assign(256, 0x11);
    auto old = collisions.copy_readonly(device, input);
    for (unsigned value = 0; value < 48; ++value) {
        input[128] = uint8_t(value);
        auto copy = collisions.copy_readonly(device, input);
        assert(copy.copied_bytes == (value == 0x11 ? 0 : input.size()));
        assert(bytes(copy.slice)[128] == value && bytes(old.slice)[128] == 0x11);
        assert(!collisions.copy_readonly(device, input).copied_bytes);
    }
    std::cout << "PASS readonly uploads: 10000 content hits + 10000 immutable-owner hits, changed live bytes/length/write-restore, forced hash collisions, deferred-batch moves/reset, bounded ownership and untouched pending slices; CPU doubles only\n";
}

// VERTEX_UPLOAD_HOOK

int main() {
    @autoreleasepool {
        using namespace renderer::metal;
        std::string error;
        auto device = Device::create(error);
        visibility_pool_test(device->native_device());
        check_commands(device->native_device());
        check_submission_pools(device->native_device());
        check_direct_cache(*device);
        check_texture_views();
        check_readonly_uploads(*device);
        check_vertex_uploads(*device);
        constexpr size_t page_size = 16384;
        auto *guest = static_cast<uint8_t *>(std::aligned_alloc(page_size, page_size * 2));
        assert(guest);
        std::memset(guest, 0x12, page_size * 2);
        std::array<GuestBufferRange, 2> ranges{{{guest + 4, 64}, {guest + page_size, 32}}};

        // Reproduce the old A -> finish -> B sequence using the real allocator.
        UploadBufferArena context;
        GuestBufferBindings old(*device, ranges, page_size, true, &context);
        const auto old_address = old.address(ranges[0]);
        assert(*reinterpret_cast<const uint8_t *>(old_address) == 0x12);
        context.reset_after_completion();
        auto overwrite = context.allocate(*device, page_size * 2, page_size);
        std::memset(overwrite.buffer.contents, 0x34, page_size * 2);
        assert(*reinterpret_cast<const uint8_t *>(old_address) == 0x34);

        // Private replay arena pins the contents even after the context resets.
        for (size_t length : {size_t(1), size_t(64), page_size, page_size * 2}) {
            std::array<GuestBufferRange, 1> input{{{guest, length}}};
            UploadBufferArena replay;
            GuestBufferBindings pinned(*device, input, page_size, true, &replay);
            const auto address = pinned.address(input[0]);
            context.reset_after_completion();
            auto next = context.allocate(*device, page_size * 2, page_size);
            std::memset(next.buffer.contents, 0x56, page_size * 2);
            const auto *bytes = reinterpret_cast<const uint8_t *>(address);
            for (size_t i = 0; i < length; ++i) assert(bytes[i] == 0x12);
        }
        std::free(guest);
        std::cout << "PASS replay lifetime: original overwrite reproduced; separate arena preserves all snapshot bytes\n";
    }
}
