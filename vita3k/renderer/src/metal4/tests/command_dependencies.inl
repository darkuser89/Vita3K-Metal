// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Loaded into the CPU wrapper test; never creates a Metal device.
@interface DependencyBuffer : TestBuffer
@property uint64_t address;
@property(strong) id<MTLHeap> heap;
@property(strong) id<MTLBuffer> remoteStorageBuffer;
@end
@implementation DependencyBuffer
- (uint64_t)gpuAddress { return _address; }
@end
@interface DependencyTexture : NSObject
@property(strong) id<MTLTexture> parentTexture, remoteStorageTexture;
@property(strong) id<MTLBuffer> buffer;
@property(strong) id<MTLHeap> heap;
@property IOSurfaceRef iosurface;
@property(getter=isShareable) BOOL shareable;
@end
@implementation DependencyTexture
@end
@interface DependencyEncoder : TestNativeEncoder
@property NSUInteger copies;
@end
@implementation DependencyEncoder
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin { ++self.copies; }
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset sourceBytesPerRow:(NSUInteger)sourceBytesPerRow sourceBytesPerImage:(NSUInteger)sourceBytesPerImage sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin { ++self.copies; }
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset sourceBytesPerRow:(NSUInteger)sourceBytesPerRow sourceBytesPerImage:(NSUInteger)sourceBytesPerImage sourceSize:(MTLSize)sourceSize toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel destinationOrigin:(MTLOrigin)destinationOrigin options:(MTLBlitOption)options { ++self.copies; }
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toBuffer:(id<MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset destinationBytesPerRow:(NSUInteger)destinationBytesPerRow destinationBytesPerImage:(NSUInteger)destinationBytesPerImage { ++self.copies; }
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel sourceOrigin:(MTLOrigin)sourceOrigin sourceSize:(MTLSize)sourceSize toBuffer:(id<MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset destinationBytesPerRow:(NSUInteger)destinationBytesPerRow destinationBytesPerImage:(NSUInteger)destinationBytesPerImage options:(MTLBlitOption)options { ++self.copies; }
- (void)generateMipmapsForTexture:(id<MTLTexture>)texture { ++self.copies; }
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value { ++self.copies; }
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture sourceSlice:(NSUInteger)sourceSlice sourceLevel:(NSUInteger)sourceLevel toTexture:(id<MTLTexture>)destinationTexture destinationSlice:(NSUInteger)destinationSlice destinationLevel:(NSUInteger)destinationLevel sliceCount:(NSUInteger)sliceCount levelCount:(NSUInteger)levelCount { ++self.copies; }
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture toTexture:(id<MTLTexture>)destinationTexture { ++self.copies; }
- (void)copyFromBuffer:(id <MTLBuffer>)sourceBuffer sourceOffset:(NSUInteger)sourceOffset toBuffer:(id <MTLBuffer>)destinationBuffer destinationOffset:(NSUInteger)destinationOffset size:(NSUInteger)size { ++self.copies; }
@end
static void check_blit_dependencies(V3KMetal4Commands *commands) {
    auto native = [DependencyEncoder new];
    V3KMetal4ComputeEncoder *encoder;
    const auto reset = [&] {
        native.blit_barriers = native.dispatch_barriers = native.copies = 0;
        encoder = [[V3KMetal4ComputeEncoder alloc] initWithNative:(id<MTL4ComputeCommandEncoder>)native commands:commands];
    };
    const auto buffer = [](uint64_t gpu, void *cpu) {
        auto result = [[DependencyBuffer alloc] initWithBytes:cpu length:4096]; result.address = gpu; return result;
    };
    std::array<uint8_t, 8192> storage{};
    auto a = buffer(0x10000, storage.data()), b = buffer(0x20000, storage.data()+4096);
    auto alias = buffer(0x30000, storage.data()+32); // different GPU address, overlapping host pages
    auto ta = [DependencyTexture new], tb = [DependencyTexture new], tc = [DependencyTexture new];
    auto view = [DependencyTexture new]; view.parentTexture = (id<MTLTexture>)ta;
    const auto fill = [&](DependencyBuffer *target, NSUInteger start, NSUInteger size) {
        [encoder fillBuffer:(id<MTLBuffer>)target range:NSMakeRange(start,size) value:1];
    };
    const auto copy = [&](DependencyTexture *source, DependencyTexture *destination) {
        [encoder copyFromTexture:(id<MTLTexture>)source toTexture:(id<MTLTexture>)destination];
    };
    reset(); copy(ta,tb); copy(ta,tc); assert(native.blit_barriers==0); // shared read
    copy(tc,ta); assert(native.blit_barriers==1); // RAW and WAR against older commands
    copy(view,tb); assert(native.blit_barriers==2); // parent view
    reset(); fill(a,0,64); fill(b,0,64); assert(native.blit_barriers==0);
    fill(alias,0,64); assert(native.blit_barriers==1); // no-copy alias across distinct GPU ranges
    alias.address = a.address; // GPU alias, distinct CPU range
    auto gpu_alias = buffer(a.address,storage.data()+4096);
    reset(); fill(a,0,64); fill(gpu_alias,0,64); assert(native.blit_barriers==1);
    reset(); fill(a,0,32); fill(a,32,32); assert(native.blit_barriers==0);
    fill(a,16,32); assert(native.blit_barriers==1);
    auto heap=(id<MTLHeap>)[NSObject new];
    a.heap=heap; tb.heap=heap;
    reset(); fill(a,0,16); copy(ta,tb); assert(native.blit_barriers==1);
    a.heap=nil; tb.heap=nil;
    ta.buffer=(id<MTLBuffer>)a;
    reset(); fill(alias,0,64); copy(view,tb); assert(native.blit_barriers==1);
    ta.buffer=nil;
    for(unsigned unknown=0;unknown<3;++unknown) {
        ta.shareable=unknown==0;
        ta.iosurface=unknown==1 ? reinterpret_cast<IOSurfaceRef>(uintptr_t(1)) : nullptr;
        ta.remoteStorageTexture=unknown==2 ? (id<MTLTexture>)tc : nil;
        reset(); fill(a,0,32); copy(ta,tb); assert(native.blit_barriers==1);
        fill(b,64,32); assert(native.blit_barriers==2);
    }
    ta.shareable=NO; ta.iosurface=nullptr; ta.remoteStorageTexture=nil;
    a.remoteStorageBuffer=(id<MTLBuffer>)b;
    reset(); fill(a,0,16); copy(ta,tb); assert(native.blit_barriers==1);
    a.remoteStorageBuffer=nil;
    // Invalid/overflow metadata must keep the opaque fallback, never silently
    // turn into a supposedly independent interval. The boundary double does not
    // perform these invalid copies/fills; no real driver sees them.
    for(unsigned invalid=0;invalid<5;++invalid) {
        a.address=invalid==0 ? UINT64_MAX-4 : invalid==1 ? 0 : 0x10000;
        reset(); copy(ta,tb);
        fill(a,invalid==2 ? 4097 : invalid==3 ? 4090 : 0,invalid==4 ? 0 : 16);
        assert(native.blit_barriers==1);
    }
    a.address=0x10000;
    // Exercise every real wrapper overload with a prior write to its source or
    // destination. This catches omitted metadata and swapped read/write roles.
    for(unsigned variant=0;variant<10;++variant) {
        reset();
        [encoder generateMipmapsForTexture:(id<MTLTexture>)ta];
        fill(a,0,4096);
        assert(native.blit_barriers==0);
        const auto origin=MTLOriginMake(0,0,0); const auto size=MTLSizeMake(1,1,1);
        switch(variant) {
        case 0: [encoder copyFromTexture:(id<MTLTexture>)ta sourceSlice:0 sourceLevel:0 sourceOrigin:origin sourceSize:size toTexture:(id<MTLTexture>)tb destinationSlice:0 destinationLevel:0 destinationOrigin:origin]; break;
        case 1: [encoder copyFromBuffer:(id<MTLBuffer>)a sourceOffset:0 sourceBytesPerRow:4 sourceBytesPerImage:4 sourceSize:size toTexture:(id<MTLTexture>)tb destinationSlice:0 destinationLevel:0 destinationOrigin:origin]; break;
        case 2: [encoder copyFromBuffer:(id<MTLBuffer>)a sourceOffset:0 sourceBytesPerRow:4 sourceBytesPerImage:4 sourceSize:size toTexture:(id<MTLTexture>)tb destinationSlice:0 destinationLevel:0 destinationOrigin:origin options:MTLBlitOptionNone]; break;
        case 3: [encoder copyFromTexture:(id<MTLTexture>)ta sourceSlice:0 sourceLevel:0 sourceOrigin:origin sourceSize:size toBuffer:(id<MTLBuffer>)b destinationOffset:0 destinationBytesPerRow:4 destinationBytesPerImage:4]; break;
        case 4: [encoder copyFromTexture:(id<MTLTexture>)ta sourceSlice:0 sourceLevel:0 sourceOrigin:origin sourceSize:size toBuffer:(id<MTLBuffer>)b destinationOffset:0 destinationBytesPerRow:4 destinationBytesPerImage:4 options:MTLBlitOptionNone]; break;
        case 5: [encoder generateMipmapsForTexture:(id<MTLTexture>)ta]; break;
        case 6: fill(a,0,16); break;
        case 7: [encoder copyFromTexture:(id<MTLTexture>)ta sourceSlice:0 sourceLevel:0 toTexture:(id<MTLTexture>)tb destinationSlice:0 destinationLevel:0 sliceCount:1 levelCount:1]; break;
        case 8: copy(ta,tb); break;
        case 9: [encoder copyFromBuffer:(id<MTLBuffer>)a sourceOffset:0 toBuffer:(id<MTLBuffer>)b destinationOffset:0 size:16]; break;
        }
        assert(native.blit_barriers==1 && native.copies==3);
    }
    std::cout << "PASS Metal 4 blit integration: all 10 overloads, texture/view/buffer/heap/host/GPU aliases, shared/remote/overflow fallback; actual wrapper with CPU Metal boundary doubles\n";
}
