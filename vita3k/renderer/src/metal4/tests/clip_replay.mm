// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <renderer/metal4/polygon_clip.h>
#include <shader/metal4/msl_recompiler.h>
#include <cassert>
#include <cstring>
#include <iostream>
using namespace renderer::metal;
@interface ClipBuffer : NSObject { @public std::vector<uint8_t> storage; }
@property(readonly) void *contents;
@end
@implementation ClipBuffer
- (void *)contents { return storage.data(); }
@end
struct Draw {
    MTLPrimitiveType type;
    size_t count;
    bool visible_front, indirect;
    float face, bias, slope;
    uint32_t stencil;
    id depth;
    std::vector<uint8_t> bytes;
};
@interface ClipEncoder : NSObject {
@public
    std::vector<Draw> draws;
    Draw current;
}
- (void)setDepthBias:(float)bias slopeScale:(float)slope clamp:(float)clamp;
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)setDepthStencilState:(id)state;
- (void)setStencilFrontReferenceValue:(uint32_t)front backReferenceValue:(uint32_t)back;
- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)setVertexBuffer:(ClipBuffer *)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)drawPrimitives:(MTLPrimitiveType)type vertexStart:(NSUInteger)start vertexCount:(NSUInteger)count;
@end
@implementation ClipEncoder
- (void)setDepthBias:(float)b slopeScale:(float)s clamp:(float)c {assert(c==0);current.bias=b;current.slope=s;}
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    assert(length==4 && index==shader::metal::RENDER_INFO_BUFFER);std::memcpy(&current.face,bytes,4);
}
- (void)setDepthStencilState:(id)state {current.depth=state;}
- (void)setStencilFrontReferenceValue:(uint32_t)front backReferenceValue:(uint32_t)back {assert(front==back);current.stencil=front;}
- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    assert(index==shader::metal::VERTEX_STREAM_BUFFER_BASE && length<=4096);
    current.bytes.assign((const uint8_t *)bytes,(const uint8_t *)bytes+length);current.indirect=false;
}
- (void)setVertexBuffer:(ClipBuffer *)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    assert(index==shader::metal::VERTEX_STREAM_BUFFER_BASE && offset==128);
    current.bytes.assign(buffer->storage.begin()+offset,buffer->storage.end());current.indirect=true;
}
- (void)drawPrimitives:(MTLPrimitiveType)type vertexStart:(NSUInteger)start vertexCount:(NSUInteger)count {
    assert(start==0 && current.bytes.size()==count*sizeof(CapturedVertexOutputs));
    current.type=type;current.count=count;draws.push_back(current);
}
@end
struct Device {};
struct Slice { ClipBuffer *buffer; size_t offset; };
struct Arena {
    std::vector<ClipBuffer *> owners;
    Slice allocate(Device &,size_t size) {auto buffer=[ClipBuffer new];buffer->storage.resize(size+128);owners.push_back(buffer);return {buffer,128};}
};
static void replay(ClipEncoder *encoder,const std::vector<RoutedClipPrimitive> &routed_clip_primitives,bool two_sided) {
    struct Stencil {uint8_t ref;};
    struct Record {float depth_bias_unit;Stencil front_stencil_state_values,back_stencil_state_values;} record{7,{31},{47}};
    struct NativeRecord {float back_depth_bias_unit;} native_record{11};
    struct Context {NativeRecord native_record;} ctx{native_record};
    struct FragmentInfo {
        struct {float point_replay_face;} base_block{};
        void copy_to(void *out) {std::memcpy(out,&base_block.point_replay_face,4);}
    } fragment_info;
    std::array<uint8_t,4> fragment_bytes{};
    const float front_depth_slope=13,back_depth_slope=17;
    id point_front_depth_state=@1,point_back_depth_state=@2;
    Device device;
    struct Impl {Device *device;} state{&device};auto *impl=&state;
    Arena replay_uploads;
    const auto select_visibility=[&](bool front) {encoder->current.visible_front=front;};
    // REPLAY_HOOK
}
int main() {
    @autoreleasepool {
        size_t cases=0;
        for(bool two_sided:{false,true}) {
            std::vector<RoutedClipPrimitive> input;
            for(auto kind:{ClipPrimitiveKind::Points,ClipPrimitiveKind::Lines,ClipPrimitiveKind::Triangles})
            for(auto face:{PolygonFace::Front,PolygonFace::Back})
            for(size_t count:{size_t(1),size_t(2),size_t(3),size_t(6),size_t(9),size_t(12),size_t(42)}) {
                if(kind==ClipPrimitiveKind::Triangles && count%3)continue;
                if(kind==ClipPrimitiveKind::Lines && count%2)continue;
                RoutedClipPrimitive p{face,kind,std::vector<CapturedVertexOutputs>(count)};
                for(size_t v=0;v<count;++v) for(size_t slot=0;slot<26;++slot)
                    p.vertices[v][slot]={float(input.size()),float(v),float(slot),float(v*slot)};
                input.push_back(std::move(p));
            }
            auto encoder=[ClipEncoder new];
            replay(encoder,{},two_sided);assert(encoder->draws.empty());
            replay(encoder,input,two_sided);assert(encoder->draws.size()==input.size());
            for(size_t i=0;i<input.size();++i) {
                const auto &p=input[i];const auto &d=encoder->draws[i];const bool front=p.face==PolygonFace::Front;
                assert(d.type==(p.kind==ClipPrimitiveKind::Points?MTLPrimitiveTypePoint:p.kind==ClipPrimitiveKind::Lines?MTLPrimitiveTypeLine:MTLPrimitiveTypeTriangle));
                assert(d.count==p.vertices.size() && d.visible_front==front && d.face==(front?1:-1));
                assert(d.bias==((front || !two_sided)?7:11) && d.slope==((front || !two_sided)?13:17));
                assert(d.stencil==((front || !two_sided)?31:47) && [d.depth intValue]==(front?1:2));
                assert(d.indirect==(p.vertices.size()*sizeof(CapturedVertexOutputs)>4096));
                assert(std::memcmp(d.bytes.data(),p.vertices.data(),d.bytes.size())==0);++cases;
            }
        }
        std::cout << "PASS clip replay encoding: " << cases << " actual production-loop cases, primitive/face order, visibility selection, face uniform, depth/stencil/bias and exact inline/arena bytes; CPU doubles, ASAN/UBSAN\n";
    }
}
