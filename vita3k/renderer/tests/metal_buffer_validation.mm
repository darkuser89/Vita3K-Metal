// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/buffers.h>
#include <shader/uniform_block.h>
#include <gxm/functions.h>
#include "../../shader/tests/metal_shader_fixture.h"
#include <sys/mman.h>
#include <unistd.h>
#include <array>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

// Main cases use the unchanged capture. The optional BFCONTROL instruction
// probe inserts OR pa2, global16, 0 before the first store in a test-only copy.
static shader::metal::Program fixture(const char *path, bool backface_probe = false) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(bool(file), "Cannot open Killzone GXP fixture");
    const auto size = file.tellg();
    check(size >= sizeof(SceGxmProgram) && size < 1024 * 1024, "Invalid GXP length");
    std::vector<uint32_t> words((size_t(size) + 3) / 4);
    file.seekg(0);
    check(bool(file.read(reinterpret_cast<char *>(words.data()), size)), "Cannot read GXP");
    const auto &original = *reinterpret_cast<const SceGxmProgram *>(words.data());
    check(original.magic == 0x00505847 && original.size <= size, "Invalid GXP header");
    if (backface_probe) {
        std::vector<uint64_t> instructions(original.primary_program_instr_count);
        std::memcpy(instructions.data(), original.primary_program_start(), instructions.size()*8);
        auto store = std::find(instructions.begin(), instructions.end(), uint64_t(0xf0a20004a0000082));
        check(store != instructions.end(), "Cannot locate probe's first STR");
        instructions.insert(store, 0x5083000a60402800);
        const size_t offset = align(words.size()*4, 8);
        words.resize(offset/4 + instructions.size()*2);
        std::memcpy(reinterpret_cast<uint8_t *>(words.data()) + offset, instructions.data(), instructions.size()*8);
        auto &gxp = *reinterpret_cast<SceGxmProgram *>(words.data());
        gxp.primary_program_offset = offset - offsetof(SceGxmProgram, primary_program_offset);
        gxp.primary_program_instr_count = instructions.size();
        gxp.size = words.size()*4;
        return compile_gxp_fixture(words, gxp.size, "global16-instruction-probe", "gxp-mapped");
    }
    return compile_gxp_fixture(words, size, "killzone-original-globals", "gxp-mapped");
}

struct Pages {
    size_t size;
    uint8_t *data;
    explicit Pages(size_t length) : size(length), data(static_cast<uint8_t *>(mmap(nullptr, length,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0))) {
        check(data != MAP_FAILED, "Cannot allocate shared test pages");
    }
    ~Pages() { munmap(data, size); }
};

int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "Usage: metal-buffer-validation <Killzone 363ba9... fragment.gxp>\n"; return 2; }
    @autoreleasepool {
        try {
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            {
                shader::metal::Program vertex{.source=R"(#include <metal_stdlib>
using namespace metal;
struct In { float4 value [[attribute(0)]]; float2 p [[attribute(1)]]; };
struct Out { float4 p [[position]]; float4 value [[user(locn0)]]; };
vertex Out broadcast(In input [[stage_in]],uint instance [[instance_id]]) {
 Out o; o.p=float4(input.p,0,1);o.value=input.value+float4(0,float(instance)/16,0,0);return o;
})",.entry_point="broadcast",.stage=shader::metal::Stage::Vertex};
                shader::metal::Program fragment{.source=R"(#include <metal_stdlib>
using namespace metal;
struct In { float4 value [[user(locn0)]]; };
fragment float4 result(In input [[stage_in]]) { return input.value; }
)",.entry_point="result",.stage=shader::metal::Stage::Fragment};
                auto v=device->compile(vertex,false,error),f=device->compile(fragment,false,error);check(bool(v)&&bool(f),error);
                auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:16 height:16 mipmapped:NO];
                td.storageMode=MTLStorageModeShared;td.usage=MTLTextureUsageRenderTarget;
                auto target=[device->native_device() newTextureWithDescriptor:td];
                const uint16_t index_values[]={0,1,65534};
                auto indices=[device->native_device() newBufferWithBytes:index_values length:sizeof(index_values) options:MTLResourceStorageModeShared];
                std::vector<float> positions(65535*2,0);
                positions[0]=-1;positions[1]=-1;positions[2]=3;positions[3]=-1;
                positions[65534*2]=-1;positions[65534*2+1]=3;
                auto geometry=[device->native_device() newBufferWithBytes:positions.data() length:positions.size()*sizeof(float) options:MTLResourceStorageModeShared];
                unsigned cases=0;
                for(bool normalized:{false,true}) for(size_t offset:{size_t(0),size_t(8)}) for(bool instanced:{false,true}) {
                    const float values[]={0.25f,0.5f,0.75f,0.125f};const uint8_t bytes[]={31,63,127,191};
                    const size_t extent=offset+(normalized?sizeof(bytes):sizeof(values));
                    auto data=[device->native_device() newBufferWithLength:extent options:MTLResourceStorageModeShared];
                    std::memset(data.contents,0xa5,extent);
                    std::memcpy(static_cast<uint8_t *>(data.contents)+offset,normalized?static_cast<const void *>(bytes):values,normalized?sizeof(bytes):sizeof(values));
                    auto layout=[MTLVertexDescriptor vertexDescriptor];layout.attributes[0].format=normalized?MTLVertexFormatUChar4Normalized:MTLVertexFormatFloat4;
                    layout.attributes[0].offset=offset;layout.attributes[0].bufferIndex=4;
                    renderer::metal::configure_vertex_stream(layout.layouts[4],0,extent,instanced);
                    layout.attributes[1].format=MTLVertexFormatFloat2;layout.attributes[1].offset=0;layout.attributes[1].bufferIndex=5;
                    renderer::metal::configure_vertex_stream(layout.layouts[5],2*sizeof(float),2*sizeof(float),false);
                    auto pd=[MTLRenderPipelineDescriptor new];pd.vertexFunction=v->function;pd.fragmentFunction=f->function;pd.vertexDescriptor=layout;
                    pd.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA32Float;pd.colorAttachments[0].writeMask=MTLColorWriteMaskAll;pd.rasterizationEnabled=YES;
                    auto pipeline=device->create_pipeline(pd,error);check(pipeline!=nil,error);
                    auto command=[device->command_queue() commandBuffer];auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
                    pass.colorAttachments[0].texture=target;pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].clearColor=MTLClearColorMake(-1,-1,-1,-1);pass.colorAttachments[0].storeAction=MTLStoreActionStore;
                    auto encoder=[command renderCommandEncoderWithDescriptor:pass];[encoder setRenderPipelineState:pipeline];[encoder setVertexBuffer:data offset:0 atIndex:4];
                    [encoder setVertexBuffer:geometry offset:0 atIndex:5];
                    [encoder setCullMode:MTLCullModeNone];[encoder setTriangleFillMode:MTLTriangleFillModeFill];
                    [encoder setViewport:MTLViewport{0,0,16,16,0,1}];
                    [encoder setScissorRect:MTLScissorRect{0,0,16,16}];
                    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:3 indexType:MTLIndexTypeUInt16 indexBuffer:indices indexBufferOffset:0 instanceCount:2];[encoder endEncoding];
                    check(device->submit_and_wait(command,error),error);
                    std::array<float,16*16*4> actual;[target getBytes:actual.data() bytesPerRow:16*16 fromRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0];
                    for(size_t i=0;i<actual.size();++i) {
                        const auto c=i%4;const float expected=(normalized?float(bytes[c])/255:values[c])+(c==1?1.f/16:0);
                        if (!(std::abs(actual[i]-expected)<1e-6f)) {
                            std::cerr << "Constant stream: normalized=" << normalized << " offset=" << offset << " instanced=" << instanced
                                      << " channel=" << i << " actual=" << actual[i] << " expected=" << expected << '\n';
                            throw std::runtime_error("Constant vertex record mismatch");
                        }
                    }
                    ++cases;
                }
                std::cout<<"PASS "<<cases<<" constant vertex streams: float/UNORM8, offsets0/8, vertex/instance sources, index65534 and two instances (8192 channels)\n";
            }
            auto fragment = fixture(argv[1]);
            check(fragment.uses_buffer_addresses, "GXP did not use native buffer addresses");
            check(fragment.writes_guest_memory,"Unchanged Killzone STR shader was incorrectly classified read-only");
            auto fs = device->compile(fragment, false, error);
            check(bool(fs), error);
            shader::metal::Program vertex{
                .source = R"(#include <metal_stdlib>
using namespace metal;
struct Out {
 float4 position [[position]];
 float4 uv0 [[user(locn4)]]; float4 uv1 [[user(locn5)]];
 float4 uv2 [[user(locn6)]]; float4 uv3 [[user(locn7)]]; float4 uv4 [[user(locn8)]];
};
vertex Out fullscreen(uint i [[vertex_id]]) {
 const float2 p[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
 Out o; o.position = float4(p[i],0,1);
 o.uv0 = o.uv1 = o.uv2 = o.uv3 = o.uv4 = float4(0.5);
 return o;
})", .entry_point = "fullscreen", .stage = shader::metal::Stage::Vertex};
            auto vs = device->compile(vertex, false, error);
            check(bool(vs), error);
            {
                shader::metal::Program read_only{.source=R"(#include <metal_stdlib>
using namespace metal;
fragment float4 read_snapshot(constant ulong2 &addresses [[buffer(0)]],float4 p [[position]]) {
    device const uint *a=reinterpret_cast<device const uint *>(addresses.x);
    device const uint *b=reinterpret_cast<device const uint *>(addresses.y);
    return float4(as_type<float>(a[uint(p.x)]),as_type<float>(b[uint(p.x)]),0,0);
})",.entry_point="read_snapshot",.stage=shader::metal::Stage::Fragment,.writes_guest_memory=false};
                auto snapshot_fs=device->compile(read_only,false,error);check(bool(snapshot_fs),error);
                auto pd=[MTLRenderPipelineDescriptor new];pd.vertexFunction=vs->function;pd.fragmentFunction=snapshot_fs->function;
                pd.colorAttachments[0].pixelFormat=MTLPixelFormatRG32Float;
                auto snapshot_pipeline=device->create_pipeline(pd,error);check(snapshot_pipeline!=nil,error);
                auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float width:64 height:8 mipmapped:NO];
                td.storageMode=MTLStorageModeShared;td.usage=MTLTextureUsageRenderTarget;
                auto output=[device->native_device() newTextureWithDescriptor:td];check(output!=nil,"Cannot allocate snapshot probe target");
                const size_t page=sysconf(_SC_PAGESIZE);
                for (bool pooled : {false, true}) {
                    renderer::metal::UploadBufferArena arena(page * 4);
                    for (unsigned batch=0;batch<3;++batch) {
                        auto commands=[device->command_queue() commandBuffer];
                        auto pass=[MTLRenderPassDescriptor renderPassDescriptor];pass.colorAttachments[0].texture=output;
                        pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
                        auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
                        [encoder setRenderPipelineState:snapshot_pipeline];
                        for (unsigned draw=0;draw<8;++draw) {
                            Pages original(page*3);
                            auto *values=reinterpret_cast<uint32_t *>(original.data+page-128);
                            for(uint32_t i=0;i<65;++i) values[i]=0x3f000000u+(batch*8+draw)*128+i;
                            const renderer::metal::GuestBufferRange source[]={{reinterpret_cast<uint8_t *>(values),64*4},{reinterpret_cast<uint8_t *>(values+1),64*4}};
                            // Leave a prefix so snapshots exercise nonzero offsets,
                            // then exhaust small blocks while older draws are queued.
                            if(pooled) arena.allocate(*device,200);
                            renderer::metal::GuestBufferBindings snapshot(*device,source,page,true,pooled ? &arena : nullptr);
                            check(snapshot.address(source[1])==snapshot.address(source[0])+4,"Snapshot lost overlapping pointer identity");
                            [encoder setViewport:MTLViewport{0,double(draw),64,1,0,1}];
                            [encoder setScissorRect:MTLScissorRect{0,draw,64,1}];
                            const uint64_t addresses[]={snapshot.address(source[0]),snapshot.address(source[1])};
                            [encoder setFragmentBytes:addresses length:sizeof(addresses) atIndex:0];snapshot.make_resident(encoder);
                            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                            std::memset(original.data,0,original.size);
                            // Wrapper and host mapping die BEFORE GPU submission.
                        }
                        [encoder endEncoding];
                        check(device->submit_and_wait(commands,error),error);
                        arena.reset_after_completion();
                        std::array<uint32_t,128*8> values{};
                        [output getBytes:values.data() bytesPerRow:64*8 fromRegion:MTLRegionMake2D(0,0,64,8) mipmapLevel:0];
                        for(unsigned draw=0;draw<8;++draw) for(uint32_t i=0;i<64;++i) {
                            const auto expected=0x3f000000u+(batch*8+draw)*128+i;
                            check(values[draw*128+i*2]==expected && values[draw*128+i*2+1]==expected+1,"Queued snapshot changed after guest overwrite/free or arena rollover/reuse");
                        }
                    }
                }
                std::cout<<"PASS 48 queued snapshots: cross-page aliases, overwrite/unmap, nonzero offsets, block rollover and reuse after GPU completion; 6144 words verified\n";
                if (std::getenv("VITA3K_METAL_BENCH_UPLOADS")) {
                    Pages original(page*8);
                    std::memset(original.data,0x3f,original.size);
                    std::array<renderer::metal::GuestBufferRange,4> ranges;
                    for(size_t i=0;i<ranges.size();++i) ranges[i]={original.data+i*page*2+16,256};
                    uint64_t guard=0;
                    for(bool pooled : {false,true}) {
                        renderer::metal::UploadBufferArena arena;
                        std::vector<double> timings;
                        for(unsigned run=0;run<6;++run) {
                            auto start=std::chrono::steady_clock::now();
                            for(unsigned draw=0;draw<2048;++draw) {
                                renderer::metal::GuestBufferBindings bindings(*device,ranges,page,true,pooled ? &arena : nullptr);
                                guard^=bindings.address(ranges[0]);
                                // This allocation-only probe submits no GPU work.
                                if(draw%64==63) arena.reset_after_completion();
                            }
                            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                            if(run) timings.push_back(ms);
                        }
                        std::sort(timings.begin(),timings.end());
                        std::cout<<"BENCH uniform snapshots pooled="<<pooled<<" draws=2048 median_ms="<<timings[2]<<" guard="<<guard<<'\n';
                    }
                }
            }
            auto pipeline_desc = [MTLRenderPipelineDescriptor new];
            pipeline_desc.vertexFunction = vs->function;
            pipeline_desc.fragmentFunction = fs->function;
            pipeline_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            // This GXP writes only buffers; its color output is undefined.
            pipeline_desc.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
            auto pipeline = device->create_pipeline(pipeline_desc, error);
            check(pipeline != nil, error);
            auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:128 height:96 mipmapped:NO];
            desc.storageMode = MTLStorageModeShared;
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
            auto target = [device->native_device() newTextureWithDescriptor:desc];
            auto white = [device->native_device() newTextureWithDescriptor:desc];
            const std::vector<uint32_t> white_pixels(128 * 96, 0xffffffff);
            [white replaceRegion:MTLRegionMake2D(0,0,128,96) mipmapLevel:0 withBytes:white_pixels.data() bytesPerRow:128*4];
            auto sampler_desc = [MTLSamplerDescriptor new];
            auto sampler = [device->native_device() newSamplerStateWithDescriptor:sampler_desc];
            const size_t page_size = sysconf(_SC_PAGESIZE);
            Pages pages(page_size * 8);
            // Deliberately non-page-aligned bindings spanning multiple pages.
            std::array<renderer::metal::GuestBufferRange, 5> ranges;
            ranges[0] = {pages.data + 32, 16};
            for (size_t i = 1; i < ranges.size(); ++i)
                ranges[i] = {pages.data + i * page_size + 64, page_size + 64};
            renderer::metal::GuestBufferBindings bindings(*device, ranges, page_size);
            for (size_t i = 1; i < ranges.size(); ++i)
                check(bindings.address({ranges[i].data + 4, 4}) == bindings.address(ranges[i]) + 4,
                    "Overlapping bindings did not preserve native address offsets");
            for (float scale : {1.0f, 2.0f}) for (bool alias : {false, true}) {
                std::memset(pages.data, 0xa5, pages.size);
                const float params[4] = {64,0,0,0};
                std::memcpy(ranges[0].data, params, sizeof(params));
                auto draw_ranges = ranges;
                if (alias) draw_ranges[2] = draw_ranges[1];
                shader::RenderFragUniformBlockExtended info{};
                info.base_block.res_multiplier = scale;
                info.set_buffer_count(draw_ranges.size());
                for (size_t i = 0; i < draw_ranges.size(); ++i)
                    info.set_buffer_address(i, bindings.address(draw_ranges[i]));
                std::vector<uint8_t> info_bytes(align(info.get_size(), 16));
                info.copy_to(info_bytes.data());
                auto commands = [device->command_queue() commandBuffer];
                auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = target;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
                [encoder setRenderPipelineState:pipeline];
                [encoder setFragmentBytes:info_bytes.data() length:info_bytes.size() atIndex:shader::metal::RENDER_INFO_BUFFER];
                [encoder setFragmentTexture:white atIndex:0];
                [encoder setFragmentTexture:white atIndex:1];
                [encoder setFragmentTexture:white atIndex:shader::metal::MASK_TEXTURE];
                [encoder setFragmentSamplerState:sampler atIndex:0];
                [encoder setFragmentSamplerState:sampler atIndex:1];
                bindings.make_resident(encoder);
                std::vector<std::pair<size_t, size_t>> pixels;
                if (scale == 1) {
                    [encoder setScissorRect:MTLScissorRect{7,9,49,31}];
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                    for (size_t y = 9; y < 40; ++y) for (size_t x = 7; x < 56; ++x)
                        pixels.emplace_back(x, y);
                } else {
                    // One native invocation per selected guest pixel avoids
                    // duplicate writes from upscaling while testing coordinates.
                    for (const auto &[x,y] : std::array<std::pair<size_t,size_t>,4>{{{1,3},{31,33},{65,63},{127,95}}}) {
                        [encoder setScissorRect:MTLScissorRect{x,y,1,1}];
                        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        pixels.emplace_back(x/2,y/2);
                    }
                }
                [encoder endEncoding];
                check(device->submit_and_wait(commands, error), error);
                // Unit texture samples produce edge=0 and base bytes=1. The
                // USSE IMA8 steps add literal bytes 8,16,24 respectively.
                std::vector<uint8_t> expected(pages.size, 0xa5);
                std::memcpy(expected.data() + 32, params, sizeof(params));
                for (size_t i = 1; i < draw_ranges.size(); ++i)
                    for (const auto &[x,y] : pixels)
                        std::memset(expected.data() + (draw_ranges[i].data - pages.data) + (y*64+x)*4, 1 + (i - 1) * 8, 4);
                for (size_t i = 0; i < pages.size; ++i)
                    check(pages.data[i] == expected[i], "Store or sentinel mismatch at byte " + std::to_string(i)
                        + ": got " + std::to_string(pages.data[i]) + ", expected " + std::to_string(expected[i]));
                std::cout << "PASS unchanged GXP STR pixel addresses across region boundaries, scale=" << scale << ", "
                          << (alias ? "aliased" : "distinct") << " buffers and every surrounding sentinel byte\n";
            }

            auto probe = device->compile(fixture(argv[1], true), false, error);
            check(bool(probe), error);
            pipeline_desc.fragmentFunction = probe->function;
            auto probe_pipeline = device->create_pipeline(pipeline_desc, error);
            check(probe_pipeline != nil, error);
            shader::metal::Program reference{
                .source = R"(#include <metal_stdlib>
using namespace metal;
fragment float4 facing(bool front [[front_facing]]) { return float4(front ? 0.0 : 1.0,0,0,1); })",
                .entry_point = "facing", .stage = shader::metal::Stage::Fragment};
            auto reference_fs = device->compile(reference, false, error);
            check(bool(reference_fs), error);
            pipeline_desc.fragmentFunction = reference_fs->function;
            pipeline_desc.colorAttachments[0].writeMask = MTLColorWriteMaskAll;
            auto reference_pipeline = device->create_pipeline(pipeline_desc, error);
            check(reference_pipeline != nil, error);
            int previous_back = -1;
            for (auto winding : {MTLWindingClockwise, MTLWindingCounterClockwise}) {
                std::memset(pages.data, 0xa5, pages.size);
                const float params[4] = {64,0,0,0};
                std::memcpy(ranges[0].data, params, sizeof(params));
                shader::RenderFragUniformBlockExtended info{};
                info.base_block.res_multiplier = 1;
                info.set_buffer_count(ranges.size());
                for (size_t i = 0; i < ranges.size(); ++i) info.set_buffer_address(i, bindings.address(ranges[i]));
                std::vector<uint8_t> info_bytes(align(info.get_size(), 16));
                info.copy_to(info_bytes.data());
                auto commands = [device->command_queue() commandBuffer];
                auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = target;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
                [encoder setFrontFacingWinding:winding];
                [encoder setScissorRect:MTLScissorRect{7,9,1,1}];
                [encoder setRenderPipelineState:reference_pipeline];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                [encoder setRenderPipelineState:probe_pipeline];
                [encoder setFragmentBytes:info_bytes.data() length:info_bytes.size() atIndex:shader::metal::RENDER_INFO_BUFFER];
                [encoder setFragmentTexture:white atIndex:0];
                [encoder setFragmentTexture:white atIndex:1];
                [encoder setFragmentTexture:white atIndex:shader::metal::MASK_TEXTURE];
                [encoder setFragmentSamplerState:sampler atIndex:0];
                [encoder setFragmentSamplerState:sampler atIndex:1];
                bindings.make_resident(encoder);
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                [encoder endEncoding];
                check(device->submit_and_wait(commands, error), error);
                uint8_t reference_pixel[4];
                [target getBytes:reference_pixel bytesPerRow:4 fromRegion:MTLRegionMake2D(7,9,1,1) mipmapLevel:0];
                const uint32_t back = reference_pixel[0] == 255 ? 1 : 0;
                check(int(back) != previous_back, "Reference winding did not flip the facing result");
                previous_back = back;
                const uint32_t expected_words[] = {back, 0x08080808 + back, 0x11111111, 0x19191919};
                std::vector<uint8_t> expected(pages.size, 0xa5);
                std::memcpy(expected.data() + 32, params, sizeof(params));
                for (size_t i = 1; i < ranges.size(); ++i)
                    std::memcpy(expected.data() + (ranges[i].data - pages.data) + (9*64+7)*4, &expected_words[i-1], 4);
                check(std::memcmp(expected.data(), pages.data, pages.size) == 0,
                    "BFCONTROL instruction probe disagrees with native front_facing or changes a sentinel");
                std::cout << "PASS SGX BFCONTROL instruction probe versus native front_facing, back=" << back << '\n';
            }
            return 0;
        } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
    }
}
