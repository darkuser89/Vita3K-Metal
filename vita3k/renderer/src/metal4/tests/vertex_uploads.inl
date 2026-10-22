// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Included in the CPU-only buffer test after injecting old/current draw loops.
#include <sstream>
@interface VertexUploadEncoder : NSObject {
@public
    std::map<NSUInteger, renderer::metal::UploadBufferSlice> bindings;
}
- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
@end
@implementation VertexUploadEncoder
- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    bindings[index] = {buffer, offset};
}
@end
struct VertexUploadFixture {
    struct Pointer {
        const uint8_t *bytes;
        explicit operator bool() const { return bytes != nullptr; }
        const uint8_t *get(int &) const { return bytes; }
    };
    struct Data { Pointer data; size_t size; };
    struct Program { struct Stream { size_t stride; }; std::vector<Stream> streams; } program;
    struct Record { std::vector<Data> vertex_streams; } record;
    struct Layout { size_t stride; MTLVertexStepFunction step_function; };
    struct Attribute { MTLVertexFormat format; uint32_t buffer_index; size_t offset, source_offset, byte_size; };
    std::vector<Layout> layouts;
    std::vector<Attribute> attributes;
    std::vector<bool> repack;
    std::vector<renderer::metal::OwnedVertexSnapshot> owners;
    std::vector<uint64_t> handles;
    std::vector<uint16_t> indices{0, 2, 3, 1};
    uint32_t instances = 3;
};
struct VertexUploadContext {
    struct Impl { renderer::metal::UploadBufferArena uploads; } storage;
    Impl *impl = &storage;
    std::vector<uint64_t> vertex_stream_snapshots;
};
struct VertexUploadResult { std::vector<std::vector<uint8_t>> captured; size_t copied; };
namespace vertex_test_fmt {
template <typename... T> std::string format(const char *, T...) { return {}; }
}
// COPY_VERTEX_HOOK
// VERTEX_LOOPS_HOOK

static void check_vertex_uploads(renderer::metal::Device &device) {
    using namespace renderer::metal;
    size_t cases = 0;
    for (bool repack : {false, true})
        for (bool snapshot : {false, true})
            for (size_t stride : {0, 1, 3, 4, 5, 8, 16})
                for (size_t attribute_bytes : {1, 2, 3, 4, 5, 6, 8, 12, 16})
                    for (auto step : {MTLVertexStepFunctionConstant, MTLVertexStepFunctionPerVertex, MTLVertexStepFunctionPerInstance}) {
                        VertexUploadFixture f;
                        std::vector<uint8_t> guest(257);
                        for (size_t i = 0; i < guest.size(); ++i) guest[i] = uint8_t(i * 73 + 19);
                        f.program.streams.push_back({stride}); f.record.vertex_streams.push_back({{guest.data()}, guest.size()});
                        f.layouts.push_back({(attribute_bytes + 7) & ~size_t(3), step});
                        f.repack.push_back(repack); f.handles.push_back(0);
                        f.attributes.push_back({MTLVertexFormatFloat, shader::metal::VERTEX_STREAM_BUFFER_BASE, 4, 1, attribute_bytes});
                        f.owners.push_back(snapshot ? std::make_shared<const std::vector<uint8_t>>(guest) : nullptr);
                        VertexUploadContext before, after;
                        before.vertex_stream_snapshots = after.vertex_stream_snapshots = f.handles;
                        for (unsigned draw = 0; draw < 4; ++draw) {
                            // Mutate guest RAM while a kick snapshot may still
                            // own the old bytes, then expire the handle so the
                            // draw must fall back to current RAM.
                            if (draw == 2) guest[10] ^= 0x81;
                            if (draw == 3) f.owners[0].reset();
                            const auto expected = original_vertex_uploads(device, f, before);
                            const auto actual = current_vertex_uploads(device, f, after);
                            assert(actual.captured == expected.captured);
                            const bool direct = !repack && (stride == 0 || stride % 4 == 0);
                            if (draw == 1 && direct) assert(actual.copied == 0);
                            else if (!direct) assert(actual.copied == expected.copied);
                            assert(actual.copied <= expected.copied);
                            ++cases;
                        }
                    }
    std::cout << "PASS vertex upload integration: " << cases
        << " old/new production-loop comparisons; raw/aligned/padded/repacked attributes, constant/vertex/instance streams, capture bytes, held/expired kick snapshots and changed live guest bytes\n";
}

// Pre-A011 production loop from 69554268, retained solely as the regression
// oracle. Tests also work in a shallow clone without historical Git objects.
static constexpr auto original_vertex_upload_loop = R"baseline(        size_t vertex_elements = 0;
        for (uint32_t stream = 0; stream < vp->streams.size(); ++stream) if (streams_used & (1u << stream)) {
            const auto &data = record.vertex_streams[stream];
            require(data.data && data.size, "Metal: empty vertex stream");
            const auto stride = vp->streams[stream].stride;
            const auto aligned = align(stride, 4);
            const auto snapshot = vertex_stream_snapshot(ctx.vertex_stream_snapshots[stream], data.size);
            const uint8_t *bytes = snapshot.empty() ? data.data.get(mem) : snapshot.data();
            size_t elements = stride && stride != aligned ? (data.size + stride - 1) / stride : 0;
            const auto &layout = vertex_stream_layouts[stream];
            if (repack_stream[stream]) {
                if (layout.step_function == MTLVertexStepFunctionConstant) {
                    elements = 1;
                } else if (layout.step_function == MTLVertexStepFunctionPerInstance) {
                    elements = instances;
                } else {
                    if (!vertex_elements) {
                        for (uint32_t i = 0; i < count; ++i) {
                            const uint32_t index = index_format == SCE_GXM_INDEX_FORMAT_U16
                                ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i];
                            vertex_elements = std::max(vertex_elements, size_t(index) + 1);
                        }
                    }
                    elements = vertex_elements;
                }
                require(elements && layout.stride && elements <= std::numeric_limits<size_t>::max() / layout.stride,
                    "Metal: invalid repacked vertex stream extent");
                const uint64_t last_record = uint64_t(elements - 1) * stride;
                for (const auto &attribute : vertex_attributes) {
                    if (attribute.format != MTLVertexFormatInvalid
                        && attribute.buffer_index == shader::metal::VERTEX_STREAM_BUFFER_BASE + stream)
                        require(last_record + attribute.source_offset + attribute.byte_size <= data.size,
                            "Metal: vertex stream does not cover the shader attribute");
                }
            }
            const size_t length = repack_stream[stream] ? elements * layout.stride
                : elements ? elements * aligned : data.size;
            const auto buffer = ctx.impl->uploads.allocate(*impl->device, length);
            auto *upload = static_cast<uint8_t *>(buffer.buffer.contents) + buffer.offset;
            if (repack_stream[stream]) {
                std::memset(upload, 0, length);
                for (const auto &attribute : vertex_attributes) {
                    if (attribute.format == MTLVertexFormatInvalid
                        || attribute.buffer_index != shader::metal::VERTEX_STREAM_BUFFER_BASE + stream)
                        continue;
                    auto *destination = upload + attribute.offset;
                    const auto *source = bytes + attribute.source_offset;
                    // These are the common Vita attribute widths. Dispatch
                    // once per attribute so the per-vertex copies have a
                    // constant size and can be inlined by the compiler.
                    switch (attribute.byte_size) {
                    case 1: copy_vertex_attribute_records<1>(destination, layout.stride, source, stride, elements); break;
                    case 2: copy_vertex_attribute_records<2>(destination, layout.stride, source, stride, elements); break;
                    case 3: copy_vertex_attribute_records<3>(destination, layout.stride, source, stride, elements); break;
                    case 4: copy_vertex_attribute_records<4>(destination, layout.stride, source, stride, elements); break;
                    case 6: copy_vertex_attribute_records<6>(destination, layout.stride, source, stride, elements); break;
                    case 8: copy_vertex_attribute_records<8>(destination, layout.stride, source, stride, elements); break;
                    case 12: copy_vertex_attribute_records<12>(destination, layout.stride, source, stride, elements); break;
                    case 16: copy_vertex_attribute_records<16>(destination, layout.stride, source, stride, elements); break;
                    default:
                        for (size_t i = 0; i < elements; ++i)
                            std::memcpy(destination + i * layout.stride,
                                source + i * stride, attribute.byte_size);
                        break;
                    }
                }
            } else if (elements) {
                // The old temporary vector zero-filled padding before copying
                // to the upload arena. Preserve those bytes while packing
                // directly into the arena's retained GPU-visible slice.
                std::memset(upload, 0, length);
                for (size_t i = 0; i < elements; ++i)
                    std::memcpy(upload + i * aligned, bytes + i * stride,
                        std::min<size_t>(stride, data.size - i * stride));
            } else std::memcpy(upload, bytes, length);
            draw_upload_bytes += length;
            [encoder setVertexBuffer:buffer.buffer offset:buffer.offset atIndex:shader::metal::VERTEX_STREAM_BUFFER_BASE + stream];
            if (capture_draw) {
                dump_bytes(fmt::format("stream-{}.bin",shader::metal::VERTEX_STREAM_BUFFER_BASE+stream),upload,length);
                draw_metadata << "stream " << shader::metal::VERTEX_STREAM_BUFFER_BASE+stream << ' ' << length << '\n';
            }
        }
)baseline";
