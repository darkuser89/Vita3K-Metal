// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// The runner inserts the unchanged production import/parser/decoder functions.
// Files and CPU decoding are real; the final GPU upload is a recording boundary.
// DDS layout: https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <gxm/types.h>
#include <util/align.h>
#include <ddspp.h>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#define LOG_ERROR(...)
#define LOG_ERROR_ONCE(...)
#define LOG_DEBUG(...)
#define BCDEC_IMPLEMENTATION
#include "@BCDEC@"
namespace fs = std::filesystem;
namespace fs_utils {
bool read_data(const fs::path &path, std::vector<uint8_t> &data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    data.assign(std::istreambuf_iterator<char>(file), {});
    return !file.bad();
}
std::string path_to_utf8(const fs::path &path) { return path.string(); }
}
uint8_t *stbi_load(...) { assert(false); return nullptr; }
void stbi_image_free(...) { assert(false); }
namespace pvr { void PVRTDecompressPVRTC(...) { assert(false); } }
namespace gxm {
@GXM@
}
namespace renderer::texture {
@DECODERS@
}
namespace renderer {
static constexpr bool log_texture_import = false;
@FORMATS@
namespace metal {
struct Info {
    uint64_t hash = 1;
    SceGxmTexture texture{};
    bool is_imported = false, is_srgb = false;
    uint32_t width = 0, height = 0;
    uint16_t mip_count = 0;
    SceGxmTextureBaseFormat format{};
};
struct Metadata { bool import_swap_rb = false; };
struct Upload {
    SceGxmTextureBaseFormat format;
    uint32_t width, height, mip, stride;
    int face;
    std::vector<uint8_t> bytes;
};
class MetalTextureCache {
public:
    Info info;
    Info *current_info = &info;
    Metadata meta;
    struct { bool is_dds = true; const fs::path *folder_path = nullptr; } loading_texture;
    bool support_astc = true, support_astc_software_import = false;
    bool support_dxt = true, support_dxt_software_import = false;
    ddspp::Descriptor *dds_descriptor = nullptr;
    std::vector<uint8_t> imported_texture_raw_data;
    uint8_t *imported_texture_decoded = nullptr;
    unsigned configurations = 0;
    std::vector<Upload> uploads;
    ~MetalTextureCache() { delete dds_descriptor; }
    Metadata &metadata(Info &) { return meta; }
    bool import_configure_texture();
    void import_upload_texture();
    void import_configure_impl(SceGxmTextureBaseFormat f, uint32_t w, uint32_t h,
        bool srgb, uint16_t components, uint16_t mips, bool swap) {
        assert(f == info.format && w == info.width && h == info.height);
        assert(srgb == info.is_srgb && mips == info.mip_count && swap == meta.import_swap_rb);
        assert(components == 4); // The fixtures use an RGBA guest descriptor.
        ++configurations;
    }
    void upload_texture_impl(SceGxmTextureBaseFormat f, uint32_t w, uint32_t h,
        uint32_t mip, const void *pixels, int face, uint32_t stride) {
        const auto [bw, bh] = gxm::get_block_size(f);
        const size_t size = gxm::is_bcn_format(f) || texture::is_astc_format(f)
            ? texture::get_compressed_size(f, stride, h)
            : size_t(stride) * h * gxm::bits_per_pixel(f) / 8;
        const auto *data = static_cast<const uint8_t *>(pixels);
        uploads.push_back({f, w, h, mip, stride, face, {data, data + size}});
    }
};
}
@IMPORT@
}

struct Format {
    ddspp::DXGIFormat dxgi;
    uint32_t bw, bh, bytes;
    bool srgb = false, swap = false;
};
// Bytes per pixel or complete compression block, independent of DDS offsets.
static const Format formats[] = {
    {ddspp::R8_UNORM,1,1,1}, {ddspp::R8_SNORM,1,1,1}, {ddspp::A8_UNORM,1,1,1},
    {ddspp::R8G8_UNORM,1,1,2}, {ddspp::R8G8_SNORM,1,1,2},
    {ddspp::R8G8B8A8_UNORM,1,1,4}, {ddspp::R8G8B8A8_UNORM_SRGB,1,1,4,true},
    {ddspp::R8G8B8A8_SNORM,1,1,4}, {ddspp::R8G8B8A8_SINT,1,1,4},
    {ddspp::B8G8R8A8_UNORM,1,1,4,false,true}, {ddspp::B8G8R8A8_UNORM_SRGB,1,1,4,true,true},
    {ddspp::B8G8R8X8_UNORM,1,1,4,false,true}, {ddspp::B8G8R8X8_UNORM_SRGB,1,1,4,true,true},
    {ddspp::R16_UNORM,1,1,2}, {ddspp::R16_SNORM,1,1,2}, {ddspp::R16_FLOAT,1,1,2}, {ddspp::D16_UNORM,1,1,2},
    {ddspp::R16G16_UNORM,1,1,4}, {ddspp::R16G16_SNORM,1,1,4}, {ddspp::R16G16_FLOAT,1,1,4},
    {ddspp::R16G16B16A16_UNORM,1,1,8}, {ddspp::R16G16B16A16_SNORM,1,1,8}, {ddspp::R16G16B16A16_FLOAT,1,1,8},
    {ddspp::R32_FLOAT,1,1,4}, {ddspp::D32_FLOAT,1,1,4}, {ddspp::R32G32_FLOAT,1,1,8},
    {ddspp::R10G10B10A2_UNORM,1,1,4}, {ddspp::R11G11B10_FLOAT,1,1,4}, {ddspp::R9G9B9E5_SHAREDEXP,1,1,4},
    {ddspp::B5G6R5_UNORM,1,1,2,false,true}, {ddspp::B5G5R5A1_UNORM,1,1,2,false,true}, {ddspp::B4G4R4A4_UNORM,1,1,2,false,true},
    {ddspp::BC1_UNORM,4,4,8}, {ddspp::BC1_UNORM_SRGB,4,4,8,true},
    {ddspp::BC2_UNORM,4,4,16}, {ddspp::BC2_UNORM_SRGB,4,4,16,true},
    {ddspp::BC3_UNORM,4,4,16}, {ddspp::BC3_UNORM_SRGB,4,4,16,true},
    {ddspp::BC4_UNORM,4,4,8}, {ddspp::BC4_SNORM,4,4,8},
    {ddspp::BC5_UNORM,4,4,16}, {ddspp::BC5_SNORM,4,4,16},
    {ddspp::BC6H_UF16,4,4,16}, {ddspp::BC6H_SF16,4,4,16},
    {ddspp::BC7_UNORM,4,4,16}, {ddspp::BC7_UNORM_SRGB,4,4,16,true},
#define ASTC_FMT(x,y) {ddspp::ASTC_##x##X##y##_UNORM,x,y,16}, {ddspp::ASTC_##x##X##y##_UNORM_SRGB,x,y,16,true},
#include "@ASTC@"
#undef ASTC_FMT
};
static void word(std::vector<uint8_t> &data, size_t at, uint32_t value) {
    for (unsigned n=0;n<4;++n) data.at(at+n) = uint8_t(value >> (n*8));
}
struct Image {
    uint32_t width, height, face, mip;
    size_t offset, size;
};
struct Fixture {
    std::vector<uint8_t> data;
    std::vector<Image> images;
};
static Fixture fixture(Format f, uint32_t w, uint32_t h, unsigned mips, bool cube, bool legacy=false) {
    Fixture out;
    out.data.resize(legacy ? 128 : 148);
    word(out.data,0,0x20534444); word(out.data,4,124); word(out.data,8,0x21007);
    word(out.data,12,h); word(out.data,16,w); word(out.data,28,mips);
    word(out.data,76,32); word(out.data,80,4);
    word(out.data,84,legacy ? 0x31545844 : 0x30315844); // DXT1 or DX10
    word(out.data,108,0x401008); word(out.data,112,cube ? 0xfe00 : 0);
    if (!legacy) {
        word(out.data,128,f.dxgi); word(out.data,132,3); // Texture2D
        word(out.data,136,cube ? 4 : 0); word(out.data,140,1);
    }
    for (unsigned face=0;face<(cube?6u:1u);++face) {
        unsigned width=w,height=h;
        for(unsigned mip=0;mip<mips;++mip) {
            const size_t bytes = size_t((width+f.bw-1)/f.bw) * ((height+f.bh-1)/f.bh) * f.bytes;
            const size_t offset=out.data.size();
            out.images.push_back({width,height,face,mip,offset,bytes});
            for(size_t n=0;n<bytes;++n) out.data.push_back(uint8_t(n*37+face*71+mip*53+unsigned(f.dxgi)*3));
            width = width>1 ? width/2 : 1; height = height>1 ? height/2 : 1;
        }
    }
    return out;
}
static void write(const fs::path &folder, const std::vector<uint8_t> &data) {
    std::ofstream file(folder/"0000000000000001.dds",std::ios::binary|std::ios::trunc);
    file.write(reinterpret_cast<const char *>(data.data()),data.size());
    assert(file.good());
}
static size_t image_checks=0, byte_checks=0;
static void check_upload(renderer::metal::MetalTextureCache &cache, const Fixture &f, Format format, bool cube, bool software) {
    cache.uploads.clear(); cache.import_upload_texture();
    assert(cache.uploads.size()==f.images.size());
    for(size_t n=0;n<f.images.size();++n) {
        const auto &expected=f.images[n]; const auto &actual=cache.uploads[n];
        assert(actual.width==expected.width && actual.height==expected.height);
        assert(actual.mip==expected.mip && actual.face==int(cube?expected.face+1:0));
        assert(actual.stride==((expected.width+format.bw-1)/format.bw)*format.bw);
        std::vector<uint8_t> bytes;
        if (!software) {
            bytes.assign(f.data.begin()+expected.offset,f.data.begin()+expected.offset+expected.size);
            assert(actual.format==cache.info.format);
        } else {
            // Separate block-by-block layout assembly using the existing codec.
            // This is an import integration oracle, not an independent BC codec.
            auto decoded=renderer::texture::get_matching_decompressed_format_metal3(cache.info.format);
            assert(actual.format==decoded);
            const auto bpp=gxm::bits_per_pixel(decoded)/8;
            bytes.resize(size_t(actual.stride)*actual.height*bpp);
            const auto columns=(expected.width+3)/4,rows=(expected.height+3)/4;
            for(unsigned by=0;by<rows;++by) for(unsigned bx=0;bx<columns;++bx) {
                std::array<uint64_t,16> block{};
                renderer::texture::decompress_compressed_texture_metal3(cache.info.format,block.data(),
                    f.data.data()+expected.offset+(by*columns+bx)*format.bytes,4,4);
                for(unsigned y=0;y<4 && by*4+y<expected.height;++y)
                    std::memcpy(bytes.data()+(size_t(by*4+y)*actual.stride+bx*4)*bpp,
                        reinterpret_cast<const uint8_t *>(block.data())+y*4*bpp,4*bpp);
            }
        }
        assert(actual.bytes==bytes);
        byte_checks+=bytes.size(); ++image_checks;
    }
}
int main(int argc,char **argv) {
    assert(argc>=2); const fs::path folder=argv[1]; fs::create_directory(folder);
    renderer::metal::MetalTextureCache cache;
    cache.loading_texture.folder_path=&folder;
    cache.info.texture.base_format=SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8>>24;
    size_t cases=0;
    for(const auto format:formats)
    for(auto [w,h]:std::array<std::pair<unsigned,unsigned>,12>{{{64,1},{1,64},{65,1},{1,65},{13,7},{7,13},{64,64},{1,1},{3,3},{5,5},{12,12},{17,17}}})
    for(bool cube:{false,true}) {
        if(cube && w!=h) continue;
        for(unsigned mips=1;mips<=std::bit_width(std::max(w,h));++mips)
        for(bool software:{false,true}) {
            auto base=renderer::dxgi_to_gxm(format.dxgi);
            if(software && !gxm::is_bcn_format(base)) continue;
            for(bool legacy:{false,true}) {
                if(legacy && format.dxgi!=ddspp::BC1_UNORM) continue;
                auto f=fixture(format,w,h,mips,cube,legacy); write(folder,f.data);
                cache.info.is_imported=false;
                cache.info.texture.type=(cube?SCE_GXM_TEXTURE_CUBE:SCE_GXM_TEXTURE_LINEAR)>>29;
                cache.support_dxt=!software; cache.support_dxt_software_import=software;
                assert(cache.import_configure_texture());
                assert(cache.info.width==w && cache.info.height==h && cache.info.mip_count==mips);
                assert(cache.info.is_srgb==format.srgb && cache.meta.import_swap_rb==format.swap);
                if (std::getenv("DDS_TRACE")) std::cerr << "DDS " << unsigned(format.dxgi) << ' ' << w << 'x' << h << " mips=" << mips << " cube=" << cube << " software=" << software << " legacy=" << legacy << '\n';
                check_upload(cache,f,format,cube,software);
                const auto count=cache.configurations;
                if(cube) cache.info.texture.type=SCE_GXM_TEXTURE_CUBE_ARBITRARY>>29;
                assert(cache.import_configure_texture() && cache.configurations==count);
                check_upload(cache,f,format,cube,software);
                ++cases;
            }
        }
    }
    std::cout<<"PASS DDS import: "<<cases<<" file/configure/reupload cases; "<<std::size(formats)
        <<" formats, "<<image_checks<<" face/mip uploads and "<<byte_checks<<" bytes; real DDS parser and BC codecs, GPU boundary recorded\n";
    size_t rejections=0,wide_sizes=0;
    const auto reject=[&](const std::vector<uint8_t> &data) {
        const auto info=cache.info; const auto swap=cache.meta.import_swap_rb;
        const auto configured=cache.configurations; const auto uploaded=cache.uploads.size();
        write(folder,data);
        assert(!cache.import_configure_texture());
        assert(cache.configurations==configured && cache.uploads.size()==uploaded);
        assert(cache.info.width==info.width && cache.info.height==info.height
            && cache.info.mip_count==info.mip_count && cache.info.format==info.format
            && cache.info.is_srgb==info.is_srgb && cache.info.is_imported==info.is_imported
            && cache.meta.import_swap_rb==swap);
        ++rejections;
    };
    cache.support_dxt=true;cache.support_dxt_software_import=true;
    for(auto format:formats) for(bool cube:{false,true}) {
        cache.info.texture.type=(cube?SCE_GXM_TEXTURE_CUBE:SCE_GXM_TEXTURE_LINEAR)>>29;
        const auto f=fixture(format,cube?7:13,7,3,cube);
        // Every short-header length, plus every subresource start/end boundary.
        for(size_t cut=0;cut<148;++cut) reject({f.data.begin(),f.data.begin()+cut});
        for(const auto &im:f.images) {
            reject({f.data.begin(),f.data.begin()+im.offset});
            reject({f.data.begin(),f.data.begin()+im.offset+im.size-1});
        }
        for(auto [at,value]:std::array<std::pair<size_t,uint32_t>,13>{{
            {0,0},{4,123},{76,31},{28,32},{28,65536},{28,0xffffffff},
            {16,16385},{12,0xffffffff},{140,0},{140,2},{132,4},{128,0},{136,cube?0u:4u}}}) {
            auto data=f.data; word(data,at,value); reject(data);
        }
        // Valid recovery after failed reads and ordinary trailing bytes.
        auto data=f.data;data.insert(data.end(),11,0xa5);write(folder,data);
        assert(cache.import_configure_texture());check_upload(cache,f,format,cube,false);
        ddspp::Descriptor desc=*cache.dds_descriptor;
        desc.width=desc.height=16384;desc.numMips=15;
        uint64_t chain=0;
        for(unsigned mip=0;mip<15;++mip) {
            const uint64_t side=uint64_t(16384)>>mip;
            const uint64_t size=((side+format.bw-1)/format.bw)*((side+format.bh-1)/format.bh)*format.bytes;
            assert(renderer::dds_mip_size(desc,side,side)==size);
            chain+=size;++wide_sizes;
        }
        const uint64_t total=chain*(cube?6:1)+desc.headerSize;
        assert(renderer::valid_dds_import(desc,total));
        assert(!renderer::valid_dds_import(desc,total-1));
        desc.numMips=16;assert(!renderer::valid_dds_import(desc,total));
    }
    // Legacy DXT1 with only one eight-byte block is shorter than MAX_HEADER_SIZE.
    auto tiny=fixture({ddspp::BC1_UNORM,4,4,8},1,1,1,false,true);
    cache.info.texture.type=SCE_GXM_TEXTURE_LINEAR>>29;
    assert(tiny.data.size()==136);
    write(folder,tiny.data);assert(cache.import_configure_texture());
    assert(cache.imported_texture_raw_data.size()==136);
    check_upload(cache,tiny,{ddspp::BC1_UNORM,4,4,8},false,false);
    for(size_t cut=128;cut<136;++cut) reject({tiny.data.begin(),tiny.data.begin()+cut});
    // Missing input, cube mismatch and the existing hardware/software gates.
    fs::remove(folder/"0000000000000001.dds");
    assert(!cache.import_configure_texture());++rejections;
    for(bool astc:{false,true}) {
        auto f=fixture(astc?Format{ddspp::ASTC_6X5_UNORM,6,5,16}:Format{ddspp::BC7_UNORM,4,4,16},13,7,4,false);
        cache.support_dxt=false;cache.support_dxt_software_import=false;
        cache.support_astc=false;cache.support_astc_software_import=false;
        reject(f.data);
        cache.support_dxt_software_import=true;cache.support_astc_software_import=true;
        write(folder,f.data);assert(cache.import_configure_texture());
        // ASTC software decoding is downstream of the recorded upload boundary.
        check_upload(cache,f,astc?Format{ddspp::ASTC_6X5_UNORM,6,5,16}:Format{ddspp::BC7_UNORM,4,4,16},false,!astc);
    }
    std::cout<<"PASS DDS bounds: "<<rejections<<" rejected files/capabilities; "<<wide_sizes
        <<" wide-size checks, tiny legacy file, retry and software gates; no device/GPU/app\n";
}
