// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// CPU conversion against an independent BT.601/BT.709 matrix reference.
static int reference(unsigned y, unsigned u, unsigned v, unsigned profile, unsigned channel) {
    const bool full=profile>=2, bt709=profile&1;
    const double kr=bt709?0.2126:0.299, kb=bt709?0.0722:0.114, kg=1-kr-kb;
    const double luma=(double(y)-(full?0:16))*(full?1:255.0/219);
    const double cu=(double(u)-128)*(full?1:255.0/224), cv=(double(v)-128)*(full?1:255.0/224);
    const double value=channel==0?luma+2*(1-kr)*cv:channel==2?luma+2*(1-kb)*cu
        :luma-2*kb*(1-kb)/kg*cu-2*kr*(1-kr)/kg*cv;
    return std::clamp(int(std::lround(value)),0,255);
}
int main() {
    renderer::metal::YUVConversionCache cache;
    size_t compared=0;
    // The official FFmpeg refresh must preserve the existing planar and
    // interleaved 4:2:0 paths as well as packed 4:2:2.
    for(unsigned width:{4u,8u})for(unsigned profile=0;profile<4;++profile)
    for(bool planar:{false,true})for(unsigned swizzle=0;swizzle<2;++swizzle) {
        constexpr unsigned height=4;
        const size_t plane=size_t(width)*height;
        std::vector<uint8_t> input(plane*3/2+64,0),output(plane*4+64,0x79);
        for(size_t i=0;i<plane;++i)input[i]=uint8_t((i*13+16)%256);
        if(planar) {
            std::fill_n(input.data()+plane,plane/4,swizzle?240:90);
            std::fill_n(input.data()+plane+plane/4,plane/4,swizzle?90:240);
        } else for(size_t i=plane;i<plane*3/2;i+=2){input[i]=swizzle?90:240;input[i+1]=swizzle?240:90;}
        renderer::texture::yuv420_texture_to_rgb_metal3(cache,output.data(),input.data(),width,height,width,height,planar,swizzle,SceGxmYuvProfile(profile));
        for(size_t i=0;i<plane;++i) {
            for(unsigned c=0;c<3;++c)assert(std::abs(int(output[i*4+c])-reference(input[i],90,240,profile,c))<=3);
            assert(output[i*4+3]==255);
        }
        assert(std::all_of(output.begin()+plane*4,output.end(),[](auto v){return v==0x79;}));
    }
    for(unsigned width:{2u,4u,16u,32u})for(unsigned height:{1u,3u})
    for(unsigned profile=0;profile<4;++profile)for(unsigned swizzle=0;swizzle<8;++swizzle) {
        std::vector<uint8_t> input(size_t(width)*height*2+64,0),output(size_t(width)*height*4+64,0x79);
        std::vector<std::array<unsigned,3>> expected;
        for(unsigned row=0;row<height;++row)for(unsigned x=0;x<width;x+=2) {
            const unsigned n=row*width+x, y0=(n*31+16)%256,y1=(n*17+235)%256,u=(n*23+128)%256,v=(n*41+128)%256;
            auto *p=input.data()+n*2;
            switch(swizzle&3) {
            case 0:p[0]=y0;p[1]=u;p[2]=y1;p[3]=v;break;
            case 1:p[0]=y0;p[1]=v;p[2]=y1;p[3]=u;break;
            case 2:p[0]=u;p[1]=y0;p[2]=v;p[3]=y1;break;
            default:p[0]=v;p[1]=y0;p[2]=u;p[3]=y1;break;
            }
            expected.push_back({y0,u,v});expected.push_back({y1,u,v});
        }
        assert(renderer::texture::yuv422_texture_to_rgb_metal3(cache,output.data(),input.data(),width,height,swizzle,SceGxmYuvProfile(profile)));
        for(size_t i=0;i<expected.size();++i) {
            for(unsigned c=0;c<3;++c) {
                const auto &p=expected[i];const int wanted=reference(p[0],p[1],p[2],profile,c);
                assert(std::abs(int(output[i*4+c])-wanted)<=3);++compared;
            }
            assert(output[i*4+3]==255);
        }
        assert(std::all_of(output.begin()+width*height*4,output.end(),[](auto v){return v==0x79;}));
    }
    uint8_t source[64]{},dest[64]{};
    for(unsigned width:{0u,1u,3u})assert(!renderer::texture::yuv422_texture_to_rgb_metal3(cache,dest,source,width,1,0,SCE_GXM_YUV_PROFILE_BT601_STANDARD));
    assert(!renderer::texture::yuv422_texture_to_rgb_metal3(cache,dest,source,2,0,0,SCE_GXM_YUV_PROFILE_BT601_STANDARD));
    assert(!renderer::texture::yuv422_texture_to_rgb_metal3(cache,dest,source,2,1,8,SCE_GXM_YUV_PROFILE_BT601_STANDARD));
    renderer::metal::release_yuv_cache(cache);
    assert(!cache.sws_context&&std::all_of(cache.sws_context_422.begin(),cache.sws_context_422.end(),[](auto p){return !p;}));
    std::cout<<"PASS YUV422: "<<compared<<" RGB channels against independent BT.601/709 limited/full reference (tolerance 3), all 8 swizzles, dimension/profile cache changes, opaque alpha, padding, invalid input and cleanup; official FFmpeg, no GPU/app\n";
    std::cout<<"PASS YUV420 preservation: NV12/NV21 and both three-plane chroma orders, all 4 profiles, dimension/profile changes, opaque alpha and padding with refreshed official FFmpeg\n";
}
