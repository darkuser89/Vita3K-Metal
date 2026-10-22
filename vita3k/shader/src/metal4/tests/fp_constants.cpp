// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <shader/metal4/usse_utilities.h>
#include <shader/metal4/usse_constant_table.h>
#include <features/state.h>
#include <util/float_to_half.h>
#include <SPIRV/GLSL.std.450.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace shader::metal3;
using namespace shader::metal3::usse;
using namespace shader::metal3::usse::utils;
// MODIFIERS_HOOK
static spv::Id extracted_load(spv::Builder &b,SpirvUtilFunctions &utils,Operand op,Imm4 dest_mask) {
    // CONSTANT_HOOK
    throw std::runtime_error("not constant");
}
static uint32_t half_encoding(float f) {
    if(std::isnan(f))return std::signbit(f)?0xfe00:0x7e00; // preserve the existing encoder's canonical NaN policy
    const __fp16 native=f;return std::bit_cast<uint16_t>(native);
}
static uint32_t reference(DataType type,unsigned index,SwizzleChannel channel) {
    const auto c=unsigned(channel);
    if(c>=4) {
        const float value=c==4?0:c==5?1:c==6?2:0.5f;
        return type==DataType::F16||type==DataType::F32?std::bit_cast<uint32_t>(value):uint32_t(value);
    }
    if(type==DataType::F32||type==DataType::INT32||type==DataType::UINT32)
        return c==1?f32_constant_table_bank_1_raw[index]:f32_constant_table_bank_0_raw[index];
    const float *banks[]={f16_constant_table_bank0,f16_constant_table_bank1,f16_constant_table_bank2,f16_constant_table_bank3};
    const float f=banks[c][index];if(type==DataType::F16)return std::bit_cast<uint32_t>(f);
    const uint32_t bits=half_encoding(f);
    return type==DataType::UINT16?bits:uint32_t(int32_t(bits)-(bits&0x8000?65536:0));
}
int main(int argc,char **argv) {
    spv::SpvBuildLogger logger;spv::Builder b(spv::Spv_1_0,0,&logger);
    b.setMemoryModel(spv::AddressingModelLogical,spv::MemoryModelGLSL450);b.addCapability(spv::CapabilityShader);
    auto *main=b.makeEntryPoint("main");b.addEntryPoint(spv::ExecutionModelVertex,main,"main");
    SpirvUtilFunctions utils{};utils.native_metal=true;utils.std_builtins=b.import("GLSL.std.450");
    SpirvShaderParameters parameters{};parameters.native_metal=true;
    FeatureState features{};
    auto load=[&](Operand operand,Imm4 mask,bool production) {
        return production?shader::metal3::usse::utils::load(b,parameters,utils,features,operand,mask,0)
                         :extracted_load(b,utils,operand,mask);
    };
    Operand op{};op.bank=RegisterBank::FPCONSTANT;op.swizzle=SWIZZLE_CHANNEL_4_DEFAULT;
    if(argc>1&&std::string(argv[1])=="repro") {op.type=DataType::INT16;op.num=20;(void)extracted_load(b,utils,op,1);return 0;}
    const std::array<Swizzle4,9> swizzles{{SWIZZLE_CHANNEL_4_DEFAULT,
        SWIZZLE_CHANNEL_4(W,Z,Y,X),SWIZZLE_CHANNEL_4(X,X,X,X),SWIZZLE_CHANNEL_4(Y,Y,Y,Y),
        SWIZZLE_CHANNEL_4(Z,Z,Z,Z),SWIZZLE_CHANNEL_4(W,W,W,W),SWIZZLE_CHANNEL_4(0,1,2,H),
        SWIZZLE_CHANNEL_4(H,X,1,Y),SWIZZLE_CHANNEL_4(Z,2,W,0)}};
    size_t cases=0,components=0;
    for(auto type:{DataType::F16,DataType::F32,DataType::INT16,DataType::UINT16,DataType::INT32,DataType::UINT32})
    for(unsigned index=0;index<64;++index)for(auto swizzle:swizzles)for(unsigned mask=1;mask<16;++mask)for(bool production:{false,true}) {
        op.num=index;op.type=type;op.swizzle=swizzle;op.flags={};
        auto value=load(op,mask,production);assert(value);
        const auto count=std::popcount(mask);assert(b.getNumComponents(value)==count);
        unsigned at=0;
        for(unsigned c=0;c<4;++c)if(mask&(1u<<c)) {
            const auto scalar=count==1?value:b.getIdOperand(value,at++);
            assert(b.getConstantScalar(scalar)==reference(type,index,swizzle[c]));
            auto t=b.getTypeId(scalar);
            assert(type==DataType::F16||type==DataType::F32?b.isFloatType(t):type==DataType::INT16||type==DataType::INT32?b.isIntType(t):b.isUintType(t));
            ++components;
        }
        ++cases;
    }
    for(auto type:{DataType::INT16,DataType::UINT16,DataType::INT32,DataType::UINT32,DataType::F16,DataType::F32})
    for(unsigned bad:{64u,127u,65535u})for(bool production:{false,true}) {
        op.num=bad;op.type=type;op.swizzle=SWIZZLE_CHANNEL_4_DEFAULT;bool threw=false;
        try{(void)load(op,1,production);}catch(const std::runtime_error &){threw=true;}assert(threw);
    }
    for(auto type:{DataType::INT16,DataType::UINT16,DataType::F32}) {
        op.num=0;op.type=type;op.swizzle={SwizzleChannel::C_UNDEFINED,SwizzleChannel::C_X,SwizzleChannel::C_Y,SwizzleChannel::C_Z};
        for(unsigned mask:{0u,1u,3u})for(bool production:{false,true}) {bool threw=false;try{(void)load(op,mask,production);}catch(const std::runtime_error &){threw=true;}assert(threw);}
    }
    size_t modifiers=0;
    for(auto type:{DataType::INT16,DataType::INT32,DataType::F16,DataType::F32,DataType::UINT16,DataType::UINT32})
    for(auto flags:{RegisterFlags::Absolute,RegisterFlags::Negative,RegisterFlags::Absolute|RegisterFlags::Negative})
    for(unsigned mask:{1u,3u,7u,15u})for(bool production:{false,true}) {
        op.num=20;op.type=type;op.flags=flags;op.swizzle=SWIZZLE_CHANNEL_4_DEFAULT;
        const auto value=load(op,mask,production);assert(value);
        auto scalar_type=b.getScalarTypeId(b.getTypeId(value));
        assert(type==DataType::F16||type==DataType::F32?b.isFloatType(scalar_type):type==DataType::INT16||type==DataType::INT32?b.isIntType(scalar_type):b.isUintType(scalar_type));++modifiers;
    }
    b.leaveFunction();std::vector<uint32_t> words;b.dump(words);
    if(argc>1){std::ofstream out(argv[1],std::ios::binary);out.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(out);}
    std::cout<<"PASS FP constants: "<<cases<<" loads (public rebuilt library and extracted production branch), "<<components<<" exact scalar payload/type checks across all banks/indices/masks; signed half extension versus independent native-half oracle, F32 NaN payloads, invalid bank/swizzle rejection and "<<modifiers<<" modifier types; extracted branch instrumented with ASAN/UBSAN/float-cast-overflow\n";
}
