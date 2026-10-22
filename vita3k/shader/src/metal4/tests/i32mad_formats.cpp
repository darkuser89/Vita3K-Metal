// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <features/state.h>
#include <shader/metal4/spirv_recompiler.h>
#include <shader/metal4/msl_recompiler.h>
#include <SPIRV/spirv.hpp>
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
using namespace shader::metal3;

// IMAE source 2 formats are independent of multiplier signedness:
// PVR_PSP2 ae4df4b723f3f1aa629b1f3b276b9cca6d663891,
// tools/intern/useasm/usedisasm.c:10778-10903, usc2/inst.h:2089-2097.
// CPU evaluation of actual generated SPIR-V, not a GPU/hardware oracle.
struct Graph {
    struct Node { unsigned op; std::vector<uint32_t> args; };
    struct Function { std::vector<uint32_t> parameters; uint32_t result = 0; };
    using Value = std::vector<uint32_t>;
    using Values = std::map<uint32_t, Value>;
    std::map<uint32_t,Node> nodes;
    std::map<uint32_t,Function> functions;
    std::map<uint32_t,std::string> names;
    std::vector<uint32_t> sums;
    explicit Graph(const std::vector<uint32_t> &words) {
        uint32_t function = 0;
        for (size_t at = 5; at < words.size();) {
            const auto *w = words.data()+at; const auto n = w[0]>>16, op=w[0]&65535;
            assert(n && at+n<=words.size());
            if(op==spv::OpName) names[w[1]]=std::string(reinterpret_cast<const char *>(w+2));
            if(op==spv::OpFunction) function=w[2];
            if(op==spv::OpFunctionEnd) function=0;
            if(op==spv::OpFunctionParameter) functions[function].parameters.push_back(w[2]);
            if(op==spv::OpReturnValue) functions[function].result=w[1];
            switch(op) {
            case spv::OpConstant: case spv::OpConstantComposite: case spv::OpCompositeConstruct:
            case spv::OpCompositeExtract: case spv::OpVectorExtractDynamic: case spv::OpBitcast:
            case spv::OpBitFieldSExtract: case spv::OpBitFieldUExtract: case spv::OpFunctionCall:
            case spv::OpIMul: case spv::OpIAdd: case spv::OpISub:
                nodes[w[2]]={op,{w+3,w+n}};break;
            }
            at+=n;
        }
        for(const auto &[id,node]:nodes)
            if((node.op==spv::OpIAdd || node.op==spv::OpISub) && nodes.contains(node.args[0]) && nodes.at(node.args[0]).op==spv::OpIMul)
                sums.push_back(id);
    }
    Value evaluate(uint32_t id,Values &values) const {
        if(values.contains(id))return values.at(id);
        const auto &node=nodes.at(id);const auto &a=node.args;Value out;
        switch(node.op) {
        case spv::OpConstant:out={a.at(0)};break;
        case spv::OpConstantComposite:case spv::OpCompositeConstruct:
            for(auto arg:a){const auto v=evaluate(arg,values);out.insert(out.end(),v.begin(),v.end());}break;
        case spv::OpCompositeExtract:{const auto v=evaluate(a.at(0),values);out={v.at(a.at(1))};break;}
        case spv::OpVectorExtractDynamic:{const auto v=evaluate(a.at(0),values);out={v.at(evaluate(a.at(1),values).at(0))};break;}
        case spv::OpBitcast:out=evaluate(a.at(0),values);break;
        case spv::OpFunctionCall:{
            const auto &fn=functions.at(a.at(0));assert(a.size()==fn.parameters.size()+1);Values local;
            for(size_t i=0;i<fn.parameters.size();++i)local[fn.parameters[i]]=evaluate(a[i+1],values);
            assert(fn.result);out=evaluate(fn.result,local);break;
        }
        case spv::OpBitFieldSExtract:case spv::OpBitFieldUExtract:{
            const auto bits=evaluate(a.at(0),values).at(0),offset=evaluate(a.at(1),values).at(0),count=evaluate(a.at(2),values).at(0);
            assert(count>0 && count<32 && offset+count<=32);
            const auto mask=(1u<<count)-1;auto value=(bits>>offset)&mask;
            if(node.op==spv::OpBitFieldSExtract && (value&(1u<<(count-1)))) value|=~mask;
            out={value};break;
        }
        case spv::OpIMul:case spv::OpIAdd:case spv::OpISub:{
            const auto lhs=evaluate(a.at(0),values),rhs=evaluate(a.at(1),values);assert(lhs.size()==rhs.size());
            for(size_t i=0;i<lhs.size();++i)out.push_back(node.op==spv::OpIMul?lhs[i]*rhs[i]:node.op==spv::OpIAdd?lhs[i]+rhs[i]:lhs[i]-rhs[i]);break;
        }
        default:assert(false);
        }
        return values[id]=out;
    }
    uint32_t raw_halfword(uint32_t id,bool sign,bool high,unsigned width=16) const {
        // A signedness bitcast is allowed after the separately selected extension.
        if(nodes.at(id).op==spv::OpBitcast)id=nodes.at(id).args.at(0);
        const auto &extract=nodes.at(id);assert(extract.op==spv::OpVectorExtractDynamic);
        Values empty;assert(evaluate(extract.args.at(1),empty).at(0)==unsigned(high));
        const auto &call=nodes.at(extract.args.at(0));assert(call.op==spv::OpFunctionCall && call.args.size()==2);
        assert(width==8 || width==16);
        assert(names.at(call.args[0]).starts_with(width==8?(sign?"unpack4xS8":"unpack4xU8"):(sign?"unpack2xS16":"unpack2xU16")));
        return call.args[1];
    }
};
static uint64_t encode(char field,unsigned value) {
    constexpr std::string_view layout="10101pps-nrcdeba0tttif00yy000kgghhjjlllllllmmmmmmmoooooooqqqqqqq";
    static_assert(layout.size()==64);uint64_t result=0;unsigned bit=0;
    for(int i=63;i>=0;--i)if(layout[i]=='1'||(layout[i]==field&&((value>>bit++)&1)))result|=uint64_t(1)<<(63-i);
    return result;
}
static uint64_t encode16(char field,unsigned value) {
    constexpr std::string_view layout="10100ppasnredbck-tttmmffoolhhgiijjqquuuuuuuvvvvvvvwwwwwwwxxxxxxx";
    static_assert(layout.size()==64);uint64_t result=0;unsigned bit=0;
    for(int i=63;i>=0;--i)if(layout[i]=='1'||(layout[i]==field&&((value>>bit++)&1)))result|=uint64_t(1)<<(63-i);
    return result;
}
static int64_t half(uint32_t word,bool high,bool sign) {
    int64_t value=(word/(high?65536u:1u))%65536;
    return sign&&value>=32768?value-65536:value;
}
static uint32_t oracle(const std::array<uint32_t,3> &words,bool sign,unsigned format,unsigned halves) {
    const int64_t a=half(words[0],halves&1,sign),b=half(words[1],halves&2,sign);
    int64_t c;
    if(format==2)c=sign&&words[2]>=0x80000000u?int64_t(words[2])-4294967296ll:int64_t(words[2]);
    else c=half(words[2],halves&4,format==1);
    return uint32_t(a*b+c);
}
static uint32_t oracle16(const std::array<uint32_t,3> &words,bool sign,unsigned f1,unsigned f2,unsigned bytes,bool negate) {
    const auto decode=[&](uint32_t word,unsigned format,bool high)->int64_t {
        if(!format)return half(word,false,sign);
        const int64_t value=(word/(high?256u:1u))%256;
        return format==2 && value>=128?value-256:value;
    };
    const int64_t a=half(words[0],false,sign),b=decode(words[1],f1,bytes&1),c=decode(words[2],f2,bytes&2);
    return uint32_t(a*b+(negate?-c:c))&65535;
}
int main(int argc,char **argv) {
    assert(argc==2);spdlog::set_level(spdlog::level::off);
    const auto directory=std::filesystem::path(argv[1]);
    struct Input {SceGxmProgram program;SceGxmProgramVertexVaryings varyings;uint64_t instruction;} input{};
    input.program.magic=0x00505847;input.program.size=sizeof(input);
    input.program.varyings_offset=offsetof(Input,varyings)-offsetof(SceGxmProgram,varyings_offset);
    input.program.primary_program_offset=offsetof(Input,instruction)-offsetof(SceGxmProgram,primary_program_offset);
    input.program.primary_program_instr_count=1;
    input.program.secondary_program_offset=offsetof(Input,instruction)-offsetof(SceGxmProgram,secondary_program_offset);
    input.program.secondary_program_offset_end=offsetof(Input,instruction)-offsetof(SceGxmProgram,secondary_program_offset_end);
    input.program.temp_reg_count1=128;
    std::vector<SceGxmVertexAttribute> attributes;Hints hints{};hints.attributes=&attributes;
    hints.color_format=SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;FeatureState features{};
    std::array<std::array<uint32_t,3>,320> cases{};
    constexpr uint32_t edges[]{0,1,0x7fff8000,0x80007fff,0x0000ffff,0xffff0000,0xffffffff,0x80000000};
    for(unsigned i=0;i<8;++i)cases[i]={edges[i],edges[(i+3)%8],edges[(i+5)%8]};
    uint32_t seed=0x32a034;
    for(unsigned i=8;i<cases.size();++i)for(auto &v:cases[i]){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;v=seed;}
    // Concrete cases that the former sign-coupled format decoding got wrong.
    assert(oracle({1,1,0xffff},true,0,0)==65536);
    assert(oracle({1,1,0xffff},false,1,0)==0);
    size_t translations=0,results=0,numeric=0,exhaustive=0,rejected=0;
    for(bool sign:{false,true})for(unsigned format=0;format<3;++format)
    for(unsigned halves=0;halves<8;++halves)for(unsigned repeat:{0u,1u,3u,7u}) {
        input.instruction=encode('i',sign)|encode('y',format)|encode('s',bool(halves&1))
            |encode('r',bool(halves&2))|encode('c',bool(halves&4))|encode('t',repeat)
            |encode('m',32)|encode('o',64)|encode('q',96);
        const auto words=convert_gxp(input.program,"imae-formats",features,Target::SpirVMetal,hints,false).spirv;
        const Graph graph(words);assert(graph.sums.size()==repeat+1);results+=graph.sums.size();
        for(auto result:graph.sums) {
            const auto &sum=graph.nodes.at(result);const auto &product=graph.nodes.at(sum.args.at(0));
            const auto a=graph.raw_halfword(product.args.at(0),sign,halves&1);
            const auto b=graph.raw_halfword(product.args.at(1),sign,halves&2);
            const auto c=format==2?sum.args.at(1):graph.raw_halfword(sum.args.at(1),format==1,halves&4);
            assert(a!=b && a!=c && b!=c);
            auto check=[&](const std::array<uint32_t,3> &in){
                Graph::Values values{{a,{in[0]}},{b,{in[1]}},{c,{in[2]}}};
                const auto value=graph.evaluate(result,values);assert(value.size()==1);
                assert(value[0]==oracle(in,sign,format,halves));
            };
            for(const auto &in:cases){check(in);++numeric;}
            if(halves==0 && repeat==0 && format<2)
                for(unsigned value=0;value<65536;++value){check({0x7fff,0x8001,value});++exhaustive;}
        }
        const auto stem=directory/("imae-"+std::to_string(translations++));
        std::ofstream out(stem.string()+".spv",std::ios::binary);out.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(out);
        if(repeat==7)std::ofstream(stem.string()+".metal")<<shader::metal::convert_spirv(words).source;
        // Reserved encoding is invalid for either signedness and saturation bit.
        input.instruction|=encode('y',3);
        for(bool sat:{false,true}){
            input.instruction|=encode('f',sat);bool threw=false;
            try{(void)convert_gxp(input.program,"imae-reserved",features,Target::SpirVMetal,hints,false);}
            catch(const std::runtime_error &e){threw=std::string(e.what()).find("Metal USSE PC 0 opcode")!=std::string::npos;}
            assert(threw);++rejected;
        }
    }
    std::cout<<"PASS IMAE source formats: "<<translations<<" public GXP translations, "<<results<<" multiply-add graphs, "<<numeric<<" raw-word numeric cases and "<<exhaustive<<" exhaustive source-2 halfword cases; actual signed/unsigned unpack functions evaluated, all high/low selectors, repeat 1/2/4/8, independent 64-bit modulo oracle\n";
    std::cout<<"PASS IMAE reserved format: "<<rejected<<" public translation rejections across signed/unsigned, high/low, repeats and saturation flag; valid saturation semantics not claimed\n";
    // IMA16 formats: useasm.c:9569-9595, usedisasm.c:10947-11077.
    // Saturation, offset-8, absolute and output shift remain outside this test.
    assert(oracle16({1,255,0},true,1,0,0,false)==255);
    assert(oracle16({1,255,0},false,2,0,0,false)==65535);
    size_t translations16=0,results16=0,numeric16=0,all_bytes=0;
    for(bool sign:{false,true})for(unsigned f1=0;f1<3;++f1)for(unsigned f2=0;f2<3;++f2)
    for(unsigned bytes=0;bytes<4;++bytes)for(bool negate:{false,true})for(unsigned repeat:{0u,1u,3u,7u}) {
        input.instruction=encode16('m',sign?2:0)|encode16('o',f1)|encode16('f',f2)
            |encode16('e',bool(bytes&1))|encode16('l',bool(bytes&2))|encode16('r',negate)
            |encode16('t',repeat)|encode16('v',32)|encode16('w',64)|encode16('x',96);
        const auto words=convert_gxp(input.program,"ima16-formats",features,Target::SpirVMetal,hints,false).spirv;
        const Graph graph(words);assert(graph.sums.size()==repeat+1);results16+=graph.sums.size();
        for(auto result:graph.sums) {
            const auto &sum=graph.nodes.at(result);const auto &product=graph.nodes.at(sum.args.at(0));
            assert(sum.op==(negate?spv::OpISub:spv::OpIAdd));
            const auto a=graph.raw_halfword(product.args.at(0),sign,false);
            const auto b=graph.raw_halfword(product.args.at(1),f1?f1==2:sign,f1 && (bytes&1),f1?8:16);
            const auto c=graph.raw_halfword(sum.args.at(1),f2?f2==2:sign,f2 && (bytes&2),f2?8:16);
            assert(a!=b && a!=c && b!=c);
            auto check=[&](const std::array<uint32_t,3> &in){
                Graph::Values values{{a,{in[0]}},{b,{in[1]}},{c,{in[2]}}};
                const auto value=graph.evaluate(result,values);assert(value.size()==1);
                assert((value[0]&65535)==oracle16(in,sign,f1,f2,bytes,negate));
            };
            for(unsigned i=0;i<64;++i){check(cases[i]);++numeric16;}
            if(repeat==0)for(unsigned value=0;value<256;++value) {
                // Every possible byte in each selected lane; the other byte
                // differs, exposing accidental halfword/byte selection.
                check({0x8001,0xabcd0000u|(value<<8)|(255-value),0x12340000u|((255-value)<<8)|value});++all_bytes;
            }
        }
        const auto stem=directory/("ima16-"+std::to_string(translations16++));
        std::ofstream out(stem.string()+".spv",std::ios::binary);out.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(out);
        if(repeat==7)std::ofstream(stem.string()+".metal")<<shader::metal::convert_spirv(words).source;
    }
    std::cout<<"PASS IMA16 source formats: "<<translations16<<" public GXP translations, "<<results16<<" real unpack/multiply/add-or-subtract graphs, "<<numeric16<<" raw-word cases plus "<<all_bytes<<" all-byte checks; both 16-bit modes, all U16/U8/S8 source pairs and high/low byte selections, repeat 1/2/4/8; saturation/offset8/abs/output-shift not claimed\n";

}
