// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Public GXP -> SPIR-V/MSL shape regression. No ISA/hardware oracle.
#include <features/state.h>
#include <shader/metal4/spirv_recompiler.h>
#include <shader/metal4/msl_recompiler.h>
#include <shader/metal4/usse_program_analyzer.h>
#include <SPIRV/spirv.hpp>
#include <SPIRV/SpvBuilder.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
using namespace shader::metal3;
using namespace shader::metal3::usse;
// TEST_PREDICATE_HOOK
// TEST_INVALIDATION_HOOK
static uint64_t field(char name,unsigned value,
    std::string_view layout="01111ppps-oydtrcevuuiizzm-aa--bbnnkkfffffffwllgggghhhhhhhjjjjjjj") {
    uint64_t out=0;unsigned bit=0;
    for(int i=63;i>=0;--i)
        if(layout[i]=='1'||(layout[i]==name&&((value>>bit++)&1)))out|=uint64_t(1)<<(63-i);
    return out;
}
struct Node { spv::Op op;uint32_t type;std::vector<uint32_t> args; };
struct Graph {
    std::map<uint32_t,Node> nodes;
    std::map<uint32_t,unsigned> widths;
    std::vector<uint32_t> predicate_stores;
    std::vector<std::pair<uint32_t,uint32_t>> predicate_targets;
    explicit Graph(const std::vector<uint32_t> &words) {
        for(size_t at=5;at<words.size();) {
            const auto count=words[at]>>16;const auto op=spv::Op(words[at]&65535);assert(count && at+count<=words.size());
            if(op==spv::OpTypeBool||op==spv::OpTypeInt||op==spv::OpTypeFloat)widths[words[at+1]]=1;
            if(op==spv::OpTypeVector)widths[words[at+1]]=words[at+3];
            if(op==spv::OpStore && nodes.contains(words[at+2])) {
                const auto value=words[at+2];const auto code=nodes.at(value).op;
                if(code==spv::OpLogicalAnd||code==spv::OpLogicalOr||code==spv::OpAll||code==spv::OpAny
                    ||(code==spv::OpCompositeExtract && nodes.contains(nodes.at(value).args[0])
                        &&(nodes.at(nodes.at(value).args[0]).op==spv::OpLogicalAnd
                            ||nodes.at(nodes.at(value).args[0]).op==spv::OpLogicalOr))) {
                    predicate_stores.push_back(value);predicate_targets.push_back({words[at+1],value});
                }
            }
            switch(op) {
            case spv::OpConstantTrue:case spv::OpConstantFalse:
            case spv::OpLogicalAnd:case spv::OpLogicalOr:case spv::OpLogicalNot:
            case spv::OpIEqual:case spv::OpINotEqual:
            case spv::OpSLessThan:case spv::OpSGreaterThanEqual:
            case spv::OpULessThan:case spv::OpUGreaterThanEqual:
            case spv::OpShiftLeftLogical:case spv::OpAll:case spv::OpAny:
            case spv::OpAccessChain:case spv::OpLoad:case spv::OpVectorShuffle:case spv::OpVectorExtractDynamic:
            case spv::OpConstant:case spv::OpConstantComposite:case spv::OpDot:case spv::OpFSub:case spv::OpFAdd:case spv::OpFMul:
            case spv::OpCompositeConstruct:case spv::OpCompositeExtract:case spv::OpSelect:case spv::OpBitcast:
            case spv::OpFOrdEqual:case spv::OpFOrdNotEqual:case spv::OpFOrdLessThan:case spv::OpFOrdLessThanEqual:
            case spv::OpFOrdGreaterThan:case spv::OpFOrdGreaterThanEqual:
                nodes.emplace(words[at+2],Node{op,words[at+1],{words.begin()+at+3,words.begin()+at+count}});break;
            default:break;
            }
            at+=count;
        }
    }
    using Value=std::vector<uint32_t>;
    using Values=std::map<uint32_t,Value>;
    Value evaluate(uint32_t id,Values &values) const {
        if(auto it=values.find(id);it!=values.end())return it->second;
        const auto &n=nodes.at(id);Value result;
        if(n.op==spv::OpConstantTrue)result={1};
        else if(n.op==spv::OpConstantFalse)result={0};
        else if(n.op==spv::OpConstant)result={n.args[0]};
        else if(n.op==spv::OpCompositeExtract)result={evaluate(n.args[0],values).at(n.args[1])};
        else if(n.op==spv::OpAll||n.op==spv::OpAny) {
            const auto v=evaluate(n.args[0],values);
            result={n.op==spv::OpAll?std::all_of(v.begin(),v.end(),[](auto x){return x!=0;}):
                std::any_of(v.begin(),v.end(),[](auto x){return x!=0;})};
        }
        else if(n.op==spv::OpConstantComposite||n.op==spv::OpCompositeConstruct) {
            for(auto arg:n.args){const auto v=evaluate(arg,values);result.insert(result.end(),v.begin(),v.end());}
        } else {
            const auto a=evaluate(n.args.at(0),values),b=evaluate(n.args.at(1),values);assert(a.size()==b.size());
            if(n.op==spv::OpDot) {
                float sum=0;for(size_t i=0;i<a.size();++i)sum+=std::bit_cast<float>(a[i])*std::bit_cast<float>(b[i]);
                return values[id]={std::bit_cast<uint32_t>(sum)};
            }
            for(size_t i=0;i<a.size();++i) {
                const auto f=std::bit_cast<float>(a[i]),g=std::bit_cast<float>(b[i]);
                if(n.op==spv::OpFAdd||n.op==spv::OpFSub||n.op==spv::OpFMul) {
                    result.push_back(std::bit_cast<uint32_t>(n.op==spv::OpFAdd?f+g:n.op==spv::OpFSub?f-g:f*g));continue;
                }
                const auto x=std::bit_cast<int32_t>(a[i]),y=std::bit_cast<int32_t>(b[i]);
                const bool ordered=!std::isnan(f)&&!std::isnan(g);bool value=false;
                switch(n.op) {
                case spv::OpLogicalAnd:value=a[i]&&b[i];break;
                case spv::OpLogicalOr:value=a[i]||b[i];break;
                case spv::OpIEqual:value=a[i]==b[i];break;
                case spv::OpINotEqual:value=a[i]!=b[i];break;
                case spv::OpSLessThan:value=x<y;break;
                case spv::OpSGreaterThanEqual:value=x>=y;break;
                case spv::OpULessThan:value=a[i]<b[i];break;
                case spv::OpUGreaterThanEqual:value=a[i]>=b[i];break;
                case spv::OpFOrdEqual:value=ordered&&f==g;break;
                case spv::OpFOrdNotEqual:value=ordered&&f!=g;break;
                case spv::OpFOrdLessThan:value=ordered&&f<g;break;
                case spv::OpFOrdGreaterThanEqual:value=ordered&&f>=g;break;
                case spv::OpFOrdLessThanEqual:value=ordered&&f<=g;break;
                case spv::OpFOrdGreaterThan:value=ordered&&f>g;break;
                default:assert(false);
                }
                result.push_back(value);
            }
        }
        return values[id]=result;
    }
    void check_dot_lanes(uint32_t id) const {
        const auto &node=nodes.at(id);
        if(node.op==spv::OpLogicalAnd||node.op==spv::OpLogicalOr) {
            check_dot_lanes(node.args[0]);check_dot_lanes(node.args[1]);return;
        }
        if(node.op==spv::OpConstantTrue||node.op==spv::OpConstantComposite)return;
        const auto &replicated=nodes.at(node.args.at(0));
        assert(replicated.op==spv::OpCompositeConstruct && replicated.args.size()==4);
        assert(replicated.args[0]==replicated.args[1] && replicated.args[0]==replicated.args[2]
            && replicated.args[0]==replicated.args[3]);
        assert(nodes.at(replicated.args[0]).op==spv::OpDot);
    }
};
static void predicate_arithmetic(const std::filesystem::path &directory) {
    spv::SpvBuildLogger logger;spv::Builder b(spv::Spv_1_0,0,&logger);
    b.setMemoryModel(spv::AddressingModelLogical,spv::MemoryModelGLSL450);b.addCapability(spv::CapabilityShader);
    auto *main=b.makeEntryPoint("main");b.addEntryPoint(spv::ExecutionModelVertex,main,"main");
    struct Case {unsigned domain,sign,zero,width;bool combine;uint32_t lhs,rhs,result;};
    std::vector<Case> cases;
    for(unsigned domain=0;domain<3;++domain)for(unsigned width=1;width<=4;++width) {
        auto type=domain==0?b.makeFloatType(32):domain==1?b.makeIntType(32):b.makeUintType(32);
        if(width>1)type=b.makeVectorType(type,width);
        const auto lhs=b.createUndefined(type),rhs=b.createUndefined(type);
        for(unsigned sign=0;sign<3;++sign)for(unsigned zero=0;zero<3;++zero)for(bool combine:{false,true})
            cases.push_back({domain,sign,zero,width,combine,lhs,rhs,test_predicate(b,lhs,rhs,domain,sign,zero,combine)});
    }
    const auto dummy=b.createUndefined(b.makeFloatType(32));size_t reserved=0;
    for(unsigned d=0;d<4;++d)for(unsigned s=0;s<4;++s)for(unsigned z=0;z<4;++z)
        if(d==3||s==3||z==3) {assert(test_predicate(b,dummy,dummy,d,s,z,false)==spv::NoResult);++reserved;}
    b.leaveFunction();std::vector<uint32_t> words;b.dump(words);const Graph graph(words);
    std::ofstream out(directory/"predicate-arithmetic.spv",std::ios::binary);
    out.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(out);
    constexpr std::array<uint32_t,16> edges={0,1,0xffffffff,0x80000000,0x7fffffff,0x8000,0xffff,0x10000,
        0xff800000,0xc0400000,0x80000001,0x40400000,0x7f800000,0x7fc12345,0xffc12345,0x007fffff};
    uint32_t random=0x3535;auto next=[&]{random^=random<<13;random^=random>>17;random^=random<<5;return random;};
    size_t numeric=0,lanes=0;
    for(const auto &c:cases)for(unsigned sample=0;sample<512;++sample) {
        Graph::Value lhs,rhs;
        for(unsigned lane=0;lane<c.width;++lane) {
            lhs.push_back(sample<256?edges[(sample+lane)%16]:next());
            rhs.push_back(sample<256?edges[(sample/16+lane*3)%16]:next());
        }
        Graph::Values values{{c.lhs,lhs},{c.rhs,rhs}};const auto result=graph.evaluate(c.result,values);assert(result.size()==c.width);
        for(unsigned lane=0;lane<c.width;++lane) {
            // Classify the relation independently, then apply the assembler's
            // two test enables and CRCOMB. Ordered NaN handling preserves the
            // existing translator contract; it is not a hardware oracle.
            int relation;
            if(c.domain==0) {
                const double a=std::bit_cast<float>(lhs[lane]),v=std::bit_cast<float>(rhs[lane]);
                relation=std::isnan(a)||std::isnan(v)?2:a<v?-1:a==v?0:1;
            } else {
                const int64_t a=c.domain==1?int64_t(std::bit_cast<int32_t>(lhs[lane])):int64_t(lhs[lane]);
                const int64_t v=c.domain==1?int64_t(std::bit_cast<int32_t>(rhs[lane])):int64_t(rhs[lane]);
                const int64_t difference=a-v;relation=difference<0?-1:difference==0?0:1;
            }
            const bool s=c.sign==0|| (c.sign==1?relation==-1:relation==0||relation==1);
            const bool z=c.zero==0|| (c.zero==1?relation==0:relation==-1||relation==1);
            assert(result[lane]==(c.combine?(s&&z):(s||z)));++lanes;
        }
        ++numeric;
    }
    assert(cases.size()==216 && reserved==37);
    std::cout<<"PASS VTST CRCOMB arithmetic: "<<numeric<<" vector cases and "<<lanes
        <<" lane results from the actual emitted helper; 216 float/signed/unsigned, 1..4-lane, sign/zero/AND/OR graphs; exhaustive 16x16 boundary pairs plus random bits, ordered NaNs and signed zero, 37 reserved rejects; ASAN/UBSAN, no hardware oracle\n";
}
static void analyzer_tests(const FeatureState &features,const Hints &hints,const std::filesystem::path &directory) {
    constexpr std::string_view layout="01001ppps-oydrceavttiizzmhhhnnbbkkffgggggggwlluuuujjjjjjjqqqqqqq";
    auto f=[&](char name,unsigned value){return field(name,value,layout);};
    size_t masks=0,blocks=0;
    for(unsigned opcode=0;opcode<32;++opcode)for(unsigned selector=0;selector<4;++selector)
    for(unsigned channel=0;channel<8;++channel)for(unsigned pdst=0;pdst<4;++pdst)for(unsigned condition=0;condition<9;++condition) {
        auto instruction=f('l',selector)|f('h',channel)|f('n',pdst);
        instruction=(instruction&((uint64_t(1)<<59)-1))|(uint64_t(opcode)<<59);
        const bool per_channel=selector==0&&channel==6;
        const unsigned index=condition>=1&&condition<=4?condition-1:condition==5?0:condition==6?1:2;
        const bool expected=opcode==9&&condition!=0&&(condition==7||per_channel||index==pdst);
        assert(invalidates_native_test_predicate(instruction,condition)==expected);++masks;
    }
    struct Input {SceGxmProgram program;SceGxmProgramVertexVaryings varyings;std::array<uint64_t,3> instructions;}in{};
    in.program.magic=0x00505847;in.program.size=sizeof(in);
    in.program.varyings_offset=offsetof(Input,varyings)-offsetof(SceGxmProgram,varyings_offset);
    in.program.primary_program_offset=offsetof(Input,instructions)-offsetof(SceGxmProgram,primary_program_offset);
    in.program.primary_program_instr_count=3;in.program.temp_reg_count1=128;
    in.program.secondary_program_offset=offsetof(Input,instructions)-offsetof(SceGxmProgram,secondary_program_offset);
    in.program.secondary_program_offset_end=offsetof(Input,instructions)-offsetof(SceGxmProgram,secondary_program_offset_end);
    for(unsigned condition=1;condition<=6;++condition) {
        in.instructions[0]=f('p',condition)|f('a',1)|f('h',6)|f('u',14)|f('j',16)|f('q',32);
        in.instructions[1]=in.instructions[2]=field('p',condition)|field('e',1)|field('w',1)
            |field('g',14)|field('h',16)|field('j',32);
        USSEBlockNode root(nullptr,0);
        analyze(root,2,[&](USSEOffset pc){return in.instructions.at(pc);},true);
        assert(root.children_count()==2);
        for(unsigned i=0;i<2;++i) {
            const auto *node=root.children_at(i);assert(node->node_type()==USSE_CODE_NODE);
            const auto &code=static_cast<const USSECodeNode &>(*node);
            assert(code.condition==condition&&code.offset==i&&code.size==(i?2u:1u));
        }
        const auto words=convert_gxp(in.program,"vtst-predicate-invalidation",features,Target::SpirVMetal,hints,false).spirv;
        std::vector<uint32_t> branches;
        for(size_t at=5;at<words.size();) {
            const auto *w=words.data()+at;const auto n=w[0]>>16,op=w[0]&65535;assert(n&&at+n<=words.size());
            if(op==spv::OpBranchConditional)branches.push_back(w[1]);
            at+=n;
        }
        const Graph graph(words);
        std::vector<uint32_t> predicate_loads;
        for(auto condition_id:branches) {
            if(!graph.nodes.contains(condition_id))continue;
            if(graph.nodes.at(condition_id).op==spv::OpLogicalNot)condition_id=graph.nodes.at(condition_id).args[0];
            const auto &load=graph.nodes.at(condition_id);if(load.op!=spv::OpLoad)continue;
            const auto &pointer=graph.nodes.at(load.args[0]);assert(pointer.op==spv::OpAccessChain);
            assert(graph.nodes.at(pointer.args[1]).args[0]==(condition<=4?condition-1:condition==5?0:1));
            predicate_loads.push_back(condition_id);
        }
        assert(predicate_loads.size()==2&&predicate_loads[0]!=predicate_loads[1]);
        const auto name="analyzer-"+std::to_string(condition);
        std::ofstream spv(directory/(name+".spv"),std::ios::binary);
        spv.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(spv);
        std::ofstream(directory/(name+".metal"))<<shader::metal::convert_spirv(words).source;
        ++blocks;
        for(unsigned pdst=0;pdst<4;++pdst) {
            in.instructions[0]=f('p',condition)|f('a',1)|f('h',0)|f('n',pdst)|f('u',14)|f('j',16)|f('q',32);
            analyze(root,2,[&](USSEOffset pc){return in.instructions.at(pc);},true);
            const unsigned index=condition<=4?condition-1:condition==5?0:1;
            assert(root.children_count()==(pdst==index?2u:1u));++blocks;
        }
    }
    in.instructions[0]=f('p',3)|f('a',1)|f('h',6)|f('u',14)|f('j',16)|f('q',32);
    in.instructions[1]=in.instructions[2]=field('p',3)|field('e',1)|field('w',1)|field('g',14)|field('h',16)|field('j',32);
    USSEBlockNode legacy(nullptr,0);analyze(legacy,2,[&](USSEOffset pc){return in.instructions.at(pc);},false);
    assert(legacy.children_count()==1&&masks==36864&&blocks==30);
    std::cout<<"PASS VTST predicate invalidation: "<<masks<<" exact native write-mask checks, "<<blocks
        <<" actual analyzer sequences; P0..P3/negated conditions reload after per-channel writes, ordinary writes split only matching predicates, mask instructions do not write predicate banks; six public shaders contain two independent conditions, legacy grouping retained\n";
}
template<class Program>
static void channel_tests(Program &in,const FeatureState &features,const Hints &hints,const std::filesystem::path &directory) {
    // Compiler contracts: PVR_PSP2 ae4df4b7 usc2/inst.h VTEST_CHANSEL,
    // useasm.c:4308-4361,4454-4465; sgxvec34usedefs.h:862 (PERCHAN=6).
    // Numerical inputs enter at actual unpacked operands, not guest storage.
    size_t cases=0,values_checked=0,msl=0;
    const auto check=[&](unsigned precision,unsigned channel,unsigned selector,unsigned opcode,
        unsigned write,unsigned sign,unsigned zero,unsigned combine) {
        constexpr std::string_view layout="01001ppps-oydrceavttiizzmhhhnnbbkkffgggggggwlluuuujjjjjjjqqqqqqq";
        auto f=[&](char name,unsigned value){return field(name,value,layout);};
        const bool per_channel=selector==0&&channel==6,dot=selector==0&&opcode==8;
        const unsigned pdst=per_channel?0:2;
        in.instruction=f('a',precision)|f('h',channel)|f('n',pdst)|f('w',write)|f('l',selector)|f('u',opcode)
            |f('i',sign)|f('z',zero)|f('m',combine)|f('j',16)|f('q',32);
        const auto words=convert_gxp(in.program,"vtst-channels",features,Target::SpirVMetal,hints,false).spirv;
        const Graph graph(words);assert(graph.predicate_targets.size()==(per_channel?4u:1u));
        for(size_t lane=0;lane<graph.predicate_targets.size();++lane) {
            const auto &pointer=graph.nodes.at(graph.predicate_targets[lane].first);assert(pointer.op==spv::OpAccessChain);
            assert(graph.nodes.at(pointer.args[1]).args[0]==(per_channel?lane:pdst));
        }
        auto logic=graph.predicate_targets[0].second;
        if(graph.nodes.at(logic).op==spv::OpAll||graph.nodes.at(logic).op==spv::OpAny
            ||graph.nodes.at(logic).op==spv::OpCompositeExtract)logic=graph.nodes.at(logic).args[0];
        const auto &condition=graph.nodes.at(logic);assert(condition.op==(combine?spv::OpLogicalAnd:spv::OpLogicalOr));
        const unsigned width=channel<4||selector!=0?1:4;
        assert(graph.widths.at(condition.type)==width);
        uint32_t lhs=0,rhs=0,alu=0;unsigned dots=0,subtractions=0;
        for(const auto &[id,node]:graph.nodes) {
            if(node.op==spv::OpDot)++dots;
            if(node.op==spv::OpFSub) {++subtractions;assert(graph.widths.at(node.type)==width);}
        }
        assert(dots==unsigned(dot));
        if(selector==0&&opcode==14)assert(subtractions==write);
        if(sign||zero) {
            auto compare=graph.nodes.at(sign?condition.args[0]:condition.args[1]);
            if((selector==0&&opcode==14) || selector==1) {lhs=compare.args[0];rhs=compare.args[1];}
            else {
                alu=compare.args[0];
                if(dot&&graph.nodes.at(alu).op==spv::OpCompositeConstruct)alu=graph.nodes.at(alu).args[0];
                if(selector==0) {lhs=graph.nodes.at(alu).args[0];rhs=graph.nodes.at(alu).args[1];}
            }
            if(lhs) {
                assert(graph.widths.at(graph.nodes.at(lhs).type)==(dot?4:width));
                assert(graph.widths.at(graph.nodes.at(rhs).type)==(dot?4:width));
            }
        }
        for(unsigned sample=0;sample<64;++sample) {
            Graph::Value a,b;std::vector<bool> expected;
            constexpr std::array<int,4> ints={-5,0,1,3};
            constexpr std::array<uint32_t,4> bits={0,1,0x80000000,0xffffffff};
            for(unsigned lane=0;lane<(dot?4:width);++lane) {
                a.push_back(selector==0?std::bit_cast<uint32_t>(float(ints[(sample+lane)%4])):bits[(sample+lane)%4]);
                b.push_back(selector==0?std::bit_cast<uint32_t>(float(ints[(sample/4+lane*3)%4])):bits[(sample/4+lane*3)%4]);
                if(sample<16) {
                    const bool high=(sample>>lane)&1;
                    a.back()=selector==0?std::bit_cast<uint32_t>(high?3.f:-5.f):high?0xffffffffu:0u;
                    b.back()=selector==1?1u:0u;
                }
            }
            Graph::Values inputs;
            if(lhs){inputs[lhs]=a;inputs[rhs]=b;}else if(alu)inputs[alu]=a;
            double product=0;
            if(dot)for(unsigned lane=0;lane<4;++lane)product+=double(std::bit_cast<float>(a[lane]))*std::bit_cast<float>(b[lane]);
            for(unsigned lane=0;lane<width;++lane) {
                const bool signed_word=selector==3||(selector==1&&opcode==14);
                double x=selector==0?std::bit_cast<float>(a[lane]):signed_word?double(std::bit_cast<int32_t>(a[lane])):double(a[lane]);
                double y=selector==0?std::bit_cast<float>(b[lane]):signed_word?double(std::bit_cast<int32_t>(b[lane])):double(b[lane]);
                if(selector==3)y=0; // Inject actual SHL result; do not execute a GPU shift.
                if(selector==0&&opcode!=14) {x=dot?product:opcode==2?x+y:x*y;y=0;}
                const bool s=sign==0||(sign==1?x<y:x>=y),z=zero==0||(zero==1?x==y:x!=y);
                expected.push_back(combine?(s&&z):(s||z));
            }
            if(!per_channel && selector==0 && channel>=4)expected={channel==4?
                std::all_of(expected.begin(),expected.end(),[](bool x){return x;}):
                std::any_of(expected.begin(),expected.end(),[](bool x){return x;})};
            for(size_t lane=0;lane<expected.size();++lane) {
                auto current=inputs;const auto value=graph.evaluate(graph.predicate_targets[lane].second,current);
                assert(value.size()==1&&value[0]==expected[lane]);++values_checked;
            }
        }
        const auto name="channels-"+std::to_string(cases++);
        std::ofstream spv(directory/(name+".spv"),std::ios::binary);
        spv.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(spv);
        if(sign==1&&zero==2&&combine==1) {std::ofstream(directory/(name+".metal"))<<shader::metal::convert_spirv(words).source;++msl;}
    };
    for(unsigned precision=0;precision<2;++precision)for(unsigned channel=4;channel<=6;++channel)
    for(unsigned opcode:{2u,8u,13u,14u})for(unsigned write=0;write<2;++write)
    for(unsigned sign=0;sign<3;++sign)for(unsigned zero=0;zero<3;++zero)for(unsigned combine=0;combine<2;++combine)
        check(precision,channel,0,opcode,write,sign,zero,combine);
    for(unsigned precision=0;precision<2;++precision)for(unsigned channel=0;channel<4;++channel)
    for(unsigned write=0;write<2;++write)for(unsigned sign=0;sign<3;++sign)
    for(unsigned zero=0;zero<3;++zero)for(unsigned combine=0;combine<2;++combine)
        check(precision,channel,0,8,write,sign,zero,combine);
    for(unsigned channel=0;channel<8;++channel)for(unsigned kind=0;kind<3;++kind)for(unsigned write=0;write<2;++write)
    for(unsigned sign=0;sign<3;++sign)for(unsigned zero=0;zero<3;++zero)for(unsigned combine=0;combine<2;++combine)
        check(0,channel,kind==2?3:1,kind==2?3:kind==1?14:15,write,sign,zero,combine);
    size_t invalid=0;
    for(unsigned precision=0;precision<2;++precision)for(unsigned opcode:{2u,8u,13u,14u})
    for(unsigned write=0;write<2;++write)for(unsigned pdst=0;pdst<4;++pdst)for(unsigned channel:{6u,7u}) {
        if(channel==6&&pdst==0)continue;
        constexpr std::string_view layout="01001ppps-oydrceavttiizzmhhhnnbbkkffgggggggwlluuuujjjjjjjqqqqqqq";
        auto f=[&](char name,unsigned value){return field(name,value,layout);};
        in.instruction=f('a',precision)|f('h',channel)|f('n',pdst)|f('w',write)|f('u',opcode)|f('j',16)|f('q',32);
        bool rejected=false;try{(void)convert_gxp(in.program,"vtst-invalid-channels",features,Target::SpirVMetal,hints,false);}
        catch(const std::runtime_error &error){rejected=std::string(error.what()).find("Metal USSE PC 0 opcode")!=std::string::npos;}
        assert(rejected);++invalid;
    }
    for(unsigned kind=0;kind<3;++kind)for(unsigned channel=4;channel<8;++channel)for(unsigned write=0;write<2;++write) {
        constexpr std::string_view layout="01001ppps-oydrceavttiizzmhhhnnbbkkffgggggggwlluuuujjjjjjjqqqqqqq";
        auto f=[&](char name,unsigned value){return field(name,value,layout);};
        in.instruction=f('h',channel)|f('w',write)|f('l',kind==2?2:1)|f('u',kind==2?8:kind==1?10:7)|f('j',16)|f('q',32);
        bool rejected=false;try{(void)convert_gxp(in.program,"vtst-packed-channel-gate",features,Target::SpirVMetal,hints,false);}
        catch(const std::runtime_error &error){rejected=std::string(error.what()).find("Metal USSE PC 0 opcode")!=std::string::npos;}
        assert(rejected);++invalid;
    }
    assert(cases==2016&&msl==112&&invalid==136);
    std::cout<<"PASS VTST channels: "<<cases<<" public GXP variants, "<<values_checked
        <<" numerical predicate results; vector AND/OR all and P0..P3 destinations, scalar/vector DOT with full four-lane operands, signed/unsigned full-word 32-bit scalar tests for all selector encodings, optional writeback, 136 reserved/per-channel/packed gates; 112 MSL variants; no hardware oracle\n";
}
template<class Program>
static void disabled_mask_tests(Program &in,const FeatureState &features,const Hints &hints,const std::filesystem::path &directory) {
    // PVR_PSP2 ae4df4b7 useasm.c:4217-4253,4389-4397: only TEST has
    // a predicate destination; WBEN-clear TESTMASK uses a read-only register.
    const auto op_counts=[](const std::vector<uint32_t> &words) {
        std::map<spv::Op,size_t> counts;
        for(size_t at=5;at<words.size();) {
            const auto n=words[at]>>16;assert(n&&at+n<=words.size());
            ++counts[spv::Op(words[at]&65535)];at+=n;
        }
        return counts;
    };
    in.instruction=0;
    for(char bit:std::string("11111----000-----------101--------------------------------------"))
        in.instruction=(in.instruction<<1)|(bit=='1');
    const auto nop=convert_gxp(in.program,"vtst-disabled-mask",features,Target::SpirVMetal,hints,false).spirv;
    auto baseline=op_counts(nop);
    std::array<std::vector<uint32_t>,8> expected;
    size_t cases=0;
    for(unsigned condition=0;condition<8;++condition)
    for(unsigned precision=0;precision<2;++precision)for(unsigned mask=0;mask<3;++mask)
    for(unsigned repeat=0;repeat<4;++repeat)for(unsigned opcode:{2u,8u,13u,14u})for(unsigned dest=0;dest<4;++dest) {
        // Vary ignored destinations, invalid-if-dereferenced source indices,
        // negate/replicate and every repeat count. No operand may be touched.
        in.instruction=field('p',condition)|field('e',precision)|field('a',mask)|field('u',repeat)
            |field('g',opcode)|field('d',dest&1)|field('b',dest)|field('f',127)|field('h',127)|field('j',127)
            |field('t',1)|field('v',1)|field('i',2)|field('z',2)|field('m',1);
        const auto words=convert_gxp(in.program,"vtst-disabled-mask",features,Target::SpirVMetal,hints,false).spirv;
        auto counts=op_counts(words);
        assert(counts[spv::OpStore]==baseline[spv::OpStore]);
        assert(counts[spv::OpLoad]==baseline[spv::OpLoad]);
        for(auto op:{spv::OpFAdd,spv::OpFSub,spv::OpFMul,spv::OpDot,spv::OpSelect,
            spv::OpFOrdEqual,spv::OpFOrdLessThan,spv::OpFOrdGreaterThanEqual,spv::OpLogicalAnd,spv::OpLogicalOr})
            assert(counts[op]==baseline[op]);
        assert(Graph(words).predicate_stores.empty());
        if(expected[condition].empty())expected[condition]=words;
        else assert(words==expected[condition]);
        assert(words==nop);
        assert(get_predicate(in.instruction,true)==0);
        assert(get_predicate(in.instruction,false)==condition);
        assert(get_predicate(in.instruction|(uint64_t(1)<<20),true)==condition);
        ++cases;
    }
    for(unsigned condition=0;condition<1;++condition) {
        const auto name="disabled-mask-"+std::to_string(condition);
        std::ofstream spv(directory/(name+".spv"),std::ios::binary);
        spv.write(reinterpret_cast<const char *>(expected[condition].data()),expected[condition].size()*4);assert(spv);
        std::ofstream(directory/(name+".metal"))<<shader::metal::convert_spirv(expected[condition]).source;
    }
    size_t invalid=0;
    for(unsigned precision=0;precision<2;++precision)for(unsigned repeat=0;repeat<4;++repeat)
    for(unsigned mask=0;mask<4;++mask)for(unsigned sign=0;sign<4;++sign)for(unsigned zero=0;zero<4;++zero) {
        if(mask<3&&sign<3&&zero<3)continue;
        in.instruction=field('e',precision)|field('a',mask)|field('u',repeat)|field('g',8)
            |field('i',sign)|field('z',zero);
        bool rejected=false;try{(void)convert_gxp(in.program,"vtst-disabled-reserved",features,Target::SpirVMetal,hints,false);}
        catch(const std::runtime_error &error){rejected=std::string(error.what()).find("Metal USSE PC 0 opcode")!=std::string::npos;}
        assert(rejected);++invalid;
    }
    assert(cases==3072&&invalid==296);
    std::cout<<"PASS VTST disabled masks: "<<cases<<" public GXP variants; all repeat counts, mask formats, all eight encoded predicates including PN and ignored operands; zero added guest stores, operand loads, ALU or predicate writes; all IR byte-identical to NOP, one distinct module and no execution predicate loads; "<<invalid<<" reserved encodings rejected; no device/GPU/app\n";
}
// Repeat encoding: PVR useasm.c EncodeTestInstruction_Repeat (VEC34).
// Compare the whole public translation to independently expanded single
// instructions. This checks register offsets/ordering, not Vita hardware.
static void word_mask_repeats(const FeatureState &features,const Hints &hints,
    const std::filesystem::path &directory) {
    struct Input {SceGxmProgram program;SceGxmProgramVertexVaryings varyings;std::array<uint64_t,4> instructions;}in{};
    in.program.magic=0x00505847;in.program.size=sizeof(in);
    in.program.varyings_offset=offsetof(Input,varyings)-offsetof(SceGxmProgram,varyings_offset);
    in.program.primary_program_offset=offsetof(Input,instructions)-offsetof(SceGxmProgram,primary_program_offset);
    in.program.secondary_program_offset=offsetof(Input,instructions)-offsetof(SceGxmProgram,secondary_program_offset);
    in.program.secondary_program_offset_end=offsetof(Input,instructions)-offsetof(SceGxmProgram,secondary_program_offset_end);
    in.program.temp_reg_count1=128;
    auto write=[&](const std::string &name,const std::vector<uint32_t>&words){
        std::ofstream out(directory/(name+".spv"),std::ios::binary);
        out.write(reinterpret_cast<const char*>(words.data()),words.size()*4);assert(out);
    };
    size_t cases=0;
    for(unsigned predicate:{0u,1u,5u})for(unsigned opcode:{0u,1u,2u})
    for(unsigned repeat=0;repeat<4;++repeat)for(unsigned overlap=0;overlap<2;++overlap) {
        const unsigned destination=overlap?33:0;
        auto instruction=[&](unsigned count,unsigned step){
            return field('p',predicate)|field('u',count)|field('w',1)|field('l',3)|field('g',opcode)
                |field('a',0)|field('i',1)|field('z',2)|field('m',1)
                |field('f',destination+step)|field('h',32+step)|field('j',64+step);
        };
        in.instructions[0]=instruction(repeat,0);in.program.primary_program_instr_count=1;
        const auto repeated=convert_gxp(in.program,"vtst-word-repeats",features,Target::SpirVMetal,hints,false).spirv;
        for(unsigned step=0;step<=repeat;++step)in.instructions[step]=instruction(0,step);
        in.program.primary_program_instr_count=repeat+1;
        const auto single=convert_gxp(in.program,"vtst-word-repeats",features,Target::SpirVMetal,hints,false).spirv;
        const auto name="wordmask-"+std::to_string(cases++);
        write(name+"-repeated",repeated);write(name+"-unrolled",single);
        if(repeat==3&&predicate==0)
            std::ofstream(directory/(name+".metal"))<<shader::metal::convert_spirv(repeated).source;
    }
    assert(cases==72);
    std::cout<<"PASS VTST word-mask repeat generation: 72 public pairs, AND/OR/XOR, repeats 1..4, NONE/P0/!P0 and disjoint/dependent overlapping register sequences; whole-IR comparison performed by CPU runner, no hardware oracle\n";
}
int main(int argc,char **argv) {
    assert(argc==2);spdlog::set_level(spdlog::level::off);const std::filesystem::path directory=argv[1];
    predicate_arithmetic(directory);
    struct Input{SceGxmProgram program;SceGxmProgramVertexVaryings varyings;uint64_t instruction;}in{};
    in.program.magic=0x00505847;in.program.size=sizeof(in);
    in.program.varyings_offset=offsetof(Input,varyings)-offsetof(SceGxmProgram,varyings_offset);
    in.program.primary_program_offset=offsetof(Input,instruction)-offsetof(SceGxmProgram,primary_program_offset);
    in.program.primary_program_instr_count=1;
    in.program.secondary_program_offset=offsetof(Input,instruction)-offsetof(SceGxmProgram,secondary_program_offset);
    in.program.secondary_program_offset_end=offsetof(Input,instruction)-offsetof(SceGxmProgram,secondary_program_offset_end);
    in.program.temp_reg_count1=128;
    std::vector<SceGxmVertexAttribute> attributes;Hints hints{};hints.attributes=&attributes;
    hints.color_format=SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;FeatureState features{};size_t cases=0;
    analyzer_tests(features,hints,directory);
    word_mask_repeats(features,hints,directory);
    channel_tests(in,features,hints,directory);
    disabled_mask_tests(in,features,hints,directory);
    for(unsigned precision=0;precision<2;++precision)for(unsigned mask=0;mask<3;++mask)
    for(unsigned sign=0;sign<3;++sign)for(unsigned zero=0;zero<3;++zero)
    for(unsigned replicate=0;replicate<2;++replicate)for(unsigned negative=0;negative<2;++negative)
    for(unsigned combine=0;combine<2;++combine) {
        in.instruction=field('e',precision)|field('a',mask)|field('w',1)|field('g',8)|field('z',zero)
            |field('i',sign)|field('v',replicate)|field('t',negative)|field('h',16)|field('j',32)|field('m',combine);
        const auto words=convert_gxp(in.program,"vtst-dot-masks",features,Target::SpirVMetal,hints,false).spirv;
        const Graph graph(words);unsigned selects=0;
        for(const auto &[id,node]:graph.nodes)if(node.op==spv::OpSelect) {
            const auto &condition=graph.nodes.at(node.args[0]);
            const auto logic=condition.op==spv::OpCompositeExtract?graph.nodes.find(condition.args[0]):graph.nodes.find(node.args[0]);
            if(logic==graph.nodes.end() || (logic->second.op!=spv::OpLogicalAnd&&logic->second.op!=spv::OpLogicalOr))continue;
            ++selects;
            if(graph.widths.at(condition.type)!=(mask==2?4:1)) {
                std::cerr<<"predicate width mismatch: precision="<<precision<<" mask="<<mask<<" sign="<<sign
                    <<" zero="<<zero<<" combine="<<combine<<" op="<<condition.op<<" width="<<graph.widths.at(condition.type)<<'\n';
                assert(false);
            }
            assert(graph.widths.at(node.type)==(mask==2?4:1));
            if(mask!=2) {
                assert(condition.op==spv::OpCompositeExtract && condition.args[1]==0);
                graph.check_dot_lanes(condition.args[0]);
            } else graph.check_dot_lanes(node.args[0]);
            uint32_t dot=0;for(const auto &[other,n]:graph.nodes)if(n.op==spv::OpDot){assert(!dot);dot=other;}
            assert(dot);
            for(uint32_t bits:{0xff800000u,0xc0400000u,0x80000001u,0x80000000u,0u,1u,0x40400000u,0x7f800000u,0x7fc12345u}) {
                Graph::Values values{{dot,{bits}}};const auto predicates=graph.evaluate(node.args[0],values);
                const float value=std::bit_cast<float>(bits);
                const bool sign_pass=sign==0|| (sign==1?value<0:value>=0);
                const bool zero_pass=zero==0|| (zero==1?value==0:!std::isnan(value)&&value!=0);
                const bool expected=combine?(sign_pass&&zero_pass):(sign_pass||zero_pass);
                for(auto predicate:predicates)assert(predicate==expected);
            }

            if(mask==0) {
                assert(graph.nodes.at(node.args[1]).args.at(0)==(precision?0xffffffffu:0xffffu));
                assert(graph.nodes.at(node.args[2]).args.at(0)==0);
                if(precision) {
                    bool bitcast=false;for(const auto &[other,n]:graph.nodes)if(n.op==spv::OpBitcast && n.args[0]==id)bitcast=true;
                    assert(bitcast);
                }
            } else if(mask==1) {
                assert(graph.nodes.at(node.args[1]).args.at(0)==0x3f800000);
                assert(graph.nodes.at(node.args[2]).args.at(0)==0);
            }
        }
        assert(selects==1);
        const auto name="dot-"+std::to_string(cases++);
        std::ofstream spv(directory/(name+".spv"),std::ios::binary);
        spv.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(spv);
        std::ofstream(directory/(name+".metal"))<<shader::metal::convert_spirv(words).source;
    }
    assert(cases==432);
    size_t predicate_cases=0,predicate_values=0;
    constexpr std::array<uint32_t,9> predicate_edges={0xff800000,0xc0400000,0x80000001,0x80000000,
        0,1,0x40400000,0x7f800000,0x7fc12345};
    for(unsigned precision=0;precision<2;++precision)for(unsigned channel=0;channel<4;++channel)
    for(unsigned sign=0;sign<3;++sign)for(unsigned zero=0;zero<3;++zero)
    for(unsigned combine=0;combine<2;++combine)for(unsigned write=0;write<2;++write) {
        constexpr std::string_view layout="01001ppps-oydrceavttiizzmhhhnnbbkkffgggggggwlluuuujjjjjjjqqqqqqq";
        auto f=[&](char name,unsigned value){return field(name,value,layout);};
        in.instruction=f('a',precision)|f('h',channel)|f('w',write)|f('u',14)
            |f('i',sign)|f('z',zero)|f('j',16)|f('q',32)|f('m',combine);
        const auto words=convert_gxp(in.program,"vtst-predicates",features,Target::SpirVMetal,hints,false).spirv;
        const Graph graph(words);assert(graph.predicate_stores.size()==1);
        const auto root=graph.predicate_stores[0];assert(graph.widths.at(graph.nodes.at(root).type)==1);
        const auto &logic=graph.nodes.at(root);assert(logic.op==(combine?spv::OpLogicalAnd:spv::OpLogicalOr));
        uint32_t lhs=0,rhs=0;unsigned subtractions=0;
        for(const auto &[id,node]:graph.nodes) {
            if(node.op==spv::OpFSub)++subtractions;
            if(id!=logic.args[0] && id!=logic.args[1])continue;
            if(node.op==spv::OpFOrdEqual||node.op==spv::OpFOrdNotEqual
                ||node.op==spv::OpFOrdLessThan||node.op==spv::OpFOrdGreaterThanEqual) {
                if(lhs)assert(lhs==node.args[0]&&rhs==node.args[1]);
                lhs=node.args[0];rhs=node.args[1];
            }
        }
        if(subtractions!=write || bool(lhs)!=bool(sign||zero)) {
            std::cerr<<"VTST mismatch precision="<<precision<<" channel="<<channel<<" sign="<<sign<<" zero="<<zero
                <<" write="<<write<<" subtractions="<<subtractions<<" lhs="<<lhs<<'\n';assert(false);
        }
        for(auto a:predicate_edges)for(auto v:predicate_edges) {
            Graph::Values values;if(lhs){values[lhs]={a};values[rhs]={v};}
            const auto result=graph.evaluate(root,values);assert(result.size()==1);
            const double x=std::bit_cast<float>(a),y=std::bit_cast<float>(v);
            const bool ordered=!std::isnan(x)&&!std::isnan(y);
            const bool s=sign==0||(ordered&&(sign==1?x<y:x>=y));
            const bool z=zero==0||(ordered&&(zero==1?x==y:x!=y));
            assert(result[0]==(combine?(s&&z):(s||z)));++predicate_values;
        }
        const auto name="predicate-"+std::to_string(predicate_cases++);
        std::ofstream spv(directory/(name+".spv"),std::ios::binary);
        spv.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(spv);
        std::ofstream(directory/(name+".metal"))<<shader::metal::convert_spirv(words).source;
    }
    assert(predicate_cases==288 && predicate_values==23328);
    std::cout<<"PASS VTST public predicates: "<<predicate_cases<<" decoded GXP variants and "<<predicate_values
        <<" emitted scalar predicate evaluations; F16/F32, all four selected channels, sign/zero/AND/OR combinations, preserved optional subtraction writeback; no hardware oracle\n";
    size_t shift_cases=0,shift_values=0;
    for(bool mask:{false,true})for(unsigned sign=0;sign<3;++sign)for(unsigned zero=0;zero<3;++zero)
    for(unsigned combine=0;combine<2;++combine)for(unsigned write=0;write<2;++write) {
        if(mask&&!write)continue; // Existing unsupported write-disabled mask.
        if(mask)in.instruction=field('l',3)|field('g',3)|field('w',1)|field('h',16)|field('j',32)
            |field('i',sign)|field('z',zero)|field('m',combine);
        else {
            constexpr std::string_view layout="01001ppps-oydrceavttiizzmhhhnnbbkkffgggggggwlluuuujjjjjjjqqqqqqq";
            auto f=[&](char name,unsigned value){return field(name,value,layout);};
            in.instruction=f('l',3)|f('u',3)|f('w',write)|f('j',16)|f('q',32)
                |f('i',sign)|f('z',zero)|f('m',combine);
        }
        const auto words=convert_gxp(in.program,"vtst-shift-sign",features,Target::SpirVMetal,hints,false).spirv;
        const Graph graph(words);uint32_t root=0,shift=0;
        for(const auto &[id,node]:graph.nodes) {
            if(node.op==spv::OpShiftLeftLogical){assert(!shift);shift=id;}
            if(node.op==spv::OpLogicalAnd||node.op==spv::OpLogicalOr){assert(!root);root=id;}
        }
        assert(root&&shift&&graph.widths.at(graph.nodes.at(root).type)==1);
        for(auto bits:{0u,1u,0x7fffffffu,0x80000000u,0xffffffffu}) {
            Graph::Values values{{shift,{bits}}};const auto result=graph.evaluate(root,values);assert(result.size()==1);
            const bool negative=bool(bits&0x80000000u),is_zero=bits==0;
            const bool s=sign==0||(sign==1?negative:!negative),z=zero==0||(zero==1?is_zero:!is_zero);
            assert(result[0]==(combine?(s&&z):(s||z)));++shift_values;
        }
        const auto name="shift-"+std::to_string(shift_cases++);
        std::ofstream spv(directory/(name+".spv"),std::ios::binary);
        spv.write(reinterpret_cast<const char *>(words.data()),words.size()*4);assert(spv);
        std::ofstream(directory/(name+".metal"))<<shader::metal::convert_spirv(words).source;
    }
    assert(shift_cases==54&&shift_values==270);
    std::cout<<"PASS VTST shift sign: "<<shift_cases<<" public VTST/VTSTMSK variants and "<<shift_values
        <<" emitted predicates from injected 32-bit ALU results; high-bit signed view, nonnegative zero, independent zero test and AND/OR preserved; no GPU/shift execution\n";
    // The SGX disassembler marks sign/zero encoding 3 as reserved:
    // PVR_PSP2 ae4df4b7 tools/intern/useasm/usedisasm.c:8714-8726.
    // Test both public instruction decoders, not only a helper's return value.
    size_t reserved=0;
    for(bool mask:{false,true})for(unsigned precision=0;precision<2;++precision)
    for(unsigned channel=0;channel<(mask?3u:4u);++channel)
    for(unsigned sign=0;sign<4;++sign)for(unsigned zero=0;zero<4;++zero) {
        if(sign!=3 && zero!=3)continue;
        if(mask) {
            in.instruction=field('e',precision)|field('a',channel)|field('w',1)|field('g',8)
                |field('i',sign)|field('z',zero)|field('h',16)|field('j',32);
        } else {
            constexpr std::string_view layout="01001ppps-oydrceavttiizzmhhhnnbbkkffgggggggwlluuuujjjjjjjqqqqqqq";
            auto f=[&](char name,unsigned value){return field(name,value,layout);};
            in.instruction=f('a',precision)|f('h',channel)|f('w',1)|f('u',8)
                |f('i',sign)|f('z',zero)|f('j',16)|f('q',32);
        }
        bool rejected=false;
        try {(void)convert_gxp(in.program,"vtst-reserved",features,Target::SpirVMetal,hints,false);}
        catch(const std::runtime_error &error) {
            rejected=std::string(error.what()).find("Metal USSE PC 0 opcode")!=std::string::npos;
        }
        assert(rejected);++reserved;
    }
    assert(reserved==98);
    std::cout<<"PASS VTST DOT masks: "<<cases
        <<" public GXP variants, scalar/vector predicate shape, identical-lane extraction proof, F16/F32 raw true bits and F32 bitcast, numeric 0/1 masks, source replication/negation and AND/OR comparison truth tables at nine DOT boundary values; "<<reserved
        <<" reserved sign/zero variants rejected through both public decoders; no hardware oracle\n";
}
