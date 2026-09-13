// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <shader/msl_recompiler.h>
#include <spirv.hpp>
#include <iostream>
#include <stdexcept>

// Instruction fragments exercise the conservative analysis, not SPIR-V
// validation. Actual translated GXP is additionally checked on the GPU.
int main() {
    try {
        std::vector<uint32_t> base={spv::MagicNumber,0x10500,0,256,0};
        auto emit=[](auto &words,spv::Op op,std::initializer_list<uint32_t> args) {
            words.push_back(uint32_t(args.size()+1)<<16|uint32_t(op));words.insert(words.end(),args);
        };
        emit(base,spv::OpTypeFloat,{30,32});
        emit(base,spv::OpTypePointer,{40,spv::StorageClassFunction,30});
        emit(base,spv::OpTypePointer,{41,spv::StorageClassPhysicalStorageBuffer,30});
        emit(base,spv::OpVariable,{40,60,spv::StorageClassFunction});
        emit(base,spv::OpConvertUToPtr,{41,61,70});
        auto check=[&](const char *name,auto words,bool expected) {
            if(shader::metal::writes_external_memory(words)!=expected) throw std::runtime_error(name);
            std::cout<<"PASS memory analysis: "<<name<<'\n';
        };
        auto words=base;emit(words,spv::OpLoad,{30,80,61});emit(words,spv::OpStore,{60,80});
        check("external read and local write",words,false);
        words=base;emit(words,spv::OpCapability,{61});emit(words,spv::OpConstant,{30,80,61});
        check("declaration literals are not pointer operands",words,false);
        words=base;emit(words,spv::OpExtInst,{30,80,90,61,81});
        check("extended instruction number is not a pointer operand",words,false);
        words=base;emit(words,spv::OpStore,{61,80});check("physical store",words,true);
        words=base;emit(words,spv::OpCopyObject,{41,62,61});emit(words,spv::OpStore,{62,80});check("copied physical pointer store",words,true);
        words=base;emit(words,spv::OpPhi,{41,62,61,90,61,91});emit(words,spv::OpStore,{62,80});check("phi physical pointer store",words,true);
        words=base;emit(words,spv::OpFunctionParameter,{41,62});emit(words,spv::OpStore,{62,80});check("parameter physical pointer store",words,true);
        words=base;emit(words,spv::OpCopyMemory,{61,60});check("external copy destination",words,true);
        words=base;emit(words,spv::OpAtomicIAdd,{30,80,61,70,70,80});check("external atomic",words,true);
        words=base;emit(words,spv::OpExtInst,{30,80,90,42,61});check("pointer-consuming extension",words,true);
        words=base;emit(words,spv::OpFunctionCall,{30,80,90,61});check("escaped external pointer",words,true);
        words=base;emit(words,spv::OpStore,{199,80});check("unknown store destination",words,true);
        words=base;emit(words,spv::OpDecorate,{42,spv::DecorationBufferBlock});check("legacy storage buffer",words,true);
        words=base;words.push_back(0);
        bool rejected=false;try {shader::metal::writes_external_memory(words);} catch(const std::invalid_argument &) {rejected=true;}
        if(!rejected) throw std::runtime_error("Malformed instruction was accepted");
        std::cout<<"PASS malformed analysis input rejected\n";
        return 0;
    } catch(const std::exception &error) {std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}
}
