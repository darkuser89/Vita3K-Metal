// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include <shader/metal4/usse_decoder_helpers.h>
#include <shader/metal4/usse_disasm.h>
#include <shader/metal4/usse_translator.h>
#include <shader/metal4/usse_types.h>
#include <util/log.h>

using namespace shader::metal3;
using namespace usse;

static Opcode decode_test_inst(const Imm2 alu_sel, const Imm4 alu_op, const Imm1 prec, DataType &filled_dt) {
    static const Opcode vf16_opcode[16] = {
        Opcode::INVALID,
        Opcode::INVALID,
        Opcode::VF16ADD,
        Opcode::VF16FRC,
        Opcode::VRCP,
        Opcode::VRSQ,
        Opcode::VLOG,
        Opcode::VEXP,
        Opcode::VF16DP,
        Opcode::VF16MIN,
        Opcode::VF16MAX,
        Opcode::VF16DSX,
        Opcode::VF16DSY,
        Opcode::VF16MUL,
        Opcode::VF16SUB,
        Opcode::INVALID,
    };

    static const Opcode vf32_opcode[16] = {
        Opcode::INVALID,
        Opcode::INVALID,
        Opcode::VADD,
        Opcode::VFRC,
        Opcode::VRCP,
        Opcode::VRSQ,
        Opcode::VLOG,
        Opcode::VEXP,
        Opcode::VDP,
        Opcode::VMIN,
        Opcode::VMAX,
        Opcode::VDSX,
        Opcode::VDSY,
        Opcode::VMUL,
        Opcode::VSUB,
        Opcode::INVALID,
    };

    static const Opcode test_ops[4][16] = {
        {
            Opcode::FADD,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::FSUBFLR,
            Opcode::FRCP,
            Opcode::FRSQ,
            Opcode::FLOG,
            Opcode::FEXP,
            Opcode::FDP,
            Opcode::FMIN,
            Opcode::FMAX,
            Opcode::FDSX,
            Opcode::FDSY,
            Opcode::FMUL,
            Opcode::FSUB,
            Opcode::INVALID,
        },
        { Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::IADD16,
            Opcode::ISUB16,
            Opcode::IMUL16,
            Opcode::IADDU16,
            Opcode::ISUBU16,
            Opcode::IMULU16,
            Opcode::IADD32,
            Opcode::IADDU32,
            Opcode::ISUB32,
            Opcode::ISUBU32 },
        { Opcode::IADD8,
            Opcode::ISUB8,
            Opcode::IADDU8,
            Opcode::ISUBU8,
            Opcode::IMUL8,
            Opcode::FPMUL8,
            Opcode::IMULU8,
            Opcode::FPADD8,
            Opcode::FPSUB8,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID },
        { Opcode::AND,
            Opcode::OR,
            Opcode::XOR,
            Opcode::SHL,
            Opcode::SHR,
            Opcode::ROL,
            Opcode::INVALID,
            Opcode::ASR,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID,
            Opcode::INVALID }
    };

    if (alu_sel == 0) {
        switch (prec) {
        case 0: {
            filled_dt = DataType::F16;
            return vf16_opcode[alu_op];
        }

        case 1: {
            filled_dt = DataType::F32;
            return vf32_opcode[alu_op];
        }

        default: return Opcode::INVALID;
        }
    }

    const auto test_op = test_ops[alu_sel][alu_op];

    switch (test_op) {
    case Opcode::IADDU32:
    case Opcode::ISUBU32: {
        filled_dt = DataType::UINT32;
        break;
    }
    case Opcode::ISUB32:
    case Opcode::IADD32: {
        filled_dt = DataType::INT32;
        break;
    }
    case Opcode::IADD16:
    case Opcode::ISUB16:
    case Opcode::IMUL16: {
        filled_dt = DataType::INT16;
        break;
    }
    case Opcode::IADDU16:
    case Opcode::ISUBU16:
    case Opcode::IMULU16: {
        filled_dt = DataType::UINT16;
        break;
    }
    case Opcode::IADD8:
    case Opcode::ISUB8:
    case Opcode::IMUL8: {
        filled_dt = DataType::INT8;
        break;
    }
    case Opcode::IADDU8:
    case Opcode::ISUBU8:
    case Opcode::FPSUB8:
    case Opcode::IMULU8: {
        filled_dt = DataType::UINT8;
        break;
    }
    default: {
        break;
    }
    }

    if (alu_sel == 2) {
        filled_dt = DataType::UINT8;
    }

    if (alu_sel == 3) {
        filled_dt = DataType::UINT32;
    }

    return test_op;
}

inline static bool is_sub_opcode(Opcode test_op) {
    return (test_op == Opcode::VSUB) || (test_op == Opcode::VF16SUB) || (test_op == Opcode::ISUB8) || (test_op == Opcode::ISUB16) || (test_op == Opcode::ISUB32) || (test_op == Opcode::ISUBU8) || (test_op == Opcode::ISUBU16) || (test_op == Opcode::ISUBU32) || (test_op == Opcode::FPSUB8);
}

// Retain the existing ordered floating-point and signed/unsigned comparison
// domains. CRCOMB combines independent sign and zero tests, not channels.
// A disabled test contributes true. The positive sign test includes zero;
// combining it with NONZERO/AND yields the ordinary strict positive test.
static spv::Id test_predicate(spv::Builder &b, spv::Id lhs, spv::Id rhs,
    unsigned domain, unsigned sign, unsigned zero, bool combine_and) {
    if (domain >= 3 || sign >= 3 || zero >= 3) return spv::NoResult;
    const auto count = b.getNumComponents(lhs);
    const auto type = count > 1 ? b.makeVectorType(b.makeBoolType(), count) : b.makeBoolType();
    const auto always = [&] {
        const auto value = b.makeBoolConstant(true);
        return count > 1 ? b.makeCompositeConstant(type, std::vector<spv::Id>(count, value)) : value;
    };
    static constexpr spv::Op negative[] = {spv::OpFOrdLessThan, spv::OpSLessThan, spv::OpULessThan};
    static constexpr spv::Op positive[] = {spv::OpFOrdGreaterThanEqual, spv::OpSGreaterThanEqual, spv::OpUGreaterThanEqual};
    static constexpr spv::Op equal[] = {spv::OpFOrdEqual, spv::OpIEqual, spv::OpIEqual};
    static constexpr spv::Op unequal[] = {spv::OpFOrdNotEqual, spv::OpINotEqual, spv::OpINotEqual};
    const auto sign_result = sign ? b.createBinOp(sign == 1 ? negative[domain] : positive[domain], type, lhs, rhs) : always();
    const auto zero_result = zero ? b.createBinOp(zero == 1 ? equal[domain] : unequal[domain], type, lhs, rhs) : always();
    return b.createBinOp(combine_and ? spv::OpLogicalAnd : spv::OpLogicalOr, type, sign_result, zero_result);
}

spv::Id USSETranslatorVisitor::vtst_impl(Instruction inst, ExtPredicate pred, int zero_test, int sign_test, Imm4 load_mask, bool mask, bool test_crcomb_and, spv::Id *alu_result, int src1_repeat_offset, int src2_repeat_offset) {
    if (m_spirv_params.native_metal && (zero_test >= 3 || sign_test >= 3)) {
        LOG_ERROR("Reserved test comparison: sign {} zero {}", sign_test, zero_test);
        return spv::NoResult;
    }
    // Usually we would expect this to have a compare behavior
    // Comparison is done by subtracting the first src by the second src, and compare the result value.
    // We currently optimize for that case first
    const DataType load_data_type = inst.opr.src1.type;
    const DataType store_data_type = inst.opr.dest.type;

    const unsigned components = std::popcount(load_mask);

    static const spv::Op tb_comp_ops[3][2][4] = {
        { { spv::OpFOrdNotEqual,
              spv::OpFOrdLessThan,
              spv::OpFOrdGreaterThan,
              spv::OpAll },
            { spv::OpFOrdEqual,
                spv::OpFOrdLessThanEqual,
                spv::OpFOrdGreaterThanEqual,
                spv::OpAll } },
        { { spv::OpINotEqual,
              spv::OpSLessThan,
              spv::OpSGreaterThan,
              spv::OpAll },
            { spv::OpIEqual,
                spv::OpSLessThanEqual,
                spv::OpSGreaterThanEqual,
                spv::OpAll } },
        { { spv::OpINotEqual,
              spv::OpULessThan,
              spv::OpUGreaterThan,
              spv::OpAll },
            { spv::OpIEqual,
                spv::OpULessThanEqual,
                spv::OpUGreaterThanEqual,
                spv::OpAll } },
    };

    // Load our compares
    spv::Id lhs = spv::NoResult;
    spv::Id rhs = spv::NoResult;

    const spv::Id pred_type = utils::make_vector_or_scalar_type(m_b, m_b.makeBoolType(), components);

    // Retain the old table only for the non-native target. Native Metal
    // evaluates both tests and CRCOMB independently below, including the
    // zero result of signed shifts in its nonnegative sign test.
    const bool compare_include_equal = (zero_test == 1);

    // Sign test number

    int index_tb_comp = 0;
    if (is_signed_integer_data_type(load_data_type)) {
        index_tb_comp = 1;
    } else if (is_unsigned_integer_data_type(load_data_type)) {
        // A 32-bit ALU result uses its high bit for the sign test. Direct
        // unsigned subtraction comparisons still compare the two operands.
        index_tb_comp = m_spirv_params.native_metal && load_data_type == DataType::UINT32 && !is_sub_opcode(inst.opcode) ? 1 : 2;
    }

    const spv::Op used_comp_op = tb_comp_ops[index_tb_comp][compare_include_equal][sign_test];

    // Optimize this case. Alternative name is CMP.
    const char *tb_comp_str[2][4] = {
        { "ne",
            "lt",
            "gt",
            "inv" },
        { "equal",
            "le",
            "ge",
            "inv" }
    };

    const char *used_comp_str = m_spirv_params.native_metal
        ? (test_crcomb_and ? "sign_and_zero" : "sign_or_zero")
        : tb_comp_str[compare_include_equal][sign_test];
    if (m_spirv_params.native_metal)
        LOG_DISASM("{:016x}: test sign={} zero={} crcomb={}", m_instr, sign_test, zero_test, test_crcomb_and ? "and" : "or");

    m_b.setDebugSourceLocation(m_recompiler.cur_pc, nullptr);

    if (is_sub_opcode(inst.opcode)) {
        if (mask) {
            LOG_DISASM("{:016x}: {}{}.{}.{}.{} {} {} {}", m_instr, disasm::e_predicate_str(pred), "CMPMSK", used_comp_str, disasm::data_type_str(store_data_type),
                disasm::data_type_str(load_data_type), disasm::operand_to_str(inst.opr.dest, load_mask), disasm::operand_to_str(inst.opr.src1, load_mask),
                disasm::operand_to_str(inst.opr.src2, load_mask));
        } else {
            LOG_DISASM("{:016x}: {}{}.{}.{} p{} {} {}", m_instr, disasm::e_predicate_str(pred), "CMP", used_comp_str, disasm::data_type_str(load_data_type),
                inst.opr.dest.num, disasm::operand_to_str(inst.opr.src1, load_mask), disasm::operand_to_str(inst.opr.src2, load_mask));
        }

        lhs = load(inst.opr.src1, load_mask, src1_repeat_offset);
        rhs = load(inst.opr.src2, load_mask, src2_repeat_offset);
    } else {
        if (mask) {
            LOG_DISASM("{:016x}: {}{}.{}zero.{}.{} {} {} {}", m_instr, disasm::e_predicate_str(pred), disasm::opcode_str(inst.opcode), used_comp_str, disasm::data_type_str(store_data_type),
                disasm::data_type_str(load_data_type), disasm::operand_to_str(inst.opr.dest, load_mask), disasm::operand_to_str(inst.opr.src1, load_mask), disasm::operand_to_str(inst.opr.src2, load_mask));
        } else {
            LOG_DISASM("{:016x}: {}{}.{}zero.{} p{} {} {}", m_instr, disasm::e_predicate_str(pred), disasm::opcode_str(inst.opcode), used_comp_str, disasm::data_type_str(load_data_type),
                inst.opr.dest.num, disasm::operand_to_str(inst.opr.src1, load_mask), disasm::operand_to_str(inst.opr.src2, load_mask));
        }

        const bool dot = inst.opcode == Opcode::VDP || inst.opcode == Opcode::VF16DP;
        lhs = do_alu_op(inst, m_spirv_params.native_metal && dot ? 0b1111 : load_mask,
            components > 1 ? load_mask : 0b1, src1_repeat_offset, src2_repeat_offset);

        const spv::Id c0_type = utils::make_vector_or_scalar_type(m_b, m_b.makeFloatType(32), components);
        spv::Id c0 = utils::make_uniform_vector_from_type(m_b, c0_type, 0.0f);

        if (is_signed_integer_data_type(load_data_type)) {
            c0 = utils::make_uniform_vector_from_type(m_b,
                utils::make_vector_or_scalar_type(m_b, m_b.makeIntType(32), components), 0);
        } else if (is_unsigned_integer_data_type(load_data_type)) {
            c0 = utils::make_uniform_vector_from_type(m_b,
                utils::make_vector_or_scalar_type(m_b, m_b.makeUintType(32), components), 0);
        }

        rhs = c0;
    }

    if (lhs == spv::NoResult || rhs == spv::NoResult) {
        LOG_ERROR("Source not loaded (lhs: {}, rhs: {})", lhs, rhs);
        return spv::NoResult;
    }

    if (alu_result) {
        // The comparison may bypass subtraction, but writeback cannot.
        *alu_result = is_sub_opcode(inst.opcode)
            ? m_b.createBinOp(is_float_data_type(load_data_type) ? spv::OpFSub : spv::OpISub, m_b.getTypeId(lhs), lhs, rhs)
            : lhs;
    }
    if (m_spirv_params.native_metal)
        return test_predicate(m_b, lhs, rhs, index_tb_comp, sign_test, zero_test, test_crcomb_and);
    return m_b.createOp(used_comp_op, pred_type, { lhs, rhs });
}

bool USSETranslatorVisitor::vtst(
    ExtPredicate pred,
    Imm1 skipinv,
    Imm1 onceonly,
    Imm1 syncstart,
    Imm1 dest_ext,
    Imm1 src1_neg,
    Imm1 src1_ext,
    Imm1 src2_ext,
    Imm1 prec,
    Imm1 src2_vscomp,
    RepeatCount rpt_count,
    Imm2 sign_test,
    Imm2 zero_test,
    Imm1 test_crcomb_and,
    Imm3 chan_cc,
    Imm2 pdst_n,
    Imm2 dest_bank,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm1 test_wben,
    Imm2 alu_sel,
    Imm4 alu_op,
    Imm7 src1_n,
    Imm7 src2_n) {
    DataType load_data_type;
    Opcode test_op = decode_test_inst(alu_sel, alu_op, prec, load_data_type);

    if (test_op == Opcode::INVALID) {
        LOG_ERROR("Unsupported comparision: {} {}", alu_sel, alu_op);
        return false;
    }

    Instruction inst;
    inst.opcode = test_op;

    const bool vector_alu = alu_sel == 0;
    const bool scalar_word = m_spirv_params.native_metal
        && (load_data_type == DataType::INT32 || load_data_type == DataType::UINT32);
    const bool per_channel = m_spirv_params.native_metal && vector_alu && chan_cc == 6;
    if ((!m_spirv_params.native_metal && chan_cc >= 4)
        || (!vector_alu && !scalar_word && chan_cc >= 4)
        || (vector_alu && chan_cc == 7) || (per_channel && pdst_n != 0)) {
        LOG_ERROR("Unsupported comparision with requested channels ordinal: {}", chan_cc);
        return false;
    }

    // Vector tests use 4/5 for all-channel AND/OR and 6 for P0..P3.
    // A 32-bit scalar ALU has one full-word result, not four neighboring
    // registers: its sign-only test reads bit 31 (PVR usc2/dce.c).
    // Packed 8/16-bit reductions need their own lane contract.
    const Imm4 load_mask = scalar_word ? 0b0001
        : chan_cc < 4 ? Imm4(1u << chan_cc) : 0b1111;

    const bool use_double_reg = alu_sel == 0;
    const uint8_t bits_max = use_double_reg ? 8 : 7;

    // Build up source
    inst.opr.src1 = decode_src12(inst.opr.src1, src1_n, src1_bank, src1_ext, use_double_reg, bits_max, m_second_program);
    inst.opr.src2 = decode_src12(inst.opr.src2, src2_n, src2_bank, src2_ext, use_double_reg, bits_max, m_second_program);

    inst.opr.src1.type = load_data_type;
    inst.opr.src2.type = load_data_type;

    inst.opr.src1.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
    inst.opr.src2.swizzle = (src2_vscomp && (alu_sel == 0)) ? (Swizzle4 SWIZZLE_CHANNEL_4(X, X, X, X)) : (Swizzle4 SWIZZLE_CHANNEL_4_DEFAULT);

    if (src1_neg) {
        inst.opr.src1.flags |= RegisterFlags::Negative;
    }

    Operand pred_op{};
    pred_op.bank = RegisterBank::PREDICATE;
    pred_op.num = pdst_n;
    inst.opr.dest = pred_op;

    spv::Id alu_result = spv::NoResult;
    const spv::Id pred_result = vtst_impl(inst, pred, zero_test, sign_test, load_mask, false, test_crcomb_and,
        m_spirv_params.native_metal && test_wben ? &alu_result : nullptr);
    if (pred_result == spv::NoResult) return false;
    if (alu_result != spv::NoResult) {
        Operand dest{};
        dest = decode_dest(dest, dest_n, dest_bank, dest_ext, use_double_reg, bits_max, m_second_program);
        dest.type = load_data_type;
        dest.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
        store(dest, alu_result, load_mask);
    }

    if (per_channel) {
        // Read the complete result before updating any predicate register.
        // This also keeps the ordinary scalar-predicate path unchanged.
        for (unsigned lane = 0; lane < 4; ++lane) {
            auto destination = inst.opr.dest;
            destination.num = lane;
            store(destination, m_b.createCompositeExtract(pred_result, m_b.makeBoolType(), lane));
        }
    } else {
        const auto result = chan_cc < 4 || scalar_word ? pred_result
            : m_b.createUnaryOp(chan_cc == 4 ? spv::OpAll : spv::OpAny,
                m_b.makeBoolType(), pred_result);
        store(inst.opr.dest, result);
    }
    return true;
}

bool USSETranslatorVisitor::vtstmsk(
    ExtPredicate pred,
    Imm1 skipinv,
    Imm1 onceonly,
    Imm1 syncstart,
    Imm1 dest_ext,
    Imm1 test_flag_2,
    Imm1 src1_ext,
    Imm1 src2_ext,
    Imm1 prec,
    Imm1 src2_vscomp,
    RepeatCount rpt_count,
    Imm2 sign_test,
    Imm2 zero_test,
    Imm1 test_crcomb_and,
    Imm2 tst_mask_type,
    Imm2 dest_bank,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm1 test_wben,
    Imm2 alu_sel,
    Imm4 alu_op,
    Imm7 src1_n,
    Imm7 src2_n) {
    DataType load_data_type;
    Opcode test_op = decode_test_inst(alu_sel, alu_op, prec, load_data_type);
    Instruction inst;

    if (test_op == Opcode::INVALID) {
        LOG_ERROR("Unsupported comparision: {} {}", alu_sel, alu_op);
        return false;
    }

    inst.opcode = test_op;

    if (tst_mask_type == 3) {
        LOG_ERROR("Invalid test mask type: 3");
        return false;
    }

    if (m_spirv_params.native_metal && !test_wben) {
        // TESTMASK has no predicate destination. With WBEN clear it has no
        // architectural result, including across repeats. The assembler uses
        // a read-only special destination, so do not decode/load/store it.
        // Keep reserved comparison encodings rejected just like enabled tests.
        return sign_test < 3 && zero_test < 3;
    }

    const bool use_double_reg = alu_sel == 0;
    const uint8_t reg_bits = use_double_reg ? 8 : 7;
    // TODO: In some op, we don't need to load src2 e.g. rsq
    inst.opr.src1 = decode_src12(inst.opr.src1, src1_n, src1_bank, src1_ext, use_double_reg, reg_bits, m_second_program);
    inst.opr.src2 = decode_src12(inst.opr.src2, src2_n, src2_bank, src2_ext, use_double_reg, reg_bits, m_second_program);
    inst.opr.dest = decode_dest(inst.opr.dest, dest_n, dest_bank, dest_ext, use_double_reg, reg_bits, m_second_program);

    inst.opr.src1.type = load_data_type;
    inst.opr.src2.type = load_data_type;
    inst.opr.dest.type = load_data_type;

    inst.opr.src1.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
    inst.opr.src2.swizzle = (alu_sel == 0 && src2_vscomp) ? (Swizzle4 SWIZZLE_CHANNEL_4(X, X, X, X)) : (Swizzle4 SWIZZLE_CHANNEL_4_DEFAULT);
    inst.opr.dest.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;

    if (test_flag_2) {
        inst.opr.src1.flags |= RegisterFlags::Negative;
    }

    if (!test_wben) {
        // TODO support it
        LOG_ERROR("Write disabled vmsktst is not supported yet.");
        return false;
    }

    // input is always 4 in case of a dot product instruction
    const bool is_vdp = (inst.opcode == Opcode::VDP || inst.opcode == Opcode::VF16DP);
    const bool output_4 = (alu_sel == 0 && tst_mask_type == 2);

    // Scalar full-word TESTMASK repeats step through ordinary register
    // words using the existing SMLSI offsets. Packed/vector test repeats and
    // predicate-destination repeats need their separate lane contract.
    const bool word_repeats = m_spirv_params.native_metal
        && (load_data_type == DataType::INT32 || load_data_type == DataType::UINT32);
    if (word_repeats) set_repeat_multiplier(1, 1, 1, 1);
    const unsigned repeats = word_repeats ? unsigned(rpt_count) + 1 : 1;
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        const int src1_repeat_offset = word_repeats
            ? get_repeat_offset(inst.opr.src1, repeat, RepeatMode::SLMSI, inst.opr.src1.bank) : 0;
        const int src2_repeat_offset = word_repeats
            ? get_repeat_offset(inst.opr.src2, repeat, RepeatMode::SLMSI, inst.opr.src2.bank) : 0;
        const int dest_repeat_offset = word_repeats
            ? get_repeat_offset(inst.opr.dest, repeat, RepeatMode::SLMSI, inst.opr.dest.bank) : 0;
        spv::Id pred_result = vtst_impl(inst, pred, zero_test, sign_test, (is_vdp || output_4) ? 0b1111 : 0b1, true, test_crcomb_and, nullptr, src1_repeat_offset, src2_repeat_offset);
        if (pred_result == spv::NoResult) return false;
        if (m_spirv_params.native_metal && !output_4 && m_b.getNumComponents(pred_result) > 1) {
            // DOT loads four source lanes and replicates its scalar result for
            // the comparison. Scalar raw/numeric masks need one of those identical
            // predicate lanes; OpSelect cannot use bool4 to select scalar values.
            pred_result = m_b.createCompositeExtract(pred_result, m_b.makeBoolType(), 0);
        }

        spv::Id output_type;
        spv::Id zeros;
        spv::Id ones;
        bool f32_mask_bits = false;
        switch (load_data_type) {
        case DataType::F16:
        case DataType::F32:
            if (m_spirv_params.native_metal && tst_mask_type == 0) {
                const bool half = load_data_type == DataType::F16;
                output_type = m_b.makeUintType(32);
                zeros = m_b.makeUintConstant(0);
                ones = m_b.makeUintConstant(half ? 0xFFFFu : 0xFFFFFFFFu);
                if (half)
                    inst.opr.dest.type = DataType::UINT16;
                else
                    f32_mask_bits = true;
                break;
            }
            output_type = type_f32;
            if (output_4)
                output_type = m_b.makeVectorType(output_type, 4);
            zeros = utils::make_uniform_vector_from_type(m_b, output_type, 0.0f);
            ones = utils::make_uniform_vector_from_type(m_b, output_type, 1.0f);
            break;
        default: {
            uint32_t ones_val = 1;
            bool is_signed = false;
            switch (load_data_type) {
            case DataType::INT8:
                is_signed = true;
                ones_val = 0x7F;
                break;
            case DataType::INT16:
                is_signed = true;
                ones_val = 0x7FFF;
                break;
            case DataType::INT32:
                is_signed = true;
                ones_val = 0x7FFFFFFF;
                break;
            case DataType::UINT8:
                ones_val = 0xFF;
                break;
            case DataType::UINT16:
                ones_val = 0XFFFF;
                break;
            case DataType::UINT32:
                ones_val = 0xFFFFFFFF;
                break;
            default:
                LOG_ERROR("Unexpected type {}", static_cast<int>(load_data_type));
                break;
            }
            output_type = is_signed ? m_b.makeIntType(32) : m_b.makeUintType(32);
            zeros = utils::make_uniform_vector_from_type(m_b, output_type, 0);
            ones = utils::make_uniform_vector_from_type(m_b, output_type, ones_val);
            break;
        }
        }

        pred_result = m_b.createOp(spv::OpSelect, output_type, { pred_result, ones, zeros });
        // Preserve all mask bits in the float register instead of storing 1.0.
        if (f32_mask_bits)
            pred_result = m_b.createUnaryOp(spv::OpBitcast, type_f32, pred_result);

        store(inst.opr.dest, pred_result, 0xFF, dest_repeat_offset);
    }
    if (word_repeats) reset_repeat_multiplier();
    return true;
}

bool USSETranslatorVisitor::br(
    ExtPredicate pred,
    Imm1 syncend,
    bool exception,
    bool pwait,
    Imm1 sync_ext,
    bool nosched,
    bool br_monitor,
    bool save_link,
    Imm1 br_type,
    Imm1 any_inst,
    Imm1 all_inst,
    uint32_t br_off) {
    // The analyzer leaves branches to the next instruction in code blocks:
    // both outcomes fall through, so they require no emitted control flow.
    if (br_off == 1)
        return true;
    assert(false && "Unreachable");

    LOG_ERROR("Branch instruction should not be recompiled here!");
    return true;
}
