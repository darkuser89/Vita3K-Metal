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

#include <shader/metal4/usse_translator.h>

#include <SPIRV/SpvBuilder.h>

#include <shader/metal4/usse_decoder_helpers.h>
#include <shader/metal4/usse_disasm.h>
#include <shader/metal4/usse_types.h>
#include <util/log.h>

using namespace shader::metal3;
using namespace usse;

static spv::Id rotate_left32(spv::Builder &builder, spv::Id value, spv::Id count) {
    const auto type = builder.makeUintType(32);
    const auto mask = builder.makeUintConstant(31);
    const auto left_count = builder.createBinOp(spv::OpBitwiseAnd, type, count, mask);
    const auto complement = builder.createBinOp(spv::OpISub, type, builder.makeUintConstant(0), left_count);
    const auto right_count = builder.createBinOp(spv::OpBitwiseAnd, type, complement, mask);
    // Mask both shifts: a zero rotation must never emit a shift by 32.
    const auto left = builder.createBinOp(spv::OpShiftLeftLogical, type, value, left_count);
    const auto right = builder.createBinOp(spv::OpShiftRightLogical, type, value, right_count);
    return builder.createBinOp(spv::OpBitwiseOr, type, left, right);
}

bool USSETranslatorVisitor::vbw(
    Imm3 op1,
    ExtPredicate pred,
    Imm1 skipinv,
    Imm1 nosched,
    bool repeat_sel,
    Imm1 sync_start,
    Imm1 dest_ext,
    Imm1 end,
    Imm1 src1_ext,
    Imm1 src2_ext,
    RepeatCount repeat_count,
    Imm1 src2_invert,
    Imm5 src2_rot,
    Imm2 src2_exth,
    Imm1 op2,
    Imm1 bitwise_partial,
    Imm2 dest_bank,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm7 src2_sel,
    Imm7 src1_n,
    Imm7 src2_n) {
    Instruction inst;

    switch (op1) {
    case 0b010: inst.opcode = op2 ? Opcode::OR : Opcode::AND; break;
    case 0b011: inst.opcode = Opcode::XOR; break;
    case 0b100: inst.opcode = op2 ? Opcode::ROL : Opcode::SHL; break;
    case 0b101: inst.opcode = op2 ? Opcode::ASR : Opcode::SHR; break;
    default: return false;
    }

    inst.opr.src1 = decode_src12(inst.opr.src1, src1_n, src1_bank, src1_ext, false, 7, m_second_program);
    inst.opr.src2 = decode_src12(inst.opr.src2, src2_n, src2_bank, src2_ext, false, 7, m_second_program);
    inst.opr.dest = decode_dest(inst.opr.dest, dest_n, dest_bank, dest_ext, false, 7, m_second_program);

    DataType type = bitwise_partial ? DataType::UINT16 : DataType::UINT32;
    inst.opr.src1.type = type;
    inst.opr.src2.type = type;
    inst.opr.dest.type = type;

    set_repeat_multiplier(1, 1, 1, 1);

    BEGIN_REPEAT(repeat_count)
    GET_REPEAT(inst, RepeatMode::SLMSI);

    bool immediate = src2_ext && inst.opr.src2.bank == RegisterBank::IMMEDIATE;
    uint32_t value = 0;

    spv::Id src2 = 0;
    if (immediate) {
        value = src2_n | (static_cast<uint32_t>(src2_sel) << 7) | (static_cast<uint32_t>(src2_exth) << 14);
        // rotate left by src2_rot
        if (src2_rot > 0) {
            if (type == DataType::UINT16) {
                src2_rot &= 15;
                uint16_t temp = static_cast<uint16_t>(value);
                temp = (temp << src2_rot) | (temp >> (16 - src2_rot));
                value = temp;
            } else {
                value = (value << src2_rot) | (value >> (32 - src2_rot));
            }
        }

        if (src2_invert)
            value = ~value;
        if (type == DataType::UINT16)
            value &= 0xFFFF;
        src2 = m_b.makeUintConstant(value);
    } else {
        src2 = load(inst.opr.src2, 0b0001, src2_repeat_offset);

        if (src2_rot) {
            if (type == DataType::UINT16)
                src2_rot &= 15;
            const uint32_t right_shift = type == DataType::UINT16 ? (16 - src2_rot) : (32 - src2_rot);

            // src2 = (src2 << src2_rot) | (src2 >> (bit_size - src2_rot))
            spv::Id left = m_b.createBinOp(spv::OpShiftLeftLogical, type_ui32, src2, m_b.makeUintConstant(src2_rot));
            spv::Id right = m_b.createBinOp(spv::OpShiftRightLogical, type_ui32, src2, m_b.makeUintConstant(right_shift));
            src2 = m_b.createBinOp(spv::OpBitwiseOr, type_ui32, left, right);

            if (type == DataType::UINT16)
                src2 = m_b.createBinOp(spv::OpBitwiseAnd, type_ui32, src2, m_b.makeUintConstant(0xFFFF));
        }

        if (src2 == spv::NoResult) {
            LOG_ERROR("Source 2 not loaded");
            return false;
        }

        if (src2_invert) {
            src2 = m_b.createUnaryOp(spv::Op::OpNot, type_ui32, src2);

            if (type == DataType::UINT16)
                src2 = m_b.createBinOp(spv::OpBitwiseAnd, type_ui32, src2, m_b.makeUintConstant(0xFFFF));
        }
    }

    spv::Id result;

    spv::Op operation;
    switch (inst.opcode) {
    case Opcode::OR: operation = spv::Op::OpBitwiseOr; break;
    case Opcode::AND: operation = spv::Op::OpBitwiseAnd; break;
    case Opcode::XOR: operation = spv::Op::OpBitwiseXor; break;
    case Opcode::ROL:
        if (bitwise_partial) {
            LOG_WARN("Partial-width Rotate Left operation unsupported.");
            return false;
        }
        operation = spv::OpNop; // ROL is lowered to two bounded shifts below.
        break;
    case Opcode::ASR: operation = spv::Op::OpShiftRightArithmetic; break;
    case Opcode::SHL: operation = spv::Op::OpShiftLeftLogical; break;
    case Opcode::SHR: operation = spv::Op::OpShiftRightLogical; break;
    default: return false;
    }
    // optimisation. (any OR 0 || any XOR 0 || any AND 0xFFFFFFFF) -> assign
    bool is_const = m_b.getOpCode(src2) == spv::Op::OpConstant;
    auto const_val = is_const ? m_b.getConstantScalar(src2) : 1; // default value is intentionally non zero
    if (((operation == spv::Op::OpBitwiseOr || operation == spv::Op::OpBitwiseXor) && is_const && const_val == 0)
        || (operation == spv::Op::OpBitwiseAnd && is_const && const_val == std::numeric_limits<decltype(const_val)>::max())) {
        result = load(inst.opr.src1, 0b0001, src1_repeat_offset);
        if (result == spv::NoResult) {
            LOG_ERROR("Source not loaded");
            return false;
        }
    } else {
        spv::Id src1 = load(inst.opr.src1, 0b0001, src1_repeat_offset);
        if (src1 == spv::NoResult) {
            LOG_ERROR("Source not loaded");
            return false;
        }
        result = inst.opcode == Opcode::ROL ? rotate_left32(m_b, src1, src2)
                                           : m_b.createBinOp(operation, type_ui32, src1, src2);
        if (m_b.isFloatType(m_b.getTypeId(src2))) {
            result = m_b.createUnaryOp(spv::Op::OpBitcast, type_f32, src2);
        }
    }

    store(inst.opr.dest, result, 0b0001, dest_repeat_offset);

    LOG_DISASM("{:016x}: {}{}{} {} {} {}", m_instr, disasm::e_predicate_str(pred), disasm::opcode_str(inst.opcode), bitwise_partial ? "16" : "",
        disasm::operand_to_str(inst.opr.dest, 0b0001, dest_repeat_offset), disasm::operand_to_str(inst.opr.src1, 0b0001, src1_repeat_offset),
        immediate ? log_hex(value) : disasm::operand_to_str(inst.opr.src2, 0b0001, src2_repeat_offset));

    END_REPEAT()

    reset_repeat_multiplier();

    return true;
}

// Sources are unpacked UINT8 vectors. Select every term from the original
// source lanes before applying that term's independent RGB/alpha complement.
static spv::Id i8mad_result(spv::Builder &b, spv::Id src0, spv::Id src1, spv::Id src2,
    unsigned csel0, bool asel0, bool csel1, bool csel2,
    bool cmod0, bool amod0, bool cmod1, bool amod1, bool cmod2, bool amod2,
    bool negate_add, bool saturate) {
    const auto type = b.getTypeId(src0);
    const auto splat = [&](uint32_t value) {
        return b.makeCompositeConstant(type, std::vector<spv::Id>(4, b.makeUintConstant(value)));
    };
    const auto complement = [&](spv::Id value, bool rgb, bool alpha) {
        if (!rgb && !alpha)
            return value;
        const auto color_mask = b.makeUintConstant(rgb ? 255 : 0);
        const auto alpha_mask = b.makeUintConstant(alpha ? 255 : 0);
        const auto mask = b.makeCompositeConstant(type, { color_mask, color_mask, color_mask, alpha_mask });
        return b.createBinOp(spv::OpBitwiseXor, type, value, mask);
    };
    const auto alpha_rgb = [&](spv::Id value, bool select_alpha) {
        return select_alpha ? b.createOp(spv::OpVectorShuffle, type,
                                  std::vector<spv::IdImmediate>{ { true, value }, { true, value },
                                      { false, 3 }, { false, 3 }, { false, 3 }, { false, 3 } })
                            : value;
    };
    spv::Id add = src0;
    if (csel0 || asel0) {
        const unsigned base = (csel0 & 1) ? 4 : 0;
        const bool alpha = (csel0 & 2) != 0;
        add = b.createOp(spv::OpVectorShuffle, type,
            std::vector<spv::IdImmediate>{ { true, src0 }, { true, src1 },
                { false, base + (alpha ? 3 : 0) }, { false, base + (alpha ? 3 : 1) },
                { false, base + (alpha ? 3 : 2) }, { false, asel0 ? 7u : 3u } });
    }
    add = complement(add, cmod0, amod0);
    const auto lhs = complement(alpha_rgb(src1, csel1), cmod1, amod1);
    const auto rhs = complement(alpha_rgb(src2, csel2), cmod2, amod2);
    const auto product = b.createBinOp(spv::OpIMul, type, lhs, rhs);
    auto result = b.createBinOp(negate_add ? spv::OpISub : spv::OpIAdd, type, product, add);
    if (saturate) {
        const auto bool_type = b.makeVectorType(b.makeBoolType(), 4);
        if (negate_add) {
            const auto underflow = b.createBinOp(spv::OpULessThan, bool_type, product, add);
            result = b.createOp(spv::OpSelect, type, { underflow, splat(0), result });
        }
        const auto maximum = splat(255);
        const auto overflow = b.createBinOp(spv::OpUGreaterThan, bool_type, result, maximum);
        result = b.createOp(spv::OpSelect, type, { overflow, maximum, result });
    }
    return result;
}

bool USSETranslatorVisitor::i8mad(
    Imm2 pred,
    Imm1 cmod1,
    Imm1 skipinv,
    Imm1 nosched,
    Imm2 csel0,
    Imm1 dest_bank_ext,
    Imm1 end,
    Imm1 src1_bank_ext,
    Imm1 src2_bank_ext,
    Imm1 cmod2,
    Imm3 repeat_count,
    Imm1 saturated,
    Imm1 cmod0,
    Imm1 asel0,
    Imm1 amod2,
    Imm1 amod1,
    Imm1 amod0,
    Imm1 csel1,
    Imm1 csel2,
    Imm1 src0_neg,
    Imm1 src0_bank,
    Imm2 dest_bank,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_num,
    Imm7 src0_num,
    Imm7 src1_num,
    Imm7 src2_num) {
    Instruction inst;

    inst.opr.src0 = decode_src0(inst.opr.src0, src0_num, src0_bank, false, false, 7, m_second_program);
    inst.opr.src1 = decode_src12(inst.opr.src1, src1_num, src1_bank, src1_bank_ext, false, 7, m_second_program);
    inst.opr.src2 = decode_src12(inst.opr.src2, src2_num, src2_bank, src2_bank_ext, false, 7, m_second_program);
    inst.opr.dest = decode_dest(inst.opr.dest, dest_num, dest_bank, dest_bank_ext, false, 7, m_second_program);

    inst.opr.src0.type = DataType::UINT8;
    inst.opr.src1.type = DataType::UINT8;
    inst.opr.src2.type = DataType::UINT8;
    inst.opr.dest.type = DataType::UINT8;

    inst.opcode = Opcode::IMA8;

    inst.opr.src0.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
    inst.opr.src1.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
    inst.opr.src2.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;

    if (m_spirv_params.native_metal)
        set_repeat_multiplier(1, 1, 1, 1);

    BEGIN_REPEAT(repeat_count);
    GET_REPEAT(inst, RepeatMode::SLMSI);

    const auto src0 = load(inst.opr.src0, 0b1111, src0_repeat_offset);
    const auto src1 = load(inst.opr.src1, 0b1111, src1_repeat_offset);
    const auto src2 = load(inst.opr.src2, 0b1111, src2_repeat_offset);
    if (!src0 || !src1 || !src2)
        return false;
    const auto result = i8mad_result(m_b, src0, src1, src2, csel0, asel0, csel1, csel2,
        cmod0, amod0, cmod1, amod1, cmod2, amod2, src0_neg, saturated);
    store(inst.opr.dest, result, 0b1111, dest_repeat_offset);

    const bool add_swizz_rgb_src0 = (csel0 & 1) == 0;
    LOG_DISASM("{:016x}: {}{}{} {} {} {} {}({}, {}) selectors={}/{}/{}/{} complements={}{}/{}{}/{}{}", m_instr,
        disasm::s_predicate_str(static_cast<usse::ShortPredicate>(pred)), disasm::opcode_str(inst.opcode), saturated ? ".sat" : "",
        disasm::operand_to_str(inst.opr.dest, 0b1111, dest_repeat_offset),
        disasm::operand_to_str(inst.opr.src1, 0b1111, src1_repeat_offset),
        disasm::operand_to_str(inst.opr.src2, 0b1111, src2_repeat_offset),
        src0_neg ? "-" : "", disasm::operand_to_str(add_swizz_rgb_src0 ? inst.opr.src0 : inst.opr.src1, 0b0111,
            add_swizz_rgb_src0 ? src0_repeat_offset : src1_repeat_offset),
        disasm::operand_to_str(asel0 ? inst.opr.src1 : inst.opr.src0, 0b1000, asel0 ? src1_repeat_offset : src0_repeat_offset),
        csel0, asel0, csel1, csel2, cmod0, amod0, cmod1, amod1, cmod2, amod2);

    END_REPEAT();

    if (m_spirv_params.native_metal)
        reset_repeat_multiplier();

    return true;
}

bool USSETranslatorVisitor::i8mad2() {
    LOG_ERROR("Metal: I8MAD2 is not implemented");
    return false;
}

bool USSETranslatorVisitor::i16mad(
    ShortPredicate pred,
    Imm1 abs,
    bool skipinv,
    bool nosched,
    Imm1 src2_neg,
    Imm1 sel1h_upper8,
    Imm1 dest_bank_ext,
    Imm1 end,
    Imm1 src1_bank_ext,
    Imm1 src2_bank_ext,
    RepeatCount repeat_count,
    Imm2 mode,
    Imm2 src2_format,
    Imm2 src1_format,
    Imm1 sel2h_upper8,
    Imm2 or_shift,
    Imm1 src0_bank,
    Imm2 dest_bank,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm7 src0_n,
    Imm7 src1_n,
    Imm7 src2_n) {
    if ((mode == 1) || (mode == 3)) {
        LOG_DISASM("Saturation in i16mad not yet supported!");
    }

    Instruction inst;
    inst.opcode = Opcode::IMA16;
    DataType operate_type = ((mode == 0) || (mode == 1)) ? DataType::UINT16 : DataType::INT16;

    inst.opr.dest = decode_dest(inst.opr.dest, dest_n, dest_bank, dest_bank_ext, false, 7, m_second_program);
    inst.opr.src0 = decode_src0(inst.opr.src0, src0_n, src0_bank, false, false, 7, m_second_program);
    inst.opr.src1 = decode_src12(inst.opr.src1, src1_n, src1_bank, src1_bank_ext, false, 7, m_second_program);
    inst.opr.src2 = decode_src12(inst.opr.src2, src2_n, src2_bank, src2_bank_ext, false, 7, m_second_program);

    inst.opr.dest.type = inst.opr.src0.type = inst.opr.src1.type = inst.opr.src2.type = operate_type;
    std::uint8_t mask_src1 = 0b1;
    std::uint8_t mask_src2 = 0b1;
    if (src1_format != 0) {
        if (src1_format == 1 || src1_format == 2) {
            inst.opr.src1.type = src1_format == 1 ? DataType::UINT8 : DataType::INT8;
        } else {
            LOG_DISASM("Offset-8 source format for IMA16 SRC1 not yet supported!");
        }

        if (sel1h_upper8) {
            mask_src1 = 0b10;
        }
    }

    if (src2_format != 0) {
        if (src2_format == 1 || src2_format == 2) {
            inst.opr.src2.type = src2_format == 1 ? DataType::UINT8 : DataType::INT8;
        } else {
            LOG_DISASM("Offset-8 source format for IMA16 SRC2 not yet supported!");
        }

        if (sel2h_upper8) {
            mask_src2 = 0b10;
        }
    }

    if (m_spirv_params.native_metal)
        set_repeat_multiplier(1, 1, 1, 1);

    BEGIN_REPEAT(repeat_count);
    GET_REPEAT(inst, RepeatMode::SLMSI);

    LOG_DISASM("{:016x}: {}{} {} {} {} {}", m_instr, disasm::s_predicate_str(pred), "IMAD16", disasm::operand_to_str(inst.opr.dest, 0b1),
        disasm::operand_to_str(inst.opr.src0, 0b1), disasm::operand_to_str(inst.opr.src1, mask_src1) + ((src1_format != 0) ? "-8bits" : ""),
        disasm::operand_to_str(inst.opr.src2, mask_src2) + ((src2_format != 0) ? "-8bits" : ""));

    inst.opr.src0.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
    spv::Id source0 = load(inst.opr.src0, 0b1, src0_repeat_offset);
    spv::Id source1 = load(inst.opr.src1, mask_src1, src1_repeat_offset);
    spv::Id source2 = load(inst.opr.src2, mask_src2, src2_repeat_offset);
    if (!source0 || !source1 || !source2)
        return false;

    spv::Id source0_type = m_b.getTypeId(source0);
    // Preserve the source format's extension before matching SPIR-V operand
    // signedness. The unsaturated UINT16/SINT16 store keeps the low 16 bits.
    if (m_b.getTypeId(source1) != source0_type)
        source1 = m_b.createUnaryOp(spv::OpBitcast, source0_type, source1);
    if (m_b.getTypeId(source2) != source0_type)
        source2 = m_b.createUnaryOp(spv::OpBitcast, source0_type, source2);

    auto mul_result = m_b.createBinOp(spv::OpIMul, source0_type, source0, source1);
    auto add_result = m_b.createBinOp(m_spirv_params.native_metal && src2_neg ? spv::OpISub : spv::OpIAdd, source0_type, mul_result, source2);

    if (add_result != spv::NoResult) {
        store(inst.opr.dest, add_result, 0b1, dest_repeat_offset);
    }

    END_REPEAT();

    if (m_spirv_params.native_metal)
        reset_repeat_multiplier();

    return true;
}

bool USSETranslatorVisitor::i32mad(
    ShortPredicate pred,
    Imm1 src0_high,
    Imm1 nosched,
    Imm1 src1_high,
    Imm1 src2_high,
    bool dest_bank_ext,
    Imm1 end,
    bool src1_bank_ext,
    bool src2_bank_ext,
    RepeatCount repeat_count,
    bool is_signed,
    bool is_sat,
    Imm2 src2_type,
    Imm1 src0_bank,
    Imm2 dest_bank,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm7 src0_n,
    Imm7 src1_n,
    Imm7 src2_n) {
    // IMAE's third-source extension is independent of the multiplier sign.
    // Encoding 3 is reserved, not another 16-bit source format.
    if (src2_type == 3) {
        LOG_ERROR("Metal: reserved IMAE source 2 format");
        return false;
    }
    Instruction inst;
    inst.opcode = Opcode::IMAD;

    inst.opr.dest = decode_dest(inst.opr.dest, dest_n, dest_bank, dest_bank_ext, false, 7, m_second_program);
    inst.opr.src0 = decode_src0(inst.opr.src0, src0_n, src0_bank, false, false, 7, m_second_program);
    inst.opr.src1 = decode_src12(inst.opr.src1, src1_n, src1_bank, src1_bank_ext, false, 7, m_second_program);
    inst.opr.src2 = decode_src12(inst.opr.src2, src2_n, src2_bank, src2_bank_ext, false, 7, m_second_program);

    const DataType inst_dt = is_signed ? DataType::INT32 : DataType::UINT32;
    const DataType inst_dt_16 = is_signed ? DataType::INT16 : DataType::UINT16;

    shader::metal3::usse::Imm4 src0_mask = 0b1;
    shader::metal3::usse::Imm4 src1_mask = 0b1;
    shader::metal3::usse::Imm4 src2_mask = 0b1;

    inst.opr.dest.type = inst_dt;
    inst.opr.src0.type = inst_dt_16;
    inst.opr.src1.type = inst_dt_16;

    if (src2_type == 2) {
        inst.opr.src2.type = inst_dt;
    } else {
        inst.opr.src2.type = src2_type == 1 ? DataType::INT16 : DataType::UINT16;
    }

    if (src0_high) {
        src0_mask = 0b10;
    }

    if (src1_high) {
        src1_mask = 0b10;
    }

    if (src2_type != 2 && src2_high) {
        src2_mask = 0b10;
    }

    set_repeat_multiplier(1, 1, 1, 1);

    BEGIN_REPEAT(repeat_count);
    GET_REPEAT(inst, RepeatMode::SLMSI);

    LOG_DISASM("{:016x}: {}{} {} {} {} {}", m_instr, disasm::s_predicate_str(pred), "IMAD2", disasm::operand_to_str(inst.opr.dest, 0b1, dest_repeat_offset),
        disasm::operand_to_str(inst.opr.src0, 0b1, src0_repeat_offset), disasm::operand_to_str(inst.opr.src1, 0b1, src1_repeat_offset), disasm::operand_to_str(inst.opr.src2, 0b1, src2_repeat_offset));

    spv::Id vsrc0 = load(inst.opr.src0, src0_mask, src0_repeat_offset);
    spv::Id vsrc1 = load(inst.opr.src1, src1_mask, src1_repeat_offset);
    spv::Id vsrc2 = load(inst.opr.src2, src2_mask, src2_repeat_offset);
    if (!vsrc0 || !vsrc1 || !vsrc2)
        return false;

    // The extension above produces the correct 32 bits. IAdd requires matching
    // signedness in SPIR-V, even though the non-saturating sum is modulo 2^32.
    const auto result_type = m_b.getTypeId(vsrc0);
    if (m_b.getTypeId(vsrc2) != result_type)
        vsrc2 = m_b.createUnaryOp(spv::OpBitcast, result_type, vsrc2);

    auto mul_result = m_b.createBinOp(spv::OpIMul, result_type, vsrc0, vsrc1);
    auto add_result = m_b.createBinOp(spv::OpIAdd, m_b.getTypeId(mul_result), mul_result, vsrc2);

    store(inst.opr.dest, add_result, 0b1, dest_repeat_offset);

    END_REPEAT();

    reset_repeat_multiplier();

    return true;
}

bool USSETranslatorVisitor::i32mad2(
    ExtPredicate pred,
    Imm1 nosched,
    Imm2 sn,
    bool dest_bank_ext,
    Imm1 end,
    bool src1_bank_ext,
    bool src2_bank_ext,
    bool src0_bank_ext,
    Imm3 count,
    bool is_signed,
    Imm1 negative_src1,
    Imm1 negative_src2,
    Imm1 src0_bank,
    Imm2 dest_bank,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm7 src0_n,
    Imm7 src1_n,
    Imm7 src2_n) {
    Instruction inst;
    inst.opcode = Opcode::IMAD;

    inst.opr.dest = decode_dest(inst.opr.dest, dest_n, dest_bank, dest_bank_ext, false, 7, m_second_program);
    inst.opr.src0 = decode_src0(inst.opr.src0, src0_n, src0_bank, src0_bank_ext, false, 7, m_second_program);
    inst.opr.src1 = decode_src12(inst.opr.src1, src1_n, src1_bank, src1_bank_ext, false, 7, m_second_program);
    inst.opr.src2 = decode_src12(inst.opr.src2, src2_n, src2_bank, src2_bank_ext, false, 7, m_second_program);

    const DataType inst_dt = is_signed ? DataType::INT32 : DataType::UINT32;

    inst.opr.dest.type = inst_dt;
    inst.opr.src0.type = inst_dt;
    inst.opr.src1.type = inst_dt;
    inst.opr.src2.type = inst_dt;

    // SGX543 IMA32 step 1 is 32 x 16 + 32. In particular, VPCK may
    // leave unrelated bits in the upper half of the second source register.
    if (m_spirv_params.native_metal && sn == 0)
        inst.opr.src1.type = is_signed ? DataType::INT16 : DataType::UINT16;

    if (negative_src1) {
        inst.opr.src1.flags |= RegisterFlags::Negative;
    }

    if (negative_src2) {
        inst.opr.src2.flags |= RegisterFlags::Negative;
    }

    // Apply the upstream I32MAD2 repeat fix to native Metal; preserve the
    // existing behavior of the Vulkan backend in this Metal-only branch.
    if (m_spirv_params.native_metal)
        set_repeat_multiplier(1, 1, 1, 1);

    BEGIN_REPEAT((m_spirv_params.native_metal ? count : 0))
    GET_REPEAT(inst, RepeatMode::SLMSI);
    if (!m_spirv_params.native_metal)
        src0_repeat_offset = src1_repeat_offset = src2_repeat_offset = dest_repeat_offset = 0;

    spv::Id vsrc0 = load(inst.opr.src0, 0b1, src0_repeat_offset);
    spv::Id vsrc1 = load(inst.opr.src1, 0b1, src1_repeat_offset);
    spv::Id vsrc2 = load(inst.opr.src2, 0b1, src2_repeat_offset);

    auto mul_result = m_b.createBinOp(spv::OpIMul, m_b.getTypeId(vsrc0), vsrc0, vsrc1);
    auto add_result = m_b.createBinOp(spv::OpIAdd, m_b.getTypeId(mul_result), mul_result, vsrc2);

    // sn is mysterious argument.
    // These are confirmed by hw testing:
    // - pa = x * y + z (sn = 0) => pa = x * y + z
    // - i = x * y + z (sn = 0) and then pa = x * y + i (sn = 1) => pa = x * y + z
    // - pa = x * y + z (sn = 1) => crash
    // TODO: properly implement this when we get more powerful fuzzer that can handle fpinternal.
    if (sn == 0) {
        store(inst.opr.dest, add_result, 0b1, dest_repeat_offset);
    } else {
        store(inst.opr.dest, vsrc2, 0b1, dest_repeat_offset);
    }

    // load() does not step an immediate, so print it unstepped
    LOG_DISASM("{:016x}: {}{} {} {} {} {} [sn={}]", m_instr, disasm::e_predicate_str(pred), "IMAD", disasm::operand_to_str(inst.opr.dest, 0b1, dest_repeat_offset),
        disasm::operand_to_str(inst.opr.src0, 0b1, src0_repeat_offset),
        disasm::operand_to_str(inst.opr.src1, 0b1, inst.opr.src1.bank == RegisterBank::IMMEDIATE ? 0 : src1_repeat_offset),
        disasm::operand_to_str(inst.opr.src2, 0b1, inst.opr.src2.bank == RegisterBank::IMMEDIATE ? 0 : src2_repeat_offset), sn);

    END_REPEAT()

    if (m_spirv_params.native_metal)
        reset_repeat_multiplier();

    return true;
}
