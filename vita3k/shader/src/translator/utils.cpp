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

#include <shader/usse_translator.h>

#include <SPIRV/SpvBuilder.h>

#include <shader/usse_types.h>
#include <util/log.h>

#include <bitset>

using namespace shader;
using namespace usse;

spv::Id USSETranslatorVisitor::load(Operand op, const Imm4 dest_mask, const int shift_offset) {
    return utils::load(m_b, m_spirv_params, m_util_funcs, m_features, op, dest_mask, shift_offset);
}

void USSETranslatorVisitor::store(Operand dest, spv::Id source, std::uint8_t dest_mask, int shift_offset) {
    if (m_spirv_params.native_metal && !m_store_from_vpck && source != spv::NoResult) {
        const int size = get_data_type_size(dest.type);
        for (int i = 0; i < 4; ++i) {
            if (!(dest_mask & (1 << i))) continue;
            const uint32_t byte = i * size;
            const uint32_t key = (uint32_t(dest.bank) << 24) | ((dest.num + shift_offset + byte / 4) & 0xFFFFFF);
            if (m_store_from_texture_sample || is_integer_data_type(dest.type))
                m_vpck_written_bytes[key] |= uint8_t((size >= 4 ? 0xFu : (1u << size) - 1) << (byte % 4));
            else
                m_vpck_written_bytes.erase(key);
        }
    }
    if (m_spirv_params.frag_output_holds_declared_type && source != spv::NoResult) {
        const int size = get_data_type_size(dest.type);
        for (int i = 0; i < 4; ++i) {
            if (!(dest_mask & (1 << i))) continue;
            const uint32_t key = (uint32_t(dest.bank) << 24) | ((dest.num + shift_offset + i * size / 4) & 0xFFFFFF);
            const auto type = m_store_is_raw_move ? m_raw_move_types[i] : dest.type;
            if (type == DataType::UNK) m_word_store_types.erase(key);
            else m_word_store_types[key] = type;
        }
    }
    utils::store(m_b, m_spirv_params, m_util_funcs, m_features, dest, source, dest_mask, shift_offset,
        m_store_is_raw_move, m_raw_move_keeps_declared);
}

void USSETranslatorVisitor::track_raw_move(const Instruction &inst, uint8_t mask, int size,
    int src1_offset, int src2_offset, int dest_offset, bool conditional) {
    if (!m_spirv_params.frag_output_holds_declared_type) return;
    const auto word_type = [&](const Operand &source, int offset, int channel) {
        const uint32_t key = (uint32_t(source.bank) << 24) | ((source.num + offset + channel * size / 4) & 0xFFFFFF);
        const auto found = m_word_store_types.find(key);
        return found == m_word_store_types.end() ? DataType::UNK : found->second;
    };
    const auto output = m_program.get_fragment_output_type();
    const auto declared = output == SCE_GXM_PARAMETER_TYPE_F16 ? DataType::F16
        : output == SCE_GXM_PARAMETER_TYPE_F32 ? DataType::F32 : DataType::UNK;
    bool copies_word0 = false;
    bool word0_declared = declared != DataType::UNK;
    for (int i = 0; i < 4; ++i) {
        m_raw_move_types[i] = DataType::UNK;
        if (!(mask & (1 << i))) continue;
        const auto channel = int(inst.opr.src1.swizzle[i]);
        if (channel < 4) {
            const auto type = word_type(inst.opr.src1, src1_offset, channel);
            if (!conditional || word_type(inst.opr.src2, src2_offset, channel) == type)
                m_raw_move_types[i] = type;
        }
        if (inst.opr.dest.bank == RegisterBank::OUTPUT && int(inst.opr.dest.num) + dest_offset + i * size / 4 == 0) {
            copies_word0 = true;
            word0_declared &= m_raw_move_types[i] == declared;
        }
    }
    m_raw_move_keeps_declared = copies_word0 && word0_declared;
    m_store_is_raw_move = true;
}

spv::Id USSETranslatorVisitor::swizzle_to_spv_comp(spv::Id composite, spv::Id type, SwizzleChannel swizzle) {
    switch (swizzle) {
    case SwizzleChannel::C_X:
    case SwizzleChannel::C_Y:
    case SwizzleChannel::C_Z:
    case SwizzleChannel::C_W:
        return m_b.createCompositeExtract(composite, type, static_cast<Imm4>(swizzle));

    // TODO: Implement these with OpCompositeExtract
    case SwizzleChannel::C_0: break;
    case SwizzleChannel::C_1: break;
    case SwizzleChannel::C_2: break;

    case SwizzleChannel::C_H: break;
    default: break;
    }

    LOG_WARN("Swizzle channel {} unsupported", static_cast<Imm4>(swizzle));
    return spv::NoResult;
}

size_t USSETranslatorVisitor::dest_mask_to_comp_count(Imm4 dest_mask) {
    std::bitset<4> bs(dest_mask);
    const auto bit_count = bs.count();
    return bit_count;
}
