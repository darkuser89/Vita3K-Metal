// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Numerical corpus checked with Richard Geldreich's independent bc7decomp:
// https://github.com/richgel999/bc7enc_rdo/tree/b9438627eef73a1157e84201b6fa6eb2ffd6d9f0
// Optional recheck: define BC7_REFERENCE and link that revision's bc7decomp.cpp.
// BC7_GENERATE_DIGESTS requires the reference and hashes its output, not bcdec's.
// No reference source or downloaded dependency is retained in this repository.
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#define BCDEC_IMPLEMENTATION
#include "../bcdec.h"
#ifdef BC7_REFERENCE
#include "bc7decomp.h"
#endif
#if defined(BC7_GENERATE_DIGESTS) && !defined(BC7_REFERENCE)
#error Digest generation requires the independent reference decoder
#endif

static constexpr const char *expected_sha256[] = {
    "046a26eb0094e41694a2a513c086ea6274de930ad15610623c9726b55f14b066",
    "b4591dee6b0233bca0820e5ff7025b1afa8bf5e25fdf61bec4f0dfb9f2d02776",
    "f26c3397c05503813dbc4e128e15301afeb54b5fdc6963b4a4adb7a71bad3d62",
    "b248d1b9603d06c24d3f504271cab16b64ed217a22709de52ff396f2febb520b",
    "942f898cbdda68142f057ac79261fc5cfbc3b8723511ded37f75ddb1201198f8",
    "8b8ca9716c29bba5791631c563de4f64818d361fdc7e37de29e14866e242f8fb",
    "941667bfcd9bd3d8d345d940c89f9c9d033e806061d684a1d072344b8d982a16",
    "81f059c6bc20abc9fde962a0379f8ba5301fd0392dd45fee56be1e36b03ddb1e",
    "437af2517d29021e8e190b6bf25ef7fa54bc2ccf755fc0bea70217abb534982e",
};

using Block = std::array<uint8_t,16>;
static uint32_t random_state = 0x0bc7a032;
static uint32_t random_word() {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}
static void field(Block &block, unsigned first, unsigned count, unsigned value) {
    for (unsigned n = 0; n < count; ++n) {
        const unsigned bit = first + n;
        block[bit/8] = uint8_t((block[bit/8] & ~(1u << (bit%8))) | (((value>>n)&1u) << (bit%8)));
    }
}
static Block payload(unsigned pattern) {
    Block block;
    for (auto &byte : block)
        byte = pattern == 0 ? 0 : pattern == 1 ? 255 : pattern == 2 ? 0x55 : pattern == 3 ? 0xaa : uint8_t(random_word());
    return block;
}
static std::string hex_digest(CC_SHA256_CTX &context) {
    std::array<uint8_t,32> digest;
    CC_SHA256_Final(digest.data(), &context);
    std::ostringstream text;
    for (auto byte : digest) text << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return text.str();
}
int main() {
    const unsigned partition_bits[] = {4,6,6,6,0,0,0,6};
    const unsigned pbit_count[] = {6,2,0,4,0,0,2,4};
    const unsigned pbit_offset[] = {77,80,0,94,0,0,63,94};
    size_t cases = 0, structured = 0, random = 0, invalid = 0;
    for (unsigned mode = 0; mode < 9; ++mode) {
        CC_SHA256_CTX digest; CC_SHA256_Init(&digest);
        const auto check = [&](const Block &block) {
            std::array<uint8_t,64> reference{};
#ifdef BC7_REFERENCE
            alignas(8) Block aligned = block;
            std::array<bc7decomp::color_rgba,16> pixels;
            const bool valid = bc7decomp::unpack_bc7(aligned.data(), pixels.data());
            assert(valid == (mode < 8));
            std::memcpy(reference.data(), pixels.data(), reference.size());
#else
            bcdec_bc7(block.data(), reference.data(), 16);
#endif
            if (mode <= 3) for (unsigned n = 0; n < 16; ++n) assert(reference[n*4+3] == 255);
            if (mode == 8) for (auto byte : reference) assert(byte == 0);
            std::array<uint8_t,32> input; input.fill(0xa5);
            const unsigned offset = cases % 8;
            std::memcpy(input.data()+offset, block.data(), block.size());
            for (unsigned pitch : {16u,19u,32u}) {
                std::array<uint8_t,128> output; output.fill(0xa5);
                bcdec_bc7(input.data()+offset, output.data()+8, int(pitch));
                for (unsigned n = 0; n < output.size(); ++n) {
                    const unsigned row = n >= 8 ? (n-8)/pitch : 4;
                    const unsigned column = n >= 8 ? (n-8)%pitch : 16;
                    assert(output[n] == (row < 4 && column < 16 ? reference[row*16+column] : 0xa5));
                }
            }
            CC_SHA256_Update(&digest, block.data(), block.size());
            CC_SHA256_Update(&digest, reference.data(), reference.size());
            ++cases;
        };
        if (mode < 8) {
            for (unsigned partition = 0; partition < (1u<<partition_bits[mode]); ++partition)
            for (unsigned pbits = 0; pbits < (1u<<pbit_count[mode]); ++pbits)
            for (unsigned rotation = 0; rotation < (mode == 4 || mode == 5 ? 4u : 1u); ++rotation)
            for (unsigned selection = 0; selection < (mode == 4 ? 2u : 1u); ++selection)
            for (unsigned pattern = 0; pattern < 32; ++pattern) {
                auto block = payload(pattern);
                field(block,0,mode+1,1u<<mode);
                field(block,mode+1,partition_bits[mode],partition);
                field(block,pbit_offset[mode],pbit_count[mode],pbits);
                if (mode == 4 || mode == 5) field(block,mode+1,2,rotation);
                if (mode == 4) field(block,mode+3,1,selection);
                check(block); ++structured;
            }
            for (unsigned n = 0; n < 100000; ++n) {
                auto block = payload(4); field(block,0,mode+1,1u<<mode);
                check(block); ++random;
            }
        } else {
            for (unsigned n = 0; n < 10000; ++n) {
                auto block = payload(n < 4 ? n : 4); block[0] = 0;
                check(block); ++invalid;
            }
        }
        const auto actual = hex_digest(digest);
#ifdef BC7_GENERATE_DIGESTS
        std::cout << "DIGEST " << mode << " " << actual << '\n';
#else
        if (actual != expected_sha256[mode]) {
            std::cerr << "BC7 reference digest mismatch for mode " << mode << ": " << actual << '\n';
            return 1;
        }
#endif
    }
    std::cout << "PASS BC7 reference: " << cases << " blocks (" << structured << " structured, " << random
              << " random, " << invalid << " reserved), all mode/partition/P-bit/rotation/index-selector combinations; exact RGBA against independent reference digests, pitches 16/19/32, source offsets 0..7 and output sentinels; ASAN/UBSAN, no GPU\n";
}
