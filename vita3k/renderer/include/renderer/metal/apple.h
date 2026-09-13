// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// macOS still exports the Carbon typedef `Ptr`. GXM has an unrelated Ptr<T>.
// Keep the legacy spelling local to Apple declarations; do not macro-rename
// guest pointers or change their ABI in Objective-C++ translation units.
#define Ptr Vita3KAppleLegacyPtr
#define ThreadState Vita3KAppleLegacyThreadState
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#undef Ptr
#undef ThreadState
