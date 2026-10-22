// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal4/apple.h>
#import <QuartzCore/CATransaction.h>
#include <memory>

namespace renderer::metal {
// GPU users have retired before this is called. Finish HUD/layer ownership on
// the UI queue while the FrameHost still owns its native view and window.
inline void retire_presentation_layer(CAMetalLayer *layer, NSView *view) {
    if (!layer) return;
    const auto retire = ^{
        struct Host {
            CAMetalLayer *layer;
            NSView *view;
            NSWindow *window;
        };
        auto host = std::make_shared<Host>(Host{layer, view, view.window});
        // Per-layer shutdown only: neither the user's HUD preference nor HUD
        // visibility during rendering is changed. Stop queries before Qt tears
        // down the native window referenced by HUDMTLLayerTracking.
        layer.developerHUDProperties = @{ @"mode" : @"disabled" };
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        // A child layer belongs to Metal; the backing layer belongs to the
        // frontend and must stay attached to its NSView until frontend teardown.
        if (view && layer != view.layer) [layer removeFromSuperlayer];
        layer.device = nil;
        [CATransaction commit];
        [CATransaction flush];
        // Already enqueued main-queue HUD callbacks must run before releasing
        // the last native host references. No event-loop pumping or sleep.
        dispatch_async(dispatch_get_main_queue(), ^{
            host->layer = nil;
            host->view = nil;
            host->window = nil;
        });
    };
    if (NSThread.isMainThread) retire();
    else dispatch_async(dispatch_get_main_queue(), retire);
}
} // namespace renderer::metal
