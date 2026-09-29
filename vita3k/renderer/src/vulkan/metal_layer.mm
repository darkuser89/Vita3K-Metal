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

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

extern "C" void *get_metal_layer_from_view(void *nsview) {
    NSView *view = (__bridge NSView *)nsview;
    if (!view)
        return nullptr;
    if (!view.layer)
        view.wantsLayer = YES;
    CALayer *root = view.layer;
    if (!root)
        return nullptr;

    CAMetalLayer *metal_layer = [root isKindOfClass:[CAMetalLayer class]] ? (CAMetalLayer *)root : nil;
    if (!metal_layer) {
        // Qt retains its QContainerLayer for VulkanSurface windows. MoltenVK
        // requires a CAMetalLayer, so keep one sized with the Qt-owned layer.
        for (CALayer *child in root.sublayers)
            if ([child isKindOfClass:[CAMetalLayer class]] && [child.name isEqualToString:@"Vita3K Vulkan"])
                metal_layer = (CAMetalLayer *)child;
        if (!metal_layer) {
            metal_layer = [CAMetalLayer layer];
            metal_layer.name = @"Vita3K Vulkan";
            metal_layer.frame = root.bounds;
            metal_layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
            [root addSublayer:metal_layer];
        }
    }
    metal_layer.contentsScale = view.window.backingScaleFactor;
    return (__bridge void *)metal_layer;
}
