// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Production retirement function; native host/layer boundary doubles, real
// Foundation ARC and main dispatch queue. No window, app or Metal device.
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
static std::vector<int> events;
static unsigned layers_removed, views_destroyed, windows_destroyed;
@class FakeWindow;
@interface FakeLayer : NSObject
@property(copy) NSDictionary *developerHUDProperties;
@property(strong) id device;
- (void)removeFromSuperlayer;
@end
@implementation FakeLayer
- (void)setDeveloperHUDProperties:(NSDictionary *)value {
    assert(NSThread.isMainThread);
    assert([value[@"mode"] isEqual:@"disabled"]);
    _developerHUDProperties = [value copy]; events.push_back(1);
}
- (void)removeFromSuperlayer { assert(NSThread.isMainThread); ++layers_removed; events.push_back(2); }
- (void)setDevice:(id)value { assert(NSThread.isMainThread); _device=value; if(!value)events.push_back(3); }
@end
@interface FakeView : NSObject
@property(strong) FakeLayer *layer;
@property(weak) FakeWindow *window;
@end
@implementation FakeView
- (void)dealloc { ++views_destroyed; }
@end
@interface FakeWindow : NSObject
@property(strong) FakeView *contentView;
@end
@implementation FakeWindow
- (void)dealloc { ++windows_destroyed; }
@end
@interface FakeTransaction : NSObject
+ (void)begin;
+ (void)setDisableActions:(BOOL)value;
+ (void)commit;
+ (void)flush;
@end
@implementation FakeTransaction
+ (void)begin { assert(NSThread.isMainThread); }
+ (void)setDisableActions:(BOOL)value { assert(value); }
+ (void)commit { events.push_back(4); }
+ (void)flush { events.push_back(5); }
@end
// RETIRE_HOOK
int main() {
    @autoreleasepool {
        FakeLayer *child=[FakeLayer new], *backing=[FakeLayer new];
        FakeView *view=[FakeView new]; view.layer=backing;
        FakeWindow *window=[FakeWindow new]; window.contentView=view; view.window=window;
        child.device=[NSObject new];
        __weak FakeView *weak_view=view; __weak FakeWindow *weak_window=window;
        // Simulate a HUD query already queued when the renderer begins cleanup.
        dispatch_async(dispatch_get_main_queue(), ^{
            assert(weak_view && weak_window && weak_view.window==weak_window);
            assert([child.developerHUDProperties[@"mode"] isEqual:@"disabled"] && !child.device);
            events.push_back(6);
        });
        retire_presentation_layer(child,view);
        retire_presentation_layer(nil,nil); // repeated / failed-init cleanup
        assert((events==std::vector<int>{1,2,3,4,5}));
        window=nil;view=nil;
        assert(weak_view && weak_window); // deferred ownership is real under ARC
        dispatch_async(dispatch_get_main_queue(), ^{
            @autoreleasepool {
                assert(!weak_view && !weak_window && views_destroyed==1 && windows_destroyed==1);
                assert(events.back()==6 && layers_removed==1);
                FakeLayer *root=[FakeLayer new];
                FakeView *host=[FakeView new];host.layer=root;
                retire_presentation_layer(root,host);
                assert(layers_removed==1); // frontend backing layer stays attached
                retire_presentation_layer([FakeLayer new],nil); // partial host setup
                dispatch_async(dispatch_get_main_queue(), ^{
                    std::cout << "PASS presentation retirement: production function, HUD disabled only at shutdown, queued HUD host lifetime, child/backing-layer ownership, device release, nil/idempotent cleanup and main-queue ordering; ARC/ASAN/UBSAN\n" << std::flush;
                    std::exit(0);
                });
            }
        });
    }
    dispatch_main();
}
