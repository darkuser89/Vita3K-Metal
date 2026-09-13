// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/overlay.h>
#include <renderer/metal/screen.h>
#include <overlay/display_manager.h>
#include <overlay/common_dialog.h>
#include <overlay/font.h>
#include <dialog/state.h>
#include <iostream>
#include <stdexcept>

struct TestOverlay : overlay::overlay {
    ::overlay::compiled_resource get_compiled() override {
        ::overlay::compiled_resource result;
        ::overlay::compiled_resource::command command;
        command.config.color = {1, 0, 0, 0.5};
        command.config.clip_region = true;
        command.config.clip_rect = {0, 0, 480, 544};
        command.verts = {{0.f,0.f,0.f,0.f}, {960.f,0.f,1.f,0.f}, {0.f,272.f,0.f,1.f}, {960.f,272.f,1.f,1.f}};
        result.draw_commands.push_back(command);
        return result;
    }
};
int main(int argc, char **argv) {
    if (argc != 2 && argc != 4) { std::cerr << "Usage: metal-overlay-validation <static-assets-dir> [firmware-font-dir output.png]\n"; return 2; }
    @autoreleasepool {
        try {
            std::string error;
            auto device = renderer::metal::Device::create(error);
            if (!device) throw std::runtime_error(error);
            renderer::metal::OverlayRenderer renderer(*device, argv[1]);
            renderer::metal::ScreenRenderer screen(*device);
            screen.set_filter(false);
            auto source_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:2 height:2 mipmapped:NO];
            source_desc.storageMode = MTLStorageModeShared;
            source_desc.usage = MTLTextureUsageShaderRead;
            auto source = [device->native_device() newTextureWithDescriptor:source_desc];
            const uint8_t blue[16] = {0,0,255,255, 0,0,255,255, 0,0,255,255, 0,0,255,255};
            [source replaceRegion:MTLRegionMake2D(0,0,2,2) mipmapLevel:0 withBytes:blue bytesPerRow:8];
            auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:64 height:64 mipmapped:NO];
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.usage = MTLTextureUsageRenderTarget;
            auto texture = [device->native_device() newTextureWithDescriptor:descriptor];
            auto commands = [device->command_queue() commandBuffer];
            auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,1);
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
            screen.render(encoder, source, {0,0,64,64,0,1});
            overlay::display_manager manager;
            manager.create<TestOverlay>()->visible = true;
            renderer.render(encoder, manager, {0,0,64,64,0,1});
            [encoder endEncoding];
            if (!device->submit_and_wait(commands, error)) throw std::runtime_error(error);
            std::vector<uint8_t> pixels(64*64*4);
            [texture getBytes:pixels.data() bytesPerRow:64*4 fromRegion:MTLRegionMake2D(0,0,64,64) mipmapLevel:0];
            for (int y = 0; y < 64; ++y) for (int x = 0; x < 64; ++x) {
                const bool drawn = x < 32 && y < 32;
                const int expected[] = {drawn ? 128 : 255, 0, drawn ? 128 : 0, 255};
                for (int c = 0; c < 4; ++c)
                    if (std::abs(int(pixels[(y*64+x)*4+c]) - expected[c]) > 1)
                        throw std::runtime_error("Overlay pixel mismatch x=" + std::to_string(x) + " y=" + std::to_string(y) + " channel=" + std::to_string(c));
            }
            std::cout << "PASS production screen + overlay pipeline, Y orientation, clip bounds, alpha blend and all BGRA pixels\n";
            if (argc == 4) {
                overlay::fontmgr::set_firmware_font_dir(argv[2]);
                overlay::resource_config::set_icons_dir(std::string(argv[1]) + "/icons/");
                overlay::display_manager dialogs;
                DialogState state{};
                state.type = MESSAGE_DIALOG;
                state.status = SCE_COMMON_DIALOG_STATUS_RUNNING;
                state.msg.message = "Native Metal 3 dialog test";
                state.msg.btn_num = 1;
                state.msg.btn[0] = "OK";
                auto dialog = dialogs.create<overlay::common_dialog_overlay>();
                if (!dialog->poll_dialog(state, 0, 1)) throw std::runtime_error("Dialog did not become active");
                const auto now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                dialog->update(now);
                dialog->update(now + 1000000);
                descriptor.width = 960; descriptor.height = 544;
                auto dialog_texture = [device->native_device() newTextureWithDescriptor:descriptor];
                pass.colorAttachments[0].texture = dialog_texture;
                pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,1);
                commands = [device->command_queue() commandBuffer];
                encoder = [commands renderCommandEncoderWithDescriptor:pass];
                renderer.render(encoder, dialogs, {0,0,960,544,0,1});
                [encoder endEncoding];
                if (!device->submit_and_wait(commands, error)) throw std::runtime_error(error);
                pixels.resize(960*544*4);
                [dialog_texture getBytes:pixels.data() bytesPerRow:960*4 fromRegion:MTLRegionMake2D(0,0,960,544) mipmapLevel:0];
                size_t bright = 0;
                for (size_t i = 0; i < pixels.size(); i += 4) if (pixels[i] > 180 && pixels[i+1] > 180 && pixels[i+2] > 180) ++bright;
                if (bright < 100) throw std::runtime_error("Native common dialog has no visible text");
                auto bitmap = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:nullptr pixelsWide:960 pixelsHigh:544 bitsPerSample:8 samplesPerPixel:4 hasAlpha:YES isPlanar:NO colorSpaceName:NSDeviceRGBColorSpace bytesPerRow:960*4 bitsPerPixel:32];
                for (size_t i = 0; i < pixels.size(); i += 4) {
                    bitmap.bitmapData[i] = pixels[i+2]; bitmap.bitmapData[i+1] = pixels[i+1];
                    bitmap.bitmapData[i+2] = pixels[i]; bitmap.bitmapData[i+3] = pixels[i+3];
                }
                if (![[bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:[NSString stringWithUTF8String:argv[3]] atomically:YES])
                    throw std::runtime_error("Cannot write dialog evidence image");
                std::cout << "PASS real common dialog, firmware font atlas and visible text (" << bright << " bright pixels)\n";
                // The same screen renderer used for CAMetalLayer presentation,
                // with a 2x game viewport inside a letterboxed framebuffer.
                constexpr uint32_t width = 2048, height = 1200;
                const MTLViewport viewport{64,56,1920,1088,0,1};
                descriptor.width = width; descriptor.height = height;
                auto composite = [device->native_device() newTextureWithDescriptor:descriptor];
                const uint8_t quadrants[16] = {255,0,0,255, 0,255,0,255, 0,0,255,255, 255,0,255,255};
                [source replaceRegion:MTLRegionMake2D(0,0,2,2) mipmapLevel:0 withBytes:quadrants bytesPerRow:8];
                pass.colorAttachments[0].texture = composite;
                commands = [device->command_queue() commandBuffer];
                encoder = [commands renderCommandEncoderWithDescriptor:pass];
                screen.render(encoder, source, viewport);
                [encoder endEncoding];
                if (!device->submit_and_wait(commands, error)) throw std::runtime_error(error);
                pixels.resize(size_t(width)*height*4);
                [composite getBytes:pixels.data() bytesPerRow:width*4 fromRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0];
                for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
                    const bool inside = x >= 64 && x < 1984 && y >= 56 && y < 1144;
                    const size_t quadrant = (y >= 600 ? 2 : 0) + (x >= 1024 ? 1 : 0);
                    const uint8_t expected[4] = {inside ? quadrants[quadrant*4+2] : uint8_t(0),
                        inside ? quadrants[quadrant*4+1] : uint8_t(0), inside ? quadrants[quadrant*4] : uint8_t(0), 255};
                    for (int c = 0; c < 4; ++c)
                        if (pixels[(size_t(y)*width+x)*4+c] != expected[c])
                            throw std::runtime_error("Screen quadrant or letterbox pixel mismatch");
                }
                commands = [device->command_queue() commandBuffer];
                encoder = [commands renderCommandEncoderWithDescriptor:pass];
                screen.render(encoder, source, viewport);
                renderer.render(encoder, dialogs, viewport);
                [encoder endEncoding];
                if (!device->submit_and_wait(commands, error)) throw std::runtime_error(error);
                [composite getBytes:pixels.data() bytesPerRow:width*4 fromRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0];
                bright = 0;
                for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
                    const size_t i = (size_t(y)*width+x)*4;
                    const bool inside = x >= 64 && x < 1984 && y >= 56 && y < 1144;
                    if (!inside && (pixels[i] || pixels[i+1] || pixels[i+2]))
                        throw std::runtime_error("Overlay escaped the letterboxed viewport");
                    if (pixels[i] > 180 && pixels[i+1] > 180 && pixels[i+2] > 180) ++bright;
                }
                if (bright < 1000) throw std::runtime_error("2x composite lost common dialog text");
                std::cout << "PASS production screen quadrants, 2x scaling, letterboxing and composed dialog (" << bright << " bright pixels)\n";

            }
            return 0;
        } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
    }
}
