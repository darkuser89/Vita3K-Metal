// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Complete production precompile_shader(), with CPU compiler/cache doubles.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <renderer/metal4/compile_jobs.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
using namespace std::chrono_literals;
using renderer::metal::CompileJobs;
#define LOG_WARN(...) ((void)0)
static void require(bool b,const std::string &error){if(!b)throw std::runtime_error(error);}
static std::string hex_string(const std::string &hash){return hash;}
namespace shader::metal {
enum class Stage {Vertex,Fragment};
struct Program {Stage stage;std::string key;};
}
struct CompiledProgram {shader::metal::Stage stage;id<MTLFunction> function;};
struct ShadersHash {std::string frag="f",vert="v";};
@interface WarmFunction : NSObject
@property(copy) NSString *name;
@end
@implementation WarmFunction
@end
@interface WarmupDescriptor : NSObject <NSCopying>
@property(strong) id<MTLFunction> vertexFunction,fragmentFunction;
@property(copy) NSString *tag;
@end
@implementation WarmupDescriptor
- (id)copyWithZone:(NSZone *)zone {
    auto copy=[WarmupDescriptor new];copy.vertexFunction=_vertexFunction;copy.fragmentFunction=_fragmentFunction;copy.tag=_tag;return copy;
}
@end
struct CachedVariant {std::string key;bool gamma_correction=false;};
struct CachedPipeline {std::string key,vertex_key="v",fragment_key="f",vertex_function="vs",fragment_function="fs";WarmupDescriptor *descriptor;};
struct Device {
    std::thread::id owner=std::this_thread::get_id();
    std::mutex mutex;std::condition_variable changed;
    unsigned active=0,maximum=0,entered=0,completed=0,shaders=0;
    bool overlap=false,release=false,abort_during_compile=false,check_workers=true;
    std::atomic_bool *abort=nullptr;
    std::vector<CachedPipeline> saved;
    std::set<std::string> failures;
    void enter(bool pipeline) {
        if(check_workers)assert(std::this_thread::get_id()!=owner);
        std::lock_guard lock(mutex);++active;maximum=std::max(maximum,active);assert(active<=2);
        if(pipeline){++entered;changed.notify_all();}
        else ++shaders;
    }
    void leave(){std::lock_guard lock(mutex);--active;++completed;changed.notify_all();}
    std::optional<shader::metal::Program> load_cached_program(const std::string &key) {
        return shader::metal::Program{key=="v"?shader::metal::Stage::Vertex:shader::metal::Stage::Fragment,key};
    }
    std::unique_ptr<CompiledProgram> compile(const shader::metal::Program &p,bool,std::string &,bool) {
        enter(false);auto f=[WarmFunction new];f.name=p.key=="v"?@"vs":@"fs";
        auto result=std::make_unique<CompiledProgram>();result->stage=p.stage;result->function=(id<MTLFunction>)f;
        leave();return result;
    }
    std::vector<CachedVariant> cached_variants(const std::string &hash){return {{hash}};}
    std::vector<CachedPipeline> cached_pipeline_templates(const std::string &,const std::string &){return saved;}
    id<MTLRenderPipelineState> create_pipeline(WarmupDescriptor *d,std::string &error) {
        enter(true);
        {
            std::unique_lock lock(mutex);
            if(entered<=2)changed.wait(lock,[&]{return release;});
            if(abort_during_compile)abort->store(true);
        }
        // Retained descriptor/function ownership must survive until the job's
        // native compiler boundary returns, including the depth-only variant.
        assert([d.vertexFunction.name isEqual:@"vs"]&&[d.fragmentFunction.name isEqual:@"fs"]);
        const std::string key=d.tag.UTF8String;
        leave();
        if(key=="4")throw std::runtime_error("driver boundary exception");
        if(failures.contains(key)){error="invalid cached descriptor";return nil;}
        return (id<MTLRenderPipelineState>)[NSObject new];
    }
};
struct MetalState {
    struct Impl {
        struct PipelineResult {id<MTLRenderPipelineState> pipeline=nil;std::string error;};
        std::unique_ptr<Device> device=std::make_unique<Device>();
        std::map<std::string,std::unique_ptr<CompiledProgram>> shaders;
        std::map<std::string,id<MTLRenderPipelineState>> pipelines;
        std::set<std::pair<std::string,std::string>> known_shader_pairs;
        std::set<std::string> warmed_vertex_hashes,warmed_fragment_hashes,warmup_failed_shaders,warmup_failed_pipelines;
        std::set<std::string> failed_shaders,failed_pipelines;
        std::string dump_draw_dir;
        void collect_shaders(bool,uint32_t &){}void collect_pipelines(bool){}
    };
    std::unique_ptr<Impl> impl=std::make_unique<Impl>();
    std::atomic_bool render_abort=false;
    uint32_t shaders_count_compiled=0,pipelines_count_precompiled=0,programs_count_pre_compiled=0;
    void precompile_shader(const ShadersHash &);
};
// WARMUP_HOOK
static void setup(MetalState &state,unsigned count) {
    auto &d=*state.impl->device;d.abort=&state.render_abort;
    for(unsigned i=0;i<count;++i){
        const auto key=std::to_string(i);auto desc=[WarmupDescriptor new];desc.tag=[NSString stringWithUTF8String:key.c_str()];
        d.saved.push_back({key,"v","f","vs","fs",desc});
    }
}
int main() { @autoreleasepool {
    for(bool cancel:{false,true}) {
        MetalState state;setup(state,10);auto &d=*state.impl->device;
        // A failed speculative descriptor must not blacklist the actual draw.
        d.failures={"1","4"};d.abort_during_compile=cancel;
        // Add one helper shader mid-sequence: pipeline jobs must retire before
        // it starts so combined compiler concurrency remains bounded by two.
        d.saved[4].fragment_key="metal-depth-only-helper";
        d.saved[8].vertex_function="mismatched-entry";
        d.saved[9].vertex_key="missing-variant";
        auto observer=std::thread([&] {
            std::unique_lock lock(d.mutex);
            d.overlap=d.changed.wait_for(lock,2s,[&]{return d.entered>=2;});
            d.release=true;d.changed.notify_all();
        });
        state.precompile_shader({});observer.join();
        assert(d.overlap&&"independent pipeline warmup calls must overlap");
        assert(d.active==0&&d.maximum==2&&state.programs_count_pre_compiled==1);
        assert(state.impl->failed_shaders.empty()&&state.impl->failed_pipelines.empty());
        assert(state.pipelines_count_precompiled==state.impl->pipelines.size());
        assert(state.pipelines_count_precompiled+state.impl->warmup_failed_pipelines.size()==d.entered);
        if(cancel)assert(d.entered<8);
        else {
            assert(d.entered==8&&state.pipelines_count_precompiled==6&&state.shaders_count_compiled==3);
            const auto before=d.entered;state.precompile_shader({});assert(d.entered==before);
        }
    }
    MetalState skipped;setup(skipped,3);skipped.render_abort=true;skipped.precompile_shader({});
    assert(skipped.impl->device->entered==0&&skipped.programs_count_pre_compiled==1);
    std::cout<<"PASS complete pipeline warmup: controlled overlap of two native calls, bounded combined shader/pipeline workers, descriptor retention, render-thread result publication, speculative failure isolation, duplicate suppression, abort drain and skipped warmup; CPU doubles only, no GPU/app\n";
} }
