// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/polygon_clip.h>
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include "baseline_polygon.inl"
using namespace renderer::metal;
static bool close(double a, double b, double epsilon = 2e-4) { return std::abs(a-b) <= epsilon * std::max({1.0,std::abs(a),std::abs(b)}); }
static CapturedVertexOutputs vertex(float x, float y, float w = 1) {
    CapturedVertexOutputs result{}; result[0] = {x,y,0.5f*w,w};
    for (size_t slot=1; slot<result.size(); ++slot) result[slot] = {1,2,3,4};
    result[15][0]=w; result[16][0]=1.5f*w; result[17][0]=0.5f*w;
    return result;
}
static double area(const std::vector<CapturedVertexOutputs> &vertices) {
    double sum=0;
    for (size_t i=0;i+2<vertices.size();i+=3) {
        const auto &a=vertices[i][0], &b=vertices[i+1][0], &c=vertices[i+2][0];
        sum+=std::abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2;
    }
    return sum;
}
static bool inside(double x, double y, const std::vector<CapturedVertexOutputs> &triangles) {
    for (size_t i=0;i+2<triangles.size();i+=3) {
        bool positive=true,negative=true;
        for (size_t corner=0;corner<3;++corner) {
            const auto &a=triangles[i+corner][0], &b=triangles[i+(corner+1)%3][0];
            const double cross=(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]);
            positive &= cross >= -1e-8; negative &= cross <= 1e-8;
        }
        if (positive || negative) return true;
    }
    return false;
}
int main() {
    const std::array<uint32_t,3> indices{0,1,2};
    std::array<CapturedVertexOutputs,3> triangle{vertex(-0.75f,-0.75f),vertex(0.75f,-0.75f),vertex(0,0.75f)};
    size_t hand=0;
    // Every captured plane, especially guest planes 5..7 that never reached
    // the original native interface, must affect all applicable routes.
    for (unsigned plane=0;plane<11;++plane) {
        auto rejected=triangle;
        for (auto &v:rejected) v[15+plane][0]=-1;
        for (auto kind:{ClipPrimitiveKind::Points,ClipPrimitiveKind::Lines,ClipPrimitiveKind::Triangles}) {
            assert(route_clip_overflow(kind,PolygonTopology::List,indices,rejected,false,false,false).empty()); ++hand;
        }
        assert(route_point_polygons(PolygonTopology::List,indices,rejected,false,false).empty());
        assert(route_wide_lines(true,PolygonTopology::List,indices,rejected,false,false,100,100,2,3).empty());
        auto clipped=triangle;
        for (auto &v:clipped) v[15+plane][0]=v[0][0]; // x >= 0: exactly half the symmetric triangle.
        const auto result=route_clip_overflow(ClipPrimitiveKind::Triangles,PolygonTopology::List,indices,clipped,false,false,false);
        assert(result.size()==1 && close(area(result[0].vertices),0.5625));
        const auto wire=route_clip_overflow(ClipPrimitiveKind::Triangles,PolygonTopology::List,indices,clipped,false,false,true);
        assert(wire.size()==1 && wire[0].kind==ClipPrimitiveKind::Lines && wire[0].vertices.size()==6);
        for (const auto &v:result[0].vertices) assert(v[0][0]>=-1e-6 && v[15+plane][0]>=0);
    }
    // Four-vertex clipped polygon: the wire boundary has no triangulation diagonal.
    auto quad=triangle;
    for(auto &v:quad) v[25][0]=0.5f-v[0][1];
    auto fill=route_clip_overflow(ClipPrimitiveKind::Triangles,PolygonTopology::List,indices,quad,false,false,false);
    auto wire=route_clip_overflow(ClipPrimitiveKind::Triangles,PolygonTopology::List,indices,quad,false,false,true);
    assert(fill.size()==1 && fill[0].vertices.size()==6 && wire[0].vertices.size()==8);
    // Source order, winding/culling, list/strip/fan and incomplete tails.
    for (auto topology:{PolygonTopology::List,PolygonTopology::Strip,PolygonTopology::Fan}) {
        const std::array<uint32_t,6> order=topology==PolygonTopology::List ? std::array<uint32_t,6>{0,1,2,2,1,0} : std::array<uint32_t,6>{0,1,2,0,1,2};
        auto polygons=route_point_polygons(topology,order,triangle,false,false);
        auto routed=route_clip_overflow(ClipPrimitiveKind::Triangles,topology,order,triangle,false,false,false);
        size_t at=0;
        for (const auto &p:polygons) if (p.points.size()>=3) { assert(routed[at].face==p.face); ++at; }
        assert(at==routed.size());
        assert(route_clip_overflow(ClipPrimitiveKind::Triangles,topology,order,triangle,true,true,false).empty());
    }
    auto point=vertex(3,0); // Point X/Y extent stays a native rasterizer decision.
    assert(route_clip_overflow(ClipPrimitiveKind::Points,PolygonTopology::List,std::array<uint32_t,1>{0},std::span<const CapturedVertexOutputs>(&point,1),false,false,false).size()==1);
    point[25][0]=std::numeric_limits<float>::quiet_NaN();
    assert(route_clip_overflow(ClipPrimitiveKind::Points,PolygonTopology::List,std::array<uint32_t,1>{0},std::span<const CapturedVertexOutputs>(&point,1),false,false,false).empty());
    bool rejected=false;
    try {(void)route_clip_overflow(ClipPrimitiveKind::Points,PolygonTopology::List,std::array<uint32_t,1>{3},triangle,false,false,false);} catch(const std::out_of_range &) {rejected=true;}
    assert(rejected);
    std::mt19937 random(0x434c4950);
    auto value=[&] {return (int(random()%2001)-1000)/1000.f;};
    // Independent parametric line interval oracle, including nonuniform W,
    // all planes, native XY bounds and interpolated output attributes.
    for (unsigned trial=0;trial<10000;++trial) {
        std::array<CapturedVertexOutputs,2> line{vertex(value()*2,value()*2,1.1f+value()),vertex(value()*2,value()*2,1.1f+value())};
        for(auto &v:line) {
            for(size_t slot=1;slot<15;++slot) for(auto &component:v[slot]) component=value()*100;
            for(size_t p=3;p<11;++p) v[15+p][0]=value();
        }
        double lo=0,hi=1;
        auto distance=[](const CapturedVertexOutputs &v,unsigned p)->double {
            if(p<4) return double(v[0][3])+(p%2?-1:1)*double(v[0][p/2]);
            return v[15+p-4][0];
        };
        for(unsigned p=0;p<15;++p) {
            double a=distance(line[0],p),b=distance(line[1],p);
            if(a<0 && b<0) {lo=2;break;}
            if(a<0) lo=std::max(lo,a/(a-b));
            if(b<0) hi=std::min(hi,a/(a-b));
        }
        auto result=clip_line_outputs(line,true);
        assert(bool(result)==(lo<=hi));
        if(result) for(size_t end=0;end<2;++end) for(size_t slot=0;slot<26;++slot) for(size_t c=0;c<4;++c) {
            double expected=line[0][slot][c]+(end?hi:lo)*(double(line[1][slot][c])-line[0][slot][c]);
            if(slot>=15 && c==0) expected=std::max(0.0,expected);
            assert(close((*result)[end][slot][c],expected));
        }
    }
    size_t samples=0;
    // A separate point-in-original-triangle + halfspace oracle checks the
    // union of replay triangles, avoiding points on rounding-sensitive edges.
    for(unsigned trial=0;trial<1000;++trial) {
        auto input=triangle;
        std::array<std::array<float,3>,8> planes;
        const unsigned count=trial%9;
        for(auto &plane:planes) plane={value(),value(),value()};
        for(unsigned p=0;p<count;++p) for(auto &v:input) v[18+p][0]=planes[p][0]*v[0][0]+planes[p][1]*v[0][1]+planes[p][2];
        auto result=route_clip_overflow(ClipPrimitiveKind::Triangles,PolygonTopology::List,indices,input,false,false,false);
        std::vector<CapturedVertexOutputs> outputs;
        for(const auto &part:result) outputs.insert(outputs.end(),part.vertices.begin(),part.vertices.end());
        std::vector<CapturedVertexOutputs> original(triangle.begin(),triangle.end());
        for(unsigned y=0;y<11;++y) for(unsigned x=0;x<11;++x) {
            double px=-0.8+x*0.1517,py=-0.8+y*0.1531;
            bool expected=inside(px,py,original),boundary=false;
            for(unsigned p=0;p<count;++p) {
                double d=planes[p][0]*px+planes[p][1]*py+planes[p][2];
                boundary |= std::abs(d)<1e-4; expected &= d>=0;
            }
            if(boundary) continue;
            assert(inside(px,py,outputs)==expected); ++samples;
        }
    }
    // Exact old/new comparison for previously supported data: eight original
    // distances and three zero extra slots. The baseline is taken from HEAD,
    // not duplicated from this implementation.
    for(unsigned trial=0;trial<1000;++trial) {
        auto input=triangle;
        std::array<baseline::CapturedVertexOutputs,3> old{};
        for(size_t corner=0;corner<3;++corner) {
            for(size_t slot=1;slot<15;++slot) for(auto &component:input[corner][slot]) component=value()*100;
            for(size_t plane=0;plane<8;++plane) input[corner][15+plane][0]=trial%2 ? value() : std::abs(value());
            for(size_t plane=8;plane<11;++plane) input[corner][15+plane][0]=0;
            std::copy_n(input[corner].begin(),23,old[corner].begin());
        }
        auto a=route_point_polygons(PolygonTopology::List,indices,input,false,false);
        auto b=baseline::route_point_polygons(baseline::PolygonTopology::List,indices,old,false,false);
        assert(a.size()==b.size());
        for(size_t i=0;i<a.size();++i) {
            assert(unsigned(a[i].face)==unsigned(b[i].face) && a[i].points.size()==b[i].points.size());
            for(size_t j=0;j<a[i].points.size();++j) assert(std::equal(b[i].points[j].begin(),b[i].points[j].end(),a[i].points[j].begin()));
        }
        auto c=route_wide_lines(true,PolygonTopology::List,indices,input,false,false,100,100,2,3);
        auto d=baseline::route_wide_lines(true,baseline::PolygonTopology::List,indices,old,false,false,100,100,2,3);
        assert(c.size()==d.size());
        for(size_t i=0;i<c.size();++i) {
            assert(unsigned(c[i].face)==unsigned(d[i].face) && c[i].primitive_index==d[i].primitive_index);
            for(size_t j=0;j<6;++j) assert(std::equal(d[i].quad[j].begin(),d[i].quad[j].end(),c[i].quad[j].begin()));
        }
    }
    std::cout << "PASS clip geometry: all 11 planes, point/line/triangle/wire routes, topology/culling/NaN/bounds, 10000 independent line intervals, " << samples << " halfspace samples and 1000 exact baseline point/wide-line routing comparisons; ASAN/UBSAN, no GPU\n";
}
