// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_GetGpuDeviceCount(int);
int TSGgml_MultiDeviceInit(int,const int*,int);
int TSGgml_SetActiveDevice(int);
const char* TSGgml_GetLastError();
void TSGgml_Shutdown();
int TSGgml_FusedPrefillAttentionF16KV(const float*,const void*,const void*,float*,int,int,int,int,int,int,int,int,float);
int TSGgml_MuseSparkAttentionF16KV2D(const float*,const void*,const void*,float*,int,int,int,int,int,int,int,int,float);
std::uint64_t TSGgml_GetMuseSparkKv2DUploadPairs();
}

// All generated values are exactly representable normal binary fractions.
static std::uint16_t half(float value) {
    if (value == 0) return 0;
    std::uint32_t bits; std::memcpy(&bits,&value,4);
    return static_cast<std::uint16_t>(((bits>>16)&0x8000)|((((bits>>23)&255)-112)<<10)|((bits>>13)&1023));
}

static bool run(int heads,int kv_heads,int rows,int keys,int first,int padding,int window,int salt) {
    constexpr int dim=64;
    const int capacity=first+keys+padding, query_start=keys-rows;
    const float scale=1/std::sqrt(float(dim));
    std::vector<std::uint16_t> k(kv_heads*capacity*dim,0x7e00),v(k.size(),0x7e00);
    std::vector<float> q(heads*rows*dim),old(q.size()+8,71.f),out(old.size(),71.f);
    auto key_value=[&](int t,int h){return ((t*17+h*3+salt)%31-15)/32.f;};
    auto value=[&](int t,int h){return ((t*7+h*5+salt)%61-30)/64.f;};
    for (int h=0;h<kv_heads;++h) for (int t=0;t<keys;++t) {
        const int at=(h*capacity+first+t)*dim;
        std::fill(k.begin()+at,k.begin()+at+dim,0);
        k[at]=half(key_value(t,h));
        for (int d=0;d<dim;++d) v[at+d]=half(value(t,h)+(d%5-2)/128.f);
    }
    for (int h=0;h<heads;++h) for (int r=0;r<rows;++r) q[(h*rows+r)*dim]=((h+r+salt)%7-3)/4.f;
    const auto original_k=k,original_v=v;
    const auto pairs_before=TSGgml_GetMuseSparkKv2DUploadPairs();
    if (!TSGgml_FusedPrefillAttentionF16KV(q.data(),k.data()+first*dim,v.data()+first*dim,old.data(),
        heads,kv_heads,dim,rows,keys,capacity,query_start,window,scale)) {
        std::fprintf(stderr,"legacy: %s\n",TSGgml_GetLastError());return false;
    }
    if (pairs_before!=TSGgml_GetMuseSparkKv2DUploadPairs()) return false;
    // Poison the exact same shape/backend session before the candidate call.
    // Without this, a missing candidate upload could reuse the immediately
    // preceding reference K/V and falsely pass equality and the scalar oracle.
    auto poison_k=k,poison_v=v;
    std::vector<float> discard(out.size(),71.f);
    for (int h=0;h<kv_heads;++h) for (int t=0;t<keys;++t) {
        const int at=(h*capacity+first+t)*dim;
        poison_k[at]=half(.75f);
        for (int d=0;d<dim;++d) poison_v[at+d]=half(.75f);
    }
    if (!TSGgml_FusedPrefillAttentionF16KV(q.data(),poison_k.data()+first*dim,poison_v.data()+first*dim,discard.data(),
        heads,kv_heads,dim,rows,keys,capacity,query_start,window,scale) ||
        pairs_before!=TSGgml_GetMuseSparkKv2DUploadPairs()) return false;
    if (!TSGgml_MuseSparkAttentionF16KV2D(q.data(),k.data()+first*dim,v.data()+first*dim,out.data(),
        heads,kv_heads,dim,rows,keys,capacity,query_start,window,scale)) {
        std::fprintf(stderr,"2D: %s\n",TSGgml_GetLastError());return false;
    }
    if (TSGgml_GetMuseSparkKv2DUploadPairs()!=pairs_before+1 || original_k!=k || original_v!=v ||
        std::memcmp(old.data(),out.data(),out.size()*sizeof(float))!=0) {
        std::fprintf(stderr,"exact legacy parity/source/sentinel/actual-use failure h%d kv%d rows%d keys%d first%d pad%d w%d\n",
            heads,kv_heads,rows,keys,first,padding,window);return false;
    }
    for (int h=0;h<heads;++h) for (int r=0;r<rows;++r) {
        const int end=query_start+r, begin=window ? std::max(0,end-window+1) : 0, kh=h/(heads/kv_heads);
        double sum=0,mean=0;
        for (int t=begin;t<=end;++t) {
            const double prob=std::exp(double(q[(h*rows+r)*dim])*key_value(t,kh)*scale);
            sum+=prob;mean+=prob*value(t,kh);
        }
        for (int d=0;d<dim;++d) {
            const double expected=mean/sum+(d%5-2)/128.;
            const float actual=out[(r*heads+h)*dim+d];
            if (!std::isfinite(actual) || std::abs(actual-expected)>1e-3) {
                std::fprintf(stderr,"oracle failure h%d rows%d keys%d row%d dim%d value%g expected%g\n",heads,rows,keys,r,d,actual,expected);
                return false;
            }
        }
    }
    return true;
}

int main(int argc,char** argv) {
    const bool multi=argc==2 && std::strcmp(argv[1],"multi")==0;
    const bool cuda=multi || (argc==2 && std::strcmp(argv[1],"cuda")==0);
    if (argc!=2 || (!cuda && std::strcmp(argv[1],"cpu")!=0)) return 2;
    if (!TSGgml_IsBackendAvailable(cuda ? 3 : 2)) return 77;
    if (multi) {
        const int devices[]={0,1};
        if (TSGgml_GetGpuDeviceCount(3)<2) return 77;
        if (!TSGgml_MultiDeviceInit(3,devices,2)) return 1;
    }
    int checks=0;
    for (int repeat=0;repeat<(multi ? 4 : 2);++repeat) {
        const int rank=multi ? repeat%2 : 0;
        if (multi && !TSGgml_SetActiveDevice(rank)) return 1;
        for (const auto geometry : {std::pair<int,int>{4,4},{4,2},{128,16}})
        for (int rows : {1,2,9,128,256}) for (int window : {0,17}) for (int padding : {0,13}) {
            if (!run(geometry.first,geometry.second,rows,std::max(rows,65),window ? 17 : 0,padding,window,checks+rank)) return 1;
            ++checks;
        }
        // Bucket growth/shrink and changed values reuse the same graph cache.
        for (int keys : {63,64,65,2047,2048,2049,8193,65,63}) {
            if (!run(128,16,1,keys,37,7,2048,checks+rank)) return 1;
            ++checks;
        }
        // Exceeds the192MiB scores threshold: native flash (or exact legacy
        // materialized fallback) must also retain the2D path and same outputs.
        if (!run(128,16,128,3073,13,7,2048,checks+rank)) return 1;
        ++checks;
    }
    std::printf("PASS %d cases: exact legacy output bytes, independent scalar attention, poisoned padding, offset windows, actual2D uploads\n",checks);
    TSGgml_Shutdown();
    return 0;
}
