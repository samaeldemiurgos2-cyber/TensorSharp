// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Equal graph shapes must retain separate backend buffers across GPU ranks.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
int TSGgml_GetGpuDeviceCount(int);
int TSGgml_MultiDeviceInit(int, const int*, int);
int TSGgml_SetActiveDevice(int);
const char* TSGgml_GetLastError();
int TSGgml_FusedPrefillAttentionF32(const float*, const float*, const float*, float*,
    int,int,int,int,int,int,int,float,int);
int TSGgml_FusedPrefillAttentionF16KV(const float*, const void*, const void*, float*,
    int,int,int,int,int,int,int,int,float);
int TSGgml_PagedAttentionForward(const float*, const float*, const float*, float*,
    const int*,const int*,const int*,const int*,const int*,int,int,int,int,int,int,int,float);
}

static std::uint16_t exact_half(float f) {
    if (f == 0) return 0;
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<std::uint16_t>((((bits >> 23) & 255)-112) << 10 | ((bits >> 13) & 1023));
}

static bool long_paged_windows(int& checks, float& maximum) {
    constexpr int heads=4, kv_heads=2, dim=64, block_size=16;
    const float scale=1/std::sqrt(float(dim));
    // These histories exceed Spark's SWA window and deliberately put its first
    // visible row inside a block. Reverse the physical pages to prevent a
    // gather implementation from accidentally depending on contiguous IDs.
    for (int rank : {0,1}) for (int keys : {8211,16393}) {
        if (!TSGgml_SetActiveDevice(rank)) return false;
        const int pages=(keys+block_size-1)/block_size;
        std::vector<int> table(pages);
        std::vector<float> k(pages*block_size*kv_heads*dim), v(k.size());
        auto key_value=[](int token,int head) { return ((token*17+head*5)%73-36)/8.f; };
        auto value=[](int token,int head,int d) {
            return (token>8000 ? 0.25f : -0.5f)+((token*13+head*7+d)%61-30)/64.f;
        };
        for (int page=0; page<pages; ++page) table[page]=pages-page-1;
        for (int token=0; token<keys; ++token) for (int head=0; head<kv_heads; ++head) {
            const int physical=table[token/block_size]*block_size+token%block_size;
            k[(physical*kv_heads+head)*dim]=key_value(token,head);
            for (int d=0; d<dim; ++d) v[(physical*kv_heads+head)*dim+d]=value(token,head,d);
        }
        for (int queries : {1,7,33}) for (int window : {0,2048,17}) {
            const int first=keys-queries, starts[]={0,queries}, lengths[]={keys}, offsets[]={0};
            std::vector<int> positions(queries);
            std::vector<float> q(queries*heads*dim), output(q.size());
            for (int row=0; row<queries; ++row) {
                positions[row]=first+row;
                for (int head=0; head<heads; ++head) q[(row*heads+head)*dim]=(head+1)/2.f;
            }
            // Exercise both explicit positions and the equivalent implicit
            // suffix position path, including several shrinking windows.
            int ok=TSGgml_PagedAttentionForward(q.data(),k.data(),v.data(),output.data(),starts,lengths,
                rank ? nullptr : positions.data(),table.data(),offsets,1,queries,heads,kv_heads,dim,block_size,window,scale);
            if (!ok) { std::fprintf(stderr,"%s\n",TSGgml_GetLastError()); return false; }
            for (int row=0; row<queries; ++row) for (int head=0; head<heads; ++head) {
                const int end=first+row, begin=window ? std::max(0,end-window+1) : 0;
                const int kv_head=head/(heads/kv_heads);
                double sum=0;
                std::vector<double> expected(dim);
                for (int token=begin; token<=end; ++token) {
                    double probability=std::exp(double((head+1)/2.f)*key_value(token,kv_head)*scale);
                    sum+=probability;
                    for (int d=0; d<dim; ++d) expected[d]+=probability*value(token,kv_head,d);
                }
                for (int d=0; d<dim; ++d) {
                    const float error=std::abs(output[(row*heads+head)*dim+d]-float(expected[d]/sum));
                    maximum=std::max(maximum,error);
                    if (!std::isfinite(error) || error>1e-3f) {
                        std::fprintf(stderr,"long-paged rank%d keys%d queries%d window%d row%d head%d error%g\n",
                            rank,keys,queries,window,row,head,error);
                        return false;
                    }
                }
            }
            ++checks;
        }
    }
    return true;
}

int main() {
    if (TSGgml_GetGpuDeviceCount(3) < 2) return 77;
    const int devices[] = {0,1};
    if (!TSGgml_MultiDeviceInit(3, devices, 2)) { std::fprintf(stderr,"%s\n",TSGgml_GetLastError()); return 1; }
    constexpr int heads=4, kv_heads=2, dim=64, queries=3, keys=7, stride=9, start=4;
    std::vector<float> q(heads*queries*dim), k(kv_heads*keys*dim), v(k.size()), out(q.size());
    std::vector<std::uint16_t> kh(kv_heads*stride*dim), vh(kh.size());
    std::vector<float> pk(8*kv_heads*dim), pv(pk.size());
    const int query_starts[]={0,queries}, lengths[]={keys}, positions[]={4,5,6}, blocks[]={1,0}, offsets[]={0};
    float maximum=0;
    int checks=0;
    for (int repeat=0; repeat<4; ++repeat) for (int rank : {0,1,0,1}) {
        if (!TSGgml_SetActiveDevice(rank)) return 1;
        for (int h=0; h<kv_heads; ++h) for (int t=0; t<keys; ++t) for (int d=0; d<dim; ++d) {
            float value=(t+h+rank+repeat+1)/16.f;
            v[(h*keys+t)*dim+d]=value;
            vh[(h*stride+t)*dim+d]=exact_half(value);
            int physical=blocks[t/4]*4+t%4;
            pv[(physical*kv_heads+h)*dim+d]=value;
        }
        for (int mode=0; mode<3; ++mode) for (int window : {0,3}) {
            int ok= mode==0 ? TSGgml_FusedPrefillAttentionF32(q.data(),k.data(),v.data(),out.data(),
                heads,kv_heads,dim,queries,keys,start,window,1/std::sqrt(float(dim)),0) :
                mode==1 ? TSGgml_FusedPrefillAttentionF16KV(q.data(),kh.data(),vh.data(),out.data(),
                heads,kv_heads,dim,queries,keys,stride,start,window,1/std::sqrt(float(dim))) :
                TSGgml_PagedAttentionForward(q.data(),pk.data(),pv.data(),out.data(),query_starts,lengths,
                positions,blocks,offsets,1,queries,heads,kv_heads,dim,4,window,1/std::sqrt(float(dim)));
            if (!ok) { std::fprintf(stderr,"%s\n",TSGgml_GetLastError()); return 1; }
            for (int row=0; row<queries; ++row) for (int head=0; head<heads; ++head) for (int d=0; d<dim; ++d) {
                int end=start+row, begin=window==0 ? 0 : std::max(0,end-window+1);
                float expected=((begin+end)*0.5f+head/(heads/kv_heads)+rank+repeat+1)/16.f;
                float error=std::abs(out[(row*heads+head)*dim+d]-expected);
                maximum=std::max(maximum,error);
                if (!std::isfinite(error) || error>2e-4f) {
                    std::fprintf(stderr,"rank%d mode%d window%d row%d head%d error%g\n",rank,mode,window,row,head,error);
                    return 1;
                }
            }
            ++checks;
        }
    }
    if (!long_paged_windows(checks,maximum)) return 1;
    std::printf("PASS %d attention graphs across 2 GPUs (F32/F16/paged, full/SWA, histories to16393), max error %g\n",checks,maximum);
    return 0;
}
