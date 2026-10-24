// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_GetGpuDeviceCount(int);
int TSGgml_MultiDeviceInit(int,const int*,int);
int TSGgml_TensorParallelInitLoopback(int,int);
int TSGgml_SetActiveDevice(int);
int TSGgml_GetActiveDevice();
int TSGgml_GetTensorParallelDegree();
int TSGgml_RecreateBackend();
void TSGgml_Shutdown();
int TSGgml_PagedAttentionSessionCount();
const char* TSGgml_GetLastError();
int TSGgml_PagedAttentionForward(const float*,const float*,const float*,float*,const int*,const int*,
    const int*,const int*,const int*,int,int,int,int,int,int,int,float);
}
static bool zero(const char* point) {
    int count=TSGgml_PagedAttentionSessionCount();
    if(count!=0)std::fprintf(stderr,"FAIL: %s retains %d paged attention sessions after teardown\n",point,count);
    return count==0;
}
static bool values_match(const std::vector<float>& actual,const std::vector<float>& expected,
    int rows,int keys,int window,int seed,bool diagnose) {
    // Unchanged ggml CUDA flash attention stores softmax numerator weights in
    // half but accumulates the denominator in float. For zero Q/K, its
    // exp(-3*.6931) weight rounds from about .1250177 to .125: about 1.416e-4
    // relative error. Cold original/fixed and post-recreate outputs were
    // identical. Keep the original absolute bound and add a small scale term.
    constexpr float absolute=2e-4f,relative=2e-4f;
    if(actual.size()!=expected.size())return false;
    bool good=true;std::size_t worst=0;float maximum=-1.f;
    for(std::size_t i=0;i<actual.size();i++) {
        float error=std::abs(actual[i]-expected[i]);
        if(!std::isfinite(actual[i])||!std::isfinite(expected[i]))error=std::numeric_limits<float>::infinity();
        if(error>maximum){maximum=error;worst=i;}
        if(error>absolute+relative*std::abs(expected[i])||!std::isfinite(error))good=false;
    }
    if(!good&&diagnose)std::fprintf(stderr,
        "numerical mismatch rows=%d keys=%d window=%d seed=%d rank=%d row=%zu head=%zu dim=%zu actual=%.9g expected=%.9g max_error=%.9g bound=%.9g\n",
        rows,keys,window,seed,TSGgml_GetActiveDevice(),worst/(4*64),(worst/64)%4,worst%64,
        actual[worst],expected[worst],maximum,absolute+relative*std::abs(expected[worst]));
    return good;
}
static bool numeric(int rows,int keys,int window,int seed) {
    constexpr int heads=4,kv=2,dim=64,block=4;
    int pages=(keys+block-1)/block,first=keys-rows;
    std::vector<int> table(pages),positions(rows);
    std::vector<float> q(rows*heads*dim),k(pages*block*kv*dim),v(k.size()),output(q.size(),-99.f);
    for(int p=0;p<pages;p++)table[p]=pages-p-1;
    for(int i=0;i<rows;i++)positions[i]=first+i;
    for(int t=0;t<keys;t++)for(int h=0;h<kv;h++)for(int d=0;d<dim;d++)
        v[((table[t/block]*block+t%block)*kv+h)*dim+d]=(t+h+seed)/16.f;
    int starts[]={0,rows},lengths[]={keys},offsets[]={0};
    if(!TSGgml_PagedAttentionForward(q.data(),k.data(),v.data(),output.data(),starts,lengths,positions.data(),
        table.data(),offsets,1,rows,heads,kv,dim,block,window,1/std::sqrt(float(dim)))) {
        std::fprintf(stderr,"numeric execution failed: %s\n",TSGgml_GetLastError());return false;
    }
    std::vector<float> expected(output.size());
    for(int row=0;row<rows;row++)for(int h=0;h<heads;h++)for(int d=0;d<dim;d++) {
        int end=first+row,begin=window?std::max(0,end-window+1):0;
        expected[(row*heads+h)*dim+d]=((begin+end)*.5f+h/(heads/kv)+seed)/16.f;
    }
    if(!values_match(output,expected,rows,keys,window,seed,true))return false;
    // Negative controls operate on actual backend outputs, so the new bound
    // must reject stale seed values, a single damaged cell, NaN and infinity.
    auto changed=output;
    for(float& value:changed)value+=1/16.f;
    if(values_match(changed,expected,rows,keys,window,seed,false)) {
        std::fprintf(stderr,"stale-seed rejection control failed\n");return false;
    }
    changed=output;changed[changed.size()/2]+=1/64.f;
    if(values_match(changed,expected,rows,keys,window,seed,false)) {
        std::fprintf(stderr,"single-cell rejection control failed\n");return false;
    }
    for(float nonfinite:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        changed=output;changed[0]=nonfinite;
        if(values_match(changed,expected,rows,keys,window,seed,false)) {
            std::fprintf(stderr,"nonfinite rejection control failed\n");return false;
        }
    }
    return TSGgml_PagedAttentionSessionCount()>0;
}
int main(int argc,char**argv) {
    if(argc!=2)return 2;
    bool cpu=std::strcmp(argv[1],"cpu")==0,multi=std::strcmp(argv[1],"multi")==0;
    if(!cpu&&!multi&&std::strcmp(argv[1],"cuda")!=0)return 2;
    int backend=cpu?2:3;
    if(!TSGgml_IsBackendAvailable(backend))return 77;
    if(multi&&TSGgml_GetGpuDeviceCount(3)<2)return 77;
    if(!zero("initial"))return 1;
    int checks=0;
    for(int cycle=0;cycle<3;cycle++) {
        if(cpu) { if(!TSGgml_TensorParallelInitLoopback(2,2))return 1; }
        else if(multi) {int devices[]={0,1};if(!TSGgml_MultiDeviceInit(3,devices,2))return 1;}
        int ranks=(cpu||multi)?2:1;
        for(int rank=0;rank<ranks;rank++) {
            if(!TSGgml_SetActiveDevice(rank))return 1;
            if(!numeric(3,7,0,cycle*7+rank)||!numeric(1,5,3,cycle*13+rank))return 1;
            checks+=2;
        }
        // Deliberately recreate from rank one for multi/CPU loopback.
        if(TSGgml_GetActiveDevice()!=ranks-1||!TSGgml_RecreateBackend())return 1;
        if(!zero("recreate"))return 1;
        if(TSGgml_GetActiveDevice()!=0||TSGgml_GetTensorParallelDegree()!=1)return 1;
        if(!numeric(1,9,0,cycle+31))return 1;
        checks++;
        TSGgml_Shutdown();if(!zero("shutdown"))return 1;
        if(cycle<2&&!TSGgml_RecreateBackend())return 1;
    }
    std::printf("PASS: %d numerical graphs, %d rejection controls, 3 recreate and 3 shutdown empty-cache checks\n",checks,4*checks);
    return 0;
}
