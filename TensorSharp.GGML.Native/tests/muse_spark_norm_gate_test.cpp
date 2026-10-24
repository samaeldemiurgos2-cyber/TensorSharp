// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
struct View { void* data; int rows, columns, row_stride, column_stride; int64_t bytes; };
extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_GetGpuDeviceCount(int);
int TSGgml_MultiDeviceInit(int,const int*,int);
int TSGgml_SetActiveDevice(int);
int TSGgml_GetActiveDevice();
int TSGgml_GetTensorParallelDegree();
int TSGgml_TensorParallelInitLoopback(int,int);
int TSGgml_MuseSparkNormGateF32(View,View,const float*,View,int,float,int,int);
int TSGgml_MuseSparkPostNormResidualF32(View,View,View,void*,void*,void*,int,float);
const char* TSGgml_GetLastError();
void TSGgml_Shutdown();
int TSGgml_RecreateBackend();
}
static View view(std::vector<float>& x,int rows,int width,int stride=0) {
    if (!stride) stride=width;
    return {x.data(),rows,width,stride,1,(int64_t(rows-1)*stride+width)*4};
}
static bool run(int rows,int width,int hd,int padding,bool gated,int salt) {
    int stride=width+padding;
    std::vector<float> input(rows*stride),gate(input.size()),weights(hd),fresh(rows*width+8,71.f),reuse(fresh);
    for (size_t i=0;i<input.size();++i) {
        const float scale=(i/stride)%3==0?0.f:(i/stride)%3==1?1024.f:1e-8f;
        input[i]=std::sin(float(i)*.031f+salt)*scale;
        gate[i]=float((i*13+salt)%81)-40.f;
    }
    for (int i=0;i<hd;++i) weights[i]=float((i+salt)%11-5)*.31f;
    const auto original_input=input,original_gate=gate,original_weights=weights;
    auto source=view(input,rows,width,stride),gv=view(gate,rows,width,stride);
    float eps=salt%2==0?1e-5f:1e-8f;
    int a=TSGgml_MuseSparkNormGateF32(view(fresh,rows,width),source,weights.data(),gv,hd,eps,gated,0);
    int b=TSGgml_MuseSparkNormGateF32(view(reuse,rows,width),source,weights.data(),gv,hd,eps,gated,1);
    if (a!=2||b!=1) {std::fprintf(stderr,"execution status %d/%d: %s\n",a,b,TSGgml_GetLastError());return false;}
    // Repack the shared allocator with an unrelated op before observing output.
    std::vector<float> x(32,.7f),y(32,-.3f),w(32,1.f),other(32);
    if (!TSGgml_MuseSparkPostNormResidualF32(view(other,1,32),view(x,1,32),view(y,1,32),w.data(),w.data(),nullptr,32,1e-5f)) return false;
    if (fresh!=reuse || input!=original_input || gate!=original_gate || weights!=original_weights) return false;
    for (int row=0;row<rows;++row) for(int head=0;head<width/hd;++head) {
        const int src=row*stride+head*hd,dst=row*width+head*hd;
        double squares=0;for(int c=0;c<hd;++c)squares+=double(input[src+c])*input[src+c];
        float inverse=float(1/std::sqrt(squares/hd+eps));
        for(int c=0;c<hd;++c) {
            float expected=input[src+c]*inverse*weights[c];
            if(gated) expected*=float(1/(1+std::exp(-double(gate[src+c]))));
            if(!std::isfinite(reuse[dst+c]) || std::abs(reuse[dst+c]-expected)>3e-5f) {
                std::fprintf(stderr,"oracle rows=%d width=%d hd=%d padding=%d gate=%d index=%d error=%g\n",rows,width,hd,padding,gated,dst+c,double(reuse[dst+c]-expected));return false;
            }
        }
    }
    return std::all_of(reuse.begin()+rows*width,reuse.end(),[](float x){return x==71.f;});
}
static bool rejection() {
    std::vector<float> input(64,.5f),weight(32,1.f),output(64,71.f),gate(64,.1f);
    auto result=view(output,2,32),source=view(input,2,32),gv=view(gate,2,32);
    auto check=[&](View r,View s,const float* w,View g,int hd,float eps,int gated,int reuse) {
        return TSGgml_MuseSparkNormGateF32(r,s,w,g,hd,eps,gated,reuse)==0 &&
            std::all_of(output.begin(),output.end(),[](float x){return x==71.f;});
    };
    for(float eps:{0.f,-1.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
        if(!check(result,source,weight.data(),gv,32,eps,1,1))return false;
    if(!check(result,source,nullptr,gv,32,1e-5f,1,1)||!check(result,source,weight.data(),gv,7,1e-5f,1,1)||
        !check(result,source,weight.data(),gv,32,1e-5f,2,1)||!check(result,source,weight.data(),gv,32,1e-5f,1,2))return false;
    auto bad=result;bad.bytes+=4;if(!check(bad,source,weight.data(),gv,32,1e-5f,1,1))return false;
    bad=source;bad.column_stride=2;if(!check(result,bad,weight.data(),gv,32,1e-5f,1,1))return false;
    bad=gv;bad.columns--;if(!check(result,source,weight.data(),bad,32,1e-5f,1,1))return false;
    auto before=input;if(TSGgml_MuseSparkNormGateF32(source,source,weight.data(),gv,32,1e-5f,1,1)!=0||input!=before)return false;
    return true;
}
static bool recreated_backend(int backend_type) {
    // Check state before IsBackendAvailable: it must not initialize the wrong
    // slot and hide a recreation bug. The next graph also proves usability.
    const int rank=TSGgml_GetActiveDevice(), degree=TSGgml_GetTensorParallelDegree();
    if(rank!=0 || degree!=1) {
        std::fprintf(stderr,"backend recreation left invalid state: active rank=%d, degree=%d; expected rank 0, degree 1\n",rank,degree);
        return false;
    }
    if(!TSGgml_IsBackendAvailable(backend_type)) {
        std::fprintf(stderr,"recreated rank-zero backend unavailable: %s\n",TSGgml_GetLastError());
        return false;
    }
    return run(1,64,64,7,true,19);
}
static bool cpu_loopback_recreation() {
    // Exercise the actual public rank lifecycle without requiring two GPUs.
    if(!TSGgml_TensorParallelInitLoopback(2,2) || !TSGgml_SetActiveDevice(1)) return false;
    if(TSGgml_GetActiveDevice()!=1 || TSGgml_GetTensorParallelDegree()!=2 || !run(1,64,64,7,true,17)) return false;
    if(!TSGgml_RecreateBackend()) {
        std::fprintf(stderr,"CPU loopback recreation failed: %s\n",TSGgml_GetLastError());
        return false;
    }
    return recreated_backend(2);
}
int main(int argc,char** argv) {
    const bool cuda=argc==2 && (std::strcmp(argv[1],"cuda")==0||std::strcmp(argv[1],"multi")==0);
    const bool multi=argc==2&&std::strcmp(argv[1],"multi")==0;
    if(argc!=2||(!cuda&&std::strcmp(argv[1],"cpu")!=0))return 2;
    if(!cuda && !cpu_loopback_recreation())return 1;
    int cases=0;
    for(int lifecycle=0;lifecycle<2;++lifecycle) {
        if(!TSGgml_IsBackendAvailable(cuda?3:2))return 77;
        if(multi) {int devices[]={0,1};if(TSGgml_GetGpuDeviceCount(3)<2)return 77;if(!TSGgml_MultiDeviceInit(3,devices,2))return 1;}
        if(!rejection())return 1;
        for(int rows:{1,2,9,128,256,1,256})for(int width:{64,4096,8192})for(int padding:{0,7})for(bool gated:{false,true}) {
            if(multi&&!TSGgml_SetActiveDevice(cases%2))return 1;
            if(!run(rows,width,gated?64:width,padding,gated,++cases))return 1;
        }
        if(lifecycle==0) {
            // Preserve the rank-one transition that exposed the production bug.
            if(multi && TSGgml_GetActiveDevice()!=1)return 1;
            if(!TSGgml_RecreateBackend()) {
                std::fprintf(stderr,"backend recreation failed: %s\n",TSGgml_GetLastError());return 1;
            }
            if(!recreated_backend(cuda?3:2))return 1;
        }
        else TSGgml_Shutdown();
    }
    std::printf("PASS: %d scalar cases, fresh/reuse exact equality, output lifetime, invalid arguments, shutdown/reinit\n",cases);
    return 0;
}
