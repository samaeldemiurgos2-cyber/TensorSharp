// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone experiment: original GGUF NVFP4 bytes, exact BF16 weights,
// rounded BF16 activations, F32 accumulation. Does not alter inference.
// nvcc -O3 -std=c++17 -arch=sm_120 -I ExternalProjects/ggml/include \
//   eng/validation/qwen-nvfp4-skinny-probe.cu \
//   -L TensorSharp.GGML.Native/build -lGgmlOps \
//   -Xlinker -rpath -Xlinker $PWD/TensorSharp.GGML.Native/build \
//   -o artifacts/nvfp4-skinny-probe
// Optional Velox comparator: add -DTS_COMPARE_VELOX and the unmodified
// target/release/build/velox-variant-*/out/velox_variant_kernels_rtx5090_skinny16.o
// from its rtx-5090-32g branch. Use the CUDA toolkit that built that object.
// The measured comparator revision was ccec38c0fadafe4e0d2d4b753044847dcaa30bbe.
// This synthetic experiment does not qualify model outputs or E2E throughput.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>
#ifdef TS_COMPARE_VELOX
// Optional independent comparator. Link the unmodified Apache-2.0 Velox
// rtx5090_skinny16.cu object; no Velox code is included in this file.
extern "C" int velox_5090_linear_skinny16(const float*,int,const uint8_t*,long long,long long,long long,int,int,float*);
#endif

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1); }} while(0)
namespace {
constexpr int Rows=64, KTile=256, Tokens=16, Threads=128, Stages=2;
constexpr int WeightStride=160, InputStride=264;
struct Shared {
    __nv_bfloat16 x[Stages][Tokens*InputStride];
    unsigned char w[Stages][Rows*WeightStride];
};
__device__ uint32_t shared_address(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ void copy16(void* d,const void* s,bool valid=true) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(shared_address(d)),"l"(s),"r"(valid?16:0));
}
__device__ void load_tile(Shared& tile,int stage,int kt,const unsigned char* w,const __nv_bfloat16* x,int k,int m,int row0) {
    // A GGUF row contains 36-byte blocks: four scales, then 32 packed bytes.
    // Four consecutive blocks restore 16-byte alignment without repacking.
    for(int i=threadIdx.x;i<Rows*9;i+=Threads) {
        int row=i/9,chunk=i%9;
        bool valid=row0+row<m;
        copy16(tile.w[stage]+row*WeightStride+chunk*16,
            w+size_t(valid?row0+row:0)*(k/64)*36+kt*144+chunk*16,valid);
    }
    for(int i=threadIdx.x;i<Tokens*32;i+=Threads) {
        int token=i/32,chunk=i%32;
        copy16(tile.x[stage]+token*InputStride+chunk*8,x+size_t(token)*k+kt*KTile+chunk*8);
    }
}
__device__ float scale_value(unsigned char v) {
    if (v==0 || v==127) return 0;
    const int exponent=(v>>3)&15, mantissa=v&7;
    return exponent ? __uint_as_float(uint32_t(exponent+120)<<23 | uint32_t(mantissa)<<20) : mantissa*(1.f/512.f);
}
__device__ float fp4_value(unsigned v) {
    unsigned exponent=(v>>1)&3, fraction=v&1;
    unsigned magnitude=exponent ? (exponent+126)<<23 | fraction<<22 : fraction*(126u<<23);
    return __uint_as_float(magnitude | (v&8)<<28);
}
__device__ uint32_t pack(float a,float b) {
    __nv_bfloat162 p=__floats2bfloat162_rn(a,b);
    return *reinterpret_cast<uint32_t*>(&p);
}
__device__ void product(float* c,const uint32_t* a,uint32_t b0,uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
__global__ void prepare_input(const float* x,__nv_bfloat16* out,int k,int n) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Tokens*k;i+=gridDim.x*blockDim.x)
        out[i]=__float2bfloat16_rn(i<n*k?x[i]:0.f);
}
__global__ __launch_bounds__(Threads) void skinny(const unsigned char* w,const __nv_bfloat16* x,float* y,int k,int m,int n,int tiles_per_split) {
    __shared__ Shared tile;
    int row0=blockIdx.x*Rows,begin=blockIdx.z*tiles_per_split,end=min(k/KTile,begin+tiles_per_split);
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    for(int stage=0;stage<Stages-1;++stage) {
        if(begin+stage<end) load_tile(tile,stage,begin+stage,w,x,k,m,row0);
        asm volatile("cp.async.commit_group;");
    }
    float acc[2][4]={{0,0,0,0},{0,0,0,0}};
    int a_row=lane&15,a_col=(lane>>4)*8;
    for(int kt=begin;kt<end;++kt) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        int i=kt-begin,stage=i%Stages,next=kt+Stages-1;
        if(next<end) load_tile(tile,(i+Stages-1)%Stages,next,w,x,k,m,row0);
        asm volatile("cp.async.commit_group;");
#pragma unroll
        for(int step=0;step<KTile/16;++step) {
            uint32_t a[4];
            uint32_t address=shared_address(tile.x[stage]+a_row*InputStride+step*16+a_col);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(address));
#pragma unroll
            for(int h=0;h<2;++h) {
                const unsigned char* block=tile.w[stage]+(warp*16+h*8+lane/4)*WeightStride+(step/4)*36;
                float s=scale_value(block[step%4]);
                const unsigned char* codes=block+4+(step%4)*8+(lane%4)*2;
                unsigned c0=codes[0],c1=codes[1];
                product(acc[h],a,pack(fp4_value(c0&15)*s,fp4_value(c1&15)*s),pack(fp4_value(c0>>4)*s,fp4_value(c1>>4)*s));
            }
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;");
    // MMA C: rows are token IDs; columns are output weight rows.
#pragma unroll
    for(int h=0;h<2;++h) {
        int row=row0+warp*16+h*8+(lane%4)*2,token=lane/4;
#pragma unroll
        for(int j=0;j<4;++j) {
            int r=row+(j&1),t=token+(j/2)*8;
            if(r<m && t<n) y[(size_t(blockIdx.z)*n+t)*m+r]=acc[h][j];
        }
    }
}
__global__ void reduce_splits(const float* partial,float* out,int count,int splits) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
        float sum=partial[i];for(int s=1;s<splits;++s)sum+=partial[size_t(s)*count+i];out[i]=sum;
    }
}
struct Graph {
    ggml_context* ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_cgraph* graph=nullptr;
    ggml_tensor *weights=nullptr,*input=nullptr,*output=nullptr;
    ~Graph(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
    void create(ggml_backend_t backend,int k,int m,int n,const std::vector<unsigned char>& raw,const std::vector<float>& x) {
        ctx=ggml_init({1024*1024,nullptr,true});weights=ggml_new_tensor_2d(ctx,GGML_TYPE_NVFP4,k,m);input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
        ggml_set_input(input);output=ggml_mul_mat(ctx,weights,input);ggml_prec_set_src(output,GGML_PREC_Q8,1);ggml_set_output(output);
        graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,output);buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        if(!buffer)std::exit(1);ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(weights,raw.data(),0,raw.size());ggml_backend_tensor_set(input,x.data(),0,x.size()*4);
    }
    double measure(ggml_backend_t backend){
        for(int i=0;i<8;++i)if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)std::exit(1);ggml_backend_synchronize(backend);
        auto start=std::chrono::steady_clock::now();for(int i=0;i<100;++i)if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)std::exit(1);ggml_backend_synchronize(backend);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100;
    }
};
float host_scale(unsigned char v){if(v==0||v==127)return 0;int e=(v>>3)&15;return e?std::ldexp(1.f+(v&7)/8.f,e-7):(v&7)/512.f;}
float host_fp4(unsigned v){constexpr float table[]={0,.5,1,1.5,2,3,4,6};return (v&8?-1.f:1.f)*table[v&7];}
}
int main(){
    ggml_backend_t backend=ggml_backend_cuda_init(0);if(!backend)return 77;
    void* flush=nullptr;constexpr size_t flush_bytes=256*1024*1024;CHECK(cudaMalloc(&flush,flush_bytes));
    auto cold_time=[&](auto operation){
        double elapsed=0;
        for(int i=0;i<20;++i){
            CHECK(cudaMemset(flush,i,flush_bytes));CHECK(cudaDeviceSynchronize());
            auto start=std::chrono::steady_clock::now();operation();CHECK(cudaDeviceSynchronize());
            elapsed+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        return elapsed/20;
    };
    // splits=0 is the optional unmodified Velox comparator, including BF16
    // input preparation and split reduction. Repacking is outside timing and
    // consumes the same weight bytes plus at most 259 bytes of padding/footer.
    // Warm timings use graph replay; cold timings evict L2 and include one
    // host submission/synchronization, so compare like columns only.
    std::puts("k,m,n,splits,mmvq_ms,skinny_bf16_ms,speedup,max_diff_mmvq,rms_diff_mmvq,max_diff_bf16_reference,mmvq_cold_ms,skinny_cold_ms,cold_speedup");
    for(auto shape:std::initializer_list<std::pair<int,int>>{{5120,17408},{17408,5120},{5120,10240}}){
        int k=shape.first,m=shape.second;uint32_t rng=1234567;
        std::vector<unsigned char> raw(size_t(k)*m/64*36);
        for(size_t i=0;i<raw.size();++i){rng=rng*1664525u+1013904223u;raw[i]=i%36<4?0x20+(rng>>30):(rng>>24);}
        for(int n:{1,4,8}){
            std::vector<float>x(k*n),base(m*n),result(m*n);for(float& v:x){rng=rng*1664525u+1013904223u;v=(int(rng>>16)-32768)/32768.f;}
            Graph normal;normal.create(backend,k,m,n,raw,x);double t0=normal.measure(backend);ggml_backend_tensor_get(normal.output,base.data(),0,base.size()*4);
            double cold0=cold_time([&]{if(ggml_backend_graph_compute(backend,normal.graph)!=GGML_STATUS_SUCCESS)std::exit(1);});
            __nv_bfloat16* input=nullptr;float *partial=nullptr,*out=nullptr;
            CHECK(cudaMalloc(&input,size_t(Tokens)*k*2));CHECK(cudaMalloc(&partial,size_t(16)*m*n*4));CHECK(cudaMalloc(&out,size_t(m)*n*4));
#ifdef TS_COMPARE_VELOX
            size_t scale_offset=(size_t(m)*k/2+255)/256*256;
            std::vector<unsigned char> repacked(scale_offset+size_t(m)*k/16+4);
            for(int row=0;row<m;++row)for(int g=0;g<k/16;++g){
                size_t original=(size_t(row)*k/64+g/4)*36;
                for(int pair=0;pair<8;++pair){
                    auto code=[&](int pos){unsigned byte=raw[original+4+(g%4)*8+pos%8];return pos<8?byte&15:byte>>4;};
                    repacked[size_t(row)*k/2+g*8+pair]=code(2*pair)|(code(2*pair+1)<<4);
                }
                size_t scale=(size_t(row/128)*(k/64)+g/4)*512+(row%32)*16+((row%128)/32)*4+g%4;
                repacked[scale_offset+scale]=raw[original+g%4];
            }
            float divisor=1.f;std::memcpy(repacked.data()+repacked.size()-4,&divisor,4);
            unsigned char* velox_weights=nullptr;CHECK(cudaMalloc(&velox_weights,repacked.size()));CHECK(cudaMemcpy(velox_weights,repacked.data(),repacked.size(),cudaMemcpyHostToDevice));
            std::vector<int> configurations={1,2,4,8,16,-1};
#else
            std::vector<int> configurations={1,2,4,8,16};
#endif
            for(int splits:configurations){
                int per=splits>0?(k/KTile+splits-1)/splits:1;int actual=(k/KTile+per-1)/per;
                cudaStream_t stream=cudaStreamPerThread;cudaGraph_t graph;cudaGraphExec_t exec;
#ifdef TS_COMPARE_VELOX
                if(splits<0){
                    actual=0;
                    int status=velox_5090_linear_skinny16(static_cast<float*>(normal.input->data),n,velox_weights,m,k,0,m,0,out);
                    if(status){std::fprintf(stderr,"Velox status %d\n",status);return 1;}CHECK(cudaStreamSynchronize(stream));
                }
#endif
                CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
#ifdef TS_COMPARE_VELOX
                if(splits<0){
                    if(velox_5090_linear_skinny16(static_cast<float*>(normal.input->data),n,velox_weights,m,k,0,m,0,out))return 1;
                }else
#endif
                {
                    prepare_input<<<(Tokens*k+255)/256,256,0,stream>>>(static_cast<float*>(normal.input->data),input,k,n);
                    skinny<<<dim3((m+Rows-1)/Rows,1,actual),Threads,0,stream>>>(static_cast<unsigned char*>(normal.weights->data),input,actual==1?out:partial,k,m,n,per);
                    if(actual>1)reduce_splits<<<(m*n+255)/256,256,0,stream>>>(partial,out,m*n,actual);
                }
                CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
                for(int i=0;i<8;++i)CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
                auto start=std::chrono::steady_clock::now();for(int i=0;i<100;++i)CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
                double t1=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100;
                double cold1=cold_time([&]{CHECK(cudaGraphLaunch(exec,stream));});
                CHECK(cudaMemcpy(result.data(),out,result.size()*4,cudaMemcpyDeviceToHost));
                float maxerr=0,maxref=0;double square=0;
                for(size_t i=0;i<base.size();++i){if(!std::isfinite(result[i]))return 1;float e=std::abs(result[i]-base[i]);maxerr=std::max(maxerr,e);square+=double(e)*e;}
                for(int token=0;token<n;++token)for(int ri=0;ri<32;++ri){
                    int row=(ri*521)%m;double expected=0;
                    for(int j=0;j<k;++j){size_t off=(size_t(row)*k/64+j/64)*36;int group=j%64/16,pos=j%16;unsigned c=raw[off+4+group*8+pos%8];
                        float w=host_fp4(pos<8?c&15:c>>4)*host_scale(raw[off+group]);float a=__bfloat162float(__float2bfloat16_rn(x[token*k+j]));expected+=double(w)*a;}
                    maxref=std::max(maxref,float(std::abs(result[token*m+row]-expected)));
                }
                std::printf("%d,%d,%d,%d,%.6f,%.6f,%.4f,%.9g,%.9g,%.9g,%.6f,%.6f,%.4f\n",k,m,n,actual,t0,t1,t0/t1,maxerr,std::sqrt(square/base.size()),maxref,cold0,cold1,cold0/cold1);std::fflush(stdout);
                if(maxref>0.002f){std::fprintf(stderr,"BF16 reference mismatch\n");return 1;}
                CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
            }
#ifdef TS_COMPARE_VELOX
            CHECK(cudaFree(velox_weights));
#endif
            CHECK(cudaFree(input));CHECK(cudaFree(partial));CHECK(cudaFree(out));
        }
    }
    CHECK(cudaFree(flush));ggml_backend_free(backend);return 0;
}
