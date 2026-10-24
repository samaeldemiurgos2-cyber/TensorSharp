// Run in separate processes with native env0/env1; no managed getenv assumption.
// Independent F64 oracle, changing weights/routes/bias/norm, 256->1->256 rows,
// retained caller-owned outputs, legacy exclusion and teardown/reinitialization.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_RecreateBackend();
void TSGgml_Shutdown();
void TSGgml_SetHostMoeThreads(int);
const char* TSGgml_GetLastError();
int TSGgml_GetMuseSparkCpuMoeScratchState(std::uint64_t*, std::uint64_t*);
int TSGgml_MuseSparkMoEFFNQuantF32(float*,float*,int,int,int,int,int,const std::int32_t*,const float*,
    void*,int,std::int64_t,std::int64_t,std::int64_t,void*,int,std::int64_t,std::int64_t,std::int64_t,
    void*,int,std::int64_t,std::int64_t,std::int64_t,const float*,const float*,const float*,int,float,float,int,const float*,float);
int TSGgml_MoEFFNPrefillSwiGLUQuantF32(float*,float*,int,int,int,int,int,const std::int32_t*,const float*,
    void*,int,std::int64_t,std::int64_t,std::int64_t,void*,int,std::int64_t,std::int64_t,std::int64_t,
    void*,int,std::int64_t,std::int64_t,std::int64_t,const float*,const float*,const float*,int,float,float,int);
}

static void require(bool good, const char* message) {
    if (!good) { std::fprintf(stderr,"FAIL %s native=%s\n",message,TSGgml_GetLastError()); std::exit(1); }
}

struct Retained { std::vector<float> result; std::vector<double> expected; };
static void check(const Retained& value) {
    double maximum=0;
    for (std::size_t i=0;i<value.result.size();++i) {
        require(std::isfinite(value.result[i]),"finite output");
        maximum=std::max(maximum,std::abs(value.result[i]-value.expected[i]));
    }
    require(maximum<4e-5,"independent F64 oracle and retained output");
}

int main(int argc,char** argv) {
    const bool enabled=argc>1 && std::strcmp(argv[1],"on")==0;
    const char* setting=std::getenv("TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH");
    require(enabled==(setting && std::strcmp(setting,"1")==0),"process-start opt-in matches expected lane");
    if (!TSGgml_IsBackendAvailable(2)) return 77;
    TSGgml_SetHostMoeThreads(2);
    std::uint64_t bytes=99,calls=99;
    require(TSGgml_GetMuseSparkCpuMoeScratchState(&bytes,&calls) && bytes==0 && calls==0,"query does not initialize scratch");
    require(!TSGgml_GetMuseSparkCpuMoeScratchState(nullptr,&calls),"null query rejected");
    constexpr int h=64,ff=96,experts=8,used=3;
    constexpr float eps=1e-5f;
    std::vector<float> gate(experts*ff*h),up(gate.size()),down(experts*h*ff),norm(h),gb(experts*ff),ub(gb.size()),db(experts*h);
    std::vector<Retained> retained;
    int checked=0;
    std::uint32_t seed=1729;
    auto random=[&]() { seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; return (int(seed%129)-64)/256.f; };
    for (int lifecycle=0;lifecycle<2;++lifecycle) {
        std::uint64_t expected_calls=0,large_bytes=0;
        // Every visit has freshly changed values, including identical pointer
        // weights. Legacy calls around arena reuse must neither allocate it nor
        // change its successful-call counter.
        for (int step=0;step<7;++step) {
            const bool spark=step!=0 && step!=4;
            const int rows=(step==2 || step==5)?1:256;
            const bool bias=(step%3)!=2;
            for (auto& x:gate) x=random(); for (auto& x:up) x=random(); for (auto& x:down) x=random();
            for (auto& x:gb) x=random()/4; for (auto& x:ub) x=random()/5; for (auto& x:db) x=random()/3;
            for (int j=0;j<h;++j) norm[j]=((j+step)%3==0?-1.f:1.f)*(.2f+((j*3+step*7)%29)/16.f);
            std::vector<float> input(rows*h),routes(rows*used);
            std::vector<std::int32_t> ids(rows*used);
            for (auto& x:input) x=random()*2;
            const auto input_before=input;
            const auto gate_before=gate;
            const std::uint64_t prior_bytes=bytes;
            Retained current{std::vector<float>(rows*h,-999.f),std::vector<double>(rows*h,0)};
            std::vector<double> activation(ff),expert(h);
            for (int r=0;r<rows;++r) for(int k=0;k<used;++k) {
                int e=ids[r*used+k]=(r*3+k*2+step+lifecycle)%experts;
                double weight=routes[r*used+k]=float(k+1+step)/float(6+3*step);
                for (int f=0;f<ff;++f) {
                    double g=bias?gb[e*ff+f]:0,u=bias?ub[e*ff+f]:0;
                    for(int j=0;j<h;++j) { g+=double(gate[(e*ff+f)*h+j])*input[r*h+j]; u+=double(up[(e*ff+f)*h+j])*input[r*h+j]; }
                    activation[f]=g/(1+std::exp(-g))*u;
                }
                double squares=0;
                for(int j=0;j<h;++j) {
                    double value=bias?db[e*h+j]:0;
                    for(int f=0;f<ff;++f) value+=double(down[(e*h+j)*ff+f])*activation[f];
                    expert[j]=spark?value*norm[j]:value; squares+=expert[j]*expert[j];
                }
                double inverse=spark?1/std::sqrt(squares/h+eps):1;
                for(int j=0;j<h;++j) current.expected[r*h+j]+=weight*expert[j]*inverse;
            }
            const auto gate_bytes=std::int64_t(gate.size()*sizeof(float)),down_bytes=std::int64_t(down.size()*sizeof(float));
            int ok=spark?TSGgml_MuseSparkMoEFFNQuantF32(input.data(),current.result.data(),rows,h,ff,experts,used,ids.data(),routes.data(),
                gate.data(),0,h,ff,gate_bytes,up.data(),0,h,ff,gate_bytes,down.data(),0,ff,h,down_bytes,
                bias?gb.data():nullptr,bias?ub.data():nullptr,bias?db.data():nullptr,0,1.702f,7.f,1,norm.data(),eps)
                :TSGgml_MoEFFNPrefillSwiGLUQuantF32(input.data(),current.result.data(),rows,h,ff,experts,used,ids.data(),routes.data(),
                gate.data(),0,h,ff,gate_bytes,up.data(),0,h,ff,gate_bytes,down.data(),0,ff,h,down_bytes,
                bias?gb.data():nullptr,bias?ub.data():nullptr,bias?db.data():nullptr,0,1.702f,7.f,1);
            require(ok!=0,"MoE dispatch"); check(current); retained.push_back(std::move(current));
            require(input==input_before && gate==gate_before,"external input and weight bytes remain caller-owned/read-only");
            for(const auto& old:retained) check(old);
            if(enabled && spark) ++expected_calls;
            require(TSGgml_GetMuseSparkCpuMoeScratchState(&bytes,&calls) && calls==expected_calls,"only successful opted-in Spark host calls counted");
            require(enabled && expected_calls?bytes>0:bytes==0,"arena allocation/default-off/legacy isolation");
            if(!spark) require(bytes==prior_bytes,"legacy call leaves retained arena unchanged");
            if(enabled && spark && rows==256) large_bytes=bytes;
            if(enabled && rows==1 && large_bytes) require(bytes>=large_bytes,"small graph retains arena capacity");
            // Pool reconfiguration must synchronize with the same compute lock
            // and leave the live arena usable with new leaves on the next call.
            if(step==3) TSGgml_SetHostMoeThreads(1+lifecycle);
            ++checked;
        }
        TSGgml_Shutdown();
        require(TSGgml_GetMuseSparkCpuMoeScratchState(&bytes,&calls) && bytes==0 && calls==0,"shutdown drops arena/counter");
        for(const auto& old:retained) check(old);
        if(lifecycle==0) require(TSGgml_RecreateBackend()!=0,"backend reinitialize");
    }
    std::printf("PASS opt_in=%d scalar_cases=%d retained_outputs=%zu lifecycle_resets=2\n",enabled,checked,retained.size());
    return 0;
}
