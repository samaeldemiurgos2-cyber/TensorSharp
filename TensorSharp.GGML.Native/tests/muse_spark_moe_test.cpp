// Muse Spark's scale -> per-expert RMSNorm -> weight -> sum is order-sensitive.
// Independent double-precision oracle with nonuniform signed scales, unequal
// routing weights, changing expert IDs, repeated calls, CPU offload and GPU.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "../ggml_ops_moe_stream_memory.h"

extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_GetGpuDeviceCount(int);
int TSGgml_MultiDeviceInit(int, const int*, int);
int TSGgml_SetActiveDevice(int);
const char* TSGgml_GetLastError();
void TSGgml_Shutdown();
int TSGgml_MuseSparkMoEFFNQuantF32(
    float*, float*, int, int, int, int, int, const std::int32_t*, const float*,
    void*, int, std::int64_t, std::int64_t, std::int64_t,
    void*, int, std::int64_t, std::int64_t, std::int64_t,
    void*, int, std::int64_t, std::int64_t, std::int64_t,
    const float*, const float*, const float*, int, float, float, int, const float*, float);
}

static void stream_opt_in(bool enabled) {
#ifdef _WIN32
    _putenv_s("TS_MUSE_SPARK_STREAM_EXPERTS", enabled ? "1" : "0");
    _putenv_s("TS_HOST_MOE_DEVICE_MIN_BATCH", "128");
#else
    setenv("TS_MUSE_SPARK_STREAM_EXPERTS", enabled ? "1" : "0", 1);
    setenv("TS_HOST_MOE_DEVICE_MIN_BATCH", "128", 1);
#endif
}

int main(int argc, char** argv) {
    const bool multi = argc > 1 && std::strcmp(argv[1], "cuda-multi") == 0;
    const bool cuda = multi || (argc > 1 && std::strcmp(argv[1], "cuda") == 0);
    if (!TSGgml_IsBackendAvailable(cuda ? 3 : 2)) return 77;
    if (multi) {
        if (TSGgml_GetGpuDeviceCount(3) < 2) return 77;
        const int devices[] = {0, 1};
        if (!TSGgml_MultiDeviceInit(3, devices, 2)) return 1;
    }
    // Test the exact reusable-buffer budget rule, including insufficient
    // headroom, equality, already-fitting buffers and overflow-sized inputs.
    const auto max = std::numeric_limits<std::size_t>::max();
    if (!tsg::moe_stream_memory_fits(100, 60, 40) || tsg::moe_stream_memory_fits(101, 60, 40)
        || !tsg::moe_stream_memory_fits(100, 0, 100) || !tsg::moe_stream_memory_fits(max, max, max)
        || tsg::moe_stream_memory_fits(max, 0, max - 1)) return 1;
    constexpr int h = 32, ff = 64, experts = 4, used = 2;
    constexpr float eps = 1e-5f;
    std::vector<float> gate(experts*ff*h), up(gate.size()), down(experts*h*ff), norm(h);
    // Independent deterministic weights avoid nearly cancelling sinusoidal
    // rows, where RMS normalization magnifies ordinary activation rounding.
    std::uint32_t random = 0x5a17c9e3u;
    auto value = [&]() {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        return (static_cast<int>(random % 61)-30)/128.f;
    };
    for (std::size_t i = 0; i < gate.size(); ++i) { gate[i] = value(); up[i] = value(); }
    for (auto& v : down) v = value();
    // Exact Q8_0 weights (half scale 1/128). Compare both unquantized and
    // quantized indexed GEMMs with the same independent dequantized oracle.
    auto quantize = [](std::vector<float>& values) {
        std::vector<std::uint8_t> bytes(values.size()/32*34);
        for (std::size_t block=0; block<values.size()/32; ++block) {
            bytes[block*34]=0; bytes[block*34+1]=0x20;
            for (int j=0; j<32; ++j) {
                const int q=static_cast<int>(std::round(values[block*32+j]*128));
                values[block*32+j]=q/128.f;
                bytes[block*34+2+j]=static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
            }
        }
        return bytes;
    };
    auto gateQ=quantize(gate), upQ=quantize(up), downQ=quantize(down);
    for (int j=0; j<h; ++j) norm[j] = (j%3 == 0 ? -1.f : 1.f)*(0.2f + j*0.06f);
    bool pass = true;
    int checks = 0;
    for (int stream : {0, 1}) for (int type : {0, 8}) for (int host : {0, 1}) for (int tokens : {1, 4, 17, 128}) for (int repeat=0; repeat<3; ++repeat) for (int rank=0; rank<(multi ? 2 : 1); ++rank) {
        if (stream && (tokens != 128 || !host)) continue;
        stream_opt_in(stream != 0);
        if (multi && !TSGgml_SetActiveDevice(rank)) return 1;
        std::vector<float> input(tokens*h), output(tokens*h), weights(tokens*used);
        std::vector<std::int32_t> ids(tokens*used);
        std::vector<double> expected(tokens*h, 0.0), active(ff), expert(h);
        // Every block has max |x| = 127/128 and lies on its Q8 grid, so
        // gate/up inputs are exactly representable by CPU and CUDA vec-dot.
        for (int i=0; i<tokens*h; ++i)
            input[i] = i%32 == 0 ? 127/128.f : ((i*37+repeat*19+rank*29)%201-100)/128.f;
        for (int t=0; t<tokens; ++t) for (int k=0; k<used; ++k) {
            // The final large-batch pass selects only half the expert stack,
            // exercising filtered uploads and reused-buffer stale contents.
            const int e = ids[t*used+k] = ((tokens == 128 && repeat == 2 ? 0 : t)+2*k+repeat)%experts;
            const double w = weights[t*used+k] = k == 0 ? 0.21f : 0.79f;
            for (int f=0; f<ff; ++f) {
                double g=0, u=0;
                for (int j=0; j<h; ++j) {
                    g += double(gate[(e*ff+f)*h+j])*input[t*h+j];
                    u += double(up[(e*ff+f)*h+j])*input[t*h+j];
                }
                active[f] = g/(1+std::exp(-g))*u;
            }
            if (type == 8) {
                // Q8 indexed GEMM also quantizes its activation input. Model
                // that format independently; comparing against unrounded
                // activations would conflate quantization with kernel errors.
                for (int block=0; block<ff; block+=32) {
                    double maximum=0;
                    for (int f=0; f<32; ++f) maximum=std::max(maximum,std::abs(active[block+f]));
                    const double scale=maximum/127;
                    int exponent=0;
                    std::frexp(scale,&exponent);
                    const double step=std::ldexp(1.0,std::max(exponent-11,-24));
                    const double storedScale=std::nearbyint(scale/step)*step;
                    for (int f=0; f<32; ++f)
                        active[block+f]=scale == 0 ? 0 : std::round(active[block+f]/scale)*storedScale;
                }
            }
            double sum=0;
            for (int j=0; j<h; ++j) {
                double v=0;
                for (int f=0; f<ff; ++f) v += double(down[(e*h+j)*ff+f])*active[f];
                expert[j] = v*norm[j];
                sum += expert[j]*expert[j];
            }
            const double inv=1/std::sqrt(sum/h+eps);
            for (int j=0; j<h; ++j) expected[t*h+j] += w*expert[j]*inv;
        }
        const int ok = TSGgml_MuseSparkMoEFFNQuantF32(input.data(), output.data(), tokens,h,ff,experts,used,
            ids.data(), weights.data(), type ? static_cast<void*>(gateQ.data()) : gate.data(),type,h,ff,type ? gateQ.size() : gate.size()*sizeof(float),
            type ? static_cast<void*>(upQ.data()) : up.data(),type,h,ff,type ? upQ.size() : up.size()*sizeof(float),
            type ? static_cast<void*>(downQ.data()) : down.data(),type,ff,h,type ? downQ.size() : down.size()*sizeof(float),
            nullptr,nullptr,nullptr,0,1.702f,7.f,host,norm.data(),eps);
        double error=0;
        for (int i=0; i<tokens*h; ++i) {
            if (!std::isfinite(output[i])) error=1e30;
            error=std::max(error,std::abs(output[i]-expected[i]));
        }
        // GPU may use reduced-precision matmul inputs. This is a numerical
        // tolerance, not a claim of token or model-quality parity.
        const bool device_compute = cuda && (!host || stream);
        const double tolerance=type ? (device_compute ? 0.003 : 0.0001) : device_compute ? 0.004 : 0.00003;
        const bool good=ok && error<tolerance;
        std::printf("%s backend=%s rank=%d type=%d cpu_offload=%d stream_opt_in=%d tokens=%d repeat=%d max_abs=%.8g%s%s\n",
            good?"PASS":"FAIL",cuda?"cuda":"cpu",rank,type,host,stream,tokens,repeat,error,
            ok?"":" error=",ok?"":TSGgml_GetLastError());
        pass &= good;
        ++checks;
    }
    std::printf("%s checks=%d memory_budget_cases=5\n", pass ? "PASS" : "FAIL", checks);
    TSGgml_Shutdown();
    return pass ? 0 : 1;
}
