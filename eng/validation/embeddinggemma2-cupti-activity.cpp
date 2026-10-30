// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone Linux LD_PRELOAD diagnostic; no inference or upstream source edits.
// CUDA 13 CUPTI Activity API: KERNEL / CONCURRENT_KERNEL records are CUpti_ActivityKernel13
// in the CUDA 13.4 cupti_activity.h (confirm it in the installed header):
// https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html
#include <cupti.h>
#include <cupti_activity.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <cxxabi.h>

namespace {
struct event { std::string name; uint64_t start, end; uint32_t device, context, stream, correlation; };
struct state {
    std::mutex mutex;
    std::string output;
    std::vector<event> events;
    std::vector<std::string> errors;
    std::atomic<uint64_t> dropped{0}, incomplete{0}, exceptions{0}, truncated{0};
    uint32_t api_version = 0;
    bool enabled = false;
};
state * data = nullptr;
constexpr size_t max_events = 65536;
std::string quoted(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c < 32) { const char hex[] = "0123456789abcdef"; result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += char(c);
    }
    return result + '"';
}
void failure(const char * operation, CUptiResult status) noexcept {
    if (status == CUPTI_SUCCESS || !data) return;
    try {
        const char * message = nullptr; cuptiGetResultString(status, &message);
        std::lock_guard<std::mutex> lock(data->mutex);
        data->errors.emplace_back(std::string(operation) + ": " + (message ? message : "unknown CUPTI error"));
    } catch (...) { ++data->exceptions; }
}
void CUPTIAPI requested(uint8_t ** buffer, size_t * size, size_t * maximum) noexcept {
    *buffer = nullptr; *size = 0; *maximum = 0;
    void * allocation = nullptr;
    if (posix_memalign(&allocation, 64, 8 * 1024 * 1024) == 0) {
        *buffer = static_cast<uint8_t *>(allocation); *size = 8 * 1024 * 1024;
    } else if (data) ++data->exceptions;
}
void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t * buffer, size_t, size_t valid) noexcept {
    try {
        CUpti_Activity * record = nullptr;
        CUptiResult result;
        while ((result = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
            if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && record->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
            const auto & kernel = *reinterpret_cast<const CUpti_ActivityKernel13 *>(record);
            if (!kernel.start || kernel.end < kernel.start || !kernel.end) { ++data->incomplete; continue; }
            std::string name = kernel.name ? kernel.name : "(unnamed)";
            int status = 0; char * demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
            if (status == 0 && demangled) name = demangled;
            std::free(demangled);
            std::lock_guard<std::mutex> lock(data->mutex);
            if (data->events.size() < max_events)
                data->events.push_back({std::move(name), kernel.start, kernel.end, kernel.deviceId, kernel.contextId, kernel.streamId, kernel.correlationId});
            else ++data->truncated;
        }
        if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) failure("next record", result);
        size_t dropped = 0;
        const auto drop_status = cuptiActivityGetNumDroppedRecords(context, stream, &dropped);
        failure("dropped records", drop_status); data->dropped += dropped;
    } catch (...) { if (data) ++data->exceptions; }
    std::free(buffer);
}
__attribute__((constructor)) void initialize() noexcept {
    const char * output = std::getenv("TSG_CUPTI_ACTIVITY_OUTPUT");
    if (!output || !*output) return;
    try {
        data = new state; data->output = output;
        failure("API version", cuptiGetVersion(&data->api_version));
        // The record layout read above is that of the headers this hook was compiled against.
        if (data->api_version != CUPTI_API_VERSION) {
            data->errors.emplace_back("Requires the CUPTI API version this hook was compiled against (CUDA 13 record layouts)."); return;
        }
        const auto registration = cuptiActivityRegisterCallbacks(requested, completed);
        failure("register buffers", registration);
        if (registration == CUPTI_SUCCESS) {
            const auto enabled = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
            failure("enable concurrent kernels", enabled); data->enabled = enabled == CUPTI_SUCCESS;
        }
    } catch (...) { if (data) ++data->exceptions; }
}
__attribute__((destructor)) void finalize() noexcept {
    if (!data) return;
    try {
        if (data->enabled) {
            failure("flush", cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
            failure("disable", cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        }
        std::lock_guard<std::mutex> lock(data->mutex);
        std::sort(data->events.begin(), data->events.end(), [](const event & a, const event & b) { return a.start < b.start; });
        std::map<std::string, std::pair<uint64_t, uint64_t>> totals;
        uint64_t total = 0;
        for (const auto & item : data->events) { auto & t = totals[item.name]; ++t.first; t.second += item.end - item.start; total += item.end - item.start; }
        const bool complete = data->enabled && !data->events.empty() && data->errors.empty() &&
            !data->dropped && !data->incomplete && !data->exceptions && !data->truncated;
        std::ofstream output(data->output);
        output << std::setprecision(17) << "{\"completed\":" << (complete ? "true" : "false")
            << ",\"diagnostic\":true,\"hardware_counters\":false,\"cupti_api_version\":" << data->api_version
            << ",\"dropped\":" << data->dropped << ",\"incomplete\":" << data->incomplete
            << ",\"callback_exceptions\":" << data->exceptions << ",\"truncated\":" << data->truncated
            << ",\"kernel_count\":" << data->events.size() << ",\"sum_kernel_ms\":" << total / 1e6 << ",\"errors\":[";
        for (size_t i = 0; i < data->errors.size(); ++i) output << (i ? "," : "") << quoted(data->errors[i]);
        output << "],\"kernels\":["; bool comma = false;
        for (const auto & item : totals) { output << (comma ? "," : "") << "{\"name\":" << quoted(item.first) << ",\"count\":" << item.second.first << ",\"sum_ms\":" << item.second.second / 1e6 << '}'; comma = true; }
        output << "],\"events\":[";
        for (size_t i = 0; i < data->events.size(); ++i) {
            const auto & e = data->events[i];
            output << (i ? "," : "") << "{\"name\":" << quoted(e.name) << ",\"start_ns\":" << e.start << ",\"end_ns\":" << e.end
                << ",\"device\":" << e.device << ",\"context\":" << e.context << ",\"stream\":" << e.stream << ",\"correlation\":" << e.correlation << '}';
        }
        output << "],\"scope\":\"All kernel activities in this process, including warmup; overlapping GPU intervals may overlap in the sum. Excludes host code, copies, and service overhead. Instrumented diagnostic, never a paired performance claim.\"}\n";
    } catch (...) {}
}
} // namespace
