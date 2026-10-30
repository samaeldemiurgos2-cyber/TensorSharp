// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Separate Linux LD_PRELOAD diagnostic derived from embeddinggemma2-cupti-activity.cpp.
// CUDA 13 CUPTI: CUpti_ActivityKernel13 and CUpti_ActivityMemcpy7, the current KERNEL and
// MEMCPY records in the CUDA 13.4 headers (confirm them in the installed cupti_activity.h).
// https://docs.nvidia.com/cupti/api/structCUpti__ActivityMemcpy7.html
// No inference, production, upstream, math-mode, or environment-selector changes.
#include <cupti.h>
#include <cupti_activity.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cxxabi.h>
#include <unistd.h>

namespace {
struct kernel_event {
    std::string name;
    uint64_t start, end;
    uint32_t device, context, stream, correlation;
};
struct memcpy_event {
    uint8_t kind, src_kind, dst_kind, flags;
    uint64_t bytes, start, end, copy_count, graph_node;
    uint32_t device, context, stream, correlation, runtime_correlation, graph;
};
struct state {
    std::mutex mutex;
    std::string output;
    int output_fd = -1;
    std::vector<kernel_event> kernels;
    std::vector<memcpy_event> copies;
    std::vector<std::string> errors;
    std::atomic<uint64_t> dropped{0}, incomplete{0}, exceptions{0}, truncated{0}, invalid_buffers{0};
    uint32_t api_version = 0;
    bool kernels_enabled = false, memcpy_enabled = false;
};
state* data = nullptr;
constexpr size_t max_events = 65536; // Combined kernel + memcpy bound.
constexpr size_t activity_buffer_bytes = 8 * 1024 * 1024;
// The record layouts below are those of the CUPTI headers this hook was compiled against.
constexpr uint32_t expected_api_version = CUPTI_API_VERSION;

std::string quoted(const std::string& value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c < 32) {
            const char hex[] = "0123456789abcdef";
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += char(c);
    }
    return result + '"';
}
const char* direction(uint8_t kind) {
    switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "PtoP";
    default: return "Unknown";
    }
}
void failure(const char* operation, CUptiResult status) noexcept {
    if (status == CUPTI_SUCCESS || !data) return;
    try {
        const char* message = nullptr; cuptiGetResultString(status, &message);
        std::lock_guard<std::mutex> lock(data->mutex);
        data->errors.emplace_back(std::string(operation) + ": " + (message ? message : "unknown CUPTI error"));
    } catch (...) { ++data->exceptions; }
}
void CUPTIAPI requested(uint8_t** buffer, size_t* size, size_t* maximum) noexcept {
    *buffer = nullptr; *size = 0; *maximum = 0;
    void* allocation = nullptr;
    if (posix_memalign(&allocation, 64, activity_buffer_bytes) == 0) {
        *buffer = static_cast<uint8_t*>(allocation); *size = activity_buffer_bytes;
    } else if (data) ++data->exceptions;
}
bool contains_record(const CUpti_Activity* record, size_t record_bytes,
                     const uint8_t* buffer, size_t valid) {
    const auto base = reinterpret_cast<uintptr_t>(buffer);
    const auto address = reinterpret_cast<uintptr_t>(record);
    return address >= base && address - base <= valid && record_bytes <= valid - (address - base);
}
void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t* buffer, size_t size, size_t valid) noexcept {
    try {
        if (!data || !buffer || valid > size) {
            if (data) ++data->invalid_buffers;
        } else {
            CUpti_Activity* record = nullptr;
            CUptiResult result;
            while ((result = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
                if (!contains_record(record, sizeof(CUpti_Activity), buffer, valid)) {
                    ++data->invalid_buffers; break;
                }
                if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
                    if (!contains_record(record, sizeof(CUpti_ActivityKernel13), buffer, valid)) {
                        ++data->invalid_buffers; break;
                    }
                    const auto& kernel = *reinterpret_cast<const CUpti_ActivityKernel13*>(record);
                    if (!kernel.start || kernel.end < kernel.start || !kernel.end) { ++data->incomplete; continue; }
                    std::string name = kernel.name ? kernel.name : "(unnamed)";
                    int status = 0;
                    char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
                    if (status == 0 && demangled) name = demangled;
                    std::free(demangled);
                    std::lock_guard<std::mutex> lock(data->mutex);
                    if (data->kernels.size() + data->copies.size() < max_events)
                        data->kernels.push_back({std::move(name), kernel.start, kernel.end,
                            kernel.deviceId, kernel.contextId, kernel.streamId, kernel.correlationId});
                    else ++data->truncated;
                } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                    if (!contains_record(record, sizeof(CUpti_ActivityMemcpy7), buffer, valid)) {
                        ++data->invalid_buffers; break;
                    }
                    const auto& copy = *reinterpret_cast<const CUpti_ActivityMemcpy7*>(record);
                    if (!copy.start || copy.end < copy.start || !copy.end) { ++data->incomplete; continue; }
                    std::lock_guard<std::mutex> lock(data->mutex);
                    if (data->kernels.size() + data->copies.size() < max_events)
                        data->copies.push_back({copy.copyKind, copy.srcKind, copy.dstKind, copy.flags,
                            copy.bytes, copy.start, copy.end, copy.copyCount, copy.graphNodeId,
                            copy.deviceId, copy.contextId, copy.streamId, copy.correlationId,
                            copy.runtimeCorrelationId, copy.graphId});
                    else ++data->truncated;
                }
            }
            if (result != CUPTI_SUCCESS && result != CUPTI_ERROR_MAX_LIMIT_REACHED) failure("next record", result);
            size_t dropped = 0;
            failure("dropped records", cuptiActivityGetNumDroppedRecords(context, stream, &dropped));
            data->dropped += dropped;
        }
    } catch (...) { if (data) ++data->exceptions; }
    std::free(buffer);
}
__attribute__((constructor)) void initialize() noexcept {
    const char* output = std::getenv("TSG_CUPTI_ACTIVITY_OUTPUT");
    if (!output || !*output) return;
    try {
        data = new state; data->output = output;
        // Reserve a new report atomically before enabling instrumentation. Keep
        // the descriptor through finalization so an existing report is never
        // truncated, including a race between the wrapper check and this hook.
        data->output_fd = ::open(output, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (data->output_fd < 0) {
            std::fprintf(stderr, "CUPTI transfer hook: cannot create fresh output '%s': %s\n", output, std::strerror(errno));
            return;
        }
        failure("API version", cuptiGetVersion(&data->api_version));
        // The versioned record layouts are those of the CUPTI headers this hook was compiled
        // against (CUDA 13); any other runtime API version could lay records out differently.
        if (data->api_version != expected_api_version) {
            data->errors.emplace_back("Requires the CUPTI API version this hook was compiled against (CUDA 13 record layouts)."); return;
        }
        const auto registration = cuptiActivityRegisterCallbacks(requested, completed);
        failure("register buffers", registration);
        if (registration == CUPTI_SUCCESS) {
            const auto kernels = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
            failure("enable concurrent kernels", kernels); data->kernels_enabled = kernels == CUPTI_SUCCESS;
            const auto copies = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
            failure("enable memcpy", copies); data->memcpy_enabled = copies == CUPTI_SUCCESS;
        }
    } catch (...) { if (data) ++data->exceptions; }
}
uint64_t add_checked(uint64_t lhs, uint64_t rhs) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) throw std::overflow_error("activity summary overflow");
    return lhs + rhs;
}
__attribute__((destructor)) void finalize() noexcept {
    if (!data || data->output_fd < 0) return;
    try {
        if (data->kernels_enabled || data->memcpy_enabled)
            failure("flush", cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
        if (data->kernels_enabled)
            failure("disable concurrent kernels", cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        if (data->memcpy_enabled)
            failure("disable memcpy", cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
        std::lock_guard<std::mutex> lock(data->mutex);
        std::sort(data->kernels.begin(), data->kernels.end(), [](const kernel_event& a, const kernel_event& b) { return a.start < b.start; });
        std::sort(data->copies.begin(), data->copies.end(), [](const memcpy_event& a, const memcpy_event& b) { return a.start < b.start; });
        std::map<std::string, std::pair<uint64_t, uint64_t>> kernel_totals;
        uint64_t kernel_ns = 0, memcpy_ns = 0, memcpy_bytes = 0;
        for (const auto& item : data->kernels) {
            auto& total = kernel_totals[item.name]; ++total.first;
            total.second = add_checked(total.second, item.end - item.start);
            kernel_ns = add_checked(kernel_ns, item.end - item.start);
        }
        struct copy_total { uint64_t records = 0, copy_count_sum = 0, bytes = 0, ns = 0; };
        std::map<std::string, copy_total> memcpy_totals;
        for (const auto& item : data->copies) {
            auto& total = memcpy_totals[direction(item.kind)]; ++total.records;
            total.copy_count_sum = add_checked(total.copy_count_sum, item.copy_count);
            total.bytes = add_checked(total.bytes, item.bytes);
            total.ns = add_checked(total.ns, item.end - item.start);
            memcpy_ns = add_checked(memcpy_ns, item.end - item.start);
            memcpy_bytes = add_checked(memcpy_bytes, item.bytes);
        }
        const bool complete = data->kernels_enabled && data->memcpy_enabled &&
            !data->kernels.empty() && !data->copies.empty() && data->errors.empty() &&
            !data->dropped && !data->incomplete && !data->exceptions && !data->truncated && !data->invalid_buffers;
        std::ostringstream output;
        output << std::setprecision(17) << "{\"schema_version\":2,\"completed\":" << (complete ? "true" : "false")
            << ",\"diagnostic\":true,\"hardware_counters\":false,\"paired_performance_claim\":false"
            << ",\"cupti_api_version\":" << data->api_version
            << ",\"kernel_record_type\":\"CUpti_ActivityKernel13\",\"memcpy_record_type\":\"CUpti_ActivityMemcpy7\""
            << ",\"kernels_enabled\":" << (data->kernels_enabled ? "true" : "false")
            << ",\"memcpy_enabled\":" << (data->memcpy_enabled ? "true" : "false")
            << ",\"dropped\":" << data->dropped << ",\"incomplete\":" << data->incomplete
            << ",\"callback_exceptions\":" << data->exceptions << ",\"truncated\":" << data->truncated
            << ",\"invalid_buffers\":" << data->invalid_buffers
            << ",\"kernel_count\":" << data->kernels.size() << ",\"sum_kernel_ms\":" << kernel_ns / 1e6
            << ",\"memcpy_record_count\":" << data->copies.size() << ",\"sum_memcpy_ms\":" << memcpy_ns / 1e6
            << ",\"memcpy_bytes\":" << memcpy_bytes << ",\"errors\":[";
        for (size_t i = 0; i < data->errors.size(); ++i) output << (i ? "," : "") << quoted(data->errors[i]);
        output << "],\"kernels\":["; bool comma = false;
        for (const auto& item : kernel_totals) {
            output << (comma ? "," : "") << "{\"name\":" << quoted(item.first)
                << ",\"count\":" << item.second.first << ",\"sum_ms\":" << item.second.second / 1e6 << '}';
            comma = true;
        }
        output << "],\"memcpy_totals\":["; comma = false;
        for (const auto& item : memcpy_totals) {
            const auto& total = item.second;
            output << (comma ? "," : "") << "{\"direction\":" << quoted(item.first)
                << ",\"records\":" << total.records << ",\"copy_count_sum\":" << total.copy_count_sum
                << ",\"bytes\":" << total.bytes << ",\"sum_ms\":" << total.ns / 1e6 << '}';
            comma = true;
        }
        output << "],\"kernel_events\":[";
        for (size_t i = 0; i < data->kernels.size(); ++i) {
            const auto& e = data->kernels[i];
            output << (i ? "," : "") << "{\"name\":" << quoted(e.name) << ",\"start_ns\":" << e.start << ",\"end_ns\":" << e.end
                << ",\"device\":" << e.device << ",\"context\":" << e.context << ",\"stream\":" << e.stream << ",\"correlation\":" << e.correlation << '}';
        }
        output << "],\"memcpy_events\":[";
        for (size_t i = 0; i < data->copies.size(); ++i) {
            const auto& e = data->copies[i];
            output << (i ? "," : "") << "{\"direction\":" << quoted(direction(e.kind))
                << ",\"copy_kind\":" << unsigned(e.kind) << ",\"src_kind\":" << unsigned(e.src_kind)
                << ",\"dst_kind\":" << unsigned(e.dst_kind) << ",\"flags\":" << unsigned(e.flags)
                << ",\"bytes\":" << e.bytes << ",\"start_ns\":" << e.start << ",\"end_ns\":" << e.end
                << ",\"device\":" << e.device << ",\"context\":" << e.context << ",\"stream\":" << e.stream
                << ",\"correlation\":" << e.correlation << ",\"runtime_correlation\":" << e.runtime_correlation
                << ",\"copy_count\":" << e.copy_count << ",\"graph_node\":" << e.graph_node << ",\"graph\":" << e.graph << '}';
        }
        output << "],\"scope\":\"All kernel and CUPTI MEMCPY activities in this process, including model loading and any warmup. Timestamps measure recorded GPU activity intervals; sums may overlap and are not wall time or host staging/API duration. Copy byte counts describe CUDA memcpy activity, not independent PCIe hardware traffic counters. Excludes MEMCPY2 peer records, memset, host work and service overhead. Bind exact forward counts and match transfer sizes/order before attributing copies to a tower. Instrumented diagnostic, never a paired performance claim.\"}\n";
        const std::string json = output.str();
        size_t offset = 0;
        while (offset < json.size()) {
            const ssize_t written = ::write(data->output_fd, json.data() + offset, json.size() - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                std::fprintf(stderr, "CUPTI transfer hook: failed writing '%s': %s\n", data->output.c_str(), std::strerror(errno));
                break;
            }
            offset += static_cast<size_t>(written);
        }
    } catch (...) {
        std::fprintf(stderr, "CUPTI transfer hook: report finalization failed for '%s'\n", data->output.c_str());
    }
    if (::close(data->output_fd) != 0)
        std::fprintf(stderr, "CUPTI transfer hook: failed closing '%s': %s\n", data->output.c_str(), std::strerror(errno));
    data->output_fd = -1;
}
} // namespace
