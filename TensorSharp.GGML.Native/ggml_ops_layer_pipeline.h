// TensorSharp-owned lossless stage channel. The transport sees opaque words;
// F16 caches and integer sparse indices retain their exact bit patterns.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tsg_layer_pipeline {
using transfer = bool (*)(void * user, int peer, int send, void * data, int bytes);

struct channel {
    // node is this executor's pipeline stage. driver is the stage whose host
    // samples: logits stay local when it is the last stage and are otherwise
    // returned to it. Peers are addressed by stage; the host maps them to nodes.
    int node = 0, nodes = 1;
    transfer exchange = nullptr;
    void * user = nullptr;
    std::vector<uint8_t> staging;
    // Last so existing positional initializers keep the default driver stage 0.
    int driver = 0;
    bool active() const { return exchange != nullptr; }
    void move(int peer, bool send, void * data, size_t bytes) {
        // Bounded chunks avoid managed/native signed-size overflow. The same
        // partition is used by both endpoints, independent of packet delivery.
        auto * p = static_cast<uint8_t *>(data);
        while (bytes) {
            const int n = static_cast<int>(bytes > 16 * 1024 * 1024 ? 16 * 1024 * 1024 : bytes);
            if (!exchange || !exchange(user, peer, send ? 1 : 0, p, n))
                throw std::runtime_error("Distributed layer pipeline transport failed; restart every node");
            p += n; bytes -= n;
        }
    }
};
}
