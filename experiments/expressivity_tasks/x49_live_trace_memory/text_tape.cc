// Exact LS literal writer. No model, target, process state, file or address inputs.
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <unordered_map>

#ifdef __FAST_MATH__
#error "Literal marks require IEEE double precision; do not build with fast-math"
#endif
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53);

namespace {
struct Row { std::uint32_t id; std::uint16_t total, distinct; };
struct Hit { std::uint64_t key; Row* row; std::uint16_t* count; };
constexpr unsigned depths[] = {1, 2, 4, 8};
constexpr std::uint64_t masks[] = {0xffULL, 0xffffULL, 0xffffffffULL, ~std::uint64_t(0)};
constexpr std::uint32_t pair_key(const Row& row, std::uint8_t b) { return (row.id << 8) | b; }
}

// ABI 1. The wrapper owns n input bytes, n output bytes, and 9 uint32 counters, respectively.
extern "C" int x49_text_tape_v1(const std::uint8_t* input, unsigned n,
                                  std::uint8_t* tape, std::uint32_t* work) noexcept {
    if (!input || !tape || !work || n == 0 || n > 4096) return 1;
    try {
        std::array<std::uint16_t, 256> order_zero{};
        // The depth namespaces distinguish leading zero bytes even when integer keys coincide.
        std::array<std::unordered_map<std::uint64_t, Row>, 4> contexts;
        std::unordered_map<std::uint32_t, std::uint16_t> pairs;
        std::uint64_t tail = 0;
        std::uint32_t lookups = 0, seen = 0, updates = 0, ids = 0, saturated = 0;
        for (unsigned j = 0; j < n; ++j) {
            const auto b = input[j];
            double p = (order_zero[b] + 0.5) / (j + 128);
            Hit hist[4]{};
            unsigned levels = 0;
            for (; levels < 4 && depths[levels] <= j; ++levels) {
                auto& hit = hist[levels];
                hit.key = tail & masks[levels];
                auto& plane = contexts[levels];
                auto it = plane.find(hit.key);
                ++lookups;
                if (it == plane.end()) continue;
                ++seen;
                auto& row = it->second;
                hit.row = &row;
                auto follower = pairs.find(pair_key(row, b));
                if (follower != pairs.end()) hit.count = &follower->second;
                p = ((hit.count ? *hit.count : 0) + row.distinct * p) / (row.total + row.distinct);
            }
            if (!(p > 0 && p <= 1)) return 2;
            tape[j] = j ? static_cast<std::uint8_t>(std::min(255., std::floor(16. * std::max(0., -std::log(p)) + 0.5))) : 0;
            saturated += (tape[j] == 255);
            // No insert until after the mark. unordered_map rehash preserves element pointers.
            for (unsigned i = 0; i < levels; ++i) {
                auto& hit = hist[i];
                if (!hit.row) {
                    auto result = contexts[i].emplace(hit.key, Row{ids++, 0, 0});
                    if (!result.second) return 3;
                    hit.row = &result.first->second;
                }
                if (hit.count) ++*hit.count;
                else {
                    if (!pairs.emplace(pair_key(*hit.row, b), 1).second) return 4;
                    ++hit.row->distinct;
                }
                ++hit.row->total;
                ++updates;
            }
            ++order_zero[b];
            tail = (tail << 8) | b;
        }
        const std::uint32_t counters[] = {n, n-1, n, lookups, updates, seen, ids,
                                          static_cast<std::uint32_t>(pairs.size()), saturated};
        std::copy(std::begin(counters), std::end(counters), work);
    } catch (...) { return 5; } // Allocation failure is not a missing context or an empty result.
    return 0;
}
