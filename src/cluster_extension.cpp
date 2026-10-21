#define DUCKDB_EXTENSION_MAIN

#include "cluster_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

namespace duckdb {

// One input row: two leaf ids and the distance between them.
struct UpgmaPair {
	int64_t a;
	int64_t b;
	double distance;
};

// The state holds a pointer, not the rows: DuckDB moves aggregate states, so they must be trivially movable.
struct UpgmaState {
	vector<UpgmaPair> *pairs;
};

// One merge as returned: an id below the leaf count is a leaf, `n + k` is merge `k`.
struct UpgmaMerge {
	int64_t left;
	int64_t right;
	double height;
	int64_t size;
};

struct UpgmaOperation {
	static void Initialize(UpgmaState &state) {
		state.pairs = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.pairs;
		state.pairs = nullptr;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		if (!source.pairs) {
			return;
		}
		if (!target.pairs) {
			target.pairs = new vector<UpgmaPair>();
		}
		target.pairs->insert(target.pairs->end(), source.pairs->begin(), source.pairs->end());
	}

	static bool IgnoreNull() {
		return false;
	}
};

static void UpgmaUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
	D_ASSERT(input_count == 3);
	UnifiedVectorFormat a_data, b_data, d_data, s_data;
	inputs[0].ToUnifiedFormat(count, a_data);
	inputs[1].ToUnifiedFormat(count, b_data);
	inputs[2].ToUnifiedFormat(count, d_data);
	state_vector.ToUnifiedFormat(count, s_data);
	auto a = UnifiedVectorFormat::GetData<int64_t>(a_data);
	auto b = UnifiedVectorFormat::GetData<int64_t>(b_data);
	auto d = UnifiedVectorFormat::GetData<double>(d_data);
	auto states = UnifiedVectorFormat::GetData<UpgmaState *>(s_data);
	for (idx_t i = 0; i < count; i++) {
		auto ai = a_data.sel->get_index(i);
		auto bi = b_data.sel->get_index(i);
		auto di = d_data.sel->get_index(i);
		if (!a_data.validity.RowIsValid(ai) || !b_data.validity.RowIsValid(bi) || !d_data.validity.RowIsValid(di)) {
			throw InvalidInputException("upgma: a pair has a NULL id or distance; every pair must be measured");
		}
		if (!std::isfinite(d[di])) {
			throw InvalidInputException("upgma: the distance between %lld and %lld is not finite", a[ai], b[bi]);
		}
		if (a[ai] == b[bi]) {
			throw InvalidInputException("upgma: a pair joins %lld to itself", a[ai]);
		}
		auto &state = *states[s_data.sel->get_index(i)];
		if (!state.pairs) {
			state.pairs = new vector<UpgmaPair>();
		}
		state.pairs->push_back({std::min(a[ai], b[bi]), std::max(a[ai], b[bi]), d[di]});
	}
}

// Index of (i, j), i < j, in the row-major upper triangle of an n x n matrix.
static inline idx_t Condensed(idx_t n, idx_t i, idx_t j) {
	return n * i - i * (i + 1) / 2 + (j - i - 1);
}

// Average linkage by the nearest-neighbour chain (Muellner 2011): O(n^2) time over the n(n-1)/2 matrix. The
// linkage is reducible, so the chain finds the same tree as merging the closest pair at every step.
static vector<UpgmaMerge> Upgma(vector<UpgmaPair> &pairs) {
	vector<int64_t> ids;
	ids.reserve(pairs.size() * 2);
	for (auto &p : pairs) {
		ids.push_back(p.a);
		ids.push_back(p.b);
	}
	std::sort(ids.begin(), ids.end());
	ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
	const idx_t n = ids.size();
	const idx_t want = n * (n - 1) / 2;
	if (pairs.size() != want) {
		throw InvalidInputException("upgma: %llu pairs for %llu leaves; every pair must appear exactly once, want %llu",
		                            (unsigned long long)pairs.size(), (unsigned long long)n, (unsigned long long)want);
	}
	auto leaf = [&](int64_t id) {
		return idx_t(std::lower_bound(ids.begin(), ids.end(), id) - ids.begin());
	};
	vector<double> dist(want, std::numeric_limits<double>::quiet_NaN());
	for (auto &p : pairs) {
		auto &slot = dist[Condensed(n, leaf(p.a), leaf(p.b))];
		if (!std::isnan(slot)) {
			throw InvalidInputException("upgma: the pair (%lld, %lld) appears twice", p.a, p.b);
		}
		slot = p.distance;
	}
	vector<UpgmaPair>().swap(pairs);
	auto d = [&](idx_t i, idx_t j) -> double & {
		return i < j ? dist[Condensed(n, i, j)] : dist[Condensed(n, j, i)];
	};

	// Merges in chain order, by the slots they joined; the survivor keeps the lower slot.
	vector<std::tuple<idx_t, idx_t, double, idx_t>> raw;
	raw.reserve(n ? n - 1 : 0);
	vector<idx_t> size(n, 1);
	vector<bool> active(n, true);
	vector<idx_t> chain;
	idx_t remaining = n;
	idx_t first = 0;
	while (remaining > 1) {
		if (chain.empty()) {
			while (!active[first]) {
				first++;
			}
			chain.push_back(first);
		}
		const idx_t a = chain.back();
		// On a tie the previous chain element wins, which is what makes the chain terminate.
		idx_t b = chain.size() >= 2 ? chain[chain.size() - 2] : n;
		double best = b < n ? d(a, b) : std::numeric_limits<double>::infinity();
		for (idx_t x = 0; x < n; x++) {
			if (x != a && active[x] && d(a, x) < best) {
				best = d(a, x);
				b = x;
			}
		}
		if (chain.size() >= 2 && b == chain[chain.size() - 2]) {
			chain.pop_back();
			chain.pop_back();
			const idx_t keep = std::min(a, b), gone = std::max(a, b);
			const double sa = double(size[keep]), sb = double(size[gone]);
			for (idx_t x = 0; x < n; x++) {
				if (active[x] && x != keep && x != gone) {
					d(keep, x) = (sa * d(keep, x) + sb * d(gone, x)) / (sa + sb);
				}
			}
			raw.emplace_back(keep, gone, best, size[keep] + size[gone]);
			size[keep] += size[gone];
			active[gone] = false;
			remaining--;
		} else {
			chain.push_back(b);
		}
	}

	// By height, stable, so at a tie a merge stays after the merges it contains; then each side is named by the
	// newest merge holding that slot, through a union-find over slots.
	vector<idx_t> order(raw.size());
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(),
	                 [&](idx_t x, idx_t y) { return std::get<2>(raw[x]) < std::get<2>(raw[y]); });
	vector<idx_t> parent(n), node(n);
	std::iota(parent.begin(), parent.end(), 0);
	std::iota(node.begin(), node.end(), 0);
	auto find = [&](idx_t x) {
		while (parent[x] != x) {
			parent[x] = parent[parent[x]];
			x = parent[x];
		}
		return x;
	};
	vector<UpgmaMerge> merges;
	merges.reserve(raw.size());
	for (idx_t k = 0; k < order.size(); k++) {
		auto &m = raw[order[k]];
		const idx_t rx = find(std::get<0>(m)), ry = find(std::get<1>(m));
		const int64_t nx = int64_t(node[rx]), ny = int64_t(node[ry]);
		merges.push_back({std::min(nx, ny), std::max(nx, ny), std::get<2>(m), int64_t(std::get<3>(m))});
		parent[ry] = rx;
		node[rx] = n + k;
	}
	return merges;
}

static LogicalType MergeType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("k", LogicalType::BIGINT);
	fields.emplace_back("left", LogicalType::BIGINT);
	fields.emplace_back("right", LogicalType::BIGINT);
	fields.emplace_back("height", LogicalType::DOUBLE);
	fields.emplace_back("size", LogicalType::BIGINT);
	return LogicalType::STRUCT(std::move(fields));
}

static void UpgmaFinalize(Vector &state_vector, AggregateFinalizeInputData &, Vector &result, idx_t count,
                          idx_t offset) {
	UnifiedVectorFormat s_data;
	state_vector.ToUnifiedFormat(count, s_data);
	auto states = UnifiedVectorFormat::GetData<UpgmaState *>(s_data);
	auto list_entries = FlatVector::GetDataMutable<list_entry_t>(result);
	auto &mask = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		const auto rid = i + offset;
		auto &state = *states[s_data.sel->get_index(i)];
		if (!state.pairs) {
			mask.SetInvalid(rid);
			continue;
		}
		auto merges = Upgma(*state.pairs);
		auto old_len = ListVector::GetListSize(result);
		ListVector::Reserve(result, old_len + merges.size());
		auto &child = ListVector::GetEntry(result);
		auto &fields = StructVector::GetEntries(child);
		auto k = FlatVector::GetDataMutable<int64_t>(fields[0]);
		auto left = FlatVector::GetDataMutable<int64_t>(fields[1]);
		auto right = FlatVector::GetDataMutable<int64_t>(fields[2]);
		auto height = FlatVector::GetDataMutable<double>(fields[3]);
		auto size = FlatVector::GetDataMutable<int64_t>(fields[4]);
		for (idx_t m = 0; m < merges.size(); m++) {
			k[old_len + m] = int64_t(m);
			left[old_len + m] = merges[m].left;
			right[old_len + m] = merges[m].right;
			height[old_len + m] = merges[m].height;
			size[old_len + m] = merges[m].size;
		}
		list_entries[rid] = list_entry_t(old_len, merges.size());
		ListVector::SetListSize(result, old_len + merges.size());
	}
}

static void LoadInternal(ExtensionLoader &loader) {
	AggregateFunction upgma("upgma", {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE},
	                        LogicalType::LIST(MergeType()), AggregateFunction::StateSize<UpgmaState>,
	                        AggregateFunction::StateInitialize<UpgmaState, UpgmaOperation>, UpgmaUpdate,
	                        AggregateFunction::StateCombine<UpgmaState, UpgmaOperation>, UpgmaFinalize, nullptr,
	                        nullptr, AggregateFunction::StateDestroy<UpgmaState, UpgmaOperation>);
	upgma.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	loader.RegisterFunction(upgma);
}

void ClusterExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string ClusterExtension::Name() {
	return "cluster";
}

std::string ClusterExtension::Version() const {
#ifdef EXT_VERSION_CLUSTER
	return EXT_VERSION_CLUSTER;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(cluster, loader) {
	duckdb::LoadInternal(loader);
}
}
