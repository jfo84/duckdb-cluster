# cluster

Hierarchical clustering for DuckDB. `upgma(a, b, distance)` is an aggregate that takes every pairwise
distance between a set of items and returns the average-linkage (UPGMA) tree.

```sql
LOAD cluster;

-- Every pair of leaves once, in either order.
CREATE TABLE pairs AS
SELECT * FROM (VALUES (0, 1, 0.1), (2, 3, 0.2), (0, 2, 0.8), (0, 3, 1.0), (1, 2, 0.9), (1, 3, 0.9)) t(a, b, distance);

SELECT unnest(upgma(a, b, distance), recursive := true) FROM pairs;
```

| k | left | right | height | size |
|---|------|-------|--------|------|
| 0 | 0    | 1     | 0.1    | 2    |
| 1 | 2    | 3     | 0.2    | 2    |
| 2 | 4    | 5     | 0.9    | 4    |

## Semantics

- **Input.** One row per unordered pair of leaves: two `BIGINT` ids and a `DOUBLE` distance. Every pair
  must appear exactly once. A missing or repeated pair, a `NULL`, a non-finite distance or a pair of an
  id with itself is an error, never a guess.
- **Leaves** are numbered `0 .. n-1` by the rank of their ids, so ids already `0 .. n-1` keep their numbers.
- **Output.** A list of `STRUCT(k, left, right, height, size)`, one per merge, ordered by height. A `left`
  or `right` below `n` is a leaf, and `n + k` is merge `k`. `height` is the average distance between the
  two clusters' members, and `size` is the number of leaves under the merge. `left` is the smaller id.
- **Groups.** It is an ordinary aggregate, so `GROUP BY` gives one tree per group. No rows give `NULL`.

To cluster by correlation, use `1 - corr` as the distance.

## Algorithm

The nearest-neighbour chain (Müllner, *Modern hierarchical, agglomerative clustering algorithms*, 2011):
O(n²) time, and O(n²) memory for the condensed n(n−1)/2 distance matrix of doubles. Average linkage is
reducible, so the chain finds the same tree as merging the closest pair at every step. Merges are then
ordered by height and named, so a parent always follows its children. Ties go to the lower index.

## Benchmark

`upgma()` against the same clustering written as plain SQL (rounds of mutual nearest pairs in a recursive
CTE). Both ran in the same DuckDB build on an Apple M-series Mac mini (12 cores, 32 GB), load average
1.6–2.5, with `MallocLargeCache=0`. Times are wall clock for one CLI run, including the n² pair generation.

| leaves | upgma() | SQL rounds |
|-------:|--------:|-----------:|
| 500 | 0.03 s, 0.02 GiB | 0.05 s, 0.16 GiB |
| 2,000 | 0.12 s, 0.11 GiB | 0.25 s, 0.55 GiB |
| 5,000 | 0.72 s, 0.86 GiB | 1.34 s, 2.62 GiB |

At every size the heights match each other and `scipy.cluster.hierarchy.linkage(method="average")`.

## Building

```sh
git clone --recurse-submodules https://github.com/jfo84/duckdb-cluster
cd duckdb-cluster
GEN=ninja make
./build/release/test/unittest --test-dir . "test/*"
```

This builds DuckDB from the `duckdb` submodule along with the extension. The loadable extension is at
`build/release/extension/cluster/cluster.duckdb_extension`.

## License

MIT.
