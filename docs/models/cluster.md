# Cluster sweeps

`dexvantage/core/cluster/` runs the independent parts of a long study on more than one process: the runs of a
designed experiment (core/experiments.py), the draws of a tolerance Monte Carlo (core/tolerance.py), the designs of
each optimiser generation (core/optimise/rig.py) and any list of (project, study, settings) jobs. It runs them on this
machine's cores (a process pool) or on a Slurm cluster (job arrays). The result is the result one process gives.

| Module | What it holds |
|---|---|
| `work.py` | the work (items, parameters, projects), the shard split, one item's computation, the assembly |
| `local.py` | the process pool: workers, memory per worker, progress, cancel, resume |
| `shards.py` | one shard's run and its output file; gather, the report, the resubmit list |
| `slurm.py` | the sbatch settings, the array script, the job folder, resubmit, submit |
| `cache.py` | the shared on-disk result cache (content keys, atomic rename, no locks) |
| `optimiser.py` | the optimiser's generations in the pool, and the optimiser on Slurm in rounds |

The batch flags are in `dexvantage/batch_cluster.py`, the Python API is `dv.cluster` (`dexvantage/api_cluster.py`) and
the pages' **Run on…** is `dexvantage/ui/cluster_run.py`.

## The model

**Work.** A work is a list of M items that do not depend on each other:

| Kind | Item 0 | Items 1 … M − 1 | Computed by |
|---|---|---|---|
| `experiment` | the rig as it is | the design's runs, in plan order | `core.experiments.evaluate_row` |
| `tolerance` | the nominal rig | draw *i* − 1 | `core.metrics.evaluate` on `core.tolerance.apply_draw` |
| `studies` | job 1 | job 2 … | `core.jobs.run_study` with the job's settings |
| `designs` | design 1 | design 2 … (one optimiser generation) | the optimiser's own evaluator (`search.Evaluator`) |

The work file `work.json` has the schema `dexvantage.cluster-work/1`. It holds the kind, the items, the parameters, the
modules a worker must import (studies and metrics added outside core), and each project's file and content hash. The
projects are saved as `.dexv` files beside it. The work id is the first 32 hex digits of the SHA-256 of the canonical
JSON of {kind, items, parameters, the projects' content hashes}. Every shard output carries the work id, so outputs
of another work in the same folder are left out (gather lists them as foreign).

**Shards.** Shard *i* of *N* takes the items *i*, *i* + *N*, *i* + 2*N*, … below M. For every M ≥ 0 and N ≥ 1, the N
shards together hold each index in range(M) exactly once. The shard sizes differ by at most one. The split is
strided, so the slow runs at the end of a design spread over every shard.

**Why the gathered result is the serial one.** Each item is computed by the same function the in-process run calls,
on the same project. The saved project's content hash is checked on load, and a work file edited by hand is refused.
The tolerance draws are made again from the seed in every process, in `monte_carlo`'s order (sensor by sensor, draw by
draw), so draw *i* is the same everywhere. The assembly calls the in-process code's own analysis:

- an experiment: `core.experiments._result`, with the plan made again from the design;
- the tolerance study: `core.studies.tolerance.run`, with an evaluator that hands back the shards' figures in draw
  order;
- the optimiser: its own search, which finds every design already scored in its memo.

Item results go through a JSON round trip, and Python's JSON writes floats exactly (the shortest repr that reads back
the same double). The only parts that differ say how the work ran:
- an experiment's `execution` block (mode, workers, seconds, peak memory);
- a study's `seconds` and `cached`;
- the optimiser's `search` counts (studies scored, results reused, seconds) and its added `cluster` block.

**The optimiser.** NSGA-II is sequential across generations, but the designs of one generation are independent.
`nsga2.run` calls a `prefetch(rows)` hook with each generation's decoded genomes before it evaluates them one by one.
`rig.optimise` passes the hook (with the base project, the factors and the evaluator) to core/cluster. The pool
scores the designs not in the memo and stores each score under the memo key `rig._key(values)`. The search then
finds each design already scored. On Slurm, each round replays the seeded search from the start:
1. Designs scored in earlier rounds are filled into the memo from the gathered rounds.
2. The current rig and the confirm step come from the shared result cache.
3. The first generation with unscored designs becomes the next round's job folder.

Both hooks are optional arguments that change no result (additive).

**Memory per worker.** Each worker holds its own project copy, robot models, the studies' caches and, on the GPU path,
an EGL context. The planning figure is `MEM_PER_WORKER_MB` = 2 048 MiB (a labelled default). The default number of
workers is min(CPUs, MemAvailable / 2 048 MiB, 64). Every run reports each worker's measured peak (`VmHWM` of `/proc/self/status` on Linux, `ru_maxrss` elsewhere: on Linux `ru_maxrss` keeps, across the exec that starts a spawned worker, the peak of the process it was forked from, so a worker of a 2.9 GB parent reported 2918 MB; found at integration).
Measured on 2026-11-07 (a spawned worker running the tolerance study's four default metrics: coverage, depth, motion
blur, VLA tokens):

| Sample | Worker peak |
|---|---|
| SO-100 (coverage only) | 360–500 MB |
| UR5e | 638 MB |
| G1 | 698 MB |
| ALOHA 2 | 999 MB |

The job script's default of 4G per task (a DexVantage default, not Slurm's: Slurm's own is the site's DefMemPerNode) is above all of these. The grasp study on large grasp sets was not measured; give
it more.

**Cancel.** The pool is terminated at once. An item in progress is discarded, as core/experiments discards a run that
a cancel interrupts. With an output folder, the items done are kept, and the next run skips them.

## The shard output and gather

Each attempt of a shard writes its own file `out/shard-<i>-of-<N>.<attempt>.jsonl`. The file holds a header line
(work id, shard, item count), one line per item (index, status, result or error, seconds) and an end line (peak
memory, seconds). Each line is a single `write`, fsynced, and the file is closed after every line. A line cut short by
a killed task is not valid JSON, so it is skipped and its item counts as not done. A resubmitted shard first reads the
earlier attempts and skips the items they finished.

`gather` sorts every shard into one of four states:

| State | Meaning |
|---|---|
| complete | every item has a result |
| failed | an item raised; the item's sentence is shown |
| incomplete | the shard started but stopped |
| missing | the shard has no output at all |

When every item has a result, gather assembles the result and writes `result.json` with provenance (an experiment
also writes `result.csv`). Otherwise it writes `resubmit.sbatch`: the same script with `--array=` set to just the
shards still to do (for example `1-2,7`).

## The Slurm job folder

    job/
      work.json  projects/base.dexv   the work and its project(s)
      run.sbatch                      the array script (bash -n clean; #SBATCH lines first)
      job.json                        shards, the settings, which of them are defaults
      out/  logs/  cache/             shard outputs, task logs (logs/shard-%A_%a.out), the shared result cache
      README.txt                      how to submit, gather and run a shard by hand

Each array task runs `python -m dexvantage --run-shard work.json --shard $SLURM_ARRAY_TASK_ID --of N --out out` in
`JOB_DIR="${DEXVANTAGE_JOB_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"`. Submit from the folder. Slurm creates no folders for
`--output`, which is why `logs/` is written with the job. The script sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
and `MKL_NUM_THREADS` to the CPUs per task (a default, `threads_from_cpus`), so a task does not oversubscribe its
allocation.

The settings below are labelled DexVantage defaults; `job.json` lists which ones were left at their default:

| Setting | Default |
|---|---|
| `cpus_per_task` | 1 |
| `mem` | 4G |
| `time` | 02:00:00 |
| `partition` | the cluster's default |
| `account` | the user's default |
| `gpus` | 0 |
| `throttle` | 0 (none) |
| `job_name` | dexvantage |
| `python` | python |
| `max_array` | 1001 |

The activation line (`activate`) has no default, so no path is ever written for the user. The other settings are
`setup` lines (for example `module load`) and `extra` #SBATCH options. Values Slurm would refuse are refused in a
sentence:
- a memory size that is not a number with an optional K, M, G or T;
- a time that is not one of Slurm's six time forms;
- an extra option that does not start with `--`;
- more shards than `max_array`.

The Slurm facts used, from the Slurm 26.05 documentation, read 2026-11-07:

| Fact | Source |
|---|---|
| `--array` takes ranges, comma lists and a `%` throttle | [job_array.html](https://slurm.schedmd.com/job_array.html) |
| `SLURM_ARRAY_TASK_ID` is "the job array index value" | job_array.html |
| `%A` and `%a` in `--output` are the array job id and the task index | job_array.html |
| "The default value of MaxArraySize is 1001" | job_array.html |
| `--mem`'s "Default units are mebibytes" | [sbatch.html](https://slurm.schedmd.com/sbatch.html) |
| the `--time` forms | sbatch.html |
| without `--cpus-per-task`, one processor per task | sbatch.html |
| `SLURM_SUBMIT_DIR` is "The directory from which sbatch was invoked" | sbatch.html |
| #SBATCH lines are read up to the first non-comment, non-whitespace line | sbatch.html |
| `--parsable` prints the job id (`;cluster`) | sbatch.html |
| `afterany` waits for the named jobs to end | sbatch.html |

**PBS, LSF and other schedulers are out of scope.** Only Slurm job arrays are written. Any scheduler can still run
the `--run-shard` lines, and gather does not care who ran them.

## The shared result cache

`DiskResultCache(folder)` is a `core.jobs.ResultCache` with a folder behind its memory. A study result is stored under
the SHA-256 of the canonical JSON of the ResultCache key: study, project content hash, settings hash, library entries'
hash, active variant, dataset fingerprint. The key is the content, so an entry is never stale; it is only rewritten
with the same bytes. The file is one header line (schema, key, payload length, payload SHA-256) followed by the
payload. A reader checks the length and the hash, so a short or corrupt file reads as a miss.

**Concurrent writers without locks.** Each writer works in four steps:
1. Write the whole file to a name no other writer uses: `.<name>.<pid>.<uuid4>.tmp` in the same folder.
2. Flush and fsync it.
3. Rename it over the final name. [rename(2)](https://man7.org/linux/man-pages/man2/rename.2.html) replaces an
   existing name atomically, "so that there is no point at which another process attempting to access newpath will
   find it missing".
4. If the rename seems to fail but the target exists, take it as done. rename(2) warns that on NFS a retransmitted
   RPC can report failure for a rename that was done.

[nfs(5)](https://man7.org/linux/man-pages/man5/nfs.5.html)'s close-to-open consistency (write-back on close, a server
check on open) makes the file visible in full on other nodes.

**Why there is no flock.**
- On NFS, [flock(2)](https://man7.org/linux/man-pages/man2/flock.2.html) is emulated with fcntl byte-range locks
  (Linux 2.6.12 and later) and needs a write-open file.
- An NFS mount with `nolock` gives locks that exclude only processes on the same client (nfs(5)).
- On SMB, flock is not propagated before Linux 5.5 and is mandatory after it (flock(2)).

A lock that silently does not lock is worse than none. (Pages read 2026-11-07.)

A result that does not survive a JSON round trip exactly (a tuple, an integer key, a NumPy scalar) stays in memory
only. `prune(folder)` removes temporary files left by writers that were killed.

## Validation

`dexvantage --validate OUT --models cluster` (validation/checks_cluster.py). Measured 2026-11-07 on the development
machine: 4 of 4 pass (full run 9.5 s, quick run 5.1 s). Each check is compared with an independent reference:

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `cluster.gather_equals_serial`: a 6-run Latin hypercube and a 6-draw tolerance Monte Carlo on the SO-100 sample (coverage), 3 shards each (full run: separate `--run-shard` processes) | the same design and seed run serially in one process | 0 differing values (the code is the serial code, so any difference is a bug) | 0 |
| `cluster.partition_exact`: m = 0 … 300, N = 1 … 40 | range(m), each index once | 0 pairs | 0 |
| `cluster.cache_concurrent_writers`: 3 processes write and read the same 24 keys (about 200 kB each) at once | each key's deterministic payload, compared in full | 0 bad reads or torn files, no temporary file left | 0 |
| `cluster.sbatch_directives`: the #SBATCH options as Slurm reads them, and `bash -n` | the settings given, bash's own parser | 0 mismatches | 0 |

The tests (tests/test_cluster.py, tests/test_cluster_ui.py) also check these:
- separate-process shards against the serial experiment and tolerance study, bit for bit as canonical JSON;
- the local pool against serial;
- the optimiser's pool and its three Slurm rounds against the serial search;
- four concurrent writers;
- cut, corrupt and foreign files;
- a mock `sbatch` on PATH (`--submit` and the optimiser's afterany chain);
- the job script run with Slurm's variables set.

On a saved SO-100 project, `--tolerance-mc` (64 draws) gives the same result through three routes: in process,
`--jobs 1` and `--jobs 4`. `--jobs 1` took 26.9 s and `--jobs 4` took 7.9 s (24 CPUs).

## Known limits

- **Slurm only:** no PBS or LSF scripts.
- **A shared filesystem:** the job folder must be on a filesystem every node mounts (no staging to node-local disk).
  Filesystems without close-to-open consistency (some object-store mounts) are not supported.
- **Like machines:** a figure that depends on the machine is identical only between like machines. For example, the
  GPU and CPU occlusion paths can differ at a silhouette.
- **The occlusion cube-map cache (fixed at integration):** `core/occlusion.CubeCache` keyed a map by its origin
  rounded to 0.1 mm, so a process that had already built a map for an origin less than 0.1 mm away reused it.
  Measured: run 1 of an 8-run Latin hypercube on the SO-100 sample read 69.2431 % after other studies had run in the
  process, and 69.2448 % from a cold cache. The origin is now keyed exactly, so a long-lived process (the window, a
  notebook) and fresh shards build the same maps; the tests and the validation check still compare cold runs.
- **What must be installed:** plugins and the modules a work names must be installed where the shards run. A dataset
  a data study reads by path must be at the same path on every node.
- **The optimiser on Slurm:** one queue wait per generation; the design agent is not distributed.
- **Pool start-up:** a worker costs the import of DexVantage (about a second) and its robot models.
- **Frozen bundles:** they need multiprocessing's freeze support (Phase 11).
- **Untested items are trusted:** gather assumes items are deterministic. When two attempts give an item, the first
  read is used.
