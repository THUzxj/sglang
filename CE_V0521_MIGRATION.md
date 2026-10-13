# Context-engineering migration to SGLang v0.5.21

Source: `codex/decode-log-server-time-v0517-ce` at `c8949f96e4`, compared with
official `v0.5.17` (`2948168546`). Destination: `codex/ce-v0521-port`, based on
official `v0.5.21` (`e00930c548`). The two upstream tags have diverged, so
this branch ports behavior by feature instead of merging the source tree.

The archive commit `d917ff2a09` is canceled by `c8949f96e4` and has no net
effect. The two merge commits `36d1114eee` and `346934b2f1` are topology, not
independent features. `f87a9a5c17` (`data_parallel_rank` alias) already exists
in v0.5.21.

| Feature group | Source commits | Destination commits | State |
| --- | --- | --- | --- |
| Pair scheduler admission, labels, configuration, compact starvation and decode limits | `0507fbcaa1`, `216b6cbb1b`, `fd73729f5d`, `2f21295bf4`, `e36de13ef2`, `1845639b4e`, `e01955a6e0` | `63eb35ae7e`, `56ada06080`, `c80a2242aa`, `1086af0824`, `c539a77738`, `4f7217d955` | Partial: admission and retract-mode limits ported; scheduler lifecycle still needs integration |
| Pair-aware KV pressure and compact pause/resume | `d5fd528f6b`, `89290ed0f4`, `2300eb3284`, `1221770f30`, `6c212d742a` | `72d612596d`, `7e27559d14` | Partial: compact-first OOM recovery and GPU-resident pause/resume ported; radix-evictable mode pending |
| Decode HiCache, Full-KV-only restore, Mamba and ownership | `b0350bfc73`, `f65ce7374f`, `64de14e6aa`, `86452ef5c7`, `8c754d7bbe`, `f8ffd6da67`, `fbd5556e16`, `3e34976738`, `fddc511146`, `ff2e533699`, `30b78f64d4`, `722b512ca1`, `c62558f756` | — | Pending architectural port to v0.5.21 unified cache; individual fixes must be compared with upstream behavior |
| PD transport and allocation | `dc1cb64e04`, `6afc37846b`, `9a0b7a62a5` | `13371a85e9`, `1ad29d8331`, `d8701e41f4` | Ported at code level; end-to-end transfer and KV pressure tests pending |
| Request control and lifecycle | `0c3905ea40`, `8a8c8cbb46` | `d2e2eb1819`, `b88eea470f` | Ported at code level; full runtime tests pending |
| Timing, scheduler status and diagnostic logs | `c63f1fdda4`, `228e669d40`, `1ad8cbb498`, `36e4cfda48`, `f8f1b21204`, `5c0119e8ba`, `affac55eeb`, `ab98cd7b47`, `04385a8577`, `7f27543fec`, `03ba01019b`, `43fc2ff420`, `ba52cfe8d0`, `284587df52`, `3b09a07078` | `ae07db12e8`, `00e5a26249`, `751585c4cd`, `52fa178024` | Partial: timing and status ports landed; prefill attribution, queue composition and detailed cache diagnostics pending |
| Design and API notes | `023f190c76` | — | Pending review against the v0.5.21 implementation; source notes describe v0.5.17 internals |

Current lightweight checks: Python syntax and `git diff --check` for each
ported change; standalone pair-scheduler helper tests and focused AST smoke
checks for request update, status logging and pair retraction. The local Python
environment lacks `orjson` and `pytest`, so package-level tests have not run.
GPU-backed PD, HiCache and pause/resume behavior remains unverified.
