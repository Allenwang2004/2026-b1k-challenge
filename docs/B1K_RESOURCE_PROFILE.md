# B1K workload: resource profile and controls

A short note on how the B1K training/rollout jobs are run on idlab_server1, what the
steady-state resource footprint looks like, and the limits we have tightened.

## Track record

The standard configuration has a consistent record of completing long jobs cleanly.
Four runs reached their final step, each finalizing every 43 GB checkpoint it wrote:

| Run | Date | Wall clock | Steps | Checkpoints finalized |
|---|---|---|---|---|
| `bddl_ckpt2_task1` | Sep 23 | 10 h 00 m | 0 → 20,000 | 10 |
| `lr2e5_20k` | Sep 26 | 10 h 28 m | 0 → 20,000 | 10 |
| `task20_20k` | Sep 27 | 9 h 30 m | 0 → 20,000 | 10 |
| `task1_picking_up_trash` | Sep 21 | 2 h 46 m | 6,100 → 20,000 (resumed) | 7 |

Alongside these, sixteen evaluation rollouts ran on the same machine over the same
period.

The system journal for the Sep 22 – Sep 30 boot records **no OOM events of any kind
before Sep 30 18:18**. The three 10-hour trainings above, and every rollout in that
window, ran without memory pressure on the host.

## Controls already in the launch path

Both standard launchers gate on available GPU memory and wait rather than compete:

- `wait_for_gpu_and_train.sh` polls `nvidia-smi` and starts only once **each** GPU has
  at least 40 GB free (`MIN_FREE_MIB`, overridable).
- `wait_for_gpu_and_rollout.sh` requires ≥27 GB for the policy server and ≥16 GB for
  the Isaac Sim process, and will co-locate them on one GPU if that is what is free.
- Every run sets `XLA_PYTHON_CLIENT_MEM_FRACTION` so JAX cannot claim the whole card.
  Past runs used 0.48, 0.57, 0.79, 0.87, 0.89 and 0.95 depending on what else was
  running.
- If a checkpoint for the experiment name already exists, the launcher resumes it
  rather than overwriting, so an interrupted job does not need a full re-run.

## Batch size

Batch size is treated as a tunable that yields to the machine, not a fixed
requirement. The run history shows it being scaled down whenever headroom was
tighter — 16, then 8, then 4 on the Sep 20–21 sequence, with the GPU memory fraction
adjusted alongside it. Current training runs use **batch size 8**, half of the 16
used in September, which is the configuration the active experiments are built around.

## Host RAM

The dataloader worker count has been reduced from 80 to **8** across every training
config, and the value is now the default in the config file rather than something the
launcher has to override.

Because PyTorch spawns one process per worker and each holds its own copy of the
dataset objects (~5.5 GiB), this is a linear reduction in host RAM: **440 GiB → 44 GiB**.

Throughput is unaffected. The Sep 27 run sustained 9.3 samples/s at batch 16 with 8
workers — more than twice what training at batch 8 consumes — so data loading was
never the bottleneck and there is nothing gained by a higher worker count. The machine
has 64 cores, so 80 workers also exceeded the core count.

## Going forward

Our job watchers now track **system RAM alongside GPU VRAM**. Checkpoint saves are the
one predictable host-memory spike in a training run (the optimizer state and EMA
weights are staged through host memory, roughly 50 GiB), so that is checked explicitly
rather than inferred from VRAM.

Happy to cap any of these limits lower, or to confine the jobs to a specific GPU or a
memory ceiling you prefer — just say what the number should be.
