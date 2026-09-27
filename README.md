# BEHAVIOR Challenge 2026

![2nd BEHAVIOR Challenge](BEHAVIOR-1K/docs/assets/challenge_teaser_frame_240.png)

Our work on the [2026 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/). A robot must complete long household tasks in the OmniGibson simulator, working only from what it sees and from its own joint readings. A run is scored by how many of the task's goal conditions it satisfies.

## Approach

**Starting point.** We build on the winning 2025 solution by Ilia Larchenko et al. ([report](https://arxiv.org/abs/2512.06951)). It is a Pi0.5 vision-language-action model that replaces the text prompt with learned task embeddings. It also has a "System 2" that predicts which stage of the task the robot is in and feeds that stage back into the model. We start from their released checkpoints and do not train from scratch.

**Porting to the 2026 data.** The 2026 demos use a newer dataset format and BEHAVIOR-1K release than the 2025 code expects. We rewrote the data loader and updated the dependencies so the original training pipeline runs on the 2026 demos without changes to the model.

**Stage labels from task goals.** The original model's stages are equal time slices of each demo, so they say nothing about what has actually been done. We derive a second stage label from each demo's BDDL goal conditions: the stage is the number of goal conditions currently satisfied. This is the same count the challenge score uses. We pass it to the model alongside the original stage and fine-tune the checkpoints with it.

**Evaluation.** Every checkpoint is tested with rollouts in the simulator on the challenge's public test instances. We run controlled ablations as well, for example fixing or shifting the stage input, to measure how much the policy depends on it.

## Repository layout

| Folder | Contents |
|---|---|
| `BEHAVIOR-1K/` | The simulator (OmniGibson), task definitions (BDDL) and the evaluation harness, with our rollout recording wrappers. |
| `b1k-train/` | Training code, adapted from the 2025 solution: model, data loading, BDDL stage labels, training configs, and scripts for serving the policy and running rollouts. |
| `b1k-evaluation/` | The official challenge baselines. Its Python environment is the one used for evaluation. |

Training and evaluation use separate Python environments because they need different versions of LeRobot. Datasets, checkpoints and rollout outputs are kept on disk and are not in the repository.

## Further reading

- [b1k-train/TRAIN_2026.md](b1k-train/TRAIN_2026.md): how to train on the 2026 demos.
- [b1k-train/ILIA_EVAL.md](b1k-train/ILIA_EVAL.md): how to evaluate a checkpoint in the simulator.
- [b1k-train/README.md](b1k-train/README.md): the original 2025 solution in detail.
