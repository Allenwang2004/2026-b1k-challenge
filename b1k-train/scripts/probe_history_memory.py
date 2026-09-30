"""How much memory would past-frame images cost? Compile-time analysis, nothing is executed.

`init_train_state(resume=True)` returns abstract shapes only and `jit(...).lower(...).compile()`
asks XLA for the memory it *would* need, so this allocates nothing, cannot OOM, and does not
disturb anything else on the GPU.

History frames are appended as extra entries in `obs.images` whose names end in `_h`; a patched
`embed_prefix` encodes them with the same (frozen) SigLIP and then average-pools the 16x16 patch
grid by `pool`, so the LLM sees (16/pool)^2 tokens per history frame while SigLIP still sees the
full 224x224. That is the configuration under discussion -- pooling the encoder's OUTPUT, not
downsampling its input.

Config 0 (no history) must reproduce the 74.92 GiB that the real task-1 run reported through XLA's
rematerialization warning. If it does not, nothing else here is trustworthy.
"""
import dataclasses, functools, re, sys
sys.path.insert(0, "src")

import jax, jax.numpy as jnp
import openpi.training.sharding as sharding
from openpi.shared import array_typing as at
from b1k.models.observation import Observation
from b1k.models.pi_behavior import PiBehavior
from b1k.training import config as _config

import importlib.util
spec = importlib.util.spec_from_file_location("trainmod", "scripts/train.py")
trainmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainmod)

CONFIG = "pi_behavior_b1k_bddl_ckpt2_lr2e5"     # task 1 -- the run that measured 74.92 GiB
BATCH = 16
GiB = 1024 ** 3
_orig_embed_prefix = PiBehavior.embed_prefix

# preprocess_observation rebuilds the images dict from `image_keys`, which defaults to the three
# fixed camera names -- so any extra history entry is silently dropped before the model ever sees it.
# Let every key through instead.
import b1k.models.pi_behavior as _pb
_orig_preprocess = _pb.preprocess_observation
def _preprocess_all_keys(rng, observation, *, train=False, **kw):
    kw.setdefault("image_keys", tuple(observation.images.keys()))
    return _orig_preprocess(rng, observation, train=train, **kw)
_pb.preprocess_observation = _preprocess_all_keys


def patch(pools: dict[str, int]):
    def embed_prefix(self, obs):
        hist = {k: v for k, v in obs.images.items() if k.endswith("_h")}
        if not hist:
            return _orig_embed_prefix(self, obs)
        base = dataclasses.replace(
            obs,
            images={k: v for k, v in obs.images.items() if not k.endswith("_h")},
            image_masks={k: v for k, v in obs.image_masks.items() if not k.endswith("_h")},
        )
        tokens, input_mask, ar_mask = _orig_embed_prefix(self, base)
        htok = []
        for k, img in hist.items():
            t, _ = self.PaliGemma.img(img, train=False)              # [B, 256, D]
            p = pools[k]
            if p > 1:
                b, n, d = t.shape
                s = int(round(n ** 0.5))
                t = t.reshape(b, s // p, p, s // p, p, d).mean(axis=(2, 4)).reshape(b, (s // p) ** 2, d)
            htok.append(t)
        h = jnp.concatenate(htok, axis=1)
        return (
            jnp.concatenate([h, tokens], axis=1),
            jnp.concatenate([jnp.ones(h.shape[:2], dtype=bool), input_mask], axis=1),
            jnp.concatenate([jnp.zeros(h.shape[1], dtype=bool), ar_mask]),
        )
    PiBehavior.embed_prefix = embed_prefix


def build(cfg, pools):
    obs_spec, act_spec = cfg.model.inputs_spec(batch_size=BATCH)
    mk = lambda s: jnp.zeros(s.shape, s.dtype)
    obs = jax.tree.map(mk, obs_spec)
    if pools:
        img = dict(obs.images); msk = dict(obs.image_masks)
        any_img = next(iter(obs.images.values())); any_msk = next(iter(obs.image_masks.values()))
        for k in pools:
            img[k] = jnp.zeros_like(any_img); msk[k] = jnp.ones_like(any_msk)
        obs = dataclasses.replace(obs, images=img, image_masks=msk)
    return obs, jax.tree.map(mk, act_spec)


def measure(name, pools):
    cfg = _config.get_config(CONFIG)
    cfg = dataclasses.replace(cfg, batch_size=BATCH, fsdp_devices=1)
    patch(pools) if pools else setattr(PiBehavior, "embed_prefix", _orig_embed_prefix)

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    dsh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # The real norm stats: correlated-noise flow matching needs the action correlation matrix, and
    # it is built from them, so passing None makes the model refuse to build.
    norm_stats = cfg.data.create(cfg.assets_dirs, cfg.model).norm_stats
    state_shape, state_sharding = trainmod.init_train_state(
        cfg, jax.random.key(0), mesh, resume=True, norm_stats=norm_stats)   # shapes only, no allocation

    batch = build(cfg, pools)
    step = jax.jit(
        functools.partial(trainmod.train_step, cfg),
        in_shardings=(rep, state_sharding, dsh),
        out_shardings=(state_sharding, rep),
        donate_argnums=(1,),
    )
    # The abstract train state holds ShapeDtypeStructs, which TrainState's runtime type annotations
    # reject; they are only there to describe real arrays, and nothing is executed here.
    #
    # XLA prints the compiled peak from its C++ rematerialization pass, so capture stderr at the file
    # descriptor level; memory_analysis() cannot be summed instead, because the output buffers alias
    # the donated train state and would be counted twice.
    import os, tempfile
    with tempfile.TemporaryFile() as cap:
        saved = os.dup(2); os.dup2(cap.fileno(), 2)
        try:
            with sharding.set_mesh(mesh), at.disable_typechecking():
                comp = step.lower(jax.random.key(0), state_shape, batch).compile()
        finally:
            os.dup2(saved, 2); os.close(saved)
        cap.seek(0); err = cap.read().decode("utf8", "replace")
    peaks = [float(x) for x in re.findall(r"only reduced to ([0-9.]+)GiB", err)]
    m = comp.memory_analysis()
    est = (m.temp_size_in_bytes + m.argument_size_in_bytes) / GiB
    n_tok = sum(256 // (p * p) for p in pools.values())
    return n_tok, (peaks[-1] if peaks else float("nan")), est


CONFIGS = [
    ("0  無歷史（基準，應為 ~74.9 GiB）",        {}),
    ("1  3 相機 x 4x4 池化  (16 tok/幀)",       {"base_0_rgb_h": 4, "left_wrist_0_rgb_h": 4, "right_wrist_0_rgb_h": 4}),
    ("2  頭部 2x2 + 腕部 4x4",                  {"base_0_rgb_h": 2, "left_wrist_0_rgb_h": 4, "right_wrist_0_rgb_h": 4}),
    ("3  3 相機 x 2x2 池化  (64 tok/幀)",       {"base_0_rgb_h": 2, "left_wrist_0_rgb_h": 2, "right_wrist_0_rgb_h": 2}),
    ("4  3 相機全解析度     (256 tok/幀)",      {"base_0_rgb_h": 1, "left_wrist_0_rgb_h": 1, "right_wrist_0_rgb_h": 1}),
]

BASELINE = []
print(f"config: {CONFIG}   batch={BATCH}   fsdp_devices=1")
print(f"{'組合':<34}{'+token':>8}{'prefix':>8}{'XLA 峰值':>12}{'vs 基準':>10}")
for name, pools in CONFIGS:
    try:
        ntok, peak, est = measure(name, pools)
        base = BASELINE[0] if BASELINE else peak
        if not BASELINE: BASELINE.append(peak)
        flag = "  ✅ 放得下" if peak < 95.6 else "  ❌ 超過 95.6 GiB"
        print(f"{name:<34}{ntok:>8}{806 + ntok:>8}{peak:>11.2f}G{peak - base:>+9.2f}G{flag}")
    except Exception as e:
        print(f"{name:<34}  失敗: {type(e).__name__}: {str(e)[:90]}")
