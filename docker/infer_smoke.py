#!/usr/bin/env python3
"""Smoke-test and benchmark the UnifoLM-WMA decision-making server.

Starts no server itself: point it at one that is already listening (the compose
service `wma-server` does that) and it POSTs real observations from a converted
G1 Dex3 episode, reporting latency and the predicted action chunk.

It also answers the question the training runs cannot: how fast is one policy
query? That number decides how often the Isaac Sim provider may call the model,
since every call runs a 16-frame DDIM world-model rollout.

The request format is the one scripts/evaluation/real_eval_server.py expects,
and both the image and the state carry a TIME axis of n_obs_steps (=2):

    {'observation.images.top': <2,3,H,W> uint8,   # (T,C,H,W)
     'observation.state':      <2,28>,             # (T,DoF) history
     'action':                 <16,28> zeros,
     'language_instruction':   <str>}

Two interface facts, both found by running this client:
- the image must be (T,C,H,W): the server does img[:, -1] and
  img.permute(0,2,1,3,4), so the model consumes (B,T,C,H,W), and the server
  obtains that from spatial_transform(images).unsqueeze(0). A (3,H,W) payload
  becomes (1,3,H,W) and the channel axis is then read as time;
- the state must be 2-D: wma_data._map_to_uni_state does
  uni_state_mask[:, :state_dim] = 1.

So the policy is conditioned on a 2-frame observation history, which the Isaac
Sim provider will have to buffer.

Usage (inside the container):
    python docker/infer_smoke.py --url http://127.0.0.1:8000/predict_action \
        --dataset g1_dex3_graspsquare_1ep --repeats 3
"""
import argparse
import json
import time
import urllib.request

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/predict_action")
    p.add_argument("--data-dir", default="/data_wma")
    p.add_argument("--dataset", default="g1_dex3_graspsquare_1ep")
    p.add_argument("--view", default="observation.images.cam_left_high")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", default=None, help="write the result JSON here")
    return p.parse_args()


def read_frames(path, indices):
    """Decode frames as (T,C,H,W) uint8 via decord (the reader WMAData uses).

    The time axis is required: scripts/evaluation/real_eval_server.py does
    img[:, -1, ...] and img.permute(0, 2, 1, 3, 4), i.e. it expects
    (B, T, C, H, W). Its own pipeline builds that with
    spatial_transform(images).unsqueeze(0), so the payload must already be
    (T, C, H, W) - one entry per observation frame, n_obs_steps_imagen of them.
    Sending (C, H, W) yields (B=1, C, H, W) and the model then treats the
    channel axis as time.
    """
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    idx = [min(i, len(vr) - 1) for i in indices]
    frames = vr.get_batch(idx).asnumpy()          # T,H,W,C uint8
    return np.transpose(frames, (0, 3, 1, 2)).copy()


def post(url, payload, timeout):
    body = json.dumps({
        "observation.images.top": payload["image"].tolist(),
        "observation.state": payload["state"].tolist(),
        "action": payload["zeros"].tolist(),
        "language_instruction": payload["instruction"],
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    args = parse_args()
    root = f"{args.data_dir}"
    mp4 = f"{root}/videos/{args.dataset}/{args.view}/0.mp4"
    h5_path = f"{root}/transitions/{args.dataset}/0.h5"

    # The state must be a (n_obs_steps, DoF) HISTORY, not a single frame:
    # wma_data._map_to_uni_state does uni_state_mask[:, :state_dim] = 1, which
    # requires 2 dimensions, and the config sets n_obs_steps: 2. The server then
    # unsqueezes to (1, 2, 28). Sending a bare (28,) vector raises
    # IndexError: too many indices for tensor of dimension 1.
    with h5py.File(h5_path, "r") as f:
        all_states = np.array(f["observation.state"][:], dtype=np.float32)
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
    n_obs = 2
    state = np.stack([all_states[0]] * n_obs) if len(all_states) < n_obs \
        else all_states[:n_obs]
    import pandas as pd
    instruction = pd.read_csv(f"{root}/{args.dataset}.csv").iloc[0]["instruction"]

    image = read_frames(mp4, [0, 1])
    print(f"image        {image.shape} {image.dtype}  (T,C,H,W)")
    print(f"state        {state.shape} {state.dtype}  (n_obs_steps={n_obs}, DoF={state.shape[-1]})")
    print(f"instruction  {instruction!r}")
    print(f"h5 attrs     {attrs}")

    # The placeholder action must be the full (horizon, DoF) chunk. Note state
    # is (n_obs_steps, DoF), so the DoF is the LAST axis: using state.shape[0]
    # silently sent (16, 2) - the observation count - and the server then padded
    # it to (16, 28) with a 2-entry mask, ran the rollout happily, and only
    # failed at the very end when it indexed the result with that mask:
    #   normolize.py:227 "tensor a (2) must match tensor b (28)"
    # So the shape was wrong at the payload boundary, not in the model.
    horizon, dof = 16, state.shape[-1]
    zeros = np.zeros((horizon, dof), dtype=np.float32)
    assert zeros.shape == (16, 28), f"placeholder action must be (16, 28), got {zeros.shape}"
    payload = {"image": image, "state": state, "zeros": zeros,
               "instruction": instruction}

    results = []
    for i in range(args.repeats):
        t0 = time.time()
        resp = post(args.url, payload, args.timeout)
        dt = time.time() - t0
        ok = resp.get("result") == "ok"
        act = resp.get("action")
        shape = np.array(act).shape if act is not None else None
        print(f"\n--- request {i+1}/{args.repeats}: {dt:.2f} s  result={resp.get('result')}"
              f"  action{shape}")
        if not ok:
            print(str(resp.get("desc"))[:1500])
        else:
            a = np.array(act)
            print(f"    action[0][:6] = {np.round(a[0][:6], 4)}")
            print(f"    per-step |delta| mean = "
                  f"{np.abs(np.diff(a, axis=0)).mean():.4f} rad")
        results.append({"seconds": dt, "result": resp.get("result"),
                        "action_shape": list(shape) if shape else None,
                        "action": act if ok else None})

    lat = [r["seconds"] for r in results if r["result"] == "ok"]
    print()
    print("=" * 70)
    if lat:
        print(f" INFERENCE SMOKE: PASS   {len(lat)}/{args.repeats} ok")
        print(f"   median latency {sorted(lat)[len(lat)//2]:.2f} s per query "
              f"(16-frame DDIM-{16} world-model rollout)")
        print(f"   => a sim provider may query at most ~"
              f"{1.0/max(sorted(lat)[len(lat)//2], 1e-9):.2f} Hz")
    else:
        print(" INFERENCE SMOKE: FAIL - no successful request")
    print("=" * 70)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"url": args.url, "dataset": args.dataset,
                       "instruction": instruction, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()