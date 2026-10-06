#!/usr/bin/env python3
"""Smoke-test and benchmark the UnifoLM-WMA decision-making server.

Starts no server itself: point it at one that is already listening (the compose
service `wma-server` does that) and it POSTs real observations from a converted
G1 Dex3 episode, reporting latency and the predicted action chunk.

It also answers the question the training runs cannot: how fast is one policy
query? That number decides how often the Isaac Sim provider may call the model,
since every call runs a 16-frame DDIM world-model rollout.

The request format is the one scripts/evaluation/real_eval_server.py expects:
    {'observation.images.top': <3,H,W> uint8,
     'observation.state':      <28,>,
     'action':                 <16,28> zeros,
     'language_instruction':   <str>}
Note the image is CHW: the server applies torchvision transforms to it and then
unsqueezes, so (3,H,W) -> (1,3,H,W) is what the UNet expects. Sending HWC
silently misinterprets the channels.

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


def read_frame(path, index=0):
    """Decode one frame as CHW uint8 via decord (the same reader WMAData uses)."""
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    frame = vr[index].asnumpy()          # H,W,C uint8
    return np.transpose(frame, (2, 0, 1)).copy()


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

    with h5py.File(h5_path, "r") as f:
        state = np.array(f["observation.state"][0], dtype=np.float32)
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
    import pandas as pd
    instruction = pd.read_csv(f"{root}/{args.dataset}.csv").iloc[0]["instruction"]

    image = read_frame(mp4, 0)
    print(f"frame        {image.shape} {image.dtype}  (CHW)")
    print(f"state        {state.shape} {state.dtype}")
    print(f"instruction  {instruction!r}")
    print(f"h5 attrs     {attrs}")

    payload = {"image": image, "state": state,
               "zeros": np.zeros((16, state.shape[0]), dtype=np.float32),
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