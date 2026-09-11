"""Optional measured INT8 candidate. The float32 paper policy remains the default."""
from pathlib import Path
import time

import numpy as np
import torch

from .compact import load_policy
from .core import Broker, Costs, atomic_json, features


def export(path, ticks, news, output):
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy, meta = load_policy(path)
    x = np.stack([features(ticks, i, Broker(Costs(**meta["costs"])), news) for i in range(63, len(ticks), max(1, len(ticks) // 128))])
    fp, quant = output / "policy-fp32.onnx", output / "policy-int8.onnx"
    torch.onnx.export(policy, torch.from_numpy(x[:1]), str(fp), input_names=["observation"], output_names=["logits", "value"], opset_version=18, dynamo=False)
    quantize_dynamic(str(fp), str(quant), weight_type=QuantType.QInt8)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sessions = [ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"]) for p in (fp, quant)]
    values, times = [], []
    for session in sessions:
        start = time.perf_counter()
        rows = [session.run(None, {"observation": row[None]})[0][0] for row in x]
        times.append((time.perf_counter() - start) * 1000 / len(x))
        values.append(np.array(rows))
    summary = {"observations": len(x), "action_agreement": float(np.mean(values[0].argmax(1) == values[1].argmax(1))), "max_logit_difference": float(np.abs(values[0] - values[1]).max()), "fp32_ms": times[0], "int8_ms": times[1], "fp32_bytes": fp.stat().st_size + sum(p.stat().st_size for p in output.glob("policy-fp32.onnx.data*")), "int8_bytes": quant.stat().st_size, "note": "A sampled conversion/latency check, not trajectory equivalence. INT8 is not automatically promoted; validate full sequential replays and target cloud CPU first. Fly native float32 kernel is unchanged."}
    atomic_json(output / "quantization.json", summary)
    return summary
