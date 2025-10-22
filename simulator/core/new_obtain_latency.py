import os


def _latency_files(gpu_type: str):
    base = "../core/Actual_latency"
    mapping = {
        "nvidia_A100": (
            os.path.join(base, "results_A100_new.txt"),
            os.path.join(base, "pairs.txt"),
        ),
        "nvidia_A6000": (
            os.path.join(base, "results_A6000_new.txt"),
            os.path.join(base, "pairs.txt"),
        ),
        "nvidia_L40S": (
            os.path.join(base, "results_L40S_new.txt"),
            os.path.join(base, "pairs.txt"),
        ),
    }
    return mapping[gpu_type]


def _read_pairs(pairs_path: str):
    pairs = []
    with open(pairs_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("(") or not line.endswith(")"):
                continue
            a, b = line.strip("()").split(",")
            pairs.append((int(a.strip()), int(b.strip())))
    return pairs


def _read_latencies(latency_path: str):
    # Each line: total_latency, prefill_latency, per_token_decode_latency
    rows = []
    with open(latency_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            total_latency = float(parts[0])
            prefill_latency = float(parts[1])
            per_token_decode_latency = float(parts[2])
            rows.append((total_latency, prefill_latency, per_token_decode_latency))
    return rows


def build_latency_dict(gpu_type: str):
    latency_file, pairs_file = _latency_files(gpu_type)
    pairs = _read_pairs(pairs_file)
    rows = _read_latencies(latency_file)

    latency_dict = {}
    for (input_length, output_length), (latency, prefill_latency, per_token_decode_latency) in zip(pairs, rows):
        latency_dict[(input_length, output_length)] = {}
        latency_dict[(input_length, output_length)]["latency"] = latency
        latency_dict[(input_length, output_length)]["prefill_latency"] = prefill_latency
        latency_dict[(input_length, output_length)]["per_token_decode_latency"] = per_token_decode_latency
    return latency_dict


