import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import json
from dataclasses import asdict
from simulator.core.global_engine import LLMGlobalEngine
from simulator.core.global_engine_optimized import OPGlobalEngine
from simulator.core.utils import load_trace
from simulator.ui import make_table
from rich.console import Console
from huggingface_hub import login
import matplotlib.pyplot as plt  # Add this import for plotting
from simulator.core.request import STANDARD_WORKFLOW, calculate_avg_empirical_time

CUR_DIR = "./simulator"

console = Console()

def run_simulation(args, mode):
    # Create server with specified mode
    server = LLMGlobalEngine(mode)

    for i in range(args.n_engines):
        server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        # server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_A6000", 4,4,4)
        # server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_L40S", 4,4,4)
    
    server.load_requests(args.input, float(args.arrival_rate), args.SLO, args.multi_tenant)
    
    server.start()
    if mode == "baseline":
        server.save_results(CUR_DIR + "/result/baseline_output.json")
    elif mode == "vtc":
        server.save_results(CUR_DIR + "/result/vtc_output.json")
    elif mode == "rr+pq":
        server.save_results(CUR_DIR + "/result/rr+pq_output.json")
    elif mode == "qlm":
        server.save_results(CUR_DIR + "/result/qlm_output.json")

    with open(args.trace_output, "w") as f:
        data = {"traceEvents": [asdict(x) for x in server.trace]}
        f.write(json.dumps(data, indent=4))
    stats = {
        "summary": server.requests_stats,
        # "failed": server.failed_requests,
        "config": server.config,
    }
    with open(args.stats_output, "w") as f:
        f.write(json.dumps(stats, indent=4))

    print(end="\n")
    print(f"--" * 10 + " Simulation Done " + "--" * 10)

    console.print(make_table("Summary", server.summary))
    
    if args.multi_tenant:
        # Show tenant-specific results
        from simulator.core.request import calculate_tenant_slo
        tenant1_slo = calculate_tenant_slo(server.hardware_lst, 0)
        tenant2_slo = calculate_tenant_slo(server.hardware_lst, 1)
        tenant1_pass_rate, tenant2_pass_rate = server.multi_tenant_SLO_pass_rate(tenant1_slo, tenant2_slo)
        print(f"\nMulti-tenant Results:")
        print(f"Tenant 1 (Base SLO): {tenant1_slo:.2f}s - Pass Rate: {tenant1_pass_rate:.2%}")
        print(f"Tenant 2 (1.5x SLO): {tenant2_slo:.2f}s - Pass Rate: {tenant2_pass_rate:.2%}")
    
    # print(f"Pass rate: {server.SLO_pass_rate(float(args.SLO))}")
    slo_scales = [round(x, 2) for x in [3 + 0.2 * i for i in range(60)]]  # 0.3 to 2.0
    pass_rates = [server.SLO_pass_rate(args.SLO * scale) for scale in slo_scales]
    return pass_rates

def run_simulation_optimized(args, w1=1, index=0, mode="hexflow"):
    # print(args)
    server = OPGlobalEngine(alpha=args.alpha, mode=mode)

    for i in range(args.n_engines):
        server.add_engine(w1, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        # server.add_engine(w1, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        server.add_engine(w1, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_A6000", 4,4,4)
        # server.add_engine(w1, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_L40S", 4,4,4)

    server.load_requests(args.input, float(args.arrival_rate), args.SLO, args.multi_tenant)

    server.start()
    if mode == "vtc":
        server.save_results(CUR_DIR + "/result/vtc_output.json")
    elif mode == "qlm":
        server.save_results(CUR_DIR + "/result/qlm_output.json")
    else:
        server.save_results(f"{CUR_DIR}/result/optimized_output{index}.json")

    with open(f"{CUR_DIR}/result/optimized_trace_output{index}.json", "w") as f:
        data = {"traceEvents": [asdict(x) for x in server.trace]}
        f.write(json.dumps(data, indent=4))
    stats = {
        "summary": server.requests_stats,
        # "failed": server.failed_requests,
        "config": server.config,
    }
    with open(f"{CUR_DIR}/result/optimized_stats_output{index}.json", "w") as f:
        f.write(json.dumps(stats, indent=4))

    print(end="\n")
    print(f"--" * 10 + " Simulation Done " + "--" * 10)

    console.print(make_table("Summary", server.summary))
    
    if args.multi_tenant:
        # Show tenant-specific results
        from simulator.core.request import calculate_tenant_slo
        tenant1_slo = calculate_tenant_slo(server.hardware_lst, 0)
        tenant2_slo = calculate_tenant_slo(server.hardware_lst, 1)
        tenant1_pass_rate, tenant2_pass_rate = server.multi_tenant_SLO_pass_rate(tenant1_slo, tenant2_slo)
        print(f"\nMulti-tenant Results:")
        print(f"Tenant 1 (Base SLO): {tenant1_slo:.2f}s - Pass Rate: {tenant1_pass_rate:.2%}")
        print(f"Tenant 2 (1.5x SLO): {tenant2_slo:.2f}s - Pass Rate: {tenant2_pass_rate:.2%}")

    slo_scales = [round(x, 2) for x in [3 + 0.2 * i for i in range(60)]]  # 0.3 to 2.0
    pass_rates = [server.SLO_pass_rate(args.SLO * scale) for scale in slo_scales]
    return pass_rates

if __name__ == "__main__":
    import argparse
    # hardware_lst = ["nvidia_A100", "nvidia_A100", "nvidia_A6000", "nvidia_L40S"]
    hardware_lst = ["nvidia_A100", "nvidia_A6000"]
    # hardware_lst = ["nvidia_A100"]
    slo = sum([calculate_avg_empirical_time(hardware_lst, s) for s in STANDARD_WORKFLOW])
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--n-engines", type=int, help="Number of engines")
    parser.add_argument("--arrival-rate", help="Arrival rate", default=None)
    parser.add_argument("--SLO", help="Text2SQL Request SLO", default=slo)
    parser.add_argument("--alpha", type=float, help="Alpha value for the optimized engine", default=0.0)
    parser.add_argument("--multi-tenant", action="store_true", help="Enable multi-tenant mode with different SLOs")

    parser.add_argument(
        "--trace-output",
        type=str,
        help="Trace file",
        default="./result/trace_output.json",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        help="Stats file",
        default="./result/stats_output.json",
    )
    args = parser.parse_args()
    # slo_scales = [round(x, 2) for x in [3 + 0.2 * i for i in range(60)]]  # 0.3 to 3.2
    
    pass_rates_vllm = run_simulation(args, "baseline")
    pass_rates_vtc = run_simulation(args, "vtc")
    pass_rate_qlm = run_simulation(args, "qlm")
    # pass_rates_vtc = run_simulation_optimized(args, 0, 0, "vtc")
    # pass_rate_qlm = run_simulation_optimized(args, 0, 0, "qlm")
    
    # pass_rates_rrpq = run_simulation(args, "rr+pq")
    # pass_rates_wbfcfs = run_simulation_optimized(args, 0, 0)
    # pass_rates_wbpq = run_simulation_optimized(args, 1, 1)
    # # Plotting the pass rates for both baseline and optimized
    # plt.figure(figsize=(10, 6))
    # # plt.plot(slo_scales, pass_rates_rrpq, marker='o', label="RR+PQ SLO Pass Rate", color='blue')
    # plt.plot(slo_scales, pass_rates_vtc, marker='o', label=" VTC SLO Pass Rate", color='blue')
    # plt.plot(slo_scales, pass_rates_wbfcfs, marker='o', label="WB+FCFS SLO Pass Rate", color='red')
    # plt.plot(slo_scales, pass_rates_wbpq, marker='o', label="WB+PQ SLO Pass Rate", color='green')
    # plt.title(f"SLO Pass Rate vs SLO Scale")
    # plt.xlabel("SLO Scale")
    # plt.ylabel("Pass Rate")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(f"./result/slo_pass_plot_1.png")
    # plt.close()
