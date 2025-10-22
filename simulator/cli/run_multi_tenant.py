import os
import shutil
import subprocess
from pathlib import Path

# Settings with their optimal alpha values
SETTINGS = [
    {"input": "input_file_trace1.json", "n_engines": 2, "arrival_rate": 0.5, "alpha": 0.1},
    {"input": "input_file_trace1.json", "n_engines": 2, "arrival_rate": 1.0, "alpha": 0.0},
    {"input": "input_file_trace2.json", "n_engines": 2, "arrival_rate": 0.5, "alpha": 0.2},
    {"input": "input_file_trace2.json", "n_engines": 2, "arrival_rate": 1.0, "alpha": 0.0},
    {"input": "input_file_trace3.json", "n_engines": 2, "arrival_rate": 0.5, "alpha": 0.2},
    {"input": "input_file_trace3.json", "n_engines": 2, "arrival_rate": 1.0, "alpha": 0.0},
]
# SETTINGS = [
#     {"input": "input_file_trace1.json", "n_engines": 1, "arrival_rate": 0.5, "alpha": 0.1},
#     {"input": "input_file_trace1.json", "n_engines": 1, "arrival_rate": 1.0, "alpha": 0.0},
#     {"input": "input_file_trace2.json", "n_engines": 1, "arrival_rate": 0.5, "alpha": 0.3},
#     {"input": "input_file_trace2.json", "n_engines": 1, "arrival_rate": 1.0, "alpha": 0.1},
#     {"input": "input_file_trace3.json", "n_engines": 1, "arrival_rate": 0.5, "alpha": 0.4},
#     {"input": "input_file_trace3.json", "n_engines": 1, "arrival_rate": 1.0, "alpha": 0.0},
# ]
# SETTINGS = [
#     {"input": "input_file_trace4.json", "n_engines": 1, "arrival_rate": 2.5, "alpha": 0.3},
#     {"input": "input_file_trace4.json", "n_engines": 1, "arrival_rate": 5.0, "alpha": 0.2},
# ]

# Create main results directory
BASE_DIR = Path("./multi_tenant_result")
BASE_DIR.mkdir(exist_ok=True)

def move_results(setting_dir):
    """Move output files to the appropriate directory"""
    result_files = [
        "baseline_output.json",
        "vtc_output.json",
        "rr+pq_output.json",
        "qlm_output.json",
        "optimized_output0.json",
        "optimized_output1.json"
    ]
    
    for file in result_files:
        src = Path("./result") / file
        if src.exists():
            shutil.move(str(src), str(setting_dir / file))

# Process each setting
for i, setting in enumerate(SETTINGS, 1):
    print(f"\nProcessing Setting {i}/{len(SETTINGS)}")
    print(f"Input: {setting['input']}, Arrival Rate: {setting['arrival_rate']}, Alpha: {setting['alpha']}")
    
    # Create directory for this setting
    setting_dir = BASE_DIR / f"setting_{i}"
    setting_dir.mkdir(exist_ok=True)
    
    # Construct command
    cmd = [
        "python", "start_simulator.py",
        "--input", setting['input'],
        "--n-engines", str(setting['n_engines']),
        "--arrival-rate", str(setting['arrival_rate']),
        "--alpha", str(setting['alpha']),
        # "--multi-tenant"  # Enable multi-tenant mode
    ]
    
    # Run the simulation
    try:
        subprocess.run(cmd, check=True)
        # Move results to the setting directory
        move_results(setting_dir)
        print(f"Results saved in {setting_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running setting {i}: {e}")
        continue

print("\nAll settings processed. Results are organized in the multi_tenant_result directory.")
