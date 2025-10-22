import re
import csv
from pathlib import Path

def process_screen_file(input_file):
    """Process the screen output format file (A100)"""
    results = []
    with open(input_file, 'r') as f_in:
        for line in f_in:
            if not line.strip() or line.strip() == 'Record':
                continue

            pattern = r'\[(\d+)\] in=(\d+) out=(\d+)\s+TTFT=([\d.]+) ms\s+Decode=([\d.]+) ms\s+toks/s=([\d.]+)'
            match = re.match(pattern, line.strip())
            
            if match:
                _, _, output_len, ttft_ms, decode_ms, _ = match.groups()
                output_len = int(output_len)
                ttft_s = float(ttft_ms) / 1000
                decode_s = float(decode_ms) / 1000
                total_s = ttft_s + decode_s
                per_token_s = decode_s / output_len if output_len > 0 else 0
                results.append((total_s, ttft_s, per_token_s))
    return results

def process_csv_file(input_file):
    """Process CSV format files (A6000 and L40S)"""
    results = []
    with open(input_file, 'r') as f_in:
        csv_reader = csv.DictReader(f_in)
        for row in csv_reader:
            output_len = int(row['output_len_req'])
            ttft_s = float(row['ttft_ms']) / 1000
            decode_s = float(row['decode_ms']) / 1000
            total_s = ttft_s + decode_s
            per_token_s = decode_s / output_len if output_len > 0 else 0
            results.append((total_s, ttft_s, per_token_s))
    return results

def save_results(results, output_file):
    """Save results to a single file with total,ttft,per_token format"""
    with open(output_file, 'w') as f:
        for total_s, ttft_s, per_token_s in results:
            f.write(f"{total_s},{ttft_s},{per_token_s}\n")

# Process A100 screen output
results_a100 = process_screen_file('./results_A100_new_screen.txt')
save_results(results_a100, './results_A100_new.txt')

# Process A6000 CSV
results_a6000 = process_csv_file('./results_A6000_new.csv')
save_results(results_a6000, './results_A6000_new.txt')

# Process L40S CSV
results_l40s = process_csv_file('./results_L40S_new.csv')
save_results(results_l40s, './results_L40S_new.txt')

print("Processing complete.")
