export PYTHONPATH=$PYTHONPATH:/home/yinhan/Hexgen-Flow
export HF_ENDPOINT=https://hf-mirror.com

python simulator/cli/start_simulator.py \
    --input ./simulator/cli/input_file_trace3.json \
    --n-engines 4 \
    --arrival-rate 1.0 \
    --trace-output ./simulator/result/trace_output.json \
    --stats-output ./simulator/result/stats_output.json