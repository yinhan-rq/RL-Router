class RWTEstimator:

    def __init__(self):
        self.workload_tokens = 200
        self.token_throughput = {"unsloth/Llama-3.2-1B-Instruct": 10000, 
                                "meta-llama/Llama-3.1-70B-Instruct": 300,
                                "meta-llama/Llama-3.1-8B-Instruct": 700}


    def get_waiting_time(self, group):
        num_requests = len(group.requests)
        est_workload_tokens = self.workload_tokens
        est_token_throughput = self.token_throughput[group.model]

        return num_requests * est_workload_tokens / est_token_throughput