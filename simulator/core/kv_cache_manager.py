import math
from collections import deque  # 导入 deque (双端队列) 用于高效的空闲块池
from typing import TYPE_CHECKING, Dict, List  # 导入 Dict 和 deque

from humanize import naturalsize
from transformers import AutoConfig

import simulator.internal.configs.llama as llama_config

if TYPE_CHECKING:
    # 我们可以继续使用 GenerationRequest，因为 FullRequest 是它的子类
    from simulator.core.request import GenerationRequest


class BlockManager:

    
    def __init__(
        self,
        model_params: AutoConfig,
        hardware_params: dict,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = 16,
        gpu_utilization: float = 0.9,
        parallel_config=None,
        block_size: int = 16,
    ):
        self.model_params = model_params
        self.parallel_config = parallel_config
        self.hardware_params = hardware_params
        self.gpu_utilization = gpu_utilization
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.block_size = block_size
        
        # --- 核心改动 ---

        # 计算最大块数
        self._max_num_blocks = self.get_max_num_blocks()

        # 2. (删除) 移除旧的记账变量
        # self._allocated_blocks = 0  (已删除)
        # self._allocation_map = {} (已删除)
        # 备注: 已分配块数 = _max_num_blocks - len(self.free_blocks)

        # 3. 空闲物理块池 (Free Block Pool)
        # 我们使用一个队列来存储所有空闲的物理块的 *编号* (ID)。
        # 编号从 0 到 _max_num_blocks - 1。
        self.free_blocks: deque[int] = deque(range(self._max_num_blocks))
        
        # 4. 引用计数器 (Reference Counter)
        # 这是一个字典，用于跟踪 *每一个* 正在被使用的物理块的引用计数。
        # Key: 物理块ID (int)
        # Value: 引用次数 (int)
        self.ref_counts: Dict[int, int] = {}

        # --- 启动信息 ---
        weights_mem = self.get_weights_memory()
        kv_cache_mem = (
            self.hardware_params["vmemory"] * self.gpu_utilization - weights_mem
        )
        print(f"--- [BlockManager] 初始化 ---")
        print(f"    模型权重显存 (Weights): {naturalsize(weights_mem)}")
        print(f"    可用KV Cache显存 (KV Cache): {naturalsize(kv_cache_mem)}")
        print(f"    每个Block大小 (Block Size): {self.block_size} tokens")
        print(f"    总物理块数 (Total Blocks): {self._max_num_blocks}")
        print(f"---------------------------------")


    def get_max_num_blocks(self):
        total_memory = self.hardware_params["vmemory"] * self.gpu_utilization
        w_memory = self.get_weights_memory()
        
        block_memory_size = (
            2
            * self.block_size
            * llama_config.get_num_key_value_heads(self.model_params)
            * llama_config.get_head_dim(self.model_params)
            * self.kv_bit
            / 8
        )
        total_block_memory_size = (
            block_memory_size * llama_config.get_num_hidden_layers(self.model_params)
        )
        
        if total_block_memory_size == 0:
            return 0
            
        available_kv_memory = total_memory - w_memory
        if available_kv_memory < 0:
            return 0
            
        return math.floor(available_kv_memory / total_block_memory_size)

    def get_weights_memory(self):
        mlp_weights = (
            3
            * llama_config.get_hidden_size(self.model_params)
            * llama_config.get_intermediate_size(self.model_params)
            * self.w_bit
            / 8
        )
        q_weights = (
            llama_config.get_hidden_size(self.model_params)
            * llama_config.get_num_attention_heads(self.model_params)
            * llama_config.get_head_dim(self.model_params)
        )
        kv_weights = (
            2
            * llama_config.get_hidden_size(self.model_params)
            * llama_config.get_head_dim(self.model_params)
            * llama_config.get_num_key_value_heads(self.model_params)
        )
        o_weights = (
            llama_config.get_hidden_size(self.model_params)
            * llama_config.get_num_attention_heads(self.model_params)
            * llama_config.get_head_dim(self.model_params)
        )
        self_attn_weights = (q_weights + kv_weights + o_weights) * self.w_bit / 8
        lm_head_weights = (
            llama_config.get_hidden_size(self.model_params)
            * llama_config.get_vocab_size(self.model_params)
            * self.w_bit
            / 8
        )
        embedding_weights = (
            llama_config.get_hidden_size(self.model_params)
            * llama_config.get_vocab_size(self.model_params)
            * self.w_bit
            / 8
        )
        return (
            (mlp_weights + self_attn_weights)
            * llama_config.get_num_hidden_layers(self.model_params)
            + lm_head_weights
            + embedding_weights
        )

    def print_status(self):
        allocated_blocks = self._max_num_blocks - len(self.free_blocks)
        print(
            f"Allocated blocks/Total blocks: {allocated_blocks}/{self._max_num_blocks} "
            f"(Free: {len(self.free_blocks)})"
        )
        
    def get_num_free_blocks(self) -> int:
        """获取当前空闲的物理块数量"""
        return len(self.free_blocks)
    
    def allocate_block(self) -> int:
        """
        分配一个物理块。
        从空闲池中取出一个块, 设置其引用计数为1, 并返回其ID
        由 调度器(Scheduler) 在需要新块时调用
        Raises:
            Exception: 如果没有空闲块 (OOM)
        """
        if not self.free_blocks:
            raise Exception("Out of memory: No free KV cache blocks.")
            
        # 1. 从空闲池中弹出一个物理块ID
        block_id = self.free_blocks.popleft()
        
        # 2. 确认该块不在引用计数中 (它应该是完全自由的)
        assert block_id not in self.ref_counts
        
        # 3. 设置其引用计数为 1 (表示它现在被1个请求占用)
        self.ref_counts[block_id] = 1
        
        return block_id

    def free_block(self, block_id: int):
        """
        (新增) 释放一个物理块。
        这会使该块的引用计数减1。
        如果引用计数降至0, 该块将被回收, 放回空闲池。
        
        由 FullRequest.release_all_blocks() 在请求完成或中止时调用。
        """
        if block_id not in self.ref_counts:
            # 这种情况在逻辑正确时本不应发生
            print(f"Warning: Attempting to free a block ({block_id}) that has no references.")
            return

        # 1. 引用计数减 1
        self.ref_counts[block_id] -= 1
        
        # 2. (关键) 检查引用计数是否归零
        if self.ref_counts[block_id] == 0:
            # 2a. 如果归零，从引用计数器中移除
            del self.ref_counts[block_id]
            # 2b. 将块ID归还到空闲池的末尾
            self.free_blocks.append(block_id)

    def increment_ref_count(self, block_id: int):
        """
        (新增) 增加一个块的引用计数。
        当一个新请求"Fork" (复用/共享) 一个已存在的物理块时调用。
        
        由 调度器(Scheduler) 在执行前缀复用时调用。
        """
        if block_id not in self.ref_counts:
            # 这种情况也不应发生 (不能复用一个未被占用的块)
            print(f"Warning: Incrementing ref count for a block ({block_id}) that is not allocated.")
            # 强行将其设置为1 (假设父请求刚释放，子请求就复用，存在竞态)
            # 在模拟器中，我们假设它至少为1
            self.ref_counts[block_id] = 1
        else:
             self.ref_counts[block_id] += 1
    
    
    # --- (删除) 旧的、与Request绑定的 API ---
    
    # def can_allocate_request(self, request: "GenerationRequest"):
    # (已删除)
    # 备注: 此逻辑现在转移到调度器(Scheduler)中。
    # 调度器会计算需要多少 *新* 块 (num_new_blocks)，
    # 然后简单地检查: block_manager.get_num_free_blocks() >= num_new_blocks

    # def allocate(self, request: "GenerationRequest"):
    # (已删除)
    # 备注: 调度器现在会调用:
    # block_id = block_manager.allocate_block()
    # request.append_block(block_id)

    # def free(self, request_ids: List[str]):
    # (已删除)
    # 备注: 调度器现在会调用:
    # request.release_all_blocks(block_manager)
    # (这个方法会循环调用 block_manager.free_block(block_id))