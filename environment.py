from multiprocessing import process
import gymnasium 
import numpy as np
from gymnasium import spaces

device_list={
       "cpu":{"speed":1.0},
       "gpu":{"speed":3.0},
       "npu":{"speed":5.0}
       }

class taskscheduler(gymnasium.Env):
    def __init__(self, task=10, cpu=4, gpu=2, npu=1):
        super().__init__()
        self.task=task
        self.cpu_num=cpu
        self.gpu_num=gpu
        self.npu_num=npu
        self.core_nums=cpu+gpu+npu
        self.observation_space=spaces.Box(
            low=0.1, high=np.inf,
            shape=(self.core_nums+2*self.task+self.task, ),
            dtype=np.float64
            )
        self.action_space=spaces.MultiDiscrete([self.task, self.core_nums])
        self.reset()
        self.core_cooldown = [0] * self.core_nums  # 新增！

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cpu_load=[0.0]*self.cpu_num
        self.gpu_load=[0.0]*self.gpu_num
        self.npu_load=[0.0]*self.npu_num
        self.task_load=np.random.rand(self.task)
        self.task_priority=np.random.rand(self.task)
        self.core_cooldown = [0] * self.core_nums  # 重置！
        return self.get_system(),{}
        

        
    def get_system(self):
        core_load = np.concatenate([self.cpu_load, self.gpu_load, self.npu_load])
        estimated_steps = np.zeros(self.task)
        for i in range(self.task):
            if self.task_load[i] > 0:
                estimated_steps[i] = max(1.0, self.task_load[i] / (0.1 * 5.0))
            else:
                estimated_steps[i] = 0.0
        return np.concatenate([core_load, self.task_load, self.task_priority, estimated_steps.astype(np.float64)])

        
    def step(self, action):
        task_index, core_index = action
        reward = 0.0
        done = False
    
        task_load = self.task_load[task_index]
        if task_load <= 0.01:
            return self.get_system(), 0.0, False, False, {}
    
        # === 1. 選擇核心 ===
        if core_index < self.cpu_num:
            speed = 1.0
            load_list = self.cpu_load
            load_idx = core_index
            core_type = "CPU"
        elif core_index < self.cpu_num + self.gpu_num:
            speed = 3.0
            load_list = self.gpu_load
            load_idx = core_index - self.cpu_num
            core_type = "GPU"
        else:
            speed = 5.0
            load_list = self.npu_load
            load_idx = core_index - self.cpu_num - self.gpu_num
            core_type = "NPU"
    
        current_load = load_list[load_idx]
        # 冷卻機制
        if self.core_cooldown[core_index] > 0:
            reward -= 7.0#冷卻的懲罰可以調小，如果策略崩潰的話
        self.core_cooldown[core_index] = max(self.core_cooldown[core_index] - 1, 0)
        if self.core_cooldown[core_index] == 0:
            self.core_cooldown[core_index] = 2  # 冷卻 3 步
    
        # === 2. 懲罰 1：執行時間（越慢越痛）===
        time_cost = task_load / speed
        reward -= time_cost * 1.0  # 基礎時間懲罰
    
        # === 3. 懲罰 2：資源稀有度（NPU 更貴！）===
        rarity_penalty = {"CPU": 1.0, "GPU": 2.0, "NPU": 5.0}
        reward -= rarity_penalty[core_type] * 0.5
    
        # === 4. 懲罰 3：負載不均（std 越大越糟）===
        all_load = np.concatenate([self.cpu_load, self.gpu_load, self.npu_load])
        load_std = np.std(all_load)
        reward -= load_std * 3.0
    
        # === 5. 懲罰 4：高負載使用（>0.7 開始痛）===
        if current_load > 0.7:
            reward -= (current_load - 0.7) * 10.0
    
        # === 6. 懲罰 5：NPU 壟斷（連續選 NPU 懲罰累加）===
        if core_type == "NPU" and hasattr(self, "last_npu_count"):
            self.npu_usage_count += 1
            if self.npu_usage_count > 3:  # 連續 3 次用 NPU
                reward -= (self.npu_usage_count - 3) * 5.0#改過，原本2
        else:
            self.npu_usage_count = 1 if core_type == "NPU" else 0
    
        # === 7. 更新負載 ===
        load_increase = 0.1 * task_load / speed
        new_load = min(1.0, current_load + load_increase)
        load_list[load_idx] = new_load
    
        # === 8. 任務衰減 ===
        decay = 0.8 ** (1.0 / speed)
        self.task_load[task_index] *= decay
    
        # === 9. 自然衰減 ===
        self.cpu_load = [max(0.1, l * 0.95) for l in self.cpu_load]
        self.gpu_load = [max(0.1, l * 0.95) for l in self.gpu_load]
        self.npu_load = [max(0.1, l * 0.95) for l in self.npu_load]
    
        # === 10. 失敗條件 ===
        if new_load >= 1.0:
            done = True
            reward -= 20.0
    
        if np.all(self.task_load < 0.01):
            done = True
            # 純懲罰：不給獎勵
    
        return self.get_system(), reward, done, False, {}

