import os
import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from main import taskscheduler
import matplotlib
matplotlib.use("TKAgg")#解決會訓練到一半堵塞的問題Agg
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter

DEFAULT_CHECKPOINT = "/Users/nick/Desktop/operating system/log/ppo_checkpoint_update_100.pth"
parser = argparse.ArgumentParser(description="PPO with TensorBoard (No Best Model Save)")
parser.add_argument('--load_checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                    help='Path to checkpoint to resume')
parser.add_argument('--total_updates', type=int, default=200,
                    help='Total training updates')
args = parser.parse_args()

LOGDIR = './log' #用相對路徑避免權限問題
os.makedirs(LOGDIR, exist_ok=True)

writer = SummaryWriter(log_dir=LOGDIR)
print(f"TensorBoard logdir: {LOGDIR}")
print(f"啟動指令：tensorboard --logdir='{LOGDIR}' --port=6006")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.actor = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

def compute_gae(rewards, dones, values, next_value, gamma=0.95, lam=0.95):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    next_value = torch.as_tensor(next_value, dtype=torch.float32)

    values = torch.cat([values, next_value.reshape(1)])
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    advantages = torch.zeros_like(deltas)
    gae = 0.0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return returns, advantages

env = taskscheduler(task=10, cpu=4, gpu=2, npu=1)
obs_dim = env.observation_space.shape[0]
nvec = env.action_space.nvec
act_dim = int(np.prod(nvec))

model = ActorCritic(obs_dim, act_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-5)   # 建議先降低學習率

GAMMA = 0.95
EPS_CLIP = 0.1#改過  0.15         # 加大 clip 範圍，提升穩定性
EPOCHS = 20
STEPS_PER_UPDATE = 2048
BATCH_SIZE = 32
CHECKPOINT_FREQ = 5
ent_coef=0.05#改過原本＝0.02

all_rewards = []
all_losses = []
all_entropy = []
raw_rewards = []
episode_rewards=[]
start_update = 0
obs, _ = env.reset()

if args.load_checkpoint and os.path.exists(args.load_checkpoint):
    cp_path = args.load_checkpoint
    if not os.path.exists(cp_path):
        raise FileNotFoundError(f"找不到檢查點: {cp_path}")
    print(f"正在載入檢查點: {cp_path}")
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    optimizer.load_state_dict(cp['optimizer_state_dict'])
    start_update = cp['update']
    all_rewards = cp.get('all_rewards', [])
    all_losses = cp.get('all_losses', [])
    all_entropy = cp.get('all_entropy', [])
    episode_rewards = cp.get('episode_rewards', [])  # 恢復！
    obs = cp.get('current_obs', None)
    if obs is None:
        obs, _ = env.reset()
    else:
        print(f"恢復觀測狀態，shape: {obs.shape}")
        print(f"從 update {start_update} 繼續，已載入 {len(all_rewards)} 筆歷史，{len(episode_rewards)} 個 episode")

else:
    print("從頭開始訓練")

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

#obs, _ = env.reset()
#episode_rewards = []        # 每個 episode 的「總懲罰」
#current_reward = 0.0
#steps_in_episode = 0

for update in range(start_update, args.total_updates):
    # --- 收集資料 ---
    states, actions, rewards_list, dones_list = [], [], [], []
    episode_rewards = [] 
    log_probs, values = [], []
    current_reward = 0

    for step in range(STEPS_PER_UPDATE):
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        logits, value = model(obs_tensor.unsqueeze(0))
        logits = logits.view(nvec[0], nvec[1])

        dists = [Categorical(logits=logits[:, i]) for i in range(2)]
        action = torch.stack([d.sample() for d in dists])
        action = torch.min(action, torch.tensor(nvec) - 1)

        log_prob = sum(d.log_prob(a) for d, a in zip(dists, action))
        next_obs, reward, done, truncated, _ = env.step(action.tolist())
        raw_rewards.append(reward)
        
        reward = np.clip(reward, -10.0, 10.0)   # 強制 clip reward，我覺得要留

        states.append(obs_tensor)
        actions.append(action)
        rewards_list.append(reward)
        dones_list.append(done or truncated)
        log_probs.append(log_prob)
        values.append(value.item())
        current_reward += reward
        #steps_in_episode += 1 #加的
        if done or truncated:
            episode_rewards.append(current_reward)
            current_reward = 0
            obs, _ = env.reset()
        obs = next_obs

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
    dones_tensor = torch.tensor(dones_list, dtype=torch.float32)
    log_probs = torch.stack(log_probs)
    values = torch.tensor(values, dtype=torch.float32)

    with torch.no_grad():
        _, next_value = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        next_value = next_value.item()

    returns, advantages = compute_gae(rewards_tensor, dones_tensor, values, next_value, GAMMA)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    actions_np = actions.cpu().numpy()
    print(f"任務動作分佈: {np.bincount(actions_np[:, 0], minlength=nvec[0])}")
    print(f"核心動作分佈: {np.bincount(actions_np[:, 1], minlength=nvec[1])}")

    # --- PPO 更新 ---
    for _ in range(EPOCHS):
        indices = torch.randperm(STEPS_PER_UPDATE)
        for start in range(0, STEPS_PER_UPDATE, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            b_states = states[idx]
            b_actions = actions[idx]
            b_log_probs = log_probs[idx]
            b_returns = returns[idx]
            b_adv = advantages[idx]

            logits, new_vals = model(b_states)
            logits = logits.view(-1, nvec[0], nvec[1])
            dists = [Categorical(logits=logits[:, :, i]) for i in range(2)]
            new_log_probs = sum(d.log_prob(b_actions[:, i]) for i, d in enumerate(dists))

            ratio = (new_log_probs - b_log_probs.detach()).exp()
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * b_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (b_returns - new_vals.squeeze(-1)).pow(2).mean()
            entropy_bonus = sum(d.entropy().mean() for d in dists)

            loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy_bonus   # 加強探索

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    # --- 記錄指標 ---
    avg_penalty = np.mean(episode_rewards[-50:]) if episode_rewards else 0
    all_rewards.append(avg_penalty)
    all_losses.append(loss.item())
    all_entropy.append(entropy_bonus.item())
    


    # --- TensorBoard ---
    global_step = update + 1
    writer.add_scalar('Train/Loss', loss.item(), global_step)
    writer.add_scalar('Train/Avg_penalty', avg_penalty, global_step)
    writer.add_scalar('Train/Entropy', entropy_bonus.item(), global_step)
    writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar('Train/Actor_Loss', actor_loss.item(), global_step)
    writer.add_scalar('Train/Critic_Loss', critic_loss.item(), global_step)
    writer.add_histogram('Action/CPU', actions[:, 0].cpu().numpy(), global_step)
    writer.add_histogram('Action/GPU', actions[:, 1].cpu().numpy(), global_step)

    ax1.clear(); ax1.plot(all_rewards, 'b-'); ax1.set_title("penalty"); ax1.legend(['Avg penalty'])
    ax2.clear(); ax2.plot(all_losses, 'orange'); ax2.set_title("Loss"); ax2.legend(['Loss'])
    ax3.clear(); ax3.plot(all_entropy, 'g-'); ax3.set_title("Entropy"); ax3.legend(['Entropy'])
    plt.tight_layout(); plt.draw(); plt.pause(0.001)

    print(f"Update {update+1}/{args.total_updates} | R: {avg_penalty:+.2f} | L: {loss.item():.3f} | Ent: {entropy_bonus.item():.3f}")

    if (update + 1) % CHECKPOINT_FREQ == 0 or (update + 1) == args.total_updates:
        cp_path = os.path.join(LOGDIR, f"ppo_checkpoint_update_{update + 1}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'update': update + 1,
            'all_rewards': all_rewards,
            'all_losses': all_losses,
            'all_entropy': all_entropy,
            'episode_rewards': episode_rewards,  # 新增這行！
            'current_obs': obs,  # 儲存當前 obs
        }, cp_path)   # 第二個參數已補上！
        print(f"Checkpoint saved: {cp_path}")

    # --- 寫入日誌 ---
    with open(os.path.join(LOGDIR, "training_log.txt"), "a") as f:
        f.write(f"{update+1},{loss.item():.6f},{avg_penalty:.6f},{entropy_bonus.item():.6f}\n")

plt.ioff()
final_plot = os.path.join(LOGDIR, "training_final.png")
plt.savefig(final_plot, dpi=300, bbox_inches='tight')
print(f"最終圖表儲存: {final_plot}")
plt.show()

# 在 update 結束後
num_episodes = len(episode_rewards)
avg_penalty = np.mean(episode_rewards) if episode_rewards else 0.0
print(f"Update {update+1} | Episodes: {num_episodes} | Penalty: {avg_penalty:+.2f} | ...")

print("Advantage 統計:", advantages.mean(), advantages.std())
print("Value target 範圍:", values.min(), values.max())
print("Policy loss (total):", loss.item())
print("Actor loss:", actor_loss.item())
print("Critic loss:", critic_loss.item())
print("Entropy bonus:", entropy_bonus.item())
print(f"原始 reward 統計: mean={np.mean(raw_rewards):.6f}, std={np.std(raw_rewards):.6f}")
print(f"最終平均 Penalty: {np.mean(all_rewards):.2f}")

writer.close()
env.close()
print("訓練結束！")
print(f"TensorBoard: tensorboard --logdir='{LOGDIR}'")
