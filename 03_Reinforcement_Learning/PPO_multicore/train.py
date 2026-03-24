import os
import torch
import numpy as np
import gymnasium as gym
from drone_env import DroneEnv
from ppo import PPOAgent

# 환경 생성 도우미 함수 (병렬 처리를 위해 필요함)
def make_env():
    def _thunk():
        return DroneEnv()
    return _thunk

def main():
    num_envs = 32  # 🚀 동시에 띄울 드론의 개수! (4060이라면 8~16개 아주 거뜬합니다)

    # 핵심: AsyncVectorEnv를 사용해 환경 64개를 병렬(백그라운드)로 띄웁니다.
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    max_episodes = 10000
    max_steps = 500
    update_timestep = 4000 # 32대가 뛰니 체감상 순식간에 업데이트 주기가 찹니다!
    
    ppo_agent = PPOAgent(
        state_dim, action_dim,
        lr_actor=0.0001, lr_critic=0.0003,
        gamma=0.99, K_epochs=10, eps_clip=0.2
    )
    
    time_step = 0
    ep_count = 0
    print_freq = 100
    episode_rewards = []
    
    print(f"🚀 PPO 병렬 학습 시작! (드론 {num_envs}대 동시 출격)")
    print(f"GPU 사용 여부: {torch.cuda.is_available()} (True여야 4060이 일하는 중입니다!)")

    # 8대의 드론 상태를 한 번에(배치로) 받아옵니다.
    states, _ = envs.reset()
    current_ep_rewards = np.zeros(num_envs)

    # 에피소드가 아니라 전체 훈련 스텝 단위로 루프를 돕니다.
    total_steps = max_episodes * max_steps // num_envs

    for t in range(1, total_steps + 1):
        time_step += num_envs # 한 번에 8스텝씩 데이터가 쌓임!
        
        # 1. 8대의 드론이 각자의 위치에서 어떤 행동을 할지 한 번에 계산
        actions = ppo_agent.select_action(states)
        
        # 2. 8대의 드론 동시에 1보씩 이동!
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
        
        # 3. 일기장(Buffer)에 8대의 기록을 통째로 압축해서 넘겨줌
        dones = np.logical_or(terminateds, truncateds)
        ppo_agent.buffer.rewards.append(torch.tensor(rewards, dtype=torch.float32))
        ppo_agent.buffer.is_terminals.append(torch.tensor(dones, dtype=torch.float32))
        
        states = next_states
        current_ep_rewards += rewards
        
        # 4. 어떤 드론이 종료(도착 or 추락)되었다면 점수 기록
        for i in range(num_envs):
            if dones[i]:
                ep_count += 1
                episode_rewards.append(current_ep_rewards[i])
                current_ep_rewards[i] = 0
                
                if ep_count % print_freq == 0:
                    avg_reward = np.mean(episode_rewards[-print_freq:])
                    print(f"Episode: {ep_count} \t Average Reward (Last 100): {avg_reward:.2f}")

        # 5. 데이터가 충분히 모였으면 GPU 풀가동 뇌 수술!
        if time_step >= update_timestep:
            ppo_agent.update()
            time_step = 0
            
        if ep_count >= max_episodes:
            break
            
    envs.close() # 훈련 끝나면 환경 닫기
    os.makedirs("./ppo_logs/", exist_ok=True)
    torch.save(ppo_agent.policy_old.state_dict(), 'ppo_drone_model_custom.pth')
    np.savetxt("./ppo_logs/custom_rewards.csv", episode_rewards, delimiter=",")
    print("🎉 병렬 학습이 완료되었습니다! 4060 수고했다!")

if __name__ == '__main__':
    main()