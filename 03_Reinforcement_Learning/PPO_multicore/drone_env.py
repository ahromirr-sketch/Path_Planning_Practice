import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        self.map_size = 600.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 11차원 관측 공간 유지
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.obstacles = [
            {'pos': np.array([50.0, 140.0]), 'radius': 50.0},
            {'pos': np.array([200.0, 230.0]), 'radius': 50.0},
            {'pos': np.array([465.0, 520.0]), 'radius': 50.0},
            {'pos': np.array([175.0, 420.0]), 'radius': 50.0},
            {'pos': np.array([375.0, 320.0]), 'radius': 100.0},
            {'pos': np.array([475.0, 120.0]), 'radius': 50.0},
        ]
        
        self.max_dist = self.map_size * 1.414
        self.max_steps = 500
        self.current_step = 0
        self.prev_min_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.array([20.0, 20.0], dtype=np.float32) # 시작점은 항상 좌하단 (20,20)으로 고정
        self.current_step = 0
        # 타겟 위치 설정: options에 'target_pos'가 있으면 그 위치로, 없으면 랜덤으로 생성
        if options and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'], dtype=np.float32)
        else:
            self.target_pos = self._get_random_target()
            
        self.prev_distance = np.linalg.norm(self.target_pos - self.drone_pos)
        self.prev_action = np.zeros(2, dtype=np.float32)
        
        self.prev_min_dist, _, _ = self._get_min_threat()
        
        return self._get_obs(), {}

    def _get_random_target(self):
        while True:
            target_x = np.random.uniform(20.0, self.map_size - 20.0)
            target_y = np.random.uniform(20.0, self.map_size - 20.0)
            # x와 y를 더한 값이 맵 크기보다 작으면 좌하단 삼각형이므로 가차없이 탈락(continue)시킵니다.
            if target_x + target_y < self.map_size + 10.0: # (+10.0을 더해서 대각선보다 살짝 더 우상단으로 밀어넣음)
                continue 
            
            target = np.array([target_x, target_y], dtype=np.float32)
            
            # 장애물/시작점 겹침 방지
            valid = True
            for obs in self.obstacles:
                if np.linalg.norm(target - obs['pos']) < obs['radius'] + 10.0:
                    valid = False
                    break
            
            # 시작점(20,20)과 거리가 너무 가깝지 않도록 (우상단이라 어차피 멀지만 혹시 몰라 유지)
            if np.linalg.norm(target - self.drone_pos) < 100.0:
                valid = False
                
            if valid:
                return target

    def _get_min_threat(self):
        """💡 NEW: 원형 장애물과 4면의 벽을 모두 포함하여 가장 가까운 위협을 계산합니다."""
        min_dist = float('inf') 
        threat_dx = 0.0
        threat_dy = 0.0

        # 1. 원형 장애물 검사
        for obs in self.obstacles:
            dist_vec = obs['pos'] - self.drone_pos 
            # 장애물 중심까지의 거리에서 반지름을 빼서 실제로 드론이 접근할 수 있는 최소 거리를 계산합니다.
            dist = np.linalg.norm(dist_vec) - obs['radius'] 
            if dist < min_dist:
                min_dist = dist
                norm_dist = np.linalg.norm(dist_vec)
                if norm_dist > 0:
                    threat_dx = dist_vec[0] / norm_dist
                    threat_dy = dist_vec[1] / norm_dist

        # 2. 4면의 벽 검사 (벽으로 향하는 벡터)
        walls = [
            (self.drone_pos[0], -1.0, 0.0),                     # 왼쪽 벽
            (self.map_size - self.drone_pos[0], 1.0, 0.0),      # 오른쪽 벽
            (self.drone_pos[1], 0.0, -1.0),                     # 아래쪽 벽
            (self.map_size - self.drone_pos[1], 0.0, 1.0)       # 위쪽 벽
        ]
        
        for w_dist, w_dx, w_dy in walls:
            if w_dist < min_dist:
                min_dist = w_dist
                threat_dx = w_dx
                threat_dy = w_dy

        return min_dist, threat_dx, threat_dy

    def _get_obs(self):
        min_dist, threat_dx, threat_dy = self._get_min_threat()
        
        tangent_dx = 0.0
        tangent_dy = 0.0
        # 벽이든 장애물이든 60픽셀 이내면 회피 접선 방향을 활성화
        if min_dist < 60.0:
            tangent_dx = -threat_dy
            tangent_dy = threat_dx

        safe_dist = min_dist / self.map_size

        target_vec = self.target_pos - self.drone_pos
        target_dist = np.linalg.norm(target_vec)
        
        target_dx = 0.0
        target_dy = 0.0
        if target_dist > 0:
            target_dx = target_vec[0] / target_dist
            target_dy = target_vec[1] / target_dist
        # 관측값 리스트: [드론 위치(x,y), 이전 행동(2), 가장 가까운 위협의 방향(2), 타겟 방향(2), 안전 거리, 회피 접선 방향(2)]
        obs_list = [
            self.drone_pos[0] / self.map_size,
            self.drone_pos[1] / self.map_size,
            self.prev_action[0],
            self.prev_action[1],
            threat_dx,     # 가장 가까운 위협의 X 방향
            threat_dy,     # 가장 가까운 위협의 Y 방향
            target_dx,  
            target_dy,   
            safe_dist,
            tangent_dx,
            tangent_dy
        ]
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        action_diff = np.linalg.norm(action - self.prev_action)
        smoothness_penalty = action_diff * 0.05  
        
        self.prev_action = action
        move_vector = action * 5.0 # 행동의 크기에 따라 이동 벡터를 조정 (5.0은 최대 이동 거리)
        self.drone_pos += move_vector
        
        terminated = False
        truncated = False
        reward = 0.0
        
        # 장애물 및 벽 최단 거리 계산
        min_dist, _, _ = self._get_min_threat()
        
        # 💡 벽이든 원이든 부딪히면 무조건 사망 (코드 대폭 단순화)
        if min_dist <= 0: 
            reward = -100.0 # 충돌 시 큰 패널티와 함께 에피소드 종료
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        distance_to_target = np.linalg.norm(self.target_pos - self.drone_pos)
        
        if distance_to_target < 5.0:  
            reward = 2000.0 # 목표에 도달하면 큰 보상과 함께 에피소드 종료
            terminated = True
        else:
            reward = (self.prev_distance - distance_to_target) * 5.0 
            
            # 💡 NEW: 스마트 자기장 척력
            # 타겟과의 거리가 60 미만으로 좁혀지면 장애물 척력을 서서히 무시하기 시작합니다!
            if min_dist < 60.0 or self.prev_min_dist < 60.0:
                threat_diff = min_dist - self.prev_min_dist
                
                # 목표와 가까울수록 척력 배율이 0에 가까워짐
                repulsion_scale = min(1.0, distance_to_target / 60.0) 
                reward += threat_diff * 15.0 * repulsion_scale
                
            reward -= 1.0 # 매 스텝마다 작은 패널티로 더 빠른 해결을 유도합니다.
            reward -= smoothness_penalty

        self.prev_distance = distance_to_target
        self.prev_min_dist = min_dist 
        
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}