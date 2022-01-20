import numpy as np
import itertools
import os


class GridworldEnv:
    def __init__(self,seed,map_name,episode_limit=30,input_rows=9, input_cols=12,penalty=True,penalty_amount=1,
                 noise=False, noise_num=1, path=None, stochastic=0., noisy_reward=0.):
        n_agents = 2
        self.noise = noise
        self.noise_num = noise_num
        self.rows, self.cols = input_rows, input_cols
        self.obs_shape = (self.rows + self.cols) * 2 + int(self.noise) * self.noise_num
        self.state_shape = self.obs_shape * 2
        self._episode_steps = 0
        self.episode_limit = episode_limit
        self.n_agents = n_agents
        self.n_actions = 5
        self.map_name=map_name   ##full
        self.heat_map = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        self.center = self.cols // 2
        ###larger gridworld
        self.visible_row=[i for i in range(self.rows//2-2,self.rows//2+3)]
        self.visible_col=[i for i in range(self.cols//2-3,self.cols//2+3)]
        self.vision_index = [[i, j] for i, j in list(itertools.product(self.visible_row, self.visible_col))]
        #self.vision_index = [[i, j] for i, j in list(itertools.product([2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]))]
        # [0, 1, 2, 3], [上，下，左，右]
        self.action_space = [0, 1, 2, 3, 4]
        self.state = None
        self.obs = None
        self.array = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        self.index = None
        self.penalty=penalty
        self.penalty_amount=penalty_amount
        self.path=path
        self.num=0

        self.stochastic = stochastic

        self.noisy_reward = noisy_reward
        self.noisy_reward_row = [i for i in range(0, self.rows // 2 - 3)]
        self.noisy_reward_index = [[i, j] for i, j in list(itertools.product(self.noisy_reward_row, self.visible_col))]



    def get_env_info(self):
        return {'state_shape': self.state_shape,
                'obs_shape': self.obs_shape,
                'episode_limit': self.episode_limit,
                'n_agents': self.n_agents,
                'n_actions': self.n_actions,
                'unit_dim': 0
                }


    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        np.save(self.path + '/heat_map_{}'.format(self.num), self.heat_map)
        # print(self.heat_map)
    def reset(self):
        self.num+=1
        self.index = [[0, 0], [self.rows - 1, self.cols - 1]]


        self._update_obs()
        self._episode_steps=0

    def _update_obs(self):
        self.array[self.index[0][0]][self.index[0][1]] += 1
        self.array[self.index[1][0]][self.index[1][1]] += 1

        obs_1 = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]
        # obs_2 = obs_1.copy()
        import copy
        obs_2 = copy.deepcopy(obs_1)

        obs_1[0][self.index[0][0]] = 1
        obs_1[1][self.index[0][1]] = 1
        obs_1 = obs_1[0] + obs_1[1]

        obs_2[0][self.index[1][0]] = 1
        obs_2[1][self.index[1][1]] = 1
        obs_2 = obs_2[0] + obs_2[1]
        if self.map_name=="origin":
            if self.index[0] in self.vision_index and self.index[1] in self.vision_index:
                temp = obs_1.copy()
                obs_1 += obs_2.copy()
                obs_2 += temp.copy()
            elif self.index[0] in self.vision_index:
                obs_1 += obs_2.copy()
                obs_2 += [0 for _ in range(self.rows + self.cols)]
            elif self.index[1] in self.vision_index:
                obs_2 += obs_1.copy()
                obs_1 += [0 for _ in range(self.rows + self.cols)]
            else:
                obs_2 += [0 for _ in range(self.rows + self.cols)]
                obs_1 += [0 for _ in range(self.rows + self.cols)]
        elif self.map_name=="full_observation":
            temp = obs_1.copy()
            obs_1 += obs_2.copy()
            obs_2 += temp.copy()
        elif self.map_name=="pomdp":
            if self.index[0] in self.vision_index and self.index[1] in self.vision_index:
                temp = obs_1.copy()
                obs_1 += obs_2.copy()
                obs_2 += temp.copy()
            else:
                obs_2 += [0 for _ in range(self.rows + self.cols)]
                obs_1 += [0 for _ in range(self.rows + self.cols)]
        elif self.map_name == "reversed":
            # the second branch and the third branch are reversed.
            if self.index[0] in self.vision_index and self.index[1] in self.vision_index:
                temp = obs_1.copy()
                obs_1 += obs_2.copy()
                obs_2 += temp.copy()
            elif self.index[0] in self.vision_index:
                obs_2 += obs_1.copy()
                obs_1 += [0 for _ in range(self.rows + self.cols)]
            elif self.index[1] in self.vision_index:
                obs_1 += obs_2.copy()
                obs_2 += [0 for _ in range(self.rows + self.cols)]
            else:
                obs_2 += [0 for _ in range(self.rows + self.cols)]
                obs_1 += [0 for _ in range(self.rows + self.cols)]

        #### add noise to state
        if self.noise:
            obs_1 += [np.random.normal() for i in range(self.noise_num)]
            obs_2 += [np.random.normal() for i in range(self.noise_num)]


        self.state = obs_1 + obs_2
        self.obs = [np.array(obs_1), np.array(obs_2)]
        # print(self.index)

    def get_state(self):
        return np.array(self.state)

    def get_obs(self):
        return self.obs


    def get_avail_actions(self):
        current_obs = self.index[0]
        avail_actions=np.ones((2,5))
        if current_obs[0] == 0:
            avail_actions[0,0] = 0
        if current_obs[0] == self.rows - 1:
            avail_actions[0,1] = 0
        if current_obs[1] == 0:
            avail_actions[0,2] = 0
        if current_obs[1] == self.center - 1:
            avail_actions[0,3] = 0
        current_obs = self.index[1]
        if current_obs[0] == 0:
            avail_actions[1,0] = 0
        if current_obs[0] == self.rows - 1:
            avail_actions[1,1] = 0
        if current_obs[1] == self.cols - 1:
            avail_actions[1,3] = 0
        if current_obs[1] == self.center:
            avail_actions[1,2] = 0
        return avail_actions.tolist()

    def step(self, actions):
        # print('State is {}, action is {}'.format(self.state, actions))
        avail_actions = self.get_avail_actions()
        for idx in range(self.n_agents):
            action = actions[idx]
            if np.random.rand() < self.stochastic:
                sum = np.sum(avail_actions[idx])
                sampled_action = np.random.randint(sum)
                for i in range(4):
                    if avail_actions[idx][i] == 1:
                        if sampled_action == 0:
                            action = i
                            break
                        else:
                            sampled_action -= 1

            if action == 0:
                self.index[idx][0] -= 1
            elif action == 1:
                self.index[idx][0] += 1
            elif action == 2:
                self.index[idx][1] -= 1
            elif action == 3:
                self.index[idx][1] += 1

        # for i in range(self.rows):
        #     print(self.array[i])
        # print('*' * 100)
        self._update_obs()
        self._episode_steps +=1

        self.heat_map[self.index[0][0]][self.index[0][1]] += 1
        self.heat_map[self.index[1][0]][self.index[1][1]] += 1



        # print('Next state is {}'.format(self.state))
        if self.penalty:
            if self.index[0] == [self.rows // 2, self.center - 1] and self.index[1] != [self.rows // 2, self.center]:
                reward = -self.penalty_amount
                Terminated = False
                env_info = {'battle_won': False}
            elif self.index[0] != [self.rows // 2, self.center - 1] and self.index[1] == [self.rows // 2, self.center]:
                reward = -self.penalty_amount
                Terminated = False
                env_info = {'battle_won': False}
            elif self.index[0] == [self.rows // 2, self.center - 1] and self.index[1] == [self.rows // 2, self.center]:
                reward= 10
                Terminated=True
                env_info={'battle_won': True}
            else:
                reward = 0
                Terminated = False
                env_info ={'battle_won': False}
        else:
            if self.index[0] == [self.rows // 2, self.center - 1] and self.index[1] == [self.rows // 2, self.center]:
                reward= 10
                Terminated=True
                env_info={'battle_won': True}
            else:
                reward = 0
                Terminated = False
                env_info ={'battle_won': False}

        if self.index[0] in self.noisy_reward_index or self.index[1] in self.noisy_reward_index:
            reward += np.random.randn() * self.noisy_reward

        if self._episode_steps >= self.episode_limit:
            Terminated= True
        if Terminated and self.path is not None:
            if self.num>1 and self.num % 500 == 0:
                self.save()
                print("save heat map")
                self.heat_map = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        return reward,Terminated,env_info

    def close(self):
        """Close StarCraft II."""
        pass



