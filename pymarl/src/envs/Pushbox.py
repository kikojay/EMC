import numpy as np
import itertools
import os


class PushBox:
    def __init__(self,seed,map_name,input_rows=9, input_cols=12, x_dim=3,y_dim=3, path=None,episode_limit=30):
        self.n_agents = 2
        self.n_fruit=1
        self.x_dim,self.y_dim=x_dim,y_dim
        self.rows, self.cols = input_rows, input_cols
        self.obs_shape = (self.rows + self.cols+self.x_dim+self.y_dim)
        self.state_shape = (self.rows + self.cols)*(self.n_agents+self.n_fruit)
        self._episode_steps = 0
        self.episode_limit = episode_limit

        self.n_actions = 5
        self.map_name=map_name   ##full
        self.heat_map = [[0 for _ in range(self.cols)] for _ in range(self.rows)]


        ###larger gridworld
        #self.visible_row=[i for i in range(self.rows//2-2,self.rows//2+3)]
        #self.visible_col=[i for i in range(self.cols//2-3,self.cols//2+3)]

        #self.vision_index = [[i, j] for i, j in list(itertools.product([2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]))]
        # [0, 1, 2, 3,4,5,6,7,8], [up，down，left，right,stay,push left,push right,push up,push down]
        self.action_space = [0, 1, 2, 3, 4,]
        self.state = None
        self.obs = None
        self.array = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.fruit = [self.rows//2,self.cols // 2]
        self.fruit_onehot=[[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]
        self.fruit_onehot[0][self.fruit[0]]=2
        self.fruit_onehot[1][self.fruit[1]] = 2

        self.fruit_catch_place=[[self.fruit[0]-1,self.fruit[1]],[self.fruit[0]+1,self.fruit[1]],
                                [self.fruit[0],self.fruit[1]-1],[self.fruit[0],self.fruit[1]+1]]

        self.index = None

        self.path=path
        self.num=0



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
        self.fruit = [self.rows // 2, self.cols // 2]

        self.fruit_onehot = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]
        self.fruit_onehot[0][self.fruit[0]] = 2
        self.fruit_onehot[1][self.fruit[1]] = 2
        self.fruit_catch_place = [[self.fruit[0] - 1, self.fruit[1]], [self.fruit[0] + 1, self.fruit[1]],
                                  [self.fruit[0], self.fruit[1] - 1], [self.fruit[0], self.fruit[1] + 1]]




        self._update_obs()
        self._episode_steps=0

    def _update_obs(self):
        self.array[self.index[0][0]][self.index[0][1]] += 1
        self.array[self.index[1][0]][self.index[1][1]] += 1
        self.fruit_onehot = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]
        self.fruit_onehot[0][self.fruit[0]] = 2
        self.fruit_onehot[1][self.fruit[1]] = 2

        self.fruit_catch_place = [[self.fruit[0] - 1, self.fruit[1]], [self.fruit[0] + 1, self.fruit[1]],
                                  [self.fruit[0], self.fruit[1] - 1], [self.fruit[0], self.fruit[1] + 1]]



        obs_1 = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)],[0 for _ in range(self.x_dim)],[0 for _ in range(self.y_dim)]]
        # obs_2 = obs_1.copy()
        import copy
        obs_2 = copy.deepcopy(obs_1)

        obs_1[0][self.index[0][0]] = 1
        obs_1[1][self.index[0][1]] = 1


        obs_2[0][self.index[1][0]] = 1
        obs_2[1][self.index[1][1]] = 1

        self.agent1_vision_x=[self.index[0][0]-1,self.index[0][0],self.index[0][0]+1]
        self.agent1_vision_y = [self.index[0][1] - 1, self.index[0][1], self.index[0][1] + 1]

        self.agent2_vision_x = [self.index[1][0] - 1, self.index[1][0], self.index[1][0] + 1]
        self.agent2_vision_y = [self.index[1][1] - 1, self.index[1][1], self.index[1][1] + 1]

        self.agent1_vision_index =[[i, j] for i, j in list(itertools.product(self.agent1_vision_x, self.agent1_vision_y))]
        self.agent2_vision_index = [[i, j] for i, j in
                                    list(itertools.product(self.agent2_vision_x, self.agent2_vision_y))]


        if self.fruit in self.agent1_vision_index:
            obs_1[2][self.fruit[0]-self.index[0][0]+1]=2
            obs_1[3][self.fruit[1] - self.index[0][1]+1]=2
        if self.index[1] in self.agent1_vision_index:
            obs_1[2][self.index[1][0] - self.index[0][0] + 1] = 1
            obs_1[3][self.index[1][1] - self.index[0][1] + 1] = 1
        if self.fruit in self.agent2_vision_index:
            obs_2[2][self.fruit[0]-self.index[1][0]+1]=2
            obs_2[3][self.fruit[1] - self.index[1][1]+1]=2
        if self.index[0] in self.agent2_vision_index:
            obs_2[2][self.index[0][0] - self.index[1][0] + 1] = 1
            obs_2[3][self.index[0][1] - self.index[1][1] + 1] = 1


        agent1_obs=obs_1[0]+obs_1[1]+obs_1[2]+obs_1[3]
        agent2_obs = obs_2[0] + obs_2[1] + obs_2[2] + obs_2[3]

        self.state = obs_1[0]+obs_1[1]+obs_2[0]+obs_2[1]+self.fruit_onehot[0]+self.fruit_onehot[1]
        self.obs = [np.array(agent1_obs), np.array(agent2_obs)]
        # print(self.index)





    def get_state(self):
        return np.array(self.state)

    def get_obs(self):
        return self.obs


    def get_avail_actions(self):
        current_obs = self.index[0]
        avail_actions=np.ones((2,5))

        if current_obs[0] == 0 :
            avail_actions[0,0] = 0
        if current_obs[0] == self.rows - 1 :
            avail_actions[0,1] = 0
        if current_obs[1] == 0 :
            avail_actions[0,2] = 0
        if current_obs[1] == self.cols-1 :
            avail_actions[0, 3] = 0



        current_obs = self.index[1]
        if current_obs[0] == 0 :
            avail_actions[1,0] = 0
        if current_obs[0] == self.rows - 1 :
            avail_actions[1,1] = 0
        if current_obs[1] == 0 :
            avail_actions[1, 2] = 0
        if current_obs[1] == self.cols - 1 :
            avail_actions[1,3] = 0




        return avail_actions.tolist()



    def step(self, actions):
        # print('State is {}, action is {}'.format(self.state, actions))
        reward=0
        self.index_old=self.index
        move=False


        if self.index[0]==self.index[1] and actions[0]==actions[1]:
            if self.index[0]==[self.fruit[0]+1,self.fruit[1]] and actions[0]==0 and self.fruit[0]>0:
                self.index[0][0] -= 1
                self.index[1][0] -= 1
                self.fruit[0] -= 1
                move = True
            elif self.index[0]==[self.fruit[0]-1,self.fruit[1]] and actions[0]==1 and self.fruit[0]<self.rows-1:
                self.index[0][0] += 1
                self.index[1][0] += 1
                self.fruit[0] += 1
                move = True
            elif self.index[0] == [self.fruit[0] , self.fruit[1]+1] and actions[0] == 2 and self.fruit[1] >0:
                self.index[0][1] -= 1
                self.index[1][1] -= 1
                self.fruit[1] -= 1
                move = True
            elif self.index[0] == [self.fruit[0], self.fruit[1] - 1] and actions[0] == 3 and self.fruit[1] < self.cols-1:
                self.index[0][1] += 1
                self.index[1][1] += 1
                self.fruit[1] += 1
                move = True

        if not move:
            for idx in range(self.n_agents):
                if actions[idx] == 0 :
                    self.index[idx][0] -= 1
                elif actions[idx] == 1:
                    self.index[idx][0] += 1
                elif actions[idx] == 2:
                    self.index[idx][1] -= 1
                elif actions[idx] == 3:
                    self.index[idx][1] += 1
                if self.index[idx]==self.fruit:
                    self.index[idx]=self.index_old[idx]








        if self.fruit[1]==self.cols-1:
            reward=10
            Terminated=True
            env_info = {'battle_won': True}
        else:
            reward=0
            Terminated = False
            env_info = {'battle_won': False}

        # for i in range(self.rows):
        #     print(self.array[i])
        # print('*' * 100)
        self._update_obs()
        self._episode_steps +=1

        self.heat_map[self.index[0][0]][self.index[0][1]] += 1
        self.heat_map[self.index[1][0]][self.index[1][1]] += 1




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



