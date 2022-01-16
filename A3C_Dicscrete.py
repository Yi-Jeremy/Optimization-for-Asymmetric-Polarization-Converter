# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 19:16
# @Author  : Chuqiao Yi
# @File    : Agent.py
# @Software: PyCharm
###### DRL packages##########
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import threading
# from A3C.plotGIF import ImprovedMethod_Improve
import multiprocessing
import numpy as np
from queue import Queue
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.keras.models import load_model
from write_txt import write_txt
import gym
from gym import spaces
import platform

# version: -exp, 无效的reward直接为0，其余的reward使用exp函数进行包装，确保可导可微

# logger folder
def create_log_folder():
    if not os.path.exists('results'):
        os.mkdir('results')
    create_file_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if platform.system() == 'Windows':
        os.mkdir(r'csv-result\%s' % (create_file_time))  # 保存每一次运行的结果的文件夹
    elif platform.system() == 'Linux':
        os.mkdir(r'csv-result/%s' % (create_file_time))
    return create_file_time


#  Arg parse
class config:
    algorithm = 'a3c'  # or 'random
    lr = 0.001
    update_freq = 20  # 20
    max_eps = 2500
    gamma = 0.99  # 0.99
    save_dir = './tmp/'
    f0 = np.linspace(420, 550, 82)

    def __init__(self):
        self.__config__()

    def __config__(self):
        print(f'\nalgorithm={self.algorithm} | lr={self.lr} | update_freq={self.update_freq} | \n'
              f'max_eps={self.max_eps} | gamma={self.gamma} | save_dir={self.save_dir}\n')


# calculate Eff of Nano
def effOfNano(state, model):
    tmp_state = [state[0] / 100, state[1] / 100, state[2] / 100, state[-1]]
    Ey, Ey1 = model.predict(np.array(tmp_state).reshape((-1, 4)))
    Ey, Ey1 = Ey[0], Ey1[0]
    multi_eff = (Ey + Ey1)/4
    argmax_index = np.argmax(multi_eff)  # 最大值的索引
    max_fre = args.f0[argmax_index]  # 频率最大值
    # '''返回值，最大的Tyx，Tyy积，最大值所在的索引，此次仿真的Ey，Ey1，最大积所在对应的频率,所有的频率'''
    return multi_eff[argmax_index], argmax_index, Ey, Ey1, max_fre, args.f0


# Nano Env class
class NanoEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        # self.observation_space_name = np.array(['index', 'hito', 'hmapbi3', 'hpcbm', 'hpedot'])  # 状态空间
        # self.observation_space_name = np.array(['index', 'Hc', 'Ha', 'Ka'])  # 状态空间, 暂时改变index,h1,h2,g2四个变量
        self.observation_space_name = np.array(['Ha', 'Hc', 'Ka', 'index'])  # 状态空间, 暂时改变index,h1,h2,g2四个变量
        self.state_lb = [100, 100, 30, 1.3]
        self.state_ub = [300, 600, 112, 1.5]
        # self.state_lb=[1,100,100,30]
        # self.state_ub=[1.5,600,300,112]
        self.action_in_state = [2,  # + Ha,
                                2,  # + Hc
                                2,  # + Ka
                                0.02,  # + index
                                -1,  # - Ha
                                -1,  # - Hc
                                -1,  # - Ka
                                -0.001  # - index
                                ]

        self.action_space = spaces.Discrete(len(self.action_in_state))  # 动作空间为2倍的状态空间
        self.observation_space = spaces.Box(np.array(self.state_lb), np.array(self.state_ub))
        self.state = None
        # self.fdtd=fdtd #init FDTD simulaiton env
        if platform.system() == 'Windows':
            self.model = load_model(r'2020_12_17_21_38_9_yy_mape_3_312393_yx_mape_4_141227.h5')
        elif platform.system() == 'Linux':
            self.model = load_model(r'model/2020_12_17_21_38_9_yy_mape_3_312393_yx_mape_4_141227.h5')

    def step0(self, action: int):
        # assert self.action_space.contains(action) ,'Invalid action!' #对于超出范围的动作幅度显示无效动作
        """
        action:
        0: index+0.1
        1: index+0.05
        2: index+0.01
        3:
        """


        action_in_state = [5, 2, 1,  # + Ha,
                           5, 2, 1,  # + Hc
                           5, 2, 1,  # + Ka
                           0.05, 0.02, 0.01,  # + index
                           -5, -2, -1,  # - Ha
                           -5, -2, -1,  # - Hc
                           -5, -2, -1,  # - Ka
                           -0.01, -0.005, -0.001  # - index
                           ]

        ac = int(np.floor(action / 3))
        if ac <= 3:
            self.state[ac] += action_in_state[action]
        else:
            self.state[ac - 4] += action_in_state[action]

        if not self.observation_space.contains(self.state):  # 超出范围直接停止
            reward = -0.1 * 5  # 超出范围的reward设置为-0.1
            done = True  # 结束动作, 结束动作之后是否有reset呢
            info = {'max_multi_eff': None,
                    'argmax_index': None,
                    'Ey': None,
                    'Ey1': None,
                    'max_freq': None,
                    'frequency': None,
                    'state': self.state}
        else:
            max_multi_eff, argmax_index, Ey, Ey1, max_fre, frequency = effOfNano(state=self.state, model=self.model)
            if max_multi_eff < 0.25:
                if Ey[argmax_index] < 0.5 and Ey1[argmax_index] < 0.5:  # 两个方向都小于0.5
                    reward = -0.25
                    done = False
                elif Ey[argmax_index] > 0.5 or Ey1[argmax_index] > 0.5:  # 其中一个方向大于0.5另一个小于0.5
                    reward = -0.05
                    done = False
            elif max_multi_eff >= 0.25:
                if Ey[argmax_index] > 0.5 and Ey1[argmax_index] > 0.5:  # 两个方向都大于0.5
                    reward = (max_multi_eff - 0.25) * 2
                    done = False
                else:  # 两个方向一个小于0.5另一个大于0.5
                    reward = -0.05
                    done = False
            info = {'max_multi_eff': max_multi_eff,
                    'argmax_index': argmax_index,
                    'Ey': Ey,
                    'Ey1': Ey1,
                    'max_freq': max_fre,
                    'frequency': frequency,
                    'state': self.state}

        return self.state, reward, done, info

    def step(self, action: int):
        # assert self.action_space.contains(action) ,'Invalid action!' #对于超出范围的动作幅度显示无效动作
        """"
        action:
        0: index+0.1
        1: index+0.05
        2: index+0.01
        3:
        """
        action_in_state = self.action_in_state
        # ac=int(np.floor(action/3))
        if action <= 3:
            self.state[action] += action_in_state[action]
        else:
            self.state[action - 4] += action_in_state[action]

        if not self.observation_space.contains(self.state):  # 超出范围直接停止
            reward = -0.1 * 5  # 超出范围的reward设置为-0.1
            done = True  # 结束动作, 结束动作之后是否有reset呢
            info = {'max_multi_eff': None,
                    'argmax_index': None,
                    'Ey': None,
                    'Ey1': None,
                    'max_freq': None,
                    'frequency': None,
                    'state': self.state}
        else:
            max_multi_eff, argmax_index, Ey, Ey1, max_fre, frequency = effOfNano(state=self.state, model=self.model)
            if max_multi_eff < 0.25:
                if Ey[argmax_index] < 0.5 and Ey1[argmax_index] < 0.5:  # 两个方向都小于0.5
                    # reward=Ey[argmax_index]+Ey1[argmax_index]-1
                    reward = 0
                    done = False
                elif Ey[argmax_index] > 0.5 or Ey1[argmax_index] > 0.5:  # 其中一个方向大于0.5另一个小于0.5
                    # reward=Ey[argmax_index]+Ey1[argmax_index]-1
                    # reward=0
                    reward = np.exp(Ey[argmax_index] + Ey1[argmax_index] - 1)
                    reward /= 2
                    done = False
            elif max_multi_eff >= 0.25:
                if Ey[argmax_index] > 0.5 and Ey1[argmax_index] > 0.5:  # 两个方向都大于0.5
                    global MAX_EFF
                    if max_multi_eff > MAX_EFF:
                        MAX_EFF = max_multi_eff
                        print('<' * 50)
                        print(f'Now, i have found the best efficiency {MAX_EFF}, state is {self.state}, '
                              f'max Ey= {Ey[argmax_index]}, max Ey1= {Ey1[argmax_index]}')
                        print('<' * 50)
                    reward = np.exp(Ey[argmax_index] + Ey1[argmax_index] - 1)
                    done = False
                else:  # 两个方向一个小于0.5另一个大于0.5
                    # reward=Ey[argmax_index]+Ey1[argmax_index]-1
                    reward = np.exp(Ey[argmax_index] + Ey1[argmax_index] - 1)
                    reward /= 2
                    done = False
            if Ey[argmax_index] < 0.3 or Ey1[argmax_index] < 0.3:
                done = True
                # reward=min((Ey[argmax_index]-0.5)*50, (Ey1[argmax_index]-0.5)*50)
                reward = np.exp(Ey[argmax_index] + Ey1[argmax_index] - 1) - 1
            info = {'max_multi_eff': max_multi_eff,
                    'argmax_index': argmax_index,
                    'Ey': Ey,
                    'Ey1': Ey1,
                    'max_freq': max_fre,
                    'frequency': frequency,
                    'state': self.state}

        return self.state, reward, done, info

    def reset(self):
        # '''
        # 清空环境
        # :return:
        # '''
        lb = np.array([100, 100, 30, 1.3])
        ub = np.array([300, 600, 112, 1.5])
        self.state = np.random.random(size=(1, 4)) * (ub - lb) + lb
        self.state = np.round(self.state[0])
        self.state[-1] = np.random.random() * (ub[-1] - lb[-1]) + lb[-1]
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None


# Actor Critic model
class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(30, activation='relu')
        # self.dense1 = layers.Dense(100, activation='relu')
        self.dense1 = layers.Dense(50, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(30, activation='relu')
        # self.dense2 = layers.Dense(100, activation='relu')
        self.dense2 = layers.Dense(50, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values


def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.

    Arguments:
      episode: Current episode
      episode_reward: Reward accumulated over the current episode
      worker_idx: Which thread (worker)
      global_ep_reward: The moving average of the global reward
      result_queue: Queue storing the moving average of the scores
      total_loss: The total loss accumualted over the current episode
      num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print('Episode: %d | Moving Average Reward: %.3f | Episode Reward: %.3f | Loss: %.3f | Steps: %d | Worker: %d' % (
        episode, global_ep_reward, episode_reward, total_loss / num_steps, num_steps, worker_idx
    ))
    # print(
    #     f"Episode: {episode} | "
    #     f"Moving Average Reward: {int(global_ep_reward*1000)/1000} | " #三位小数
    #     f"Episode Reward: {int(episode_reward*1000)/1000} | " # 三位小数
    #     f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
    #     f"Steps: {num_steps} | "
    #     f"Worker: {worker_idx}"
    # )
    result_queue.put(global_ep_reward)
    return global_ep_reward


class RandomAgent:
    """Random Agent that will play the specified game

      Arguments:
        env_name: Name of the environment to be played
        max_eps: Maximum number of episodes to run agent for.
    """

    def __init__(self, env_name, max_eps):
        self.env = NanoEnv()
        self.max_episodes = max_eps
        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def run(self):
        each_reward_list = []
        each_reward_global_avg_list = []
        reward_avg = 0
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                state, reward, done, info = self.env.step(self.env.action_space.sample())
                steps += 1
                reward_sum += reward
                print(
                    f'Step {steps}, Step reward {reward}, Current reward sum {reward_sum}, Done is {done}, Current is {state}')
            # Record statistics
            self.global_moving_average_reward = record(episode,
                                                       reward_sum,
                                                       0,
                                                       self.global_moving_average_reward,
                                                       self.res_queue, 0, steps)
            each_reward_global_avg_list.append(self.global_moving_average_reward)
            # each_episode_reward_sum=list(self.res_queue.queue)
            reward_avg += reward_sum
            each_reward_list.append(reward_avg)
            # if False and len(each_reward_global_avg_list)>1 and len(each_reward_global_avg_list)%2==0:
            #     ImprovedMethod_Improve(each_reward_global_avg_list,title='Avg reward sum from current episode',figNum=0)
            print('--' * 50)
            print(
                f'Episode {episode}, Episode reward {reward_sum}, Reward average {int(100 * reward_sum / (episode + 1)) / 100}')
            print('--' * 50)
        final_avg = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
        return final_avg


class MasterAgent():
    def __init__(self):
        # -----
        # self.fdtd = fdtd  # init FDTD simulaiton env
        self.game_name = 'Nano'
        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # env = gym.make(self.game_name)
        env = NanoEnv()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.opt = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
        # print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self):
        if args.algorithm == 'random':
            random_agent = RandomAgent(self.game_name, args.max_eps)
            random_agent.run()
            return

        res_queue = Queue()

        workers = [Worker(self.state_size,
                          self.action_size,
                          self.global_model,
                          self.opt, res_queue,
                          i, game_name=self.game_name,
                          save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
                # if len(moving_average_rewards)>1 and len(moving_average_rewards)%2==0:
                #     ImprovedMethod_Improve(moving_average_rewards,title='Avg reward sum from current episode',figNum=0)
            else:
                break
        [w.join() for w in workers]

        # plt.plot(moving_average_rewards)
        # plt.ylabel('Moving average ep reward')
        # plt.xlabel('Step')
        # plt.savefig(os.path.join(self.save_dir,
        #                          '{} Moving Average.png'.format(self.game_name)))
        # plt.show()
        # ImprovedMethod_Improve(moving_average_rewards,figNum=1)
        return moving_average_rewards

    def play(self):
        base_dir = self.save_dir
        base_dir = '/tmp/'
        # env = gym.make(self.game_name).unwrapped
        env = NanoEnv().unwrapped
        state = env.reset()
        model = self.global_model
        model_path = os.path.join(base_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                # env.render(mode='rgb_array')
                policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("Step: {}. Reward: {}, action: {}, state: {}".format(step_counter, reward_sum, action, state))
                step_counter += 1

        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 state_size,
                 action_size,
                 global_model,
                 opt,
                 result_queue,
                 idx,
                 game_name='Nano',
                 save_dir=r'\tmp',
                 ):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = NanoEnv().unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                logits, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy().reshape((1, self.action_size))[0])
                new_state, reward, done, info = self.env.step(action)
                # if done:
                #     reward = -1
                worker_episode_file = r'csv-results/{}/{}'.format(create_file_time, self.worker_idx)

                ## file record
                if info['Ey'] is None:
                    max_Eyy = None
                    max_Eyx = None
                    multi_eff = None
                    max_freq = None
                else:
                    max_Eyy = info['Ey'][info['argmax_index']]
                    max_Eyx = info['Ey1'][info['argmax_index']]
                    multi_eff = info['max_multi_eff']
                    max_freq = info['frequency'][info['argmax_index']]
                write_txt(episode=self.global_episode, step=ep_steps, time_file=create_file_time,
                          worker_index=self.worker_idx, Ha=new_state[0], Hc=new_state[1],
                          Ka=new_state[2], index=new_state[3], max_Eyy=max_Eyy,
                          max_Eyx=max_Eyx, multi_eff=multi_eff,
                          max_freq=max_freq, reward=reward)

                ep_reward += reward  # 只要未完成就把reward累加算在一个episode里面的reward
                mem.store(current_state, action, reward)

                if time_count == args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       args.gamma)
                    self.ep_loss += total_loss  # 一个episode当中的loss总和
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                 self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        print('-' * 100)
                        print(new_state)
                        print('-' * 100)
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward, self.result_queue,
                                   self.ep_loss, ep_steps)
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            if info['Ey'] is None:
                                max_Eyy = None
                                max_Eyx = None
                                multi_eff = None
                                max_freq = None
                            else:
                                max_Eyy = info['Ey'][info['argmax_index']]
                                max_Eyx = info['Ey1'][info['argmax_index']]
                                multi_eff = info['max_multi_eff']
                                max_freq = info['frequency'][info['argmax_index']]
                            print('-' * 100)
                            print(f'Worker_index-{self.worker_idx} Episode-{Worker.global_episode} Step-{ep_steps} '
                                  f'State= {new_state[0]} {new_state[1]} {new_state[2]} {new_state[3]}, '
                                  f'max_Eyy/Eyx= {max_Eyy} {max_Eyx} {multi_eff} max_freq= {max_freq}, '
                                  f'This episode reward {ep_reward}\n')
                            with Worker.save_lock:
                                print("Saving best model to {}, "
                                      "episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}_{}.h5'.format(self.game_name, create_file_time))
                                )  # 按时间戳保存模型
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                     logits=logits)
        # policy_loss *= tf.stop_gradient(advantage)
        policy_loss *= advantage
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


if __name__ == '__main__':
    global MAX_EFF
    MAX_EFF = 0
    create_file_time = create_log_folder()
    args = config()
    env = NanoEnv()
    master = MasterAgent()
    # moving_average_rewards = master.train()
    # np.save(f'moving_average_rewards_{create_file_time}.npy', np.array(moving_average_rewards))
    # np.save(f'./result/moving_average_rewards_{create_file_time}.npy',np.array(moving_average_rewards))
