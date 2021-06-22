# -*- coding: utf-8 -*-
from gym import spaces
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model, Model
import gym
import argparse
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
import platform

if platform.system().lower() == 'windows':
    import matplotlib.pyplot as plt
from queue import Queue

tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.005)
parser.add_argument('--critic_lr', type=float, default=0.01)

args = parser.parse_args()
GLOBAL_REWARD_QUEUE = Queue()
CUR_EPISODE = 0


#  Arg parse
class config:
    algorithm = 'A3C Discrete'
    gamma = 0.99
    update_interval = 5
    actor_lr = 0.0001
    critic_lr = 0.0001
    max_eps = 100000
    save_dir = 'tmp'
    f0 = np.linspace(420, 550, 82)

    def __init__(self):
        self.__config__()

    def __config__(self):
        print(
            f'\nalgorithm={self.algorithm} | actor_lr={self.actor_lr} | critic_lr={self.critic_lr} |update_freq={self.update_interval} | \n'
            f'max_eps={self.max_eps} | gamma={self.gamma} | save_dir={self.save_dir}\n')


# calculate Eff of Nano
def effOfNano(state, model):
    tmp_state = [state[0] / 100, state[1] / 100, state[2] / 100, state[-1]]
    Ey, Ey1 = model.predict(np.array(tmp_state).reshape((-1, 4)))
    Ey, Ey1 = Ey[0], Ey1[0]
    multi_eff = Ey * Ey1
    argmax_index = np.argmax(multi_eff)  # 最大值的索引
    max_fre = args.f0[argmax_index]  # 频率最大值
    '''返回值，最大的Tyx，Tyy积，最大值所在的索引，此次仿真的Ey，Ey1，最大积所在对应的频率,所有的频率'''
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
        self.state_lb = [100, 100, 30, 1.1]
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
        if platform.system().lower() == 'windows':
            self.model = load_model(r'../bi-network/result-log/2020_12_17_21_38_9_yy_mape_3_312393_yx_mape_4_141227.h5')
        elif platform.system().lower() == 'linux':
            self.model = load_model(r'2020_12_17_21_38_9_yy_mape_3_312393_yx_mape_4_141227.h5')

    def step(self, action: int):
        # assert self.action_space.contains(action) ,'Invalid action!' #对于超出范围的动作幅度显示无效动作
        '''
        action:
        0: index+0.1
        1: index+0.05
        2: index+0.01
        3:
        '''
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

    def step0(self, action: int):
        # assert self.action_space.contains(action) ,'Invalid action!' #对于超出范围的动作幅度显示无效动作
        '''
        action:
        0: index+0.1
        1: index+0.05
        2: index+0.01
        3:
        '''
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
                    reward = Ey[argmax_index] + Ey1[argmax_index] - 1
                    done = False
                elif Ey[argmax_index] > 0.5 or Ey1[argmax_index] > 0.5:  # 其中一个方向大于0.5另一个小于0.5
                    reward = Ey[argmax_index] + Ey1[argmax_index] - 1
                    done = False
            elif max_multi_eff >= 0.25:
                if Ey[argmax_index] > 0.5 and Ey1[argmax_index] > 0.5:  # 两个方向都大于0.5
                    reward = (Ey[argmax_index] + Ey1[argmax_index] - 1) * 100
                    done = False
                else:  # 两个方向一个小于0.5另一个大于0.5
                    reward = Ey[argmax_index] + Ey1[argmax_index] - 1
                    done = False
            if Ey[argmax_index] < 0.3 or Ey1[argmax_index] < 0.3:
                done = True
                reward = min((Ey[argmax_index] - 0.5) * 50, (Ey1[argmax_index] - 0.5) * 50)
            info = {'max_multi_eff': max_multi_eff,
                    'argmax_index': argmax_index,
                    'Ey': Ey,
                    'Ey1': Ey1,
                    'max_freq': max_fre,
                    'frequency': frequency,
                    'state': self.state}

        return self.state, reward, done, info

    def reset(self):
        '''
        清空环境
        :return:
        '''
        lb = np.array([100, 100, 30, 1.1])
        ub = np.array([300, 600, 112, 1.5])
        self.state = np.random.random(size=(1, 4)) * (ub - lb) + lb
        self.state = np.round(self.state[0])
        self.state[-1] = np.random.random() * (ub[-1] - lb[-1]) + lb[-1]
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None


class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        inputs = Input(shape=(self.state_dim,))
        layer = Dense(8, activation='relu')(inputs)
        layer = Dense(5, activation='relu')(layer)
        layer = Dense(3, activation='relu')(layer)
        outputs = Dense(self.action_dim, activation='softmax')(layer)
        return Model(inputs, outputs)

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(8, activation='relu'),
            Dense(3, activation='relu'),
            Dense(3, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env_name=None):
        env = NanoEnv()
        self.env_name = env_name
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.global_actor = Actor(self.state_dim, self.action_dim)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = cpu_count()

    def train(self, max_episodes=10000):
        workers = []

        for i in range(self.num_workers):
            env = NanoEnv()
            workers.append(WorkerAgent(
                env, self.global_actor, self.global_critic, max_episodes))

        for worker in workers:
            worker.start()

        if platform.system().lower() == 'windows' and not GLOBAL_REWARD_QUEUE.empty():
            rewards = list(GLOBAL_REWARD_QUEUE.queue)
            if len(rewards) % 5 == 0 and len(rewards) > 0:
                plt.cla()
                plt.figure(0)
                plt.plot(rewards)

        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes):
        Thread.__init__(self)
        self.lock = Lock()
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        mu, siga = np.mean(td_targets, axis=0), np.std(td_targets, axis=0)
        td_targets = (td_targets - mu) / siga
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        global CUR_EPISODE

        while self.max_episodes >= CUR_EPISODE:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                probs = self.actor.model.predict(
                    np.reshape(state, [1, self.state_dim]))
                try:
                    action = np.random.choice(self.action_dim, p=probs[0])
                except:
                    action = np.random.choice(self.action_dim, p=[1 / self.action_dim for _ in range(self.action_dim)])

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= args.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(
                        rewards, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)

                    with self.lock:
                        actor_loss = self.global_actor.train(
                            states, actions, advantages)
                        critic_loss = self.global_critic.train(
                            states, td_targets)

                        self.actor.model.set_weights(
                            self.global_actor.model.get_weights())
                        self.critic.model.set_weights(
                            self.global_critic.model.get_weights())

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    td_target_batch = []
                    advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print('index {} EP {} EpisodeReward={}'.format(self.getName(), CUR_EPISODE, episode_reward))
            GLOBAL_REWARD_QUEUE.put(episode_reward)
            CUR_EPISODE += 1

    def run(self):
        self.train()


def main():
    env_name = 'CartPole-v1'
    agent = Agent(env_name)
    agent.train(max_episodes=args.max_eps)


if __name__ == "__main__":
    args = config()
    main()
