import tensorflow as tf
import numpy as np

class Network:
    def __init__(self, x_shape, y_shape, learning_rate, gamma=0.95, restore_path=None):

        self.gamma = gamma
        self.build_network(learning_rate=learning_rate, x_shape=x_shape, y_shape=y_shape)
        self.sess = tf.Session()
        tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if restore_path is not None:
            self.saver.restore(self.sess, restore_path)
            print("Restore model from {}".format(restore_path))


    def predict(self, observation):
        observation = observation[:, np.newaxis]
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.X: observation})
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def fit(self, epoche_observations, epoche_actions, epoche_rewards, repl_buffer):
        self.sess.run(self.train_op, feed_dict={
            self.X: np.vstack(epoche_observations).T,
            self.Y: np.vstack(np.array(epoche_actions)).T,
            self.epoch_rewards: self.calc_reward(epoche_rewards),
        })
        repl_buffer_observations, repl_buffer_actions, repl_buffer_rewards = repl_buffer.get_data(10)
        self.sess.run(self.train_op, feed_dict={
            self.X: np.vstack(repl_buffer_observations).T,
            self.Y: np.vstack(np.array(repl_buffer_actions)).T,
            self.epoch_rewards: np.array(repl_buffer_rewards),
        })


    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)
        print("Save model to {}".format(save_path))
   
    def calc_reward(self, episode_rewards):
        discounted_reward = np.zeros_like(episode_rewards)
        running_add = 0
        for t in reversed(range(len(episode_rewards))):
            running_add = running_add * self.gamma + episode_rewards[t]
            discounted_reward[t] = running_add
       
       # discounted_reward -= np.mean(discounted_reward)
       # discounted_reward /= np.std(discounted_reward)	
        return discounted_reward


    def build_network(self, x_shape, y_shape, learning_rate=0.01):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(x_shape, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(y_shape, None), name="Y")
            self.epoch_rewards = tf.placeholder(tf.float32, [None, ], name="rewards")

        units_layers = [10, 10, 10]

        with tf.name_scope('layer_1'):
            W1 = tf.get_variable("W1", [units_layers[0], x_shape],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layers[0], 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            Z1 = tf.add(tf.matmul(W1, self.X), b1)
            A1 = tf.nn.relu(Z1)

        with tf.name_scope('layer_2'):
            W2 = tf.get_variable("W2", [units_layers[1], units_layers[0]],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layers[1], 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)

        with tf.name_scope('layer_3'):
            W3 = tf.get_variable("W3", [units_layers[2], units_layers[1]],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [units_layers[2], 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            Z3 = tf.add(tf.matmul(W3, A2), b3)
            A3 = tf.nn.relu(Z3)

        with tf.name_scope('layer_4'):
            W4 = tf.get_variable("W4", [y_shape, units_layers[2]],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b4 = tf.get_variable("b4", [y_shape, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            Z4 = tf.add(tf.matmul(W4, A3), b4)
            A4 = tf.nn.softmax(Z4)

        logits = tf.transpose(Z4)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A4')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.epoch_rewards)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

