# coding: utf-8

import os
import sys
import cv2
import time
import traceback
import threading
import numpy as np
import cPickle as pic
import tensorflow as tf

TAG_SIZE = 20
IMG_SIZE = 256
IMG_SIZE_HEIGHT = 456
BATCH_SIZE = 64
MAX_CAP = 20
MAX_EP_STEP=20000

def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def line(x, w, b):
    return tf.matmul(x, w) + b


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


class Pix2code:
    tag_size = 20
    img_size = IMG_SIZE
    img_size_height = IMG_SIZE_HEIGHT
    lstm_layer_count = 2
    lstm1_units = 128
    lstm2_units = 512
    learn_rate = 1e-4
    batch_size = 64
    step_num = 48
    train_data_cap = 1000

    def __init__(self):
        pass
        # self.create_graph()
        # self.optimizer()
        # self.sess = tf.InteractiveSession()
        # tf.global_variables_initializer().run()

    def create_graph(self):
        # shapes
        tag_input_shape = [None, self.tag_size]
        tag_target_input_shape = [None, self.step_num, self.tag_size]
        img_input_shape = [None, self.img_size_height, self.img_size, 1]
        # tag one hot vec input
        self.tag_input = tag_input = tf.placeholder(tf.float32, tag_input_shape, "tag")
        # img data input
        self.tag_target_input = tag_target_input = tf.placeholder(
            tf.float32, tag_target_input_shape, "tag_target")
        self.img_input = tf.placeholder(
            tf.float32, img_input_shape, "img")

        self.batch_input = batch_input = tf.Variable(
            BATCH_SIZE, trainable=False, name="batch_size")

        queue = tf.FIFOQueue(
            self.batch_size*5, [tf.float32, tf.float32, tf.float32],
            shapes=[img_input_shape[1:], tag_input_shape[1:], tag_target_input_shape[1:]])
        
        self.enqueue = queue.enqueue_many(
                [self.img_input, self.tag_input, self.tag_target_input])

        # maxpool k arg
        self.mpk1 = mpk1 = 4
        self.mpk2 = mpk2 = 2
        self.mpk3 = mpk3 = 2
        self.mpk4 = mpk4 = 2

        mpk_scale = mpk1*mpk2*mpk3*mpk4

        self.cnn_out_size = (self.img_size_height/mpk_scale+1)*(self.img_size/mpk_scale) * 256

        self.weights = {
            "conv1":
            tf.Variable(tf.random_normal([3, 3, 1, 32])),
            "conv2":
            tf.Variable(tf.random_normal([3, 3, 32, 64])),
            "conv3":
            tf.Variable(tf.random_normal([3, 3, 64, 128])),
            "conv4":
            tf.Variable(tf.random_normal([3, 3, 128, 256])),
            "line5":
            tf.Variable(
                tf.random_normal([self.cnn_out_size, 1024])),
            "rnn1_line":
            tf.Variable(tf.random_normal([self.tag_size, self.lstm1_units])),
            "rnn2_line":
            tf.Variable(
                tf.random_normal([self.lstm1_units + 1024, self.lstm2_units])),
            "sm":
            tf.Variable(tf.random_normal([self.lstm2_units, self.tag_size])),
        }
        self.biases = {
            "conv1": tf.Variable(tf.random_normal([32])),
            "conv2": tf.Variable(tf.random_normal([64])),
            "conv3": tf.Variable(tf.random_normal([128])),
            "conv4": tf.Variable(tf.random_normal([256])),
            "line5": tf.Variable(tf.random_normal([1024])),
            "rnn1_line": tf.Variable(tf.random_normal([self.lstm1_units])),
            "rnn2_line": tf.Variable(tf.random_normal([self.lstm2_units])),
            "sm": tf.Variable(tf.random_normal([self.tag_size])),
        }

        # first rnn layer
        def cell1():
            return tf.contrib.rnn.BasicLSTMCell(
                self.lstm1_units)

        self.lstm1 = tf.contrib.rnn.MultiRNNCell(
            [cell1() for _ in range(self.lstm_layer_count)])

        def cell2():
            return tf.contrib.rnn.BasicLSTMCell(
                self.lstm2_units)

        self.lstm2 = tf.contrib.rnn.MultiRNNCell(
            [cell2() for _ in range(self.lstm_layer_count)])

        state1 = self.lstm1.zero_state(self.batch_size, tf.float32)
        state2 = self.lstm2.zero_state(self.batch_size, tf.float32)

        tag_input = self.tag_input

        cost_i = tf.constant(0.)
        i = tf.constant(1)
        while_condition = lambda i, t, s1, s2, c: tf.less(i, self.step_num)

        optimizer = tf.train.RMSPropOptimizer(self.learn_rate)
        
        img_que, tag_que, tag_target_que = queue.dequeue_many(self.batch_size) 
        def body(i, tag_input, state1, state2, cost_i):
            conv = conv2d(img_que, self.weights["conv1"],
                          self.biases["conv1"])
            conv = maxpool2d(conv, k=mpk1)
            conv = conv2d(conv, self.weights["conv2"], self.biases["conv2"])
            conv = maxpool2d(conv, k=mpk2)
            conv = conv2d(conv, self.weights["conv3"], self.biases["conv3"])
            conv = maxpool2d(conv, k=mpk3)
            conv = conv2d(conv, self.weights["conv4"], self.biases["conv4"])
            conv = maxpool2d(conv, k=mpk4)
            val_conv = tf.reshape(conv, [-1, self.cnn_out_size])
            val_conv = tf.nn.relu(
                line(val_conv, self.weights["line5"], self.biases["line5"]))

            val_rnn = tf.nn.relu(
                line(tag_input, self.weights["rnn1_line"], self.biases[
                    "rnn1_line"]))
            val_rnn, state1 = self.lstm1(val_rnn, state1, scope="lstm1")
            val = tf.concat([val_rnn, val_conv], 1)
            val = tf.nn.relu(
                line(val, self.weights["rnn2_line"], self.biases["rnn2_line"]))
            val, state2 = self.lstm2(val, state2, scope="lstm2")
            tag_out = line(val, self.weights["sm"], self.biases["sm"])

            tag_input = tag_target_que[:, i, :]
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tag_target_que[:, i, :], logits=tag_out))

            gvs = optimizer.compute_gradients(cost)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                          for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs)

            cost_i = tf.add(cost_i, cost)
            # train_op = optimizer.minimize(cost)
            with tf.control_dependencies([train_op]):
                return [tf.add(i, 1), tag_input, state1, state2, cost_i]

        _, _, _, _, cost_i = tf.while_loop(
            while_condition, body, [i, tag_que, state1, state2, cost_i])
        self.cost = cost_i / self.step_num

        self.create_test_graph()

    def create_test_graph(self):

        state1 = self.lstm1.zero_state(1, tf.float32)
        state2 = self.lstm2.zero_state(1, tf.float32)

        conv = conv2d(self.img_input, self.weights["conv1"],
                      self.biases["conv1"])
        conv = maxpool2d(conv, k=self.mpk1)
        conv = conv2d(conv, self.weights["conv2"], self.biases["conv2"])
        conv = maxpool2d(conv, k=self.mpk2)
        conv = conv2d(conv, self.weights["conv3"], self.biases["conv3"])
        conv = maxpool2d(conv, k=self.mpk3)
        conv = conv2d(conv, self.weights["conv4"], self.biases["conv4"])
        conv = maxpool2d(conv, k=self.mpk4)
        val_conv = tf.reshape(conv, [-1, self.cnn_out_size])
        val_conv = tf.nn.relu(
            line(val_conv, self.weights["line5"], self.biases["line5"]))

        i_val = tf.constant(1)
        while_condition_val = lambda i, t, s1, s2, to: tf.less(i, self.step_num)
        tag_input = self.tag_input

        def body_val(i, tag_input, state1, state2, tags_out):
            tf.get_variable_scope().reuse_variables()
            val_rnn = tf.nn.relu(
                line(tag_input, self.weights["rnn1_line"], self.biases[
                    "rnn1_line"]))
            val_rnn, state1 = self.lstm1(val_rnn, state1, scope="lstm1")
            val = tf.concat([val_rnn, val_conv], 1)
            val = tf.nn.relu(
                line(val, self.weights["rnn2_line"], self.biases["rnn2_line"]))
            val, state2 = self.lstm2(val, state2, scope="lstm2")
            tag_out = line(val, self.weights["sm"], self.biases["sm"])
            tag_out = tf.nn.softmax(tag_out)
            tag_input = tag_out
            tags_out = tf.concat([tags_out, tag_input], 0)
            return [tf.add(i, 1), tag_input, state1, state2, tags_out]

        _, _, _, _, tags_out = tf.while_loop(while_condition_val, body_val, [
            i_val, tag_input, state1, state2, tag_input
        ])

        self.tags_out = tf.identity(tags_out, name="tags_out")
        self.tags_out = tags_out


class DateReader:

    train_scope = 0.9

    def __init__(self, tag_file, img_dir):
        self.img_dir = img_dir
        self._load_tag(tag_file)
        # self._load_img(img_dir)

    def _load_tag(self, tag_file):
        f = open(tag_file, "r")
        s = f.read()
        lines = s.split("\n")[:-1]
        count = len(lines)

        tags = []
        for line in lines:
            tags.append(line.split(","))

        dic = {"start": 0, "end": 1}
        i = 2
        max_cap = 0
        for line_tags in tags:
            # print len(line_tags)
            if len(line_tags) > max_cap:
                max_cap = len(line_tags)
            for tag in line_tags:
                if not dic.has_key(tag):
                    dic[tag] = i
                    i += 1

        print "tag max len", max_cap
        max_cap = MAX_CAP

        size = len(dic)
        dic_vec = {}
        dic_tag = [None] * size
        for k, v in dic.items():
            vec = np.zeros(size, dtype=np.float32)
            vec[v] = 1
            dic_vec[k] = vec
            dic_tag[v] = k

        tags_arr = np.zeros((count, max_cap, size), dtype=np.float32)
        tags_arr[:, :, 1] = 1  # set to end tag
        for i in range(len(tags)):
            line_tags = tags[i]
            for j in range(max_cap):
                if j >= len(line_tags): continue
                tags_arr[i, j, :] = dic_vec[line_tags[j]]

        self.tags_arr = tags_arr
        self.dic_vec = dic_vec
        self.dic_i = dic
        self.dic_tag = dic_tag

        self.count = len(lines)
        self.tags = tags
        self.max_cap = max_cap
        self.tag_size = size

    def _load_img(self, img_dir):
        imgs = np.zeros((self.count, IMG_SIZE_HEIGHT, IMG_SIZE), dtype=np.float32)
        for i in range(self.count):
            im = cv2.imread(img_dir + "/" + str(i) + ".png")
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE_HEIGHT))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = im.astype(np.float32)
            im = im / 256 * 2 - 1

            imgs[i] = im

            if i % 500 == 0:
                print i, "loaded ..."
                sys.stdout.flush()

        imgs = imgs.reshape(self.count, IMG_SIZE_HEIGHT, IMG_SIZE, 1)
        self.imgs_arr = imgs

    def _read_img(self, idx):
        imgs = np.zeros((len(idx), IMG_SIZE_HEIGHT, IMG_SIZE)) 
        for n in range(len(idx)):
            i = idx[n]
            im = cv2.imread(self.img_dir + "/" + str(i) + ".png")
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE_HEIGHT))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = im.astype(np.float32)
            im = im / 256 * 2 - 1
            imgs[n] = im
            
        imgs = imgs.reshape(list(imgs.shape)+[1])
        return imgs

    def decode(self, arr):
        tags = []
        for t in arr:
            i = np.argmax(t)
            tags.append(self.dic_tag[i])
        return tags

    def encode(self):
        pass

    def train_random(self, batch):
        imgs_arr = self.imgs_arr[:self.train_scope]
        tags_arr = self.tags_arr[:self.train_scope]
        i = 0
        while 1:
            tag_list = np.zeros(
                (batch, self.max_cap, self.tag_size), dtype=np.float32)
            img_list = np.zeros(
                (batch, IMG_SIZE_HEIGHT, IMG_SIZE, 1), dtype=np.float32)
            idx = np.random.randint(self.train_scope, size=batch)
            img_list[:] = imgs_arr[idx]
            tag_list[:] = tags_arr[idx]
            yield (i, tag_list, img_list)
            i += 1

    def train_random_load(self, batch):
        scope = int(self.count*self.train_scope)
        idx = np.random.randint(scope, size=batch)
        tag_list = np.zeros(
            (batch, self.max_cap, self.tag_size), dtype=np.float32)
        tag_list[:] = self.tags_arr[idx] 
        img_list = self._read_img(idx)
        return (tag_list, img_list)
        
    def all_train_data(self):
        
        imgs = self.imgs_arr.copy()
        tag_list = self.tags_arr.copy()
        tag_input = self.tags_arr[:, 0, :].copy() 

        return [imgs, tag_input, tag_list]

    def test_sample(self, n=1):
        scope = int(self.train_scope*self.count)
        imgs_arr = self.imgs_arr[scope:]
        tags_arr = self.tags_arr[scope:]

        tag_list = np.zeros((n, self.max_cap, self.tag_size), dtype=np.float32)
        img_list = np.zeros((n, IMG_SIZE_HEIGHT, IMG_SIZE, 1), dtype=np.float32)

        idx = np.random.randint(self.count - scope, size=n)
        img_list[:] = imgs_arr[idx]
        tag_list[:] = tags_arr[idx]

        for i in range(n):
            yield (i, tag_list[i:i + 1], img_list[i:i + 1])

    def test_sample_load(self, n=1):
        scope = int(self.train_scope*self.count)
        tag_list = np.zeros((n, self.max_cap, self.tag_size), dtype=np.float32)

        idx = np.random.randint(self.count - scope, size=n) + scope
        tag_list[:] = self.tags_arr[idx]
        img_list = self._read_img(idx)
        
        for i in range(n):
            yield (i, tag_list[i:i+1], img_list[i: i+1])


def train(model, reader):
    f = open("test_sample.txt", "w")

    batch = BATCH_SIZE
    saver = tf.train.Saver()
    with tf.Session() as sess:

        print "init variables ..."
        sess.run(tf.global_variables_initializer())

        print "start thread of enqueue train data ..."
        def enqueue_th():
            while 1:
                try:
                    tags, imgs = reader.train_random_load(batch)
                    start_tag_input = tags[:, 0, :]
                    sess.run(model.enqueue, feed_dict={
                        model.img_input: imgs,
                        model.tag_input: start_tag_input,
                        model.tag_target_input: tags,
                    })
                except BaseException, err:
                    traceback.print_exc()
                    # raise BaseException(e)
                    os._exit(1)
            
        threading.Thread(target=enqueue_th).start() 

        print "start train model ..."

        for i in range(MAX_EP_STEP):
            stm = time.time()
            cost = sess.run(model.cost)
            print "train", i, "cost:", cost, "use time %.4fs"%(time.time()-stm)
            sys.stdout.flush()

            if i % 100 == 0:
                f.write("train %d: \n\n" % (i))
                for _, tag, img in reader.test_sample_load(5):
                    out = sess.run(
                        [
                            model.tags_out,
                        ],
                        feed_dict={
                            model.img_input: img,
                            model.tag_input: tag[:, 0],
                        })

                    f.write("target: %s \n" %
                            (",".join(reader.decode(tag[0]))))
                    f.write("output: %s \n\n" %
                            (",".join(reader.decode(out[0]))))
                f.flush()

                saver.save(sess, "models/pix2code", global_step=i)
                print "saved ..."


def main():
    pc = Pix2code()
    reader = DateReader("tags.txt", "imgs")
    with open("dict.pik", "w") as f:
        pic.dump(reader.dic_tag, f)

    pc.batch_size = BATCH_SIZE
    pc.tag_size = reader.tag_size
    pc.step_num = MAX_CAP
    pc.train_data_cap = reader.count
    pc.create_graph()
    # pc.optimizer()
    print "tag size:", reader.tag_size, "max step", reader.max_cap

    train(pc, reader)


if __name__ == "__main__":
    main()
