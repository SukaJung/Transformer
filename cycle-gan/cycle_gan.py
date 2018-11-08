import tensorflow as tf
import tensorflow.contrib.layers as tcl

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time


img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width


batch_size = 1
pool_size = 50
ngf = 32
ndf = 64



def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d",
                   do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(inputconv, o_d, f_w, s_w, padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID",
                     name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                  biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")


    return conv

def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


def build_generator_resnet_6blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
        o_c2 = general_conv2d(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv2d(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1")
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2")
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3")
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4")
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5")
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6")

        o_c4 = general_deconv2d(o_r6, [batch_size, 64, 64, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv2d(o_c4, [batch_size, 128, 128, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5")
        o_c5_pad = tf.pad(o_c5, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c6 = general_conv2d(o_c5_pad, img_layer, f, f, 1, 1, 0.02, "VALID", "c6", do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def build_generator_resnet_9blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
        o_c2 = general_conv2d(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv2d(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1")
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2")
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3")
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4")
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5")
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6")
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7")
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8")
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9")

        o_c4 = general_deconv2d(o_r9, [batch_size, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv2d(o_c4, [batch_size, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5")
        o_c6 = general_conv2d(o_c5, img_layer, f, f, 1, 1, 0.02, "SAME", "c6", do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6, "t1")


    return out_gen
def build_gen_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return o_c5


def patch_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f= 4

        patch_input = tf.random_crop(inputdisc,[1,70,70,3])
        o_c1 = general_conv2d(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm="False", relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

    return o_c5

def plot(samples):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(5, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample+1)/2
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig



def init():
    X = tf.placeholder(tf.float32, shape=[batch_size, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])
    X_reshape = tf.reshape(X, shape = [-1,28,28,1])

    z_sample = tf.placeholder(tf.float32, shape=[None, 100])

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    G_output = generator(z_sample, Y)

    D_real,D_real_logits = discriminator(X_reshape, Y)
    D_fake,D_fake_logits = discriminator(G_output, Y,reuse=True)

    d_fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_fake), logits=D_fake_logits))
    d_real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_real), logits=D_real_logits))

    d_loss = d_fake_loss+d_real_loss
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_fake), logits=D_fake_logits))

    t_vars = tf.trainable_variables()
    d_var = [var for var in t_vars if 'dis' in var.name]
    g_var = [var for var in t_vars if 'gen' in var.name]

    d_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(d_loss, var_list=d_var)
    g_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(g_loss, var_list=g_var)

    sess.run(tf.global_variables_initializer())

    num_img = 0

    print("Start training batch_size : {} total_batch : {}".format(batch_size, total_batch))

    for epoch in range(10000):
        start =  time.time()
        for i in range(total_batch):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            z_variable = np.random.uniform(-1., 1., [batch_size, num_latent_variable])
            sess.run(d_optimizer, feed_dict={X: batch_X, Y: batch_Y, z_sample: z_variable})
            sess.run(g_optimizer, feed_dict={Y: batch_Y, z_sample: z_variable})



            if i % 100 == 0:
                d_l, g_l = sess.run([d_loss, g_loss],
                                    feed_dict={X: batch_X, Y: batch_Y,
                                               z_sample: np.random.uniform(-1., 1., [batch_size, num_latent_variable])})

                print("epoch : {} d_loss : {} g_loss : {} time : {}".format(epoch, d_l, g_l, time.time() - start))

                samples = sess.run(G_output,
                                       feed_dict={z_sample: np.random.uniform(-1., 1., [batch_size, num_latent_variable])})
                fig = plot(samples[:50])
                plt.savefig('output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
                num_img += 10
                plt.close(fig)


def main():
    init()
    return


if __name__ == "__main__":
    main()
