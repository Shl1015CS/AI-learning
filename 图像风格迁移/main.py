"""
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os


IMAGE_W = 800 
IMAGE_H = 600 
CONTENT_IMG =  './images/Taipei101.jpg'
STYLE_IMG = './images/StarryNight.jpg'
OUTOUT_DIR = './results'
OUTPUT_IMG = 'results.png'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
INI_NOISE_RATIO = 0.7
STYLE_STRENGTH = 500
ITERATION = 5000

CONTENT_LAYERS =[('conv4_2',1.)]
STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]


MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))

def build_net(ntype, nin, nwb=None):
  if ntype == 'conv':
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME')+ nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                  strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i,):
  weights = vgg_layers[i][0][0][0][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][0][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias

def build_vgg19(path):
  net = {}
  vgg_rawnet = scipy.io.loadmat(path)
  vgg_layers = vgg_rawnet['layers'][0]
  net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
  net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
  net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
  net['pool1']   = build_net('pool',net['conv1_2'])
  net['conv2_1'] = build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5))
  net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
  net['pool2']   = build_net('pool',net['conv2_2'])
  net['conv3_1'] = build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10))
  net['conv3_2'] = build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12))
  net['conv3_3'] = build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14))
  net['conv3_4'] = build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16))
  net['pool3']   = build_net('pool',net['conv3_4'])
  net['conv4_1'] = build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19))
  net['conv4_2'] = build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21))
  net['conv4_3'] = build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23))
  net['conv4_4'] = build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25))
  net['pool4']   = build_net('pool',net['conv4_4'])
  net['conv5_1'] = build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28))
  net['conv5_2'] = build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30))
  net['conv5_3'] = build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32))
  net['conv5_4'] = build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34))
  net['pool5']   = build_net('pool',net['conv5_4'])
  return net

def build_content_loss(p, x):
  M = p.shape[1]*p.shape[2]
  N = p.shape[3]
  loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((x - p),2))  
  return loss


def gram_matrix(x, area, depth):
  x1 = tf.reshape(x,(area,depth))
  g = tf.matmul(tf.transpose(x1), x1)
  return g

def gram_matrix_val(x, area, depth):
  x1 = x.reshape(area,depth)
  g = np.dot(x1.T, x1)
  return g

def build_style_loss(a, x):
  M = a.shape[1]*a.shape[2]
  N = a.shape[3]
  A = gram_matrix_val(a, M, N )
  G = gram_matrix(x, M, N )
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
  return loss



def read_image(path):
  image = scipy.misc.imread(path)
  image = scipy.misc.imresize(image,(IMAGE_H,IMAGE_W))
  image = image[np.newaxis,:,:,:] 
  image = image - MEAN_VALUES
  return image

def write_image(path, image):
  image = image + MEAN_VALUES
  image = image[0]
  image = np.clip(image, 0, 255).astype('uint8')
  scipy.misc.imsave(path, image)


def main():
  net = build_vgg19(VGG_MODEL)
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  noise_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
  content_img = read_image(CONTENT_IMG)
  style_img = read_image(STYLE_IMG)

  sess.run([net['input'].assign(content_img)])
  cost_content = sum(map(lambda l,: l[1]*build_content_loss(sess.run(net[l[0]]) ,  net[l[0]])
    , CONTENT_LAYERS))

  sess.run([net['input'].assign(style_img)])
  cost_style = sum(map(lambda l: l[1]*build_style_loss(sess.run(net[l[0]]) ,  net[l[0]])
    , STYLE_LAYERS))

  cost_total = cost_content + STYLE_STRENGTH * cost_style
  optimizer = tf.train.AdamOptimizer(2.0)

  train = optimizer.minimize(cost_total)
  sess.run(tf.initialize_all_variables())
  sess.run(net['input'].assign( INI_NOISE_RATIO* noise_img + (1.-INI_NOISE_RATIO) * content_img))

  if not os.path.exists(OUTOUT_DIR):
      os.mkdir(OUTOUT_DIR)

  for i in range(ITERATION):
    sess.run(train)
    if i%100 ==0:
      result_img = sess.run(net['input'])
      print sess.run(cost_total)
      write_image(os.path.join(OUTOUT_DIR,'%s.png'%(str(i).zfill(4))),result_img)
  
  write_image(os.path.join(OUTOUT_DIR,OUTPUT_IMG),result_img)
  

if __name__ == '__main__':
  main()
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.io
import cv2
import scipy.misc

HEIGHT = 300
WIGHT = 450
LEARNING_RATE = 1.0
NOISE = 0.5
ALPHA = 1
BETA = 500

TRAIN_STEPS = 200

OUTPUT_IMAGE = "/kaggle/working"
STYLE_LAUERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]
CONTENT_LAYERS = [('conv4_2', 0.5), ('conv5_2',0.5)]


def vgg19():
    layers=(
        'conv1_1','relu1_1','conv1_2','relu1_2','pool1',
        'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
        'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',
        'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',
        'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','conv5_4','relu5_4','pool5'
    )
    vgg = scipy.io.loadmat('/kaggle/input/imagenetvggverydeep19/imagenet-vgg-verydeep-19.mat')
    weights = vgg['layers'][0]

    network={}
    net = tf.Variable(np.zeros([1, 300, 450, 3]), dtype=tf.float32)
    network['input'] = net
    for i,name in enumerate(layers):
        layer_type=name[:4]
        if layer_type=='conv':
            kernels = weights[i][0][0][0][0][0]
            bias = weights[i][0][0][0][0][1]
            conv=tf.nn.conv2d(net,tf.constant(kernels),strides=(1,1,1,1),padding='SAME',name=name)
            net=tf.nn.relu(conv + bias)
        elif layer_type=='pool':
            net=tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
        network[name]=net
    return network


# 求gamm矩阵
def gram(x, size, deep):
    x = tf.reshape(x, (size, deep))
    g = tf.matmul(tf.transpose(x), x)
    return g


def style_loss(sess, style_neck, model):
    style_loss = 0.0
    for layer_name, weight in STYLE_LAUERS:
        # 计算特征矩阵
        a = style_neck[layer_name]
        x = model[layer_name]
        # 长x宽
        M = a.shape[1] * a.shape[2]
        N = a.shape[3]

        # 计算gram矩阵
        A = gram(a, M, N)
        G = gram(x, M, N)

        # 根据公式计算损失，并进行累加
        style_loss += (1.0 / (4 * M * M * N * N)) * tf.reduce_sum(tf.pow(G - A, 2)) * weight
        # 将损失对层数取平均
    style_loss /= len(STYLE_LAUERS)
    return style_loss


def content_loss(sess, content_neck, model):
    content_loss = 0.0
    # 逐个取出衡量内容损失的vgg层名称及对应权重

    for layer_name, weight in CONTENT_LAYERS:
        # 计算特征矩阵
        p = content_neck[layer_name]
        x = model[layer_name]
        # 长x宽xchannel

        M = p.shape[1] * p.shape[2]
        N = p.shape[3]

        lss = 1.0 / (M * N)
        content_loss += lss * tf.reduce_sum(tf.pow(p - x, 2)) * weight
        # 根据公式计算损失，并进行累加

    # 将损失对层数取平均
    content_loss /= len(CONTENT_LAYERS)
    return content_loss


def random_img(height, weight, content_img):
    noise_image = np.random.uniform(-20, 20, [1, height, weight, 3])
    random_img = noise_image * NOISE + content_img * (1 - NOISE)
    return random_img


def get_neck(sess, model, content_img, style_img):
    sess.run(tf.compat.v1.assign(model['input'], content_img))
    content_neck = {}
    for layer_name, weight in CONTENT_LAYERS:
        # 计算特征矩阵
        p = sess.run(model[layer_name])
        content_neck[layer_name] = p
    sess.run(tf.assign(model['input'], style_img))
    style_content = {}
    for layer_name, weight in STYLE_LAUERS:
        # 计算特征矩阵
        a = sess.run(model[layer_name])
        style_content[layer_name] = a
    return content_neck, style_content


def main():
    model = vgg19()
    content_img = cv2.imread('/kaggle/input/images/images/Taipei101.jpg')
    content_img = cv2.resize(content_img, (450, 300))
    content_img = np.reshape(content_img, (1, 300, 450, 3)) - [128.0, 128.2, 128.0]
    style_img = cv2.imread('/kaggle/input/images/images/StarryNight.jpg')
    style_img = cv2.resize(style_img, (450, 300))
    style_img = np.reshape(style_img, (1, 300, 450, 3)) - [128.0, 128.2, 128.0]

    # 生成图片
    rand_img = random_img(HEIGHT, WIGHT, content_img)

    with tf.compat.v1.Session() as sess:
        # 计算loss值
        content_neck, style_neck = get_neck(sess, model, content_img, style_img)
        cost = ALPHA * content_loss(sess, content_neck, model) + BETA * style_loss(sess, style_neck, model)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.compat.v1.assign(model['input'], rand_img))
        for step in range(TRAIN_STEPS):
            print(step)
            # 训练
            sess.run(optimizer)

            if step % 10 == 0:
                img = sess.run(model['input'])
                img += [128, 128, 128]
                img = np.clip(img, 0, 255).astype(np.uint8)
                name = OUTPUT_IMAGE + "//" + str(step) + ".jpg"
                img = img[0]
                cv2.imwrite(name, img)

        img = sess.run(model['input'])
        img += [128, 128, 128]
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite("/kaggle/working", img[0])

main()
