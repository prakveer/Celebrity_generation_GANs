#LSGAN on CelebA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

#import spatial_transformer

import tensorflow as tf

import utils
import matplotlib.pyplot as plt
import pdb
import sys
import numpy as np



def plotting(entry, namer):
        #plotting
    plt.plot(range(len(entry)), entry)
    plt.ylabel(namer)
    plt.xlabel('iterations')
    plt.show()
    return
  
train_disc_loss=[]
train_gen_loss=[]



class ConvNet(object):
    def __init__(self):
        self.batch_size = 200
        #self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.skip_step = 200
        self.no_display_imgs = 10
        self.training=False

        self.lr = 0.0001
        self.beta1 = 0.5
        self.gen_train = tf.Variable(True, dtype= tf.bool, trainable = False)
        self.disc_assign = self.gen_train.assign(False)
        self.gen_assign = self.gen_train.assign(True)
        #self.clip_value = 0.01
        #self.n_critic = 5
    
        

    def image_plotter(self, imgs, n = 1):
        for i in range(0,self.batch_size, int(self.batch_size/n)):
            
                im = np.reshape(imgs[i,:,:,0], [64,64])
                im = ((im+1)*255.0)/2
                plt.imshow(im.astype(np.uint8), cmap='gray')
                plt.show()



    def get_data(self):
        with tf.name_scope('data'):
            self.true_imgs_dataset = utils.get_celeba_dataset( option =0)
            #self.image_plotter(self.true_imgs_dataset)
            self.true_imgs =  tf.placeholder(dtype = np.float32, shape=[None, 64, 64, 1])




    def generator(self):
        z1 = tf.random_normal([self.batch_size, 1, 1,100], mean=0, stddev=0.02)
        filters_start = 512

        with tf.variable_scope("generator"):

            t_conv1 = utils.transpose_conv_bn_relu('t_conv1',z1, filter=int(filters_start), k_size=[4,4], padd="VALID")

            t_conv2 = utils.transpose_conv_bn_relu('t_conv2',t_conv1, filter=int(filters_start/2), k_size=[4,4], stride = [2,2], padd ='SAME')


            t_conv3 = utils.transpose_conv_bn_relu('t_conv3',t_conv2, filter=int(filters_start/4), k_size=[4,4], stride = [2,2], padd ='SAME')


            t_conv4 = utils.transpose_conv_bn_relu('t_conv4',t_conv3, filter=int(filters_start/8), k_size=[4,4], stride = [2,2], padd ='SAME')

            t_conv5 = tf.layers.conv2d_transpose(t_conv4, 1, [4,4], [2,2], 'SAME', activation=None, name ='t_conv5')
            
            self.fake_imgs = tf.nn.tanh(t_conv5, name='fake_imgs')

            
            
            
    def disc_layers(self,x):
        filters_start = 128
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):

            
            #conv1 =utils.conv_bn_leaky_relu('conv1', imgs, filter=int(filters_start), k_size=[4,4], stride=[2,2], padd='SAME')
            
            conv1_without_activation = tf.layers.conv2d(inputs=x,
                                filters=int(filters_start),
                                kernel_size=[4,4],
                                strides=[2,2],
                                padding='SAME', kernel_initializer = tf.random_normal_initializer(0, 0.02), bias_initializer =tf.zeros_initializer(), name='conv1_without_activation')
            conv1 = tf.nn.leaky_relu(conv1_without_activation, name='conv1')
            
            
            
            conv2 =utils.conv_bn_leaky_relu('conv2', conv1, filter=int(filters_start*2), k_size=[4,4], stride=[2,2], padd='SAME')

            conv3 =utils.conv_bn_leaky_relu('conv3', conv2, filter=int(filters_start*4), k_size=[4,4], stride=[2,2], padd='SAME')

            conv4 =utils.conv_bn_leaky_relu('conv4', conv3, filter=int(filters_start*8), k_size=[4,4], stride=[2,2], padd='SAME')

            disc_out = tf.squeeze(tf.layers.conv2d(inputs=conv4,
                                    filters=1,
                                    kernel_size= [4,4],
                                    strides=[1,1],
                                    padding='VALID', activation=None), axis=[2,3])
        return disc_out

    def discriminator(self):
        
        
        disc_out_real = self.disc_layers(self.true_imgs)
        disc_out_fake = self.disc_layers(self.fake_imgs)

        d_r_loss = tf.losses.mean_squared_error(tf.ones_like(disc_out_real), disc_out_real)
        d_f_loss = tf.losses.mean_squared_error(tf.zeros_like(disc_out_fake), disc_out_fake)
        self.disc_loss = (d_r_loss + d_f_loss) / 2.0
        self.gen_loss = tf.losses.mean_squared_error(tf.ones_like(disc_out_fake), disc_out_fake)
   
    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.disc_opt = tf.train.AdamOptimizer(self.lr, beta1= self.beta1).minimize(self.disc_loss,
                                                global_step=self.gstep, var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'))

        self.gen_opt = tf.train.AdamOptimizer(self.lr, beta1= self.beta1).minimize(self.gen_loss,
                                        global_step=self.gstep, var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
        
#         with tf.control_dependencies([self.disc_opt]):
#             self.disc_clip = tf.group(*(tf.assign(var, tf.clip_by_value(var, -self.clip_value, self.clip_value)) 
#                                     for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')))


    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            #tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('loss', self.loss)            
            self.summary_op = tf.summary.merge_all()
    

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        print('loaded data')
        self.generator()
        self.discriminator()
        self.optimize()
        #self.summary()

        
    def train_one_epoch(self, sess, saver, writer, epoch, step):
        start_time = time.time()
        self.training = True
        n_batches = 0
        total_disc_loss = 0
        total_gen_loss =0
  
        for i in range (int( (1*len(self.true_imgs_dataset) )/ self.batch_size)):
            if((i+1) * self.batch_size < len(self.true_imgs_dataset)):
                st = i * self.batch_size
                en = (i+1) * self.batch_size
            else:
                st = i * self.batch_size
                en = len(self.true_imgs_dataset)    
            #print('end:', en)

            ###########################
            ##(1) Update D network: 
            ##########################
            #sess.run(self.disc_assign)
            #for i in range(self.n_critic):
            _, disc_l=  sess.run([self.disc_opt, self.disc_loss], {self.true_imgs: self.true_imgs_dataset[st: en]})


            ###########################
            ##(2) Update G network:
            ##########################
            #sess.run(self.gen_assign)

            _, gen_l=  sess.run([self.gen_opt, self.gen_loss], {self.true_imgs: self.true_imgs_dataset[st: en]})

            #writer.add_summary(summaries, global_step=step)
            if (step + 1) % self.skip_step == 0:
                print('step: {0}, Discriminator Loss: {1} generator Loss: {2}'.format(step, disc_l, gen_l))
                fake_out = sess.run(self.fake_imgs)
                self.image_plotter(fake_out, 2)               

            step += 1
            total_disc_loss += disc_l
            total_gen_loss += gen_l

            n_batches += 1
            
        train_disc_loss.append(total_disc_loss/n_batches)   
        train_gen_loss.append(total_gen_loss/n_batches) 

        return step


    

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        
        utils.safe_mkdir( '../../checkpoints')
        utils.safe_mkdir('../../checkpoints/convnet_layers')
        writer = tf.summary.FileWriter('../.././graphs/convnet_layers', tf.get_default_graph())
        print('starting to train')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
#             # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/cifar-convnet1'))
#             # if ckpt and ckpt.model_checkpoint_path:
#             #     saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, writer, epoch, step)
#                 #self.eval_once(sess, self.test_init, writer, epoch, step)
            fake_out = sess.run(self.fake_imgs)              
            self.image_plotter(fake_out, self.no_display_imgs)               
                      
        writer.close()
        


if __name__ == '__main__':
    tf.reset_default_graph()
    model = ConvNet()
    model.build()
    print('built model')    
    #print(model.true_imgs_dataset[0,:,:,0])
    model.train(n_epochs=10)
    print('done')
