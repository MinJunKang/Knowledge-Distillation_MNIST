import tensorflow as tf
import numpy as np
import os
import shutil
import warnings
import math

# for no warning sign
warnings.filterwarnings('ignore')

# for plot the graph
import matplotlib.pyplot as plt

# Check os type
if(os.name == 'nt' or os.name == 'posix'): # windows or ubuntu linux
    import tensorflow as tf
    from tensorflow.contrib.keras import backend as K
    from tensorflow.contrib.keras import layers as l
    from tensorflow.contrib.keras import models as m
    from tensorflow.contrib.keras import optimizers as op
    from tensorflow.contrib.keras import callbacks as call
    from tensorflow.contrib.keras import regularizers as rz
    from tensorflow.contrib.keras import initializers as lz
    from tensorflow.contrib.keras import utils as ut
    from tensorflow.contrib.keras import applications as ap
else:
    print('The os type is not windows or linux. This code supports only windows and linux')
    exit(1)

print('Checking os type is done!! Start Program')
#C:/Users/minjun/Desktop/
# Dir to save the model
dir_tmp = "./tmp/"
dir_best = "./best/"
dir_log = "./log/"
dir_summary = os.path.join(dir_tmp,"summary/")
model_name = "DNN_model"
result_file = "Result_log.txt"
summary_file = "model_summary"
softy_file = "soft_y.pickle"


class Small_DNN(object):
    def __init__(self,model_type):
        self.model_type = model_type
        
        # Data for Training
        self.train_x = None
        self.train_y = None

        # Data for Test
        self.valid_x = None
        self.valid_y = None

        # Data of soft_targets
        self.soft_targets = None

        # State Checking
        self.trainable = False
        self.evaluable = False

        self.num_input = []
        self.num_output =[]

    def get_batch_data(self,x, y, batch):
        if(batch > len(x)):
            batch = len(x)
        batch_xs = []
        batch_ys = []

        num_part = int(len(x) / batch)

        tmpx = []
        tmpy = []
        for i in range(len(x)):
            tmpx.append(x[i,:])
            tmpy.append(y[i,:])
            if((i+1) % batch == 0):
                batch_xs.append(np.asarray(tmpx))
                batch_ys.append(np.asarray(tmpy))
                tmpx = []
                tmpy = []
        if(len(tmpx)!=0):
            batch_xs.append(np.asarray(tmpx))
            batch_ys.append(np.asarray(tmpy))
            tmpx = []
            tmpy = []
        return batch_xs, batch_ys

    # Initialize the weight and bias
    def get_stddev(self,in_dim, out_dim):
        return 1.3 / math.sqrt(float(in_dim) + float(out_dim))

    # You have to modify this part!!
    def Build_Model(self,hidden_units=[256,256]):

        input_dim = np.shape(self.num_input)[0]
        output_dim = np.shape(self.num_output)[0]
        
        input_dim = np.cast[int](input_dim)
        output_dim = np.cast[int](output_dim)

        input_shape = [None]
        output_shape = [None]

        hidden = []

        for i in range(input_dim):
            input_shape.append(self.num_input[i])
        for i in range(output_dim):
            output_shape.append(self.num_output[i])
        
        # For just training and test the model
        self.X = tf.placeholder(tf.float32, input_shape, name="%s_%s" % (self.model_type, "xinput"))
        self.Y = tf.placeholder(tf.float32, output_shape, name="%s_%s" % (self.model_type, "yinput"))
        # Select Model to train model with soft_Y
        self.flag = tf.placeholder(tf.bool, None, name="%s_%s" % (self.model_type, "flag"))
        self.soft_Y = tf.placeholder(tf.float32, output_shape, name="%s_%s" % (self.model_type, "softy"))
        # Temperature
        self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemperature"))
      

        # Calculate input dimension
        dim_input = []
        sum_dim = 1
        for i in range(input_dim):
            dim_input.append(self.num_input[i])
            sum_dim = sum_dim * self.num_input[i]
        weight_dim = dim_input
        weight_dim.append(hidden_units[0])

        with tf.name_scope(name = "%s_input_layer" % (self.model_type)):

            weights = tf.Variable(tf.truncated_normal(weight_dim,stddev=self.get_stddev(sum_dim, hidden_units[0])),name="%s" % ("input_weight"))
            bias = tf.Variable(tf.zeros([hidden_units[0]]),name="%s" % ("input_biases"))
            input = tf.matmul(self.X ,weights) + bias

        
        # hidden layer
        with tf.name_scope(name = "%s_hidden_layer" % (self.model_type)):
            for index, num_hidden in enumerate(hidden_units):
                if index == len(hidden_units) - 1: break
                with tf.name_scope("hidden{}".format(index+1)):
                    weights = tf.Variable(tf.truncated_normal([num_hidden, hidden_units[index+1]], stddev=self.get_stddev(num_hidden, hidden_units[index+1])), name='weights')
                    biases = tf.Variable(tf.zeros([hidden_units[index+1]]), name='biases')
                    inputs = input if index == 0 else hidden[index-1]
                    hidden.append(tf.nn.relu(tf.matmul(inputs, weights) + biases, name="hidden{}".format(index+1)))
        

        
        # Calculate output dimension
        dim_output = []
        weight_dim = [ hidden_units[-1] ]
        sum_dim = 1
        for i in range(output_dim):
            dim_output.append(self.num_output[i])
            sum_dim = sum_dim * self.num_output[i]
            weight_dim.append(self.num_output[i])
            

        # output layer
        with tf.name_scope(name = "%s_output_layer" % (self.model_type)):
            weights = tf.Variable(tf.truncated_normal(weight_dim, stddev=self.get_stddev(hidden_units[-1], sum_dim)), name='weights')
            biases = tf.Variable(tf.zeros(dim_output), name='biases')
            # output
            if(len(hidden) == 0):
                logits = tf.matmul(tf.nn.relu(input), weights) + biases
            else:
                logits = tf.matmul(hidden[-1], weights) + biases

        return logits

    def Data_Prepare(self,Train_x,Train_y,Valid_x,Valid_y,Soft_targets = None,Soft_targets_v = None):
        self.train_x = Train_x
        self.train_y = Train_y
        self.valid_x = Valid_x
        self.valid_y = Valid_y
        self.soft_targets = Soft_targets
        self.soft_targets_v = Soft_targets_v

        if(self.train_x == [] or self.valid_x == [] or self.train_y == [] or self.valid_y == []):
            print("Data is empty!!\n")
            # Initialize the parameter
            self.train_x = None
            self.valid_x = None
            self.train_y = None
            self.valid_y = None
            return
        if(self.train_x.shape[0] != self.train_y.shape[0] or self.valid_x.shape[0] != self.valid_y.shape[0]):
            print("Data num is not matched between x and y!!\n")
            # Initialize the parameter
            self.train_x = None
            self.valid_x = None
            self.train_y = None
            self.valid_y = None
            return

        # Calculate number of data of train and test
        self.num_train = self.train_x.shape[0]
        self.num_test = self.valid_x.shape[0]

        # Calculate input_dim and output_dim
        self.num_input = []
        self.num_output =[]

        for i in range(np.shape(self.train_x.shape)[0]-1):
            self.num_input.append(self.train_x.shape[i+1])
        for i in range(np.shape(self.train_y.shape)[0]-1):
            self.num_output.append(self.train_y.shape[i+1])

        # Set trainable variable to True
        self.trainable = True

    def loss(self,pred):

        if(self.Objective == "classifier"):
            loss_op_standard = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=pred, labels=self.Y))

            total_loss = loss_op_standard

            loss_op_soft = tf.cond(self.flag,
                                        true_fn=lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits=pred / self.softmax_temperature, labels=self.soft_Y)),
                                        false_fn=lambda: 0.0)

            total_loss += tf.square(self.softmax_temperature) * loss_op_soft
        else:
            loss_op_soft = tf.cond(self.flag,
                                        true_fn=lambda: tf.reduce_mean(tf.square(pred / self.softmax_temperature - self.soft_Y)),
                                        false_fn=lambda: 0.0)
            total_loss = tf.reduce_mean(tf.square(pred - self.Y)) + tf.square(self.softmax_temperature) * loss_op_soft

        return total_loss

    def Train(self,args,teacher_flag = False):
        if(self.trainable == False):
            print("Please call Data Prepare function first!! Data is not prepared\n")
            return
        if(teacher_flag == True and (self.soft_targets == [] or self.soft_targets_v == [])):
            print("We need teacher's soft_y data for knowledge distillation to student model")
            return

        # Initialize the parameter
        self.epoch = args.epoch

        self.min_batch_size = args.batch_size
        self.batch_size_increment = 2
        self.max_batch_size = 512

        self.min_learning_rate = args.start_learning_rate
        self.learning_rate_increment = args.learning_rate_increment
        self.max_learning_rate = 0.1

        self.Objective = args.Objective
        self.temperature = args.temperature
        self.max_overfit = args.max_overfit
        self.last_epoch = 0

        self.min_node = args.start_node
        self.node_increment = args.node_increment
        self.max_node = args.max_node

        self.min_layer = 1
        self.max_layer = 4

        temperature = args.temperature

        f = open(os.path.join(dir_log,result_file),'w')
        f.write("Temperature : %f\n" % self.temperature)
        f.close()

        # Change the structure
        for layer_num in range(self.min_layer,self.max_layer):
            hidden_node = []
            for layer in range(layer_num):
                hidden_node.append(self.min_node)
            for layer in range(layer_num):
                while(hidden_node[layer] <= self.max_node):

                    f = open(os.path.join(dir_log,result_file),'a')
                    result=self.Optimize(hidden_node,teacher_flag)
                    f.write("-------------------------------\n")
                    f.write("Model : [")
                    for i in range(len(hidden_node)):
                        f.write("%d  "%hidden_node[i])
                    f.write("],     Training_info : [")
                    f.write("learning_rate : %f     "%result[0])
                    f.write("batch_size : %d    " % result[1])
                    f.write("Last epoch : %d]\n" % self.last_epoch)
                    f.write("Result(accuracy) : %f\n" % result[2])
                    f.write("-------------------------------\n")
                    f.close()
                    if(result[2] == 1.00):
                        print("Found the best model!! Finish the training")
                        return

                    hidden_node[layer] *= self.node_increment

    def Optimize(self,hidden_node,teacher_flag):
        rate = self.min_learning_rate
        result = 0
        data = []
        data.append(rate)
        data.append(self.min_batch_size)
        data.append(result)

        while(rate <= self.max_learning_rate):
            batch = self.min_batch_size
            while(batch <= self.max_batch_size):
                tmp_result = self.Run(hidden_node,rate,batch,result,teacher_flag)
                if(result < tmp_result):
                    data[0] = rate
                    data[1] = batch
                    data[2] = tmp_result
                    result = tmp_result
                batch *= self.batch_size_increment
                if(result == 1.00):
                    return data

            rate *= self.learning_rate_increment
        return data

    def Run(self,hidden_node,learning_rate,batch_size,result,teacher_flag):

        sess = tf.Session()

        # Build Model
        logits = self.Build_Model(hidden_node)
        if(self.Objective == "classifier"):
            prediction = tf.nn.softmax(logits)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(self.Y,1)),"float32"),name = "Accuracy_classifier")
        else:
            prediction = logits
            accuracy = tf.reduce_mean(tf.square(prediction - self.Y,name = "squared_accuracy"),name = "Accuracy_regression")

        loss = self.loss(logits)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Save the model
        saver = tf.train.Saver()

        # Generate batch data
        batch_x,batch_y = self.get_batch_data(self.train_x,self.train_y,batch_size)
        
        if teacher_flag:
            _,soft_targets = self.get_batch_data(self.train_x,self.soft_targets,batch_size)
        else:
            soft_targets = batch_y
            soft_targets_v = self.valid_y

        test_acc_save = 0;

        # Initialize parameter
        sess.run(tf.global_variables_initializer())

        overfit_count = 0
        last_epoch = 0

        for i in range(self.epoch):
            # Training
            loss_tank = []
            for j in range(len(batch_x)):
                _, loss_value, acc,logit = sess.run([train_op,loss,accuracy,logits],feed_dict = {self.X : batch_x[j],
                                                                      self.Y : batch_y[j],
                                                                      self.flag : teacher_flag,
                                                                      self.soft_Y : soft_targets[j],
                                                                      self.softmax_temperature: self.temperature}
                                         )
                loss_tank.append(acc)
            # Test
            loss_value, predict,test_acc = sess.run([loss,prediction,accuracy],feed_dict = {self.X : self.valid_x,
                                                                          self.Y : self.valid_y,
                                                                          self.flag : teacher_flag,
                                                                          self.soft_Y : self.soft_targets_v,
                                                                          self.softmax_temperature: self.temperature}
                                           )
            # Evaulate the accuracy and compare with the previous result to find stop point
            if(test_acc > test_acc_save):
                test_acc_save = test_acc
                # Take the best result
                saver.save(sess, os.path.join(dir_tmp,model_name))

                overfit_count = 0
                print("The Local Model info : ")
                print("Structure : ",hidden_node)
                print("Train_parameter = learning_rate : ",learning_rate,"    batch_size : ",batch_size)
                print("Step : %d    Test_accuracy : %f      Test_loss : %f"% (i+1,test_acc,loss_value))
                print("Train_accuracy : %f",np.mean(np.asarray(loss_tank)))
                print("-------------------------------")
            overfit_count += 1

            # Count the overfit number and check if it is stop point
            if(overfit_count > self.max_overfit):
                print("Overfitted! Go to the next Model!")
                last_epoch = i+1
                break

        # if the model has better result than before, save the model

        if(result < test_acc_save):
            for file in os.listdir(dir_tmp):
                src_file = os.path.join(dir_tmp,file)
                dst_file = os.path.join(dir_best,file)
                shutil.copy(src_file,dst_file)
            self.last_epoch = last_epoch
            distil_writer = tf.summary.FileWriter(dir_log + model_name,sess.graph)
        
        # Close the session
        sess.close()

        # return the result
        return test_acc_save
    def predict(self,args,test_x,test_y,soft_y,flag_):

        with tf.Session() as sess:
            if not os.path.isfile(os.path.join(dir_best,model_name+".meta")):
                print("There is no saved meta file")
                exit(1)

            new_saver = tf.train.import_meta_graph(os.path.join(dir_best,model_name+".meta"))
            new_saver.restore(sess,tf.train.latest_checkpoint(dir_best))
    

            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name("%s_%s" % (self.model_type,"xinput:0"))
            Y = graph.get_tensor_by_name("%s_%s" % (self.model_type,"yinput:0"))
            flag = graph.get_tensor_by_name("%s_%s" % (self.model_type,"flag:0"))
            soft_Y = graph.get_tensor_by_name("%s_%s" % (self.model_type, "softy:0"))
            softmax_temperature = graph.get_tensor_by_name("%s_%s" % (self.model_type, "softmaxtemperature:0"))
            new_prec = graph.get_tensor_by_name("Accuracy_classifier:0")
    
            
            test_prec = sess.run(new_prec,feed_dict={X: test_x,
                                                     Y: test_y, 
                                                     flag: flag_,
                                                     soft_Y: soft_y,
                                                     softmax_temperature: args.temperature})
            print('test_prec: ',test_prec)
































# Deprecated Model

class SmallModel:
    def __init__(self, args, model_type):
        self.learning_rate = 0.001
        self.num_steps = 500
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.n_hidden_1 = 256  # 1st layer number of neurons
        self.n_hidden_2 = 256  # 2nd layer number of neurons
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.temperature = args.temperature
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "smallmodel"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        self.max_checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + "max")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1]),
                              name="%s_%s" % (self.model_type, "h1")),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]),
                              name="%s_%s" % (self.model_type, "h2")),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.num_classes]),
                               name="%s_%s" % (self.model_type, "out")),
            'linear': tf.Variable(tf.random_normal([self.num_input, self.num_classes]),
                                  name="%s_%s" % (self.model_type, "linear"))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1]), name="%s_%s" % (self.model_type, "b1")),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2]), name="%s_%s" % (self.model_type, "b2")),
            'out': tf.Variable(tf.random_normal([self.num_classes]), name="%s_%s" % (self.model_type, "out")),
            'linear': tf.Variable(tf.random_normal([self.num_classes]), name="%s_%s" % (self.model_type, "linear"))
        }

        self.build_model()

        self.saver = tf.train.Saver()

    # Create model
    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))

        self.flag = tf.placeholder(tf.bool, None, name="%s_%s" % (self.model_type, "flag"))
        self.soft_Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "softy"))
        self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemperature"))

        with tf.name_scope("%sfclayer" % (self.model_type)), tf.variable_scope("%sfclayer" % (self.model_type)):
            # Hidden fully connected layer with 256 neurons
            layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
            # # Hidden fully connected layer with 256 neurons
            layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
            # # Output fully connected layer with a neuron for each class
            logits = (tf.matmul(layer_2, self.weights['out']) + self.biases['out'])
            #logits = tf.add(tf.matmul(self.X, self.weights['linear']), self.biases['linear'])

        with tf.name_scope("%sprediction" % (self.model_type)), tf.variable_scope("%sprediction" % (self.model_type)):
            self.prediction = tf.nn.softmax(logits)

            self.correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with tf.name_scope("%soptimization" % (self.model_type)), tf.variable_scope(
                        "%soptimization" % (self.model_type)):
            # Define loss and optimizer
            self.loss_op_standard = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.Y))

            self.total_loss = self.loss_op_standard

            self.loss_op_soft = tf.cond(self.flag,
                                        true_fn=lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits=logits / self.softmax_temperature, labels=self.soft_Y)),
                                        false_fn=lambda: 0.0)

            self.total_loss += tf.square(self.softmax_temperature) * self.loss_op_soft

            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(0.05)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss)

        with tf.name_scope("%ssummarization" % (self.model_type)), tf.variable_scope(
                        "%ssummarization" % (self.model_type)):
            tf.summary.scalar("loss_op_standard", self.loss_op_standard)
            tf.summary.scalar("total_loss", self.total_loss)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Merge all summaries into a single op

            # If using TF 1.6 or above, simply use the following merge_all function
            # which supports scoping
            # self.merged_summary_op = tf.summary.merge_all(scope=self.model_type)

            # Explicitly using scoping for TF versions below 1.6

            def mymergingfunction(scope_str):
                with tf.name_scope("%s_%s" % (self.model_type, "summarymerger")), tf.variable_scope(
                                "%s_%s" % (self.model_type, "summarymerger")):
                    from tensorflow.python.framework import ops as _ops
                    key = _ops.GraphKeys.SUMMARIES
                    summary_ops = _ops.get_collection(key, scope=scope_str)
                    if not summary_ops:
                        return None
                    else:
                        return tf.summary.merge(summary_ops)

            self.merged_summary_op = mymergingfunction(self.model_type)

    def start_session(self):
        self.sess = tf.Session()

    def close_session(self):
        self.sess.close()

    def train(self, dataset, teacher_model=None):
        teacher_flag = False
        if teacher_model is not None:
            teacher_flag = True

        # Initialize the variables (i.e. assign their default value)
        self.sess.run(tf.global_variables_initializer())
        train_data = dataset.get_train_data()
        train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

        max_accuracy = 0

        print("Starting Training")

        def dev_step():
            validation_x, validation_y = dataset.get_validation_data()
            loss, acc = self.sess.run([self.loss_op_standard, self.accuracy], feed_dict={self.X: validation_x,
                                                                                         self.Y: validation_y,
                                                                                         # self.soft_Y: validation_y,
                                                                                         self.flag: False,
                                                                                         self.softmax_temperature: 1.0})

            if acc > max_accuracy:
                save_path = self.saver.save(self.sess, self.checkpoint_path)
                print("Model Checkpointed to %s " % (save_path))

            print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))

        for step in range(1, self.num_steps + 1):
            batch_x, batch_y = train_data.next_batch(self.batch_size)
            soft_targets = batch_y
            if teacher_flag:
                soft_targets = teacher_model.predict(batch_x, self.temperature)

            # self.sess.run(self.train_op,
            _, summary = self.sess.run([self.train_op, self.merged_summary_op],
                                       feed_dict={self.X: batch_x,
                                                  self.Y: batch_y,
                                                  self.soft_Y: soft_targets,
                                                  self.flag: teacher_flag,
                                                  self.softmax_temperature: self.temperature}
                                       )
            if (step % self.display_step) == 0 or step == 1:
                dev_step()
        else:
            # Final Evaluation and checkpointing before training ends
            dev_step()

        train_summary_writer.close()

        print("Optimization Finished!")

    def predict(self, data_X, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.X: data_X, self.flag: False, self.softmax_temperature: temperature})

    def run_inference(self, dataset):
        test_images, test_labels = dataset.get_test_data()
        print("Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                                           self.Y: test_labels,
                                                                           # self.soft_Y: test_labels,
                                                                           self.flag: False,
                                                                           self.softmax_temperature: 1.0
                                                                           }))

    def load_model_from_file(self, load_path):
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
