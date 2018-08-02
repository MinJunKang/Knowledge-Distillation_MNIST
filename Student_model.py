
import Header

# No Dropout, Simple DNN Structure
class Student_model(object):
    def __init__(self, args, model_name):
        # Flag info
        self.flag = False

        self.model_name = model_name

        # Define Argument
        self.Define_Arg(args)

        # Training Parameters
        self.num_parameter = 0

        # To determine if it is trainable
        self.trainable = False

        self.last_epoch = 0

    def Define_Arg(self,args):
        # Temperature
        self.temperature = args.temperature

        # Learning parameter
        self.min_learning_rate = args.min_learning_rate
        self.learning_rate_increment = args.learning_rate_increment
        self.max_learning_rate = args.max_learning_rate
        self.min_batch = args.min_batch
        self.batch_increment = args.batch_increment
        self.max_batch = args.max_batch

        # Structure parameter
        self.min_layer = args.min_layer
        self.layer_increment = args.layer_increment
        self.max_layer = args.max_layer

        self.min_node = args.min_node
        self.node_increment = args.node_increment
        self.max_node = args.max_node
        
        # Set epoch and stop point criterion
        self.max_epoch = args.max_epoch
        self.max_overfit = args.max_overfit

        # model type
        self.model_type = args.model_type

        # Saving place
        self.dir = args.dir
        self.checkpoint_dir = args.checkpoint
        self.final_dir = args.final
        self.log_dir =args.log
        self.model_dir = args.model
        self.tmp_dir = args.tmp

        self.checkpoint_file = Header.os.path.join(self.checkpoint_dir,"checkpoint_weight.h5")
        self.final_file = Header.os.path.join(self.final_dir,"trained_weight.h5")
        self.log_file_1 = Header.os.path.join(self.log_dir,"Result_model.txt")
        self.model_file = Header.os.path.join(self.model_dir,"model.json")
        self.tmp_file_1 = Header.os.path.join(self.tmp_dir,"model.json")
        self.tmp_file_2 = Header.os.path.join(self.tmp_dir,"trained_weight.h5")
        self.tmp_file_3 = Header.os.path.join(self.tmp_dir,"info.txt")

    # Initialize the weight and bias
    def get_stddev(self,in_dim, out_dim):
        return 1.3 / Header.math.sqrt(float(in_dim) + float(out_dim))

    # Loss function
    def loss(self,y_true,y_pred):
        if(self.model_type == "classifier"):
            pred = Header.K.softmax(y_pred)

            total_loss = Header.tf.reduce_mean(Header.K.categorical_crossentropy(y_true,pred))

            loss_op_soft = Header.tf.cond(Header.tf.constant(self.flag),
                                        true_fn=lambda: Header.tf.reduce_mean(Header.K.categorical_crossentropy(
                                           y_true, Header.K.softmax(y_pred / self.temperature))),
                                        false_fn=lambda: 0.0)
        else:
            pred = Header.K.tanh(y_pred)

            total_loss = Header.tf.reduce_mean(Header.tf.square(pred - y_true))

            loss_op_soft = Header.tf.cond(Header.tf.constant(self.flag),
                                        true_fn=lambda: Header.tf.reduce_mean(Header.tf.square(Header.K.tanh(y_pred/ self.temperature) - y_true)),
                                        false_fn=lambda: 0.0)

        total_loss += Header.tf.square(self.temperature) * loss_op_soft

        return total_loss

    def Data_Prepare(self,Train_x=[],Train_y=[],Valid_x=[],Valid_y=[],soft_targets_t = []):
        self.train_x = Train_x
        self.train_y = Train_y
        self.valid_x = Valid_x
        self.valid_y = Valid_y
        self.soft_y = soft_targets_t

        if(self.soft_y == []):
            self.flag = False
        else:
            self.flag = True

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
        num_input = []
        num_output =[]

        for i in range(Header.np.shape(self.train_x.shape)[0]-1):
            num_input.append(self.train_x.shape[i+1])
        for i in range(Header.np.shape(self.train_y.shape)[0]-1):
            num_output.append(self.train_y.shape[i+1])

        self.input_shape =[]
        self.output_shape =[]
        self.in_dim = 1
        self.out_dim = 1

        for i in range(Header.np.shape(num_input)[0]):
            self.input_shape.append(num_input[i])
            self.in_dim *= num_input[i]
        for i in range(Header.np.shape(num_output)[0]):
            self.output_shape.append(num_output[i])
            self.out_dim *= num_output[i]

        # Set trainable variable to True
        self.trainable = True

    def Load_Text(self,dst_file):
        contents = []
        with open(dst_file,'rb') as myloaddata:
            contents = Header.pickle.load(myloaddata)
        return contents

    def Build_DNN(self,hidden_units=[256,256]):

        soft_y = Header.l.Input(shape=self.output_shape,name="%s_soft_y" % self.model_name)

        model = Header.m.Sequential(name = "%s_DNN_Model" % self.model_name)

        # Hidden Layer 1
        model.add(Header.l.Dense(hidden_units[0],activation = "relu",kernel_initializer=Header.lz.TruncatedNormal(mean=0.0, stddev=self.get_stddev(self.in_dim,hidden_units[0]), seed=Header.seed),name = "Hidden_1",input_shape = self.input_shape))

        # Hidden Layer 2 ~ N
        for i in range(1,len(hidden_units)):
            model.add(Header.l.Dense(hidden_units[i],activation = "relu",kernel_initializer=Header.lz.TruncatedNormal(mean=0.0, stddev=self.get_stddev(hidden_units[i-1],hidden_units[i]), seed=Header.seed),name = "Hidden_%s" % str(i+1)))
        
        # Last Output Layer
        model.add(Header.l.Dense(self.output_shape[0],activation=None,kernel_initializer=Header.lz.TruncatedNormal(mean=0.0, stddev=self.get_stddev(hidden_units[-1],self.out_dim), seed=Header.seed),name = "Output"))
        
        model = Header.m.Model([model.input,soft_y],model.output)

        return model

    def Train(self,args,teacher_flag = False):

        # Define Argument
        self.Define_Arg(args)

        if(self.trainable == False):
            print("Please call Data Prepare function first!! Data is not prepared\n")
            return
        if(teacher_flag == True and self.soft_y == []):
            print("We need teacher's soft_y data for knowledge distillation to student model")
            return

        f = open(self.log_file_1,'w')
        f.write("Temperature : %f\n" % self.temperature)
        f.close()
        
        # Change the structure
        for layer_num in range(self.min_layer,self.max_layer+1):
            hidden_node = []
            for layer in range(layer_num):
                hidden_node.append(self.min_node)
            for layer in range(layer_num):
                if(layer != 0):
                    hidden_node[layer] *= self.node_increment
                while(hidden_node[layer] <= self.max_node):
                    f = open(self.log_file_1,'a')
                    result=self.Optimize(hidden_node,teacher_flag)
                    f.write("-------------------------------\n")
                    f.write("Model : [")
                    for i in range(len(hidden_node)):
                        f.write("%d  "%hidden_node[i])
                    f.write("]\nTraining_info : [")
                    f.write("learning_rate : %f     "%result[0])
                    f.write("batch_size : %d    " % result[1])
                    f.write("Last epoch : %d]\n" % self.last_epoch)
                    f.write("Train accuracy : %f        Validation accuracy : %f\n" % (result[3],result[2]))
                    f.write("Number of parameter : %d\n" % self.num_parameter)
                    f.write("-------------------------------\n")
                    f.close()
                    hidden_node[layer] *= self.node_increment
                    if(result[2] == 1.00):
                        print("Found the best model!! Finish the training")
                        return
                hidden_node[layer] = int(hidden_node[layer] / self.node_increment)


    def Optimize(self,hidden_node,teacher_flag):
        rate = self.min_learning_rate
        result = 0
        data = []
        data.append(rate)
        data.append(self.min_batch)
        data.append(result) # Test accuracy
        data.append(result) # Train accuracy

        while(rate <= self.max_learning_rate):
            batch = self.min_batch
            while(batch <= self.max_batch):
                [train_result,tmp_result] = self.Run(hidden_node,rate,batch,result,teacher_flag)
                if(result < tmp_result):
                    data[0] = rate
                    data[1] = batch
                    data[2] = tmp_result
                    data[3] = train_result
                    result = tmp_result
                batch *= self.batch_increment
                if(result == 1.00):
                    return data

            rate *= self.learning_rate_increment
        return data

    def acc_model(self,y_true,y_pred):
        if(self.model_type == "classifier"):
            prediction = Header.K.softmax(y_pred)
            accuracy = Header.tf.reduce_mean(Header.tf.cast(Header.tf.equal(Header.tf.argmax(prediction,1),Header.tf.argmax(y_true,1)),"float32"),name = "Accuracy_classifier")
        else:
            accuracy = Header.tf.reduce_mean(Header.tf.square(y_pred - y_true,name = "squared_accuracy"),name = "Accuracy_regression")
        return accuracy

    # Still Programming
    def Run(self,hidden_node,learning_rate,batch_size,result,teacher_flag):

        # Build Model
        model = self.Build_DNN(hidden_node)

        if teacher_flag:
            soft_targets = self.soft_y
            self.flag = True
        else:
            soft_targets = self.train_y
            self.flag = False

        test_acc_save = 0
        last_epoch = 0

        # compile model
        adam = Header.op.Adam(lr=learning_rate,clipvalue=1.5)

        model.compile(adam,loss=self.loss,metrics=[self.acc_model])

        # Model information
        model.summary()
        self.num_parameter = int(Header.np.sum([Header.K.count_params(p) for p in set(model.trainable_weights)]))

        checkpointer = Header.call.ModelCheckpoint(filepath=self.checkpoint_file,monitor = "val_acc_model", verbose=1, save_best_only=True, save_weights_only=True)
        earlyStopping=Header.call.EarlyStopping(monitor='val_acc_model', patience=self.max_overfit, verbose=0, mode='auto')

        history = model.fit([self.train_x,soft_targets],self.train_y,batch_size = batch_size,epochs = self.max_epoch,verbose=2,validation_data=([self.valid_x,self.valid_y],self.valid_y),callbacks=[checkpointer,earlyStopping])
        
        # Get the value
        test_acc_save = Header.np.max(history.history['val_acc_model'])
        last_epoch = Header.np.argmax(history.history['val_acc_model'])+1
        test_loss_save = history.history['val_loss'][last_epoch-1]
        train_acc_save = history.history['acc_model'][last_epoch-1]
        train_loss_save = history.history['loss'][last_epoch-1]
        
        print("\n\nTrained Model Structure : ")
        print("Structure : ",hidden_node, "     Number of parameter : ",self.num_parameter)
        print("Train_parameter = learning_rate : ",learning_rate,"    batch_size : ",batch_size)
        print("Last Epoch : %d" % last_epoch)
        print("Validation Accuracy : ",test_acc_save,"      Validation loss : ",test_loss_save)
        print("Trainset Accuracy : ",train_acc_save,"       Trainset loss : ",train_loss_save)

        if(result < test_acc_save):
            self.last_epoch = last_epoch
            model.save(self.model_file)
            Header.shutil.copy(self.checkpoint_file,self.final_file)

        # return the result
        return [train_acc_save,test_acc_save]
    def Inference(self,test_x,test_y,soft_y = []):
        if(Header.os.path.isfile(self.model_file) == False):
            print("Train the student model first!!")
            exit(1)
        loaded_model_json = Header.m.load_model(self.model_file,custom_objects = {'loss':self.loss,'acc_model':self.acc_model})

         # Load Model
        loaded_model = Header.m.model_from_json(loaded_model_json.to_json())
        loaded_model.load_weights(self.final_file)

        loaded_model.compile(loss = self.loss, optimizer = 'adam',metrics = [self.acc_model])
        loaded_model.summary()
        self.num_parameter = int(Header.np.sum([Header.K.count_params(p) for p in set(loaded_model.trainable_weights)]))
        
        if(soft_y != []):
            soft_t = soft_y
        else:
            soft_t = test_y

        return loaded_model.evaluate(test_x,test_y,verbose = 3)

    def Num_Parameter(self):
        if(self.num_parameter != 0):
            return self.num_parameter

    def Fit_SingleModel(self,train_x,train_y,valid_x,valid_y,test_x,test_y,hidden_node = [256,256],learn_rate = 0.001,batch = 128,soft = []):
        # Prepare data
        self.Data_Prepare(train_x,train_y,valid_x,valid_y,soft)
        
        # Build Model
        model = self.Build_DNN(hidden_node)

        soft_target = train_y

        if(soft == []):
            flag_point = False
        else:
            flag_point = True
            soft_target = soft

        # compile model
        adam = Header.op.Adam(lr=learn_rate,clipvalue=1.5)

        model.compile(adam,loss=self.loss,metrics=[self.acc_model])

        # Model information
        model.summary()
        parameter_num = int(Header.np.sum([Header.K.count_params(p) for p in set(model.trainable_weights)]))
        self.num_parameter = parameter_num

        checkpointer = Header.call.ModelCheckpoint(filepath=self.tmp_file_2,monitor = "val_acc_model", verbose=1, save_best_only=True, save_weights_only=True)
        earlyStopping=Header.call.EarlyStopping(monitor='val_acc_model', patience=self.max_overfit, verbose=0, mode='auto')

        history = model.fit([train_x,soft_target],train_y,batch_size = batch,epochs = self.max_epoch,verbose=2,validation_data=([valid_x,valid_y],valid_y),callbacks=[checkpointer,earlyStopping])

        # Get the value
        valid_acc_save = Header.np.max(history.history['val_acc_model'])
        last_epoch = Header.np.argmax(history.history['val_acc_model'])+1
        valid_loss_save = history.history['val_loss'][last_epoch-1]
        train_acc_save = history.history['acc_model'][last_epoch-1]
        train_loss_save = history.history['loss'][last_epoch-1]

        # Save model
        model.save(self.tmp_file_1)

        # Inference
        result = model.evaluate([test_x,test_y],test_y,verbose = 3)
        test_acc_save = result[1]

        # Result view
        model.summary()

        # Save the result to txt file
        with open(self.tmp_file_3,'wt') as f:
            f.write("Temperature : %f\n" % self.temperature)
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("----------------------------------------\n")
            f.write("Training Parameter Information : \n")
            f.write("learning_rate : %f     batch_size : %d\n" % (learn_rate,batch))
            f.write("Last Epoch : %d\n" % last_epoch)
            f.write("----------------------------------------\n")
            f.write("Train Accuracy : %f    Train loss : %f\n" % (train_acc_save,train_loss_save))
            f.write("Valid Accuracy : %f    Test Accuracy : %f\n" % (valid_acc_save,test_acc_save))
            f.write("----------------------------------------\n")
        return [test_acc_save,parameter_num]

    def Temp_file3_add(self,data):
        with open(self.tmp_file_3,'at') as f:
            f.write("Complexity_gain(Teacher params / Student params) : %f\n" % data[0])
            f.write("Accuracy_gain(Teacher acc - Student acc) : %f\n" % data[1])
            f.write("Overall_gain(Complexity_gain / Accuracy_gain) : %f\n" % data[2])

    def View_Temp_log(self):
        if(Header.os.path.isfile(self.tmp_file_3) == False):
            print("You must first train the temporary student model to get spec")
            return False
        else:
            with open(self.tmp_file_3) as f:
                for line in f:
                    print(line)
            return True

    def View_Train_log(self):
        if(Header.os.path.isfile(self.log_file_1) == False):
            print("You must first train the student model to get train log")
            return False
        else:
            with open(self.log_file_1) as f:
                for line in f:
                    print(line)
            return True

