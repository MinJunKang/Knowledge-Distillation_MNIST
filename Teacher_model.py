
import Header

class Teacher_model(object):
    def __init__(self, args, model_name):

        # model name
        self.model_name = model_name

        # Define Args
        self.Define_Args(args)

        # Training Parameters
        self.num_parameter = 0

        self.test_loss = []
        self.train_loss = []
    def Define_Args(self,args):
        # Temperature
        self.temperature = args.temperature

        # Learning parameter
        self.min_learning_rate = args.min_learning_rate
        self.learning_rate_increment = args.learning_rate_increment
        self.max_learning_rate = args.max_learning_rate
        self.min_batch = args.min_batch
        self.batch_increment = args.batch_increment
        self.max_batch = args.max_batch
        
        # Set epoch and stop point criterion
        self.max_epoch = args.max_epoch
        self.max_overfit = args.max_overfit

        # model type
        self.model_type = args.model_type

        # dropout rate
        # self.dropout = args.dropout

        # Saving place
        self.dir = args.dir
        self.checkpoint_dir = args.checkpoint
        self.final_dir = args.final
        self.log_dir =args.log
        self.model_dir = args.model

        self.checkpoint_file = Header.os.path.join(self.checkpoint_dir,"checkpoint_weight.h5")
        self.final_file = Header.os.path.join(self.final_dir,"trained_weight.h5")
        self.log_file_1 = Header.os.path.join(self.log_dir,"soft_y.pickle")
        self.log_file_2 = Header.os.path.join(self.log_dir,"loss.pickle")
        self.log_file_3 = Header.os.path.join(self.log_dir,"teacher_model_info.txt")
        self.log_file_4 = Header.os.path.join(self.log_dir,"Accuracy.pickle")
        self.model_file = Header.os.path.join(self.model_dir,"model.json")

    def Save_Text(self,contents,dst_file):
        with open(dst_file,'wb') as mysavedata:
            Header.pickle.dump(contents,mysavedata)
        return True

    def Load_Text(self,dst_file):
        contents = []
        with open(dst_file,'rb') as myloaddata:
            contents = Header.pickle.load(myloaddata)
        return contents

    def Build_model(self,input_shape,num_classes):
        model = Header.m.Sequential()
        model.add(Header.l.Conv2D(32, kernel_size = (5,5), padding = 'valid', activation='relu', input_shape=input_shape,name = "Conv2D_1"))
        model.add(Header.l.Conv2D(64, kernel_size = (5,5), padding = 'valid', activation='relu',name = "Conv2D_2"))
        model.add(Header.l.MaxPooling2D(pool_size = (2,2),name = "MaxPooling"))
        model.add(Header.l.Flatten(name = "Flatten_Layer"))
        model.add(Header.l.Dense(1024, activation = 'relu',name = "FC_Layer_1"))
        model.add(Header.l.Dense(num_classes, activation = None,name = "FC_Layer_2"))
        return model

    def Make_model(self):
        self.input_shape = (28,28,1)
        self.num_classes = 10

        model = self.Build_model(self.input_shape,self.num_classes)

        self.num_parameter = int(Header.np.sum([Header.K.count_params(p) for p in set(model.trainable_weights)]))

        return model

    def Reshape_input(self,x):
        return Header.np.reshape(x,(-1,28,28,1))

    def Reshape_output(self,x):
        return Header.np.reshape(x,(-1,10))

    def Train(self,args,train_x,train_y,valid_x,valid_y):

        self.Define_Args(args)
        self.train_x = self.Reshape_input(train_x)
        self.train_y = self.Reshape_output(train_y)
        self.valid_x = self.Reshape_input(valid_x)
        self.valid_y = self.Reshape_output(valid_y)

        model = self.Make_model()
        result = 0
        train_acc = 0
        good_batch = 0
        good_rate = 0

        f = open(self.log_file_3,'w')
        f.close()

        rate = self.min_learning_rate
        while(rate <= self.max_learning_rate):
            batch = self.min_batch
            while(batch <= self.max_batch):
                [train_result,tmp_result] = self.Run(model,rate,batch,result)
                if(tmp_result > result):
                    result = tmp_result
                    good_batch = batch
                    good_rate = rate
                    train_acc = train_result
                batch *= self.batch_increment
            rate *= self.learning_rate_increment

        self.Save_Text([self.train_loss,self.test_loss],self.log_file_2)
        self.Save_Acc([train_acc,result,0])
        f = open(self.log_file_3,'a')
        f.write("-------------------------------\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\nTraining Parameter : \n")
        f.write("learning_rate : %f     batch_size : %d\n" % (good_rate,good_batch))
        f.write("Last Epoch : %d\n" % self.last_epoch)
        f.write("Validation Acc : %f\n" % result)
        f.write("Train Acc : %f\n" % train_acc)
        f.write("-------------------------------")
        f.close()

    def acc_model(self,y_true,y_pred):
        if(self.model_type == "classifier"):
            prediction = Header.K.softmax(y_pred)
            accuracy = Header.tf.reduce_mean(Header.tf.cast(Header.tf.equal(Header.tf.argmax(prediction,1),Header.tf.argmax(y_true,1)),"float32"),name = "Accuracy_classifier")
        else:
            accuracy = Header.tf.reduce_mean(Header.tf.square(y_pred - y_true,name = "squared_accuracy"),name = "Accuracy_regression")
        return accuracy

    def Run(self,model,learning_rate,batch_size,result):
        # compile model
        adam = Header.op.Adam(lr=learning_rate,clipvalue=1.5)

        checkpointer = Header.call.ModelCheckpoint(filepath=self.checkpoint_file,monitor = "val_acc_model", verbose=1, save_best_only=True, save_weights_only=True)
        earlyStopping=Header.call.EarlyStopping(monitor='val_acc_model', patience=self.max_overfit, verbose=0, mode='auto')

        model.summary()

        self.temperature = 1
        model.compile(optimizer = adam,loss = self.loss,metrics = [self.acc_model])

        model.save(self.model_file)

        history = model.fit(self.train_x,self.train_y,batch_size = batch_size,epochs = self.max_epoch,verbose = 2,callbacks = [checkpointer,earlyStopping],validation_data = (self.valid_x,self.valid_y))

        # Get the value
        test_acc_save = Header.np.max(history.history['val_acc_model'])
        last_epoch = Header.np.argmax(history.history['val_acc_model'])+1
        test_loss_save = history.history['val_loss'][last_epoch-1]
        train_acc_save = history.history['acc_model'][last_epoch-1]
        train_loss_save = history.history['loss'][last_epoch-1]

        print("\n\nTrained Model Structure : ")
        print("Structure : CNN ", "     Number of parameter : ",self.num_parameter)
        print("Train_parameter = learning_rate : ",learning_rate,"    batch_size : ",batch_size)
        print("Last Epoch : %d" % last_epoch)
        print("Validation Accuracy : ",test_acc_save,"      Validation loss : ",test_loss_save)
        print("Trainset Accuracy : ",train_acc_save,"       Trainset loss : ",train_loss_save)

        if(result < test_acc_save):
            self.last_epoch = last_epoch
            self.train_loss = history.history['loss'][0:last_epoch]
            self.test_loss = history.history['val_loss'][0:last_epoch]
            Header.shutil.copy(self.checkpoint_file,self.final_file)

        # return the result
        return [train_acc_save,test_acc_save]

    def Save_soft_value(self,train_x,valid_x,test_x,temperature):
        if(Header.os.path.isfile(self.final_file) == False):
            print("Train the Teacher model first to get soft target")
            return false
        else:
            train_x = self.Reshape_input(train_x)
            valid_x = self.Reshape_input(valid_x)
            test_x = self.Reshape_input(test_x)
            soft_train = self.Predict(train_x,temperature)
            soft_valid = self.Predict(valid_x,temperature)
            soft_test = self.Predict(test_x,temperature)
            self.Save_Text([soft_train,soft_valid,soft_test],self.log_file_1)
        return True
    def Get_soft_value(self):
        data = []

        if(Header.os.path.isfile(self.log_file_1) == True):
            data = self.Load_Text(self.log_file_1)

        return data

    # Loss function
    def loss(self,y_true,y_pred):
        if(self.model_type == "classifier"):
            pred = Header.K.softmax(y_pred / self.temperature)

            total_loss = Header.tf.reduce_mean(Header.K.categorical_crossentropy(y_true,pred))
        else:
            pred = Header.K.tanh(y_pred)

            total_loss = Header.tf.reduce_mean(Header.tf.square(pred - y_true))

        return total_loss

    def Predict(self,x,temperature):

        if(Header.os.path.isfile(self.model_file) == False):
            print("Train the teacher model first!!")
            exit(1)
        loaded_model_json = Header.m.load_model(self.model_file,custom_objects = {'loss':self.loss,'acc_model':self.acc_model})

        # Load Model
        loaded_model = Header.m.model_from_json(loaded_model_json.to_json())
        loaded_model.load_weights(self.final_file)

        self.temperature = temperature

        if(self.model_type == "classifier"):
            new_layer = Header.l.Lambda(lambda x : Header.K.softmax(x/self.temperature))
            new_model = Header.m.Model(loaded_model.input,new_layer(loaded_model.output))
        else:
            new_layer = Header.l.Lambda(lambda x : Header.K.tanh(x))
            new_model = Header.m.Model(loaded_model.input,new_layer(loaded_model.output))

        new_model.compile(loss = self.loss, optimizer = 'adam')
        
        y = new_model.predict(x)

        return y

    def Inference(self,test_x,test_y):

        if(Header.os.path.isfile(self.model_file) == False):
            print("Train the teacher model first!!")
            exit(1)

        test_x = self.Reshape_input(test_x)
        test_y = self.Reshape_output(test_y)

        loaded_model_json = Header.m.load_model(self.model_file,custom_objects = {'loss':self.loss,'acc_model':self.acc_model})

         # Load Model
        loaded_model = Header.m.model_from_json(loaded_model_json.to_json())
        loaded_model.load_weights(self.final_file)

        loaded_model.compile(loss = self.loss, optimizer = 'adam',metrics = [self.acc_model])

        return loaded_model.evaluate(test_x,test_y,verbose = 3)

    def Num_Parameter(self):
        if(self.num_parameter == 0):
            self.Make_model()
            return self.num_parameter
        else:
            return self.num_parameter

    def Get_Acc(self):
        if(Header.os.path.isfile(self.log_file_4) == False):
            print("You must first train the teacher model to get accuracy")
            return
        else:
            return self.Load_Text(self.log_file_4)
    def Save_Acc(self,data):
        if(len(data) != 3):
            print("Acc data type must be the list of length 3")
            return
        else:
            self.Save_Text(data,self.log_file_4)
            return

    def View_Loss(self):
        if(Header.os.path.isfile(self.log_file_2) == False):
            print("You must first train the teacher model to get loss")
            return
        else:
            [self.train_loss,self.test_loss] = self.Load_Text(self.log_file_2)
            fig = Header.plt.figure()
            Header.plt.plot(self.train_loss,label = 'train_loss')
            Header.plt.plot(self.test_loss,label = 'validation_loss')
            Header.plt.title(self.model_name)
            Header.plt.xlabel("Epoch")
            Header.plt.ylabel("loss_value")
            Header.plt.legend(loc='upper left')
            Header.plt.show()
    def View_Train_log(self):
        if(Header.os.path.isfile(self.log_file_3) == False):
            print("You must first train the teacher model to get train log")
            return False
        else:
            with open(self.log_file_3) as f:
                for line in f:
                    print(line)
            return True

