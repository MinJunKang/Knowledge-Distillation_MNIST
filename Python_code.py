import os
import argparse
import model
import data
import tensorflow as tf
import numpy as np
import random
import shutil
import pickle
import Teacher_model

# Randomness setting
seed = int(os.getenv("SEED", 12))
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Dir maker
def check_and_makedir(path,choice=False):
    if(choice==True):
        shutil.rmtree(path, ignore_errors=True)
    if not os.path.isdir(path):
        os.makedirs(path)

# For boolean
def convert_str_to_bool(text):
    if text.lower() in ["true", "yes", "y", "1"]:
        return True
    else:
        return False

def get_basic_info():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # for hardware acceleration
    parser.add_argument('--gpu', type=int, default=0, choices=[None, 0, 1])

    # Training Parameters
    parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student"])

    return parser

def get_parameter():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--start_node',type=int,default=16)
    parser.add_argument('--node_increment',type=int,default=2)
    parser.add_argument('--max_node',type=int,default=256)

    # Model Parameters
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--start_learning_rate',type=float,default=0.001)
    parser.add_argument('--learning_rate_increment',type=float,default=10)
    parser.add_argument('--max_overfit',type=int,default=50)

    # Model Type
    parser.add_argument('--Objective',type=str,default="classifier",choices=["classifier", "regression"])

    return parser

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # checkpoint step & loss showing step
    parser.add_argument('--display_step', type=int, default=100)
    # dir for saving checkpoint data
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint")
    # dir for saving log data
    parser.add_argument('--log_dir', type=str, default="logs")
    # dir for saving small model's log
    parser.add_argument('--small_dir',type=str,default="smallmodel")
    # for hardware acceleration
    parser.add_argument('--gpu', type=int, default=None, choices=[None, 0, 1])

    # Training Parameters
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--load_teacher_from_checkpoint', type=str, default="true")
    parser.add_argument('--load_teacher_checkpoint_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument('--batch_size', type=int, default=128)

    # Model Parameters
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dropoutprob', type=float, default=0.75)

    return parser

def setup(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (args.gpu)

    args.load_teacher_from_checkpoint = convert_str_to_bool(args.load_teacher_from_checkpoint)
    check_and_makedir(args.log_dir)
    check_and_makedir(args.checkpoint_dir)

def main():

    parser_teacher = get_parser()
    args_teacher = parser_teacher.parse_args()
    setup(args_teacher)

    parser_student = get_basic_info()
    args_student = parser_student.parse_args()
    # Set cuda environment
    if args_student.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (args_student.gpu)

    parser_train = get_parameter()
    args_train = parser_train.parse_args()

    dataset = data.Dataset(args_teacher)

    # Get dataset
    train_x,train_y = dataset.get_train_np()
    test_x,test_y = dataset.get_train_np()
    valid_x,valid_y = dataset.get_validation_np()

    # Make saving place
    check_and_makedir(model.dir_tmp,True)
    check_and_makedir(model.dir_best,True)
    check_and_makedir(model.dir_log)

    print("\n[Runnning Options]\nTrain Teacher : 1   /   Train Student : 2    /    Train Student without softy : 3\n<You should first train Teacher to get third option>")
    uinput = input("입력하세요 : ")

    if(int(uinput) == 2 or int(uinput) == 3):
        args_teacher.model_type = "student"

    # Train the teacher model
    if args_teacher.model_type == "teacher":
        tf.reset_default_graph()
        teacher_model = Teacher_model.BigModel(args_teacher, "teacher")
        teacher_model.start_session()
        teacher_model.train(dataset)

        # Testing teacher model on the best model based on validation set
        teacher_model.load_model_from_file(args_teacher.checkpoint_dir)
        teacher_model.run_inference(dataset)

        softy_path = os.path.join(model.dir_log,model.softy_file)

        if(os.path.isfile(softy_path)==False):
            soft_y = teacher_model.predict(train_x, args_student.temperature)
            soft_y_t = teacher_model.predict(test_x, args_student.temperature)
            soft_y_v = teacher_model.predict(valid_x, args_student.temperature)

            with open(softy_path,'wb') as mysavedata:
                pickle.dump([soft_y,soft_y_t,soft_y_v],mysavedata)
            print("Data is ready!, Please Start Again")
            teacher_model.close_session()
            exit(1)

        teacher_model.close_session()
    else:
        tf.reset_default_graph()

        softy_path = os.path.join(model.dir_log,model.softy_file)

        if(os.path.isfile(softy_path)==False):

            teacher_model = Teacher_model.BigModel(args_teacher, "teacher")
            teacher_model.start_session(False)
            teacher_model.load_model_from_file(args_teacher.checkpoint_dir)
            print("Verify Teacher State before Training Student")
            teacher_model.run_inference(dataset)

            soft_y = teacher_model.predict(train_x, args_student.temperature)
            soft_y_t = teacher_model.predict(test_x, args_student.temperature)
            soft_y_v = teacher_model.predict(valid_x, args_student.temperature)

            with open(softy_path,'wb') as mysavedata:
                pickle.dump([soft_y,soft_y_t,soft_y_v],mysavedata)
            print("Data is ready!, Please Start Again")
            teacher_model.close_session()
            exit(1)

        
        with open(softy_path,'rb') as myloaddata:
            [soft_y,soft_y_t,soft_y_v] = pickle.load(myloaddata)

        student_model = model.Small_DNN("student")
        student_model.Data_Prepare(train_x,train_y,valid_x,valid_y,soft_y)
        if(int(uinput) == 2):
            student_model.Train(args_train,False)
        else:
            student_model.Train(args_train,True)
    

    return 0



#def main():
#    parser = get_parser()
#    args = parser.parse_args()
#    setup(args)

#    dataset = data.Dataset(args)

#    args.model_type = "student"

#    tf.reset_default_graph()
#    if args.model_type == "student":
#        teacher_model = None
#        if args.load_teacher_from_checkpoint:
#            teacher_model = Teacher_model.BigModel(args, "teacher")
#            teacher_model.start_session()
#            teacher_model.load_model_from_file(args.load_teacher_checkpoint_dir)
#            print("Verify Teacher State before Training Student")
#            teacher_model.run_inference(dataset)
#        student_model = model.SmallModel(args, "student")
#        student_model.start_session()
#        student_model.train(dataset, teacher_model)

#        # Testing student model on the best model based on validation set
#        student_model.load_model_from_file(args.checkpoint_dir)
#        student_model.run_inference(dataset)

#        if args.load_teacher_from_checkpoint:
#            print("Verify Teacher State After Training student Model")
#            teacher_model.run_inference(dataset)
#            teacher_model.close_session()
#        student_model.close_session()
#    else:
        #teacher_model = Teacher_model.BigModel(args, "teacher")
        #teacher_model.start_session()
        #teacher_model.train(dataset)

        ## Testing teacher model on the best model based on validation set
        #teacher_model.load_model_from_file(args.checkpoint_dir)
        #teacher_model.run_inference(dataset)
        #teacher_model.close_session()


#    return 0

if __name__ == '__main__':
    main()
