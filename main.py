

import Header
import data
import argparse
from Student_model import Student_model as Student
from Teacher_model import Teacher_model as Teacher


# Execution part
def main():

                                                        ## Get DataSet ##
    dataset = data.Dataset()
    train_x,train_y = dataset.get_train_np() # Train
    valid_x,valid_y = dataset.get_validation_np() # Validation
    test_x,test_y = dataset.get_test_np() # Test
                                                    ## Check the options ##

    print("\n[Runnning Options]\n")
    print("1. Train Teacher")
    print("2. Prepare Soft Target")
    print("3. Train Student")
    print("4. Train Student with Soft Target")
    print("5. Get the best student and test accuracy")
    print("6. Show all the training log of Student model")
    print("7. Show training log of Teacher model")
    print("8. Simulation of Student Model")
    print("9. Show the temporary student model spec")
    print("10. Analyze the Student model data")
    print("\nNotice : <You should first train Teacher to get 4th option>\n")
    uinput = int(input("입력하세요: "))

    # Set argument for setup
    parser_setup = Header.Set_Model_Info()
    
    args_setup = parser_setup.parse_args()

    # Option Check
    if(uinput == 1 or uinput == 2 or uinput == 7): # Teacher model
        args_setup.gpu = 0
        args_setup.model_usage = "teacher"
        args_setup.model_lang = "keras"
    elif(uinput == 3 or uinput == 4 or uinput == 5 or uinput == 6 or uinput == 8 or uinput == 9 or uinput == 10): # Student model
        args_setup.gpu = 2
        args_setup.model_usage = "student"
        args_setup.model_lang = "keras"
    else:
        print("Wrong Options. Please Start Again!")
        exit(1)

    # Setup
    Header.set_up(args_setup)

    # Set argument for training
    parser_train = Header.Set_Parameter(args_setup)
    args_train = parser_train.parse_args()

    # Teacher model
    if(args_setup.model_usage == "teacher"):
        # Train Teacher
        if(uinput == 1):
            Header.check_and_makedir(args_train.dir)
            Header.check_and_makedir(args_train.checkpoint,True)
            Header.check_and_makedir(args_train.model,True)
            Header.check_and_makedir(args_train.final,True)
            Header.check_and_makedir(args_train.log,True)

            args_train.max_overfit = 10
            teacher_model = Teacher(args_train,args_setup.model_usage)
            teacher_model.Train(args_train,train_x,train_y,valid_x,valid_y)
            teacher_model.View_Loss()

        # Prepare Soft Target
        elif(uinput == 2):
            Header.check_and_makedir(Header.temp_dir)
            choice = -1
            # Get Temperature
            while(choice <= 0):
                choice = float(input("Temperature >> "))
            args_train.temperature = choice

            teacher_model = Teacher(args_train,args_setup.model_usage)
            teacher_model.Save_Text(args_train.temperature,Header.temp_file) # Save the temperature file
            result = teacher_model.Inference(test_x,test_y)
            # Save accuracy
            [train_acc,valid_acc,dummy] = teacher_model.Get_Acc()
            teacher_model.Save_Acc([train_acc,valid_acc,result[1]])
            print("Test loss : %f       Test accuracy : %f" %(result[0],result[1]))
            teacher_model.Save_soft_value(train_x,valid_x,test_x,args_train.temperature)
            teacher_model.View_Loss()
        else:
            teacher_model = Teacher(args_train,args_setup.model_usage)
            if(teacher_model.View_Train_log() == False):
                return -1
    else:
        if(uinput == 3):
            Header.check_and_makedir(args_train.dir)
            Header.check_and_makedir(args_train.checkpoint,True)
            Header.check_and_makedir(args_train.model,True)
            Header.check_and_makedir(args_train.final,True)
            Header.check_and_makedir(args_train.log,True)

            student_model = Student(args_train,args_setup.model_usage)
            student_model.Data_Prepare(train_x,train_y,valid_x,valid_y)
            student_model.Train(args_train)
        elif(uinput == 4):
            Header.check_and_makedir(args_train.dir)
            Header.check_and_makedir(args_train.checkpoint,True)
            Header.check_and_makedir(args_train.model,True)
            Header.check_and_makedir(args_train.final,True)
            Header.check_and_makedir(args_train.log,True)

            student_model = Student(args_train,args_setup.model_usage)
            # Get the soft target's temperature data
            args_train.temperature = student_model.Load_Text(Header.temp_file)
            if(args_train.temperature == []):
                print("First train the Teacher model to get soft targets")
                exit(1)
            else:
                student_model.Define_Arg(args_train)
            # Prepare soft y data
            if(Header.os.path.isfile(args_train.softy_file) == False):
                print("First train the Teacher model to get soft targets")
                exit(1)
            [soft_train,soft_valid,soft_test] = student_model.Load_Text(args_train.softy_file)
            
            student_model.Data_Prepare(train_x,train_y,valid_x,valid_y,soft_train)

            student_model.Train(args_train,True)

        elif(uinput == 5):
            student_model = Student(args_train,args_setup.model_usage)
            teacher_model = Teacher(args_train,args_setup.model_usage)
            if(Header.os.path.isfile(args_train.softy_file) == False):
                result  = student_model.Inference(test_x,test_y)
                print("Test accuracy : %f" %(result[1]))
            if(Header.os.path.isfile(args_train.softy_file) == True):
                [soft_train,soft_valid,soft_test] = student_model.Load_Text(args_train.softy_file)
                [train_t_acc,valid_t_acc,test_t_acc] = student_model.Load_Text(args_train.teacher_acc_file)
                result  = student_model.Inference(test_x,test_y,soft_test)
                print("Student model Test accuracy : %f" %(result[1]))
                print("Teacher model Test accuracy : %f" %test_t_acc)
                print("# Student_params : %d" % student_model.Num_Parameter())
                print("# Teacher_params : %d" % teacher_model.Num_Parameter())
                print("Complexity_gain(Student params / Teacher params) : %f" % (float(student_model.Num_Parameter())) / float(teacher_model.Num_Parameter()))
                print("Accuracy_gain(Teacher acc - Student acc) : %f" % float(test_t_acc - result[1]))
                print("Overall_gain(1 / (Complexity_gain * Accuracy_gain)) : %f" % 1 / Header.np.mutliply(float(teacher_model.Num_Parameter()) / float(student_model.Num_Parameter()),test_t_acc - result[1]))
        elif(uinput == 6):
            student_model = Student(args_train,args_setup.model_usage)
            if(student_model.View_Train_log() == False):
                return -1
        elif(uinput == 8):
            Header.check_and_makedir(args_train.tmp,True)

            student_model = Student(args_train,args_setup.model_usage)
            teacher_model = Teacher(args_train,args_setup.model_usage)

            choice = 0
            layer = 0
            hidden_node = []
            learn_rate = 0.001
            batch = 128
            
            # Get Layer number
            while(choice <= 0):
                choice = int(input("Number of Hidden Layer >> "))
            layer = choice
            choice = -1

            # Get Node number
            for i in range(layer):
                while(choice <= 0):
                    choice = int(input("Number of Hidden node of layer %s >> " % str(i+1)))
                hidden_node.append(choice)
                choice = -1

            # Get Learning rate
            while(choice <= 0):
                choice = float(input("Learning_rate >> "))
            learn_rate = choice
            choice = -1

            # Get Batch size
            while(choice <= 0):
                choice = int(input("Batch size >> "))
            batch = choice
            choice = " "

            soft_target = []
            if(Header.os.path.isfile(args_train.softy_file) == True):
                # If the model has to use the soft target
                while(choice.lower() != "yes" and choice.lower() != "no"):
                    choice = input("Soft Target Found. Shall we use it? (Yes or No) >> ")
                if(choice.lower() == "yes"):
                    [soft_train,soft_valid,soft_test] = student_model.Load_Text(args_train.softy_file)
                    args_train.temperature = student_model.Load_Text(Header.temp_file)
                    if(args_train.temperature == []):
                        print("Temperature information is lost! Please try option 2 again")
                        exit(1)
                    else:
                        student_model.Define_Arg(args_train)
                    soft_target = soft_train

            [acc,num_param] = student_model.Fit_SingleModel(train_x,train_y,valid_x,valid_y,test_x,test_y,hidden_node,learn_rate,batch,soft_target)
            [train_t_acc,valid_t_acc,test_t_acc] = student_model.Load_Text(args_train.teacher_acc_file)

            complexity_gain = Header.np.divide(float(num_param) , float(teacher_model.Num_Parameter()))
            accuracy_gain = float(test_t_acc - acc)
            overall_gain = 1 / Header.np.multiply(complexity_gain,accuracy_gain)
            student_model.Temp_file3_add([complexity_gain,accuracy_gain,overall_gain])

            print("Student model Test accuracy : %f" %(acc))
            print("Teacher model Test accuracy : %f" %test_t_acc)
            print("# Student_params : %d" % num_param)
            print("# Teacher_params : %d" % teacher_model.Num_Parameter())
            print("Complexity_gain(Student params / Teacher params) : %f" % complexity_gain)
            print("Accuracy_gain(Teacher acc - Student acc) : %f" % accuracy_gain)
            print("Overall_gain(1 / (Accuracy_gain * Complexity_gain)) : %f" % overall_gain)
        elif(uinput == 9):
            student_model = Student(args_train,args_setup.model_usage)
            if(student_model.View_Temp_log() == False):
                return -1
        else:
            from Analyze_Student import Analyzer as Analyzer

            analyze = Analyzer("./Student_model/log/Result_model.txt","./Result.xlsx")

            student_model = Student(args_train,args_setup.model_usage)
            teacher_model = Teacher(args_train,args_setup.model_usage)

            teacher_params = teacher_model.Num_Parameter()

            analyze.Excelgraph(teacher_params)

    return 0

if __name__ == '__main__':
    main()
