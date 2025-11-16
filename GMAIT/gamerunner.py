import time
import gamehandler as gh
import os

def static_runner(function, rounds, inpt_dict):
    null, number_of_rounds = rounds
    start = time.time()
    pts = 0
    for i in range(number_of_rounds):
        string, res, conv_method = function(inpt_dict)
        st = time.time()

        entry = conv_method(input("%s"%string))
        en = time.time()
        if entry == res:
            print("Correct.")
            pts += 1
        
        else:
            print("Incorrect. The answer was :\n%s \n" % str(res))
        
        print("Input took %d seconds. " % round(en - st))
    
    end = time.time()
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def dynamic_runner(function, time_tuple, inpt_dict):
    time_per, rounds = time_tuple
    pts = 0
    number_of_rounds = 0
    while number_of_rounds < rounds:
        string, res, conv_function = function(inpt_dict)
        number_of_rounds += 1
        start = time.time()
        print(string)
        time.sleep(time_per)
        os.system("clear")
        entry = conv_function(input("> "))
        end = time.time()
        
        if entry == res:
            print("Correct.")
            pts += 1
        
        else:
            print("Incorrect. The answer was :\n%f \n" % res)
            
    
    end = time.time()
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds, time_per]

def general_runner(function, parameter, inpt_dict, md):
    if md == 1:
        return static_runner(function, parameter, inpt_dict)
    
    elif md == 2:
        return dynamic_runner(function, parameter, inpt_dict)