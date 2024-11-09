import time
import gamehandler as gh

def static_runner(function, number_of_rounds, inpt_dict):
    start = time.time()
    pts = 0
    for i in range(number_of_rounds):
        string, res, conv_method = function(inpt_dict)

        entry = conv_method(input("%s"%string))
        if entry == res:
            print("Correct.")
            pts += 1
        
        else:
            print("Incorrect. The answer was :\n%s \n" % str(res))
    
    end = time.time()
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def dynamic_runner(function, total_time, inpt_dict):
    start = time.time()
    pts = 0
    number_of_rounds = 0
    while time.time() - start < total_time:
        string, res, conv_function = function(inpt_dict)
        number_of_rounds += 1
        entry = conv_function(input(string))
        end = time.time()
        if time.time() - start > total_time:
            print("Time Elapsed before entry.")
            return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds, total_time]
        if entry == res:
            print("Correct.")
            print("Remaining time : ", round(total_time - (time.time() - start)))
            pts += 1
        
        else:
            print("Incorrect. The answer was :\n%f \n" % res)
            print("Remaining time : ", round(total_time - (time.time() - start)))
    
    end = time.time()
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds, total_time]

def general_runner(function, parameter, inpt_dict, md):
    if md == 1:
        return static_runner(function, parameter, inpt_dict)
    
    elif md == 2:
        return dynamic_runner(function, parameter, inpt_dict)