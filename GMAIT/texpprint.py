'''
The purpose of this code is to 
convert LaTeX to pprint.
'''

x = [["x"]]

def connect(arr1, arr2):
    arr3 = []
    for i in range(len(arr1)):
        arr3.append(arr1[i] + arr2[i])
    
    return arr3[:]

