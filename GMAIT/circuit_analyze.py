import cmath
import utils as ut 
import random

def node_analysis(incidence_array, admittance_array, es_vect, is_vect):
    ay = incidence_array * admittance_array
    admittance_matrix = ay * incidence_array.transpose()
    inps = ay * es_vect - incidence_array * is_vect
    return admittance_matrix, inps, admittance_matrix.inverse() * inps

class inductor:
    def __init__(self, inductance, init_val=0):
        self.val = inductance
        self.init = init_val
    
