import coneff
import numpy as np
import pandas as pd

def main():
    '''
    eff_conn is the direct connectivity- effective connectivity matrix
    eff_conn + M_2nstep is the direct + 1st degree separation connectivity- effective connectivity matrix
    '''   
    averaged_bold_seq = pd.read_csv("test_coneff_input.csv").fillna(0)
    averaged_bold_seq.drop(columns=averaged_bold_seq.columns[0], axis=1, inplace=True)
    averaged_bold_seq = averaged_bold_seq.to_numpy()
    
    struc_mat = np.random.randint(0,10,(averaged_bold_seq.shape[0], averaged_bold_seq.shape[0]))
    #norm_opt = 1
    #use_multistep = 0
    #use_deconv = 0
    #lr = 10**(-4)
    eff_conn, M_2nstep, etem, etem_2 = coneff.structured_G_causality(struc_mat, averaged_bold_seq, use_multistep=1, maxiter=1000)
    np.savetxt("direct_eff.csv", eff_conn, delimiter=",")
    np.savetxt("1degree_eff.csv", eff_conn + M_2nstep, delimiter=",")

if __name__ == "__main__":
    main()