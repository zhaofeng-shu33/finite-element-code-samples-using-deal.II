## @package hollowSphere_pp
# Data post-processing for hollowSphereEighth
# used package:
#    matplotlib
#    pandas
#    numpy

import code
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
## radial and hoop stress calculation.
#  displacement calculation
def dp_stress_cal(csvDoc):
    csvDoc = csvDoc
    sigma = np.matrix([[csvDoc.tensor_xx, csvDoc.tensor_xy, csvDoc.tensor_xz],
                       [csvDoc.tensor_xy, csvDoc.tensor_yy, csvDoc.tensor_yz],
                       [csvDoc.tensor_xz, csvDoc.tensor_yz, csvDoc.tensor_zz]])
    
    r = np.sqrt(csvDoc['Point:0']**2 + csvDoc['Point:1']**2 + csvDoc['Point:2']**2)
    r_vector = np.array([[csvDoc['Point:0']/r, csvDoc['Point:1']/r, csvDoc['Point:2']/r]]).T
    
    r_stress = np.diag((r_vector.T)*sigma*r_vector)
    
    phi = np.sqrt(csvDoc['Point:0']**2 + csvDoc['Point:1']**2)
    fi_vector = np.array([[-csvDoc['Point:1']/phi, csvDoc['Point:0']/phi, 0]]).T
    #theta_vector=np.cross(r_vector.T,fi_vector.T).T
    fi_stress = np.diag((fi_vector.T)*sigma*fi_vector)
    #theta_stress = np.linalg.norm(sigma*theta_vector)
    
    return r_stress, fi_stress#,theta_stress


if __name__ == '__main__':    
    fileName=sys.argv[1]
    print(fileName, 'saved')
    csvDoc = pd.read_csv('build/'+fileName+'.csv')
    csvDoc['point'] =np.sqrt((csvDoc['Point:0'])**2 + (csvDoc['Point:1'])**2 + (csvDoc['Point:2'])**2)
    csvDoc['displacement'] =  np.sqrt((csvDoc['x_displacement'])**2 + (csvDoc['y_displacement'])**2 + (csvDoc['z_displacement'])**2)

    csvDoc.plot.scatter('point', 'displacement')
    plt.savefig(fileName+'.png')
    plt.show()

    fileName=fileName+'2';
    csvDoc = pd.read_csv('build/'+fileName+'.csv')
    csvDoc['point'] = np.sqrt((csvDoc['Point:0'])**2 + (csvDoc['Point:1'])**2 + (csvDoc['Point:2'])**2)    
    csvDoc.insert(loc=9, column='r_stress', value=0)
    csvDoc.insert(loc=10, column='fi_stress', value=0)
    #csvDoc.insert(loc=11, column='theta_stress', value=0)
    for i in range(csvDoc.shape[0]):
        csvDoc.ix[i, 'r_stress'], csvDoc.ix[i, 'fi_stress'] = dp_stress_cal(csvDoc.ix[i])

    #csvDoc.to_csv(fileName+'3.csv')


    csvDoc.plot.scatter('point', 'r_stress')
    plt.savefig(fileName+'r'+'.png')
    plt.show()

    csvDoc.plot.scatter('point', 'fi_stress')
    plt.savefig(fileName+'p'+'.png')
    plt.show()


    #code.interact(local=locals())