import os
from os.path import split, join, exists
from psbody.mesh import Mesh
import numpy as np

def L2_distance(vector1, vector2):
    e = 0
    if len(vector1) == len(vector2):
        N = len(vector1)
        print('N:',N)
        for n in range(N):
            e += np.sqrt(np.sum(np.square(vector1[n] - vector2[n])))
        return e / N

if __name__ == '__main__':
    #path = './cq_output/'
    path = '/media/sdc/qingchen/mgn_output_epoch/'
    
    source_name = 'cq_data_0_finetuned_garment0_epoch_200.obj'
    for n in range(20,100,20):
        target_name = 'cq_data_0_finetuned_garment0_epoch_{}.obj'.format(n)
        scan_source = Mesh(filename=join(path, source_name))
        scan_target = Mesh(filename=join(path, target_name))

        closest_points1 = scan_target.closest_points(scan_source.v)
        closest_points2 = scan_source.closest_points(scan_target.v)
        #print(scan_source.v)
        #print(closest_points1)
        #print(scan_target.v)
        #print(closest_points2)
        VertexToSurfaceError = L2_distance(closest_points1, scan_target.v) + L2_distance(closest_points2, scan_source.v) 
        print(source_name, target_name)
        print(VertexToSurfaceError, '\n')
