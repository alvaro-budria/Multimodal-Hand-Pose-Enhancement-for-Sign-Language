import glob
import os
import numpy as np


def load_results(res_path="Multi/sample/chemistry_test/seq1/sample_results/test_predicted_body_3d_frontal/"):
    filelist = glob.glob(os.path.join(res_path, "*.txt"))
    
    res = np.array([])

    for infile in sorted(filelist):
        print(str(infile))
        count = 0
        with open(infile) as fp:
            for line in fp:
                a = [ float(x) for x in line.split(" ")[:-1] ]
                print(len(a), a)




load_results()

