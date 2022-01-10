from evaluations import *
from utils import *

def distance(gps1,gps2):
    x1,y1 = gps1
    x2,y2 = gps2
    return np.sqrt((x1-x2)**2+(y1-y2)**2 )

    
    

def gen_matrix(data='geolife'):
    train_data = read_data_from_file('../data/%s/real.data'%data)
    gps = get_gps('../data/%s/gps'%data)
    if data=='mobile':
        max_locs = 8606
    else:
        max_locs = 23768

    reg1 = np.zeros([max_locs,max_locs])
    for i in range(len(train_data)):
        line = train_data[i]
        for j in range(len(line)-1):
            reg1[line[j],line[j+1]] +=1
    reg2 = np.zeros([max_locs,max_locs])
    for i in range(max_locs):
        for j in range(max_locs):
            if i!=j:
                reg2[i,j] = distance((gps[0][i],gps[1][i]),(gps[0][j],gps[1][j]))
    

    np.save('../data/%s/M1.npy'%data,reg1)
    np.save('../data/%s/M2.npy'%data,reg2)

    print('Matrix Generation Finished')

    

    




    
