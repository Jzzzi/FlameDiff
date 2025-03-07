import numpy as np
names = ['T', 'OH', 'HO2']
field_number = len(names)
frames = 200
thetas = [str(i) for i in range(0,179,36)]
#thetas = [str(i) for i in range(0,1,36)] #for fast Debug
theta_number = len(thetas)
dir_names = ['T1030', 'T1035', 'T1040', 'T1045', 'T1050', 'T1055', 'T1060', 'T1070', 'T1080']

rootdir = '/work/e283/e283/sipeiwu/OF3D/cabra/laminar_LEMOS/cases_snap3_H1R2/datasets/'
Train_series = np.zeros((len(dir_names)*theta_number,frames,field_number,192,256),dtype=np.float32)
for (i, dir_name) in enumerate(dir_names):
    for (j,theta) in enumerate(thetas):
        for (k,physicalFieldName) in enumerate(names):
            Train_series[i*theta_number+j,:,k,:,:] = np.load(rootdir+'/9cases_slice_5_200_192_256/'+dir_name+'/grade_192_256_npdata_'+theta+'/'+physicalFieldName+'.npy')[:,:,:] #(frames, X, Y)




