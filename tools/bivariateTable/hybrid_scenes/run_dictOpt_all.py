import numpy as np
import math
import os
import subprocess
import scipy.io as sio
import matplotlib.pyplot as plt
from os import listdir

ext_render = True;

os.system('rm -rf mitsuba-diff')
os.system("cp -rf 'mitsuba-diff-dict' 'mitsuba-diff'")

MainFolder = "/home/kfirs/runAll_dict"
BRDFfolder = MainFolder + "/BRDF_target/";
print(BRDFfolder);
loc = listdir(BRDFfolder);
loc = sorted(loc);

os.system('pkill -9 mtssrv');
os.system('pkill -9 mtstensor');
os.system('pkill -9 MATLAB');
os.system('rm -rf ' + MainFolder + '/sceneDict*');
os.system('cp -rf target.ply /home/kfirs/scene_dictOpt_2L/meshes/target.ply');
os.system('cp -rf target.ply /home/kfirs/scene_dictOpt_2L/meshes/photometric.ply');
os.system('python "/home/kfirs/mitsuba-diff/optimizers/multisrv.py" -n 2 -l')

for filename in loc:
    os.chdir("/home/kfirs");  
    print(filename);
    # rendering the target
    # copy inputs: target table, lighting, config file
    os.system('cp ' + BRDFfolder + filename + '/BRDFtable.mat ' + '/home/kfirs/BRDFtable.mat');

    mat_content = sio.loadmat("/home/kfirs/BRDFtable.mat");
    BRDF=mat_content['BRDFtable'];
    shape = (180, 90, 90);
    BRDFVals = np.zeros((180, 90, 90, 3));
    
    # BRDF=np.reshape(X,(180,90,90),order='F');
    BRDFVals[:, :, :, 0] = BRDF;
    
    vec = np.reshape(np.swapaxes(BRDFVals, 1, 2), (-1), 'F')
    vec[vec < 1e-20] = 1e-20;
    shape = [shape[2], shape[1], shape[0]];
    BRDF_filename = "/home/kfirs/currRec.binary";
    f = open(BRDF_filename, "wb")
    np.array(shape).astype(np.int32).tofile(f)
    vec.astype(np.float64).tofile(f)
    f.close();

    if ext_render:
      os.system("mitsuba '/home/kfirs/intensity-scene.xml' -Dwidth=256 -Dheight=256 -DlightX=0 -DlightY=0 -DlightZ=-1.0 -Dirradiance=5.0 -Ddepth=-1 -DsampleCount=64000 -Dalpha=0.1"); 
      f=open('intensity-scene.hds','r');
      vals = np.fromfile(f, np.float32, -1)
      renderedImage = np.squeeze(np.reshape(vals[3:],[256,256]));
  
      target = np.zeros((256,256,2));
      target[:,:,0] = renderedImage ;
      target[:,:,1] = renderedImage ;
      np.save("/home/kfirs/target.npy", target);
        
      os.system("cp -rf '/home/kfirs/target.npy' '/home/kfirs/scene_dictOpt_2L/target.npy'");      
      
      np.save(BRDFfolder + filename + "/target.npy", target);
      targetPic = renderedImage;
      sio.savemat(BRDFfolder + filename + "/targetPic.mat" , {'targetPic':targetPic})    
      

    os.system("cp -rf " + BRDFfolder + filename + "/target.npy " + '/home/kfirs/scene_dictOpt_2L/target.npy');
    dictScene_loc = "/home/kfirs/scene_dictOpt_2L";
    os.chdir(dictScene_loc);
    os.system('rm -rf input* images render* err* log* output*');
    
    currScene = MainFolder + '/sceneDict_' + filename;
    os.system("cp -rf " + "/home/kfirs/scene_dictOpt_2L " + currScene);
    
    os.chdir(currScene);   
    
    os.system('MTSTF_REDUCTOR_ENABLE=true python /home/kfirs/mitsuba-diff/optimizers/optimize.py ' + currScene + '/config.json');