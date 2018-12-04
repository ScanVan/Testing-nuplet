#  Scanvan
#
#      Vincent Buntinx - vbuntinx@shogaku.ch
#      Copyright (c) 2016-2018 DHLAB, EPFL
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
    #! \file   test_prior_model_nuple.py
    #  \author Vincent Buntinx <vbuntinx@shogaku.ch>
    #
    #  Scanvan - https://github.com/ScanVan

import numpy as np
import spherical_algo_nuple
from plyfile import PlyData, PlyElement

def importation_data(path):
    fichier=open(path,'r')
    text=fichier.read()
    fichier.close()
    text=text.split('\n')
    data=[]
    for i in range(len(text[:-1])):
        els=text[i].split('\t')
        data.append(np.array([float(els[0]),float(els[1]),float(els[2])]))
    return data

def importation_centers(path,num):
    fichier=open(path,'r')
    text=fichier.read()
    fichier.close()
    text=text.split('\n')
    data=[]
    for i in range(num):
        els=text[i].split('\t')
        data.append(np.array([float(els[0]),float(els[1]),float(els[2])]))
    return data

def projection(p3d,center):
    res=[]
    for i in range(len(p3d)):
        vec=np.subtract(p3d[i],center)
        vec/=np.linalg.norm(vec)
        point=vec
        res.append(point)
    return res

def generate_unit_spheres(p3d,centers):
    spheres=[]
    for center in centers:
        p3d_proj=projection(p3d,center)
        spheres.append(p3d_proj)
    return spheres

def save_ply(scene,name):
    scene_ply=[]
    for elem in scene:
        scene_ply.append(tuple(elem))
    scene_ply=np.array(scene_ply,dtype=[('x','f4'),('y','f4'),('z','f4')])
    el=PlyElement.describe(scene_ply,'vertex',comments=[name])
    PlyData([el],text=True).write(name+'.ply')

def svd_rotation(v,u):
    vu=np.dot(v,u)
    det=round(np.linalg.det(vu),4)
    m=np.identity(3)
    m[2,2]=det
    vm=np.dot(v,m)
    vmu=np.dot(vm,u)
    return vmu

def miseaechelle(data1,data2):
    if len(data1)==len(data2):
        longueur=len(data1)
    trans1=-data1[0]
    trans2=-data2[0]
    for i in range(longueur):
        data1[i]+=trans1
        data2[i]+=trans2
    scale=0.0
    for i in range(1,longueur):
        scale+=(np.linalg.norm(data1[i])/np.linalg.norm(data2[i]))
    scale/=(longueur-1)
    for i in range(1,longueur):
        data2[i]*=scale
    return data1,data2

def superposition(data1,data2):
    if len(data1)==len(data2):
        longueur=len(data1)
    data1,data2=miseaechelle(data1,data2)
    sv_corr_12=np.zeros((3,3))
    sv_cent_1=np.zeros(3)
    sv_cent_2=np.zeros(3)
    for i in range(longueur):
        sv_cent_1+=data1[i]
        sv_cent_2+=data2[i]
    sv_cent_1/=longueur
    sv_cent_2/=longueur
    for i in range(longueur):
        sv_diff_1=data1[i]-sv_cent_1
        sv_diff_2=data2[i]-sv_cent_2
        sv_corr_12+=np.outer(sv_diff_1,sv_diff_2)
    svd_U_12,svd_s_12,svd_Vt_12=np.linalg.svd(sv_corr_12)
    rotation=svd_rotation(svd_Vt_12.transpose(),svd_U_12.transpose())
    translation=sv_cent_2-np.dot(rotation,sv_cent_1)
    new_data=[]
    for i in range(longueur):
        point=translation+np.dot(rotation,data1[i])
        new_data.append(point)
    return new_data,data2

def simulation(num_model,num_cam,stop_crit):
    dirname=str(num_model)
    folder='prior_generated_test_models/'
    data=importation_data(folder+dirname+'/model.dat')
    centers_ori=importation_centers(folder+dirname+'/centers_R1.txt',num_cam)
    spheres=generate_unit_spheres(data,centers_ori)
    x=spherical_algo_nuple.pose_estimation(spheres,stop_crit)
    centers_ori,centers_est=superposition(centers_ori,x[0])
    models_ori,models_est=superposition(data,x[1])
    save_ply(models_est,'models_est_'+dirname+'_'+str(num_cam))
    save_ply(models_ori,'models_ori_'+dirname)
    error=0.0
    for i in range(len(centers_ori)):
        error+=np.linalg.norm(centers_est[i]-centers_ori[i])
    print(num_model,num_cam,stop_crit,'done: error =>',error)

#sphere (triplet, 6-uple, 9-uple)
simulation(42,3,10**-8)
simulation(42,6,10**-8)
simulation(42,9,10**-8)
#dolphin (triplet, 6-uple, 9-uple)
simulation(16,3,10**-8)
simulation(16,6,10**-8)
simulation(16,9,10**-8)
#canstick (triplet, 6-uple, 9-uple)
simulation(10,3,10**-8)
simulation(10,6,10**-8)
simulation(10,9,10**-8)
