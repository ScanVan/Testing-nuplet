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
    #! \file   test_prior_model_nuple_measure_merge.py
    #  \author Vincent Buntinx <vbuntinx@shogaku.ch>
    #
    #  Scanvan - https://github.com/ScanVan

import numpy as np
from plyfile import PlyData, PlyElement

import spherical_algo_nuple

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
    rayons=[]
    for i in range(len(p3d)):
        vec=np.subtract(p3d[i],center)
        vec/=np.linalg.norm(vec)
        point=vec
        res.append(point)
        rayons.append(np.linalg.norm(p3d[i]-center))
    return res,rayons

def generate_unit_spheres(p3d,centers):
    spheres=[]
    for center in centers:
        p3d_proj,rayons_part=projection(p3d,center)
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

def fusion(model_1,model_2,nb_com):
    pos_1=model_1[0]
    rot_1=model_1[1]
    sce_1=model_1[2]
    pos_2=model_2[0]
    rot_2=model_2[1]
    sce_2=model_2[2]
    len_1=len(pos_1)
    len_2=len(pos_2)
    if len_2>nb_com:
        nb_rot=len_1-nb_com
        scale_factor=np.linalg.norm(pos_1[nb_rot+1]-pos_1[nb_rot])/np.linalg.norm(pos_2[1]-pos_2[0])
        for i in range(len(pos_2)):
            pos_2[i]*=scale_factor
        for i in range(len(sce_2)):
            sce_2[i]*=scale_factor
        translation=(pos_1[nb_rot]-pos_2[0])
        rotation=np.identity(3)
        for k in range(nb_rot-1,-1,-1):
            rotation=np.dot(rot_1[k].transpose(),rotation)
        new_positions=[]
        for i in range(len_2):
            new_pos=np.squeeze(np.asarray(translation+np.dot(rotation,pos_2[i])))
            new_positions.append(new_pos)
        positions=pos_1[:nb_rot]+new_positions
        rotations=rot_1[:nb_rot]+rot_2
        scene=[]
        for i in range(len(sce_1)):
            point=np.squeeze(np.asarray(sce_1[i]))
            scene.append(point)
        for i in range(len(sce_2)):
            point=np.squeeze(np.asarray(translation+np.dot(rotation,sce_2[i])))
            scene.append(point)
        model=[positions,rotations,scene]
        return model
    else:
        return 'error'

def fusion_totale(x):
    model=x[0]
    for i in range(1,len(x)):
        model=fusion(model,x[i],2)
    return model

def simulation(spheres,num_cam,stop_crit):
    ran=len(spheres)-num_cam+1
    models=[]
    for i in range(ran):
        spheres_list=[]
        for j in range(num_cam):
            spheres_list.append(spheres[i+j])
        positions,rotations,scene,log=spherical_algo_nuple.pose_estimation(spheres_list,stop_crit)
        models.append([positions,rotations,scene])
    model=fusion_totale(models)
    return model
    
def test_by_model(num_model,num_cam,num_total_cam,stop_crit):
    dirname=str(num_model)
    folder='prior_generated_test_models/'
    data=importation_data(folder+dirname+'/model.dat')
    centers_ori=importation_centers(folder+dirname+'/centers_R1.txt',num_total_cam)
    spheres=generate_unit_spheres(data,centers_ori)
    positions,rotations,scene=simulation(spheres,num_cam,stop_crit)
    centers_ori,centers_est=superposition(centers_ori,positions)
    save_ply(scene,'models_est_'+dirname+'_'+str(num_cam)+'_'+str(num_total_cam))
    save_ply(data,'models_ori_'+dirname)
    error=0.0
    for i in range(len(centers_ori)):
        error+=np.linalg.norm(centers_est[i]-centers_ori[i])
    print(num_model,num_cam,stop_crit,'done: error =>',error)

test_by_model(42,3,100,10**-12)
