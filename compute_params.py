
import numpy as np
import cc3d
import nibabel as nib
import pandas as pd


def connected_component_params_compute(seg_image_path, params_file_center_coord, params_file_component_voxel):
    volume = nib.load(seg_image_path)
    arr = volume.get_data()
    bin_arr = np.zeros_like(arr, dtype=np.int8)
    bin_arr[arr>0]=1
    del arr
    del volume
    labels_out = cc3d.connected_components(bin_arr, connectivity=26)
    del bin_arr
    stats = cc3d.statistics(labels_out)
    np.save(params_file_center_coord, stats['centroids'])
    np.save(params_file_component_voxel, stats['voxel_counts'])



tls_image_file = r'' # path of TLS mask image
meta_image_file = r'' # path of metastasis mask image
params_path_out = r'' # path to save results
connected_component_params_compute(tls_image_file, params_path_out + 'tls_center_coord.npy', params_path_out + 'tls_voxels.npy')
connected_component_params_compute(meta_image_file, params_path_out + 'meta_center_coord.npy', params_path_out + 'meta_voxels.npy')

img_res = [1.625, 1.625, 8]

tls_center_coord = np.load(params_path_out + 'tls_center_coord.npy')[1:]
tls_component_size = np.load(params_path_out + 'tls_voxels.npy')[1:]
meta_center_coord = np.load(params_path_out + 'meta_center_coord.npy')[1:]
meta_component_size = np.load(params_path_out + 'meta_voxels.npy')[1:]

tls_volume = np.prod(img_res) * tls_component_size
meta_volume = np.prod(img_res) * meta_component_size
pd.DataFrame(tls_volume).to_csv('tls_volume.csv')
pd.DataFrame(meta_volume).to_csv('meta_volume.csv')

tls_dis_mat = np.zeros((len(tls_center_coord), len(tls_center_coord)-1), dtype= np.float)

for i in range(len(tls_center_coord)):
    cur_center = tls_center_coord[i]
    other_centers = np.delete(tls_center_coord, i, axis=0)
    for j in range(len(other_centers)):
        dis_tmp = np.sqrt(np.sum(np.square(img_res*(cur_center - other_centers[j]))))
        tls_dis_mat[i,j]=dis_tmp

tls_near_dis = np.min(tls_dis_mat, axis=-1)
pd.DataFrame(tls_near_dis).to_csv('tls_tls_dis.csv')

meta_dis_mat = np.zeros((len(meta_center_coord), len(meta_center_coord)-1), dtype= np.float)

for i in range(len(meta_center_coord)):
    cur_center = meta_center_coord[i]
    other_centers = np.delete(meta_center_coord, i, axis=0)
    for j in range(len(other_centers)):
        dis_tmp = np.sqrt(np.sum(np.square(img_res*(cur_center - other_centers[j]))))
        meta_dis_mat[i,j]=dis_tmp

meta_near_dis = np.min(meta_dis_mat, axis=-1)
pd.DataFrame(meta_near_dis).to_csv('meta_meta_dis.csv')



tls_meta_dis_mat = np.zeros((len(tls_center_coord), len(meta_center_coord)), dtype= np.float)

for i in range(len(tls_center_coord)):
    cur_center = tls_center_coord[i]
    # other_centers = np.delete(tls_center, i, axis=0)
    for j in range(len(meta_center_coord)):
        dis_tmp = np.sqrt(np.sum(np.square(img_res*(cur_center - meta_center_coord[j]))))
        tls_meta_dis_mat[i,j]=dis_tmp

tls_meta_near_dis = np.min(tls_meta_dis_mat, axis=-1)
pd.DataFrame(tls_meta_near_dis).to_csv('tls_meta_dis.csv')
tls_meta_near_id = np.argmin(tls_meta_dis_mat, axis=-1)
tls_meta_near_volume = meta_volume[tls_meta_near_id]
pd.DataFrame(tls_meta_near_volume).to_csv('tls_near_meta_volume.csv')


meta_tls_dis_mat = np.zeros((len(meta_center_coord), len(tls_center_coord)), dtype= np.float)

for i in range(len(meta_center_coord)):
    cur_center = meta_center_coord[i]
    for j in range(len(tls_center_coord)):
        dis_tmp = np.sqrt(np.sum(np.square(img_res*(cur_center - tls_center_coord[j]))))
        meta_tls_dis_mat[i,j]=dis_tmp

meta_tls_near_dis = np.min(meta_tls_dis_mat, axis=-1)
pd.DataFrame(meta_tls_near_dis).to_csv('meta_tls_dis.csv')
meta_tls_near_id = np.argmin(meta_tls_near_dis, axis=-1)
meta_tls_near_volume = tls_volume[meta_tls_near_id]
pd.DataFrame(meta_tls_near_volume).to_csv('meta_near_tls_volume.csv')
