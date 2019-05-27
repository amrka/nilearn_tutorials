#!/usr/bin/env python
# coding: utf-8

# # <font color=red> 1. Introduction: nilearn in a nutshell
# </font>

# In[7]:


import nilearn
print(nilearn.__version__)


# In[8]:


from nilearn import plotting


# In[9]:


ls


# In[10]:


plotting.plot_glass_brain('dr_stage3_ic0001_tfce_corrp_tstat1.nii.gz')


# In[12]:


from nilearn import image


# In[13]:


smoothed_img = image.smooth_img('dr_stage3_ic0001_tfce_corrp_tstat1.nii.gz', fwhm=5)
print(type(smoothed_img))


# In[15]:


smoothed_img.to_filename('smoothed_image.nii.gz')


# now the image is saved

# In[17]:


index_0 = image.index_img('dr_stage2_subject00007_Z.nii.gz', 0)


# In[18]:


index_0.to_filename('index_0.nii.gz')


# In[22]:


for volume in image.iter_img('dr_stage2_subject00007_Z.nii.gz'):
    print(volume.shape)
    smoothed_img = image.smooth_img(volume, fwhm=5)


# # excercises 

# In[28]:


#download adhd
data = nilearn.datasets.fetch_adhd()


# In[37]:


print(data.keys())
print(data.func[0])


# In[38]:


first_subj = data.func[0]


# In[39]:


avg_1st_subj = image.mean_img(first_subj)


# In[40]:


avg_1st_subj.to_filename('1st_subj_avg.nii.gz')


# In[46]:


for i in range(0,25,5):
    
    print(i)
    smoothed_vol = image.smooth_img(avg_1st_subj, i)
    smoothed_vol.to_filename('1st_subj_avg_smoothed_{0}'.format(i))
    plotting.plot_epi(smoothed_vol)


# # <font color=red>8.1.2. Basic nilearn example: manipulating and looking at data
# 
# </font>

# In[48]:


from nilearn.datasets import MNI152_FILE_PATH


# In[49]:


MNI152_FILE_PATH


# In[51]:



print('Path to MNI152 template: {0}'.format(MNI152_FILE_PATH))


# In[58]:


plotting.plot_img(MNI152_FILE_PATH)


# Functions containing ‘img’ can take either a filename or an image as input. 
# 
# e.g smooth_img

# In[60]:


from nilearn import image


# In[61]:


smooth_anat_img = image.smooth_img(MNI152_FILE_PATH, fwhm=3)


# In[62]:


plotting.plot_anat(smooth_anat_img)


# In[67]:


#to get the header
get_ipython().run_line_magic('pinfo2', 'smooth_anat_img')
# or
print(smooth_anat_img)


# In[69]:



more_smooth_anat_img = image.smooth_img(smooth_anat_img, fwhm=3)


# In[70]:


plotting.plot_anat(more_smooth_anat_img)


# # <font color=red>8.1.3. 3D and 4D niimgs: handling and visualizing</font>

# In[73]:


from nilearn import datasets

print('Datasets are stored in:{0}'.format(datasets.get_data_dirs()))


# In[74]:


motor_images = datasets.fetch_neurovault_motor_task()
print(motor_images.images)


# In[81]:



type(motor_images.images)


# In[90]:


#you cannot plot a list, that's why you need the single file
tmap_filename = motor_images.images[0]


# In[97]:


plotting.plot_stat_map(tmap_filename)


# In[101]:


get_ipython().system('cp /Users/amr/nilearn_data/neurovault/collection_658/image_10426.nii.gz .')


# In[103]:


plotting.plot_stat_map(tmap_filename, threshold=3)


# In[104]:


plotting.plot_stat_map(tmap_filename, threshold=0)


# ## visualize 4D

# In[105]:


rsn = datasets.fetch_atlas_smith_2009()['rsn10']


# In[111]:


print(rsn)
print(type(rsn))
#a 4d map with 10 ICs


# In[113]:


get_ipython().system('fsleyes /Users/amr/nilearn_data/smith_2009/PNAS_Smith09_rsn10.nii.gz')


# In[120]:


print(image.load_img(rsn).shape)


# In[121]:


first_rsn = image.index_img(rsn, 0)


# In[123]:


type(first_rsn)


# In[127]:


plotting.plot_stat_map(first_rsn, threshold=1)


# In[129]:


for img in image.iter_img(rsn):
    plotting.plot_stat_map(img, threshold=3, display_mode='z', cut_coords=1,colorbar=False)


# In[130]:


import numpy as np


# In[131]:


t = np.linspace(1, 10, 2000)


# In[134]:


np.cos(t)


# In[135]:


import matplotlib.pyplot as plt


# In[136]:


plt.plot(t, np.cos(t))


# In[137]:


from scipy import ndimage
t_smooth = ndimage.gaussian_filter(t, sigma=2)


# In[138]:


from scipy import signal
t_detrended = signal.detrend(t)


# In[142]:


plt.plot(np.cos(t_detrended))


# In[143]:


import sklearn
print(sklearn.__version__)


# In[ ]:




