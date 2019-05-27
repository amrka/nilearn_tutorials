#!/usr/bin/env python
# coding: utf-8

# # <font color=red>3.5.3.2. Compressed representation 

# ## 8.4.12. Clustering methods to learn a brain parcellation from rest fMRI 

# In[13]:


from nilearn import datasets, image, plotting
dataset = datasets.fetch_adhd(n_subjects=1)


# ### <font color=green> Brain parcellations with Ward Clustering 

# In[4]:


from nilearn.regions import Parcellations


# In[75]:


import time
start = time.time()


# In[76]:


ward = Parcellations(method='ward', n_parcels=1000, standardize=False, smoothing_fwhm=2.,
                    memory='nilearn_cache/', memory_level=1, verbose=1)


# In[77]:


ward.fit(dataset.func)
print('Ward agglomeration 1000 clusters: {0}'.format(time.time() - start))


# In[114]:


start = time.time()
ward = Parcellations(method='ward', n_parcels=2000, standardize=False,
                    smoothing_fwhm=2., memory='nilearn_cache/', memory_level=1, verbose=1)
ward.fit(dataset.func)
print('Ward agglomeration 2000 clusters: {0}'.format(time.time()- start))
         


# <font color=blue>because of caching, the first one with 1000 took 164 sec, the second took only 13 secs

# In[115]:


ward_labels_img = ward.labels_img_


# In[116]:


ward_labels_img.to_filename('ward_parcellation.nii.gz')


# In[124]:


first_plot = plotting.plot_roi(ward_labels_img, title='Ward parcellation', display_mode='xz')
first_plot.savefig('ward2_display.png')
cut_coords = first_plot.cut_coords


# In[82]:


import numpy as np
original_voxels = np.sum(ward.mask_img_.get_data())
print('Number of voxels in the original image is: {0}'.format(original_voxels))


# In[83]:


#mean functional image to use it later as a background
mean_func_img = image.mean_img(dataset.func[0])


# In[84]:


plotting.plot_epi(mean_func_img)


# In[85]:


vmin = np.min(mean_func_img.get_data())
vmax = np.max(mean_func_img.get_data())


# In[86]:


plotting.plot_epi(mean_func_img, cut_coords=cut_coords, 
                  title='original ({0} voxels)'.format(original_voxels), vmax=vmax,
                 vmin=vmin, display_mode='xz')


# In[87]:


fmri_reduced = ward.transform(dataset.func)
fmri_compressed = ward.inverse_transform(fmri_reduced)


# <font color=blue>it clusters on all the 176 volumes of the functional 4D, that is why we calculated mean_func to compare it to one of the volumes here

# In[88]:


plotting.plot_epi(image.index_img(fmri_compressed,0),
                 cut_coords=cut_coords,
                 title='Ward compressed represenation (2000 parcels)',
                 vmin=vmin, vmax=vmax, display_mode='xz')


# ### <font color=green>8.4.12.6. Brain parcellations with KMeans Clustering

# In[91]:


kmeans = Parcellations(method='kmeans', n_parcels=50, standardize=True, smoothing_fwhm=10., memory='nilearn_cache/',
                      memory_level=1, verbose=1)
kmeans.fit(dataset.func)
print('KMeans 50 clusters: {0}'.format(time.time()-start))


# In[92]:


kmeans_labels_img = kmeans.labels_img_


# In[101]:


second_plot = plotting.plot_roi(kmeans_labels_img, mean_func_img, title='KMeans parcellation',
                 display_mode='xz')

cut_coords = second_plot.cut_coords


# In[102]:


kmeans_labels_img.to_filename('kmeans_parcellation.nii.gz')


# In[103]:


fmri_reduced = kmeans.transform(dataset.func)
fmri_compressed = kmeans.inverse_transform(fmri_reduced)

plotting.plot_epi(image.index_img(fmri_compressed,0),
                 cut_coords=cut_coords,
                 title='Ward compressed represenation (2000 parcels)',
                 vmin=vmin, vmax=vmax, display_mode='xz')


# In[108]:


get_ipython().run_line_magic('pinfo', 'plotting.plot_epi')


# 
# <font color=blue>
# Warning
# 
# Opening too many figures without closing
# 
# Each call to a plotting function creates a new figure by default. When used in non-interactive settings, such as a script or a program, these are not displayed, but still accumulate and eventually lead to slowing the execution and running out of memory.
# 
# To avoid this, you must close the plot as follow:
# >>>
# 
# >>> from nilearn import plotting
# >>> display = plotting.plot_stat_map(img)     
# >>> display.close()     
# 
# 

# In[125]:


from nilearn import plotting, datasets     
img = datasets.fetch_localizer_button_task()['tmap']     
view = plotting.view_img_on_surf(img, threshold='90%', surf_mesh='fsaverage') 


# In[126]:


view


# In[ ]:




