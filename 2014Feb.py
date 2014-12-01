# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 21:24:22 2014

@author: kazuharu
"""

import numpy as np
import scipy.ndimage as scindi
import matplotlib.pyplot as plt
import struct
import folder_location 

#%%

def USD_open_for_STS(fol_n, fil_n, imagexsize, specxsize, datasize, specflag):

    file_name = "%s/%s.usd" % (fol_n, fil_n);
    ifile = open(file_name, 'r')
    
    #global header
    dumy = ifile.read(848)

    def load_image_from_usd_file(ifile, xsize):
        #xsize = imagexsize
        total_pixel = xsize * xsize;
        dumy = ifile.read(648)
        
        y2 = ifile.read(total_pixel*2)
        val = struct.unpack('%dH'% total_pixel, y2)
        #y2 = t(as.integer(y));
        Im = np.array(val)
        Im = Im.reshape(xsize, xsize)
        return Im
        
    #Page1: CH1 image for
    Im1f = load_image_from_usd_file(ifile, imagexsize);    
    #Dum
    Imdum = load_image_from_usd_file(ifile, imagexsize);
    #Page2: CH3 image for
    Im2f = load_image_from_usd_file(ifile, imagexsize);
    # #Dum
    Imdum = load_image_from_usd_file(ifile, imagexsize);
    #Page3: TOPO image for
    Im3f = load_image_from_usd_file(ifile, imagexsize);
    
    #Page4: CH1 image back
    Im1b = load_image_from_usd_file(ifile, imagexsize);
    #Dum
    Imdum = load_image_from_usd_file(ifile, imagexsize);
    #Page5: CH3 image back
    Im2b = load_image_from_usd_file(ifile, imagexsize);
    # #Dum
    Imdum = load_image_from_usd_file(ifile, imagexsize);
    #Page6: TOPO image back
    Im3b = load_image_from_usd_file(ifile, imagexsize);

    #####
    ##### Spectrum Part #####
    #####
    
    #flag = 1    
    
    def load_spectrum_from_usd_file_STS(ifile, xsize, datasize, flag):

        if flag == 0:
        
            #simple
            total_point = xsize * xsize
            total_data = total_point * datasize
            sp = np.zeros(total_data)
            #Sp = c(matrix(0, total_data, 1));
            for i in range(0, total_point):
                dumy = ifile.read(648)
                #dumy = ifile.read(324)
    
                y2 = ifile.read(datasize*2)
                val = struct.unpack('%dH' % datasize, y2)
                sp_dum = np.array(val)
                sp[(i*datasize):((i+1)*datasize)] = sp_dum
            
            sp = sp.reshape((xsize*xsize),datasize)
            
            return sp

        else:
        
            #324?
            dumy = ifile.read(648)
            total_point = xsize * xsize
            total_data = total_point * datasize
    	
            sp = np.zeros(total_data)
            
            for i in range(0,(total_point - 1)):
                y2 = ifile.read(datasize*2)
                val = struct.unpack('%dH' % datasize, y2)
                sp_dum = np.array(val)
                sp[(i*datasize):((i+1)*datasize)] = sp_dum
                #dumy = ifile.read(324)
    
            i = total_point-1;
            y2 = ifile.read(datasize*2)
            val = struct.unpack('%dH' % datasize, y2)
            sp_dum = np.array(val)
            sp[(i*datasize):((i+1)*datasize)] = sp_dum
            
            sp = sp.reshape((xsize*xsize),datasize)
            
            return sp

        
    #Page1: CH1 spectrum for
    Sp1f = load_spectrum_from_usd_file_STS(ifile, specxsize, datasize, specflag);    
    #Page2: CH2 spectrum for
    Sp2f = load_spectrum_from_usd_file_STS(ifile, specxsize, datasize, specflag);
    #Page3: CH3 spectrum for
    Sp3f = load_spectrum_from_usd_file_STS(ifile, specxsize, datasize, specflag);
    #Dum
    Spdum = load_spectrum_from_usd_file_STS(ifile, specxsize, datasize, specflag);

    #Page4: CH1 spectrum back
    Sp1b = load_spectrum_from_usd_file_STS(ifile, specxsize, datasize, specflag);
    #Page5: CH2 spectrum back
    Sp2b = load_spectrum_from_usd_file_STS(ifile, specxsize, datasize, specflag);
    #Page6: CH3 spectrum back
    Sp3b = load_spectrum_from_usd_file_STS(ifile, specxsize, datasize, specflag);
    #Dum
    Spdum = load_spectrum_from_usd_file_STS(ifile, specxsize, datasize, specflag);

    ifile.close()	

    #plt.figure()
    #plt.plot(np.arange(0, datasize), Sp3f[100,:])
    
    return Im1f,Im1b, Im2f, Im2b, Im3f, Im3b, Sp1f, Sp1b, Sp2f, Sp2b, Sp3f, Sp3b

#%%

fol_n = folder_location.location("dropbox") + "R/2014Feb"; #01_12, 01_23, 02_04, 02_07
fil_n = "u4tp0013"; #1-17, 18-48, 49-73, 74-
imagexsize = 512
specxsize = 64
datasize = 2048

#flag is 0 for STS, 1 for CITS
specflag = 0

Im1f,Im1b, Im2f, Im2b, Im3f, Im3b, Sp1f, Sp1b, Sp2f, Sp2b, Sp3f, Sp3b = USD_open_for_STS(fol_n, fil_n, imagexsize, specxsize, datasize, specflag)

#ifile.close()
plt.figure()
plt.subplot(2,3,1)
plt.imshow(Im1f)    
plt.subplot(2,3,2)
plt.imshow(Im2f)    
plt.subplot(2,3,3)
plt.imshow(Im3f)    
plt.subplot(2,3,4)
plt.imshow(Im1b)    
plt.subplot(2,3,5)
plt.imshow(Im2b)    
plt.subplot(2,3,6)
plt.imshow(Im3b)    

bias = range(0, datasize)

#%%
#CH3


k_Sp3f = (Sp3f < 60000)

i = 1000

plt.figure()
plt.subplot(2,1,1)
plt.plot(bias, Sp3f[i,:])
plt.subplot(2,1,2)
plt.plot(bias, k_Sp3f[i,:])

Im_Sp3f = np.sum(k_Sp3f, axis = 1)
Im_Sp3f = Im_Sp3f.reshape(specxsize, specxsize)

plt.figure()
plt.imshow(Im_Sp3f)


k_Sp3b = (Sp3b < 60000)

i = 3000

plt.figure()
plt.subplot(2,1,1)
plt.plot(bias, Sp3b[i,:])
plt.subplot(2,1,2)
plt.plot(bias, k_Sp3b[i,:])

Im_Sp3b = np.sum(k_Sp3b, axis = 1)
Im_Sp3b = Im_Sp3b.reshape(specxsize, specxsize)

plt.figure()
plt.imshow(Im_Sp3b)

#%%
sigma = 2


plt.figure()
plt.imshow(scindi.filters.gaussian_filter(Im_Sp3f.astype(np.float),sigma))
plt.figure()
plt.imshow(scindi.filters.gaussian_filter(Im_Sp3b.astype(np.float),sigma))
