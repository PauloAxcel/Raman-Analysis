# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:12:26 2018
 
@author: PAULO GOMES
"""
 
import scipy as scipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solveh_banded
from adjustText import adjust_text
import chart_studio.plotly as py
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy import stats
from collections import OrderedDict
from scipy import optimize
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline, BSpline
from brokenaxes import brokenaxes
from sklearn.preprocessing import LabelEncoder
 
def _1LOR_GAU(x, amp, cen, wid):
    return (amp/np.pi)*wid/((x-cen)**2+wid**2)
 
def shift_right(lst):
    try:
        return lst[1:] + [lst[0]]
    except IndexError:
        return lst
 
def despike(yi):
    
    y = np.copy(yi) # use y = y1 if it is OK to modify input array
    n = len(y)
    x = np.arange(n)
    c = 503
    s = 40
    popt_lorentz, pcov_lorentz = scipy.optimize.curve_fit(_1LOR_GAU, x, y, p0=[y.max(), c ,1])
    lorentz_peak_1 = _1LOR_GAU(x, popt_lorentz[0],popt_lorentz[1],popt_lorentz[2])
    diff = abs(y-lorentz_peak_1)
    diff[c-s:c+s] =  np.nan
 
    return (diff)
 
def als_baseline(intensities, asymmetry_param=0.001, smoothness_param=1e6,
                 max_iters=10, conv_thresh=1e-5, verbose=False):
  '''Computes the asymmetric least squares baseline.
  * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
  smoothness_param: Relative importance of smoothness of the predicted response.
  asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                       Setting p=1 is effectively a hinge loss.
  '''
  smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
  # Rename p for concision.
  p = asymmetry_param
  # Initialize weights.
  w = np.ones(intensities.shape[0])
  for i in range(max_iters):
    z = smoother.smooth(w)
    mask = intensities > z
    new_w = p*mask + (1-p)*(~mask)
    conv = np.linalg.norm(new_w - w)
    if verbose:
      print (i+1, conv)
    if conv < conv_thresh:
      break
    w = new_w
  else:
    print ('ALS did not converge in %d iterations' % max_iters)
  return z
 
 
class WhittakerSmoother(object):
  def __init__(self, signal2, smoothness_param, deriv_order=1):
    self.y = signal2
    assert deriv_order > 0, 'deriv_order must be an int > 0'
    # Compute the fixed derivative of identity (D).
    d = np.zeros(deriv_order*2 + 1, dtype=int)
    d[deriv_order] = 1
    d = np.diff(d, n=deriv_order)
    n = self.y.shape[0]
    k = len(d)
    s = float(smoothness_param)
 
    # Here be dragons: essentially we're faking a big banded matrix D,
    # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
    diag_sums = np.vstack([
        np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
        for i in range(1, k+1)])
    upper_bands = np.tile(diag_sums[:,-1:], n)
    upper_bands[:,:k] = diag_sums
    for i,ds in enumerate(diag_sums):
      upper_bands[i,-i-1:] = ds[::-1][:i+1]
    self.upper_bands = upper_bands
 
  def smooth(self, w):
    foo = self.upper_bands.copy()
    foo[-1] += w  # last row is the diagonal
    return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)
 
 
 
# From MatrixExp
def matrix_exp_eigen(U, s, t, x):
    exp_diag = np.diag(np.exp(s * t), 0)
    return U.dot(exp_diag.dot(U.transpose().dot(x)))
 
# From LineLaplacianBuilder
def get_line_laplacian_eigen(n):
    assert n > 1
    eigen_vectors = np.zeros([n, n])
    eigen_values = np.zeros([n])
 
    for j in range(1, n + 1):
        theta = np.pi * (j - 1) / (2 * n)
        sin = np.sin(theta)
        eigen_values[j - 1] = 4 * sin * sin
        if j == 0:
            sqrt = 1 / np.sqrt(n)
            for i in range(1, n + 1):
                eigen_vectors[i - 1, j - 1] = sqrt
        else:
            for i in range(1, n + 1):
                theta = (np.pi * (i - 0.5) * (j - 1)) / n
                math_sqrt = np.sqrt(2.0 / n)
                eigen_vectors[i - 1, j - 1] = math_sqrt * np.cos(theta)
    return eigen_vectors, eigen_values
 
def smooth2(t, signal3):
    dim = signal3.shape[0]
    U, s = get_line_laplacian_eigen(dim)
    return matrix_exp_eigen(U, -s, t, signal3)
 
 
################################
#####HISTOGRAM PART#############
#####lOAD DATA FILE RAMAN#######
##########INPUTS################
################################
        
#file = [
#        '19_07_2019\\SiAu\\SiAu_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '19_07_2019\\SiAuDSA\\SiAuDSA_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSAComp\\SiAuDSAComp_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST75\\SiAuST75_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST95\\SiAuST95_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv'
#        ]
 
#file = ['19_07_2019\\SiAuDSAComp\\SiAuDSAComp_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSAComp\\SiAuDSACompS2_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSACompW\\SiAuDSACompS2W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        ]
    
#file = [
#        '19_07_2019\\SiAu\\SiAu_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '19_07_2019\\SiAu\\SiAuS1_512map_static950_1sec_10%_1acc_785nm_pinhole_01111.csv',
#        '19_07_2019\\SiAu\\SiAuS2_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAu\\SiAuS3_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAu\\SiAuS4_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv'
#        ]
    
 
#file = ['19_07_2019\\SiAu\\SiAu_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '19_07_2019\\SiAuW\\SiAuS1W_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '19_07_2019\\SiAuW\\SiAuS2W_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '19_07_2019\\SiAuW\\SiAuS3W_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '19_07_2019\\SiAuW\\SiAuSCW_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv'
#        ]
 
#file = ['19_07_2019\\SiAuST75\\SiAuST75_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST75W\\SiAuST75S1W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST75W\\SiAuST75S2W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST75W\\SiAuST75S3W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST75W\\SiAuST75S4W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv'
#        ]
#    
#file = ['19_07_2019\\SiAuDSA\\SiAuDSA_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSAW\\SiAuDSAS1W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSAW\\SiAuDSAS2W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSAW\\SiAuDSAS3W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSAW\\SiAuDSAS4W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv'
#        ]
 
#file = ['19_07_2019\\SiAuST95\\SiAuST95_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST95W\\SiAuST95S1W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST95W\\SiAuST95S2W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST95W\\SiAuST95S3W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuST95W\\SiAuST95S4W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv'
#        ]
 
 
#file = ['19_07_2019\\SiAuDSAComp\\SiAuDSAComp_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSACompW\\SiAuDSACompS1W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSACompW\\SiAuDSACompS2W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSACompW\\SiAuDSACompS3W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv',
#        '19_07_2019\\SiAuDSACompW\\SiAuDSACompS4W_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.csv'
#        ]
#    
#file = ['17_08_2019\\ST75\\SiAuST75_rep1_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
##        '17_08_2019\\ST75\\SiAuST75_rep2_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '17_08_2019\\ST75\\SiAuST75_rep3_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '17_08_2019\\ST75\\SiAuST75_rep1_STA_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
##        '17_08_2019\\ST75\\SiAuST75_rep2_STA_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '17_08_2019\\ST75\\SiAuST75_rep3_STA_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv'
#        ]
 
#file = [
##        '17_08_2019\\ST95\\SiAuST95_rep1_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '17_08_2019\\ST95\\SiAuST95_rep2_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '17_08_2019\\ST95\\SiAuST95_rep3_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
##        '17_08_2019\\ST95\\SiAuST95_rep1_STA_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '17_08_2019\\ST95\\SiAuST95_rep2_STA_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv',
#        '17_08_2019\\ST95\\SiAuST95_rep3_STA_512map_static950_1sec_10%_1acc_785nm_pinhole_01.csv'
#        ]
 
#file = [
#        '31_08_2019\\ST75_EHD_r1_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
#        '31_08_2019\\ST75_EHD_r2_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
#        '31_08_2019\\ST75_EHD_r3_1s_10%_1acc_785nm_950static_100map_pinholein_011.csv',
#        '31_08_2019\\ST75_EHD_STA_r1_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
#        '31_08_2019\\ST75_EHD_STA_r2_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
#        '31_08_2019\\ST75_EHD_STA_r3_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv'        
#        ]
 
file = [
        '31_08_2019\\ST95_EHD_r1_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
        '31_08_2019\\ST95_EHD_r2_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
        '31_08_2019\\ST95_EHD_r3_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
        '31_08_2019\\ST95_EHD_STA_r1_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
        '31_08_2019\\ST95_EHD_STA_r2_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
        '31_08_2019\\ST95_EHD_STA_r3_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv'        
        ]
    
#file = [
#        '31_08_2019\\gold_EHD_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
#        '31_08_2019\\gold_EHD_r2_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv',
#        '31_08_2019\\gold_EHD_r3_1s_10%_1acc_785nm_950static_100map_pinholein_01.csv'
#        ]
 
# FILE FORMAT HAS THE FORM 
#####################################
#CENTER          # INTENSITY        #
#      ...       #       ...        #
#      ...       #       ...        #
#      ...       #       ...        #
#      ...       #       ...        #
#      ...       #       ...        #
#####################################
 
 
smoo = 5
 
tol = 5
 
#label = [
#        'SA',
#         'SADSA',
#         'SADSAC',
#         'SAST75',
#         'SAST95'
#         ]
#
#label = ['SADSAC',
#         'SADSACS',
#         'SADSACSW'
#         ]
 
#label = ['SA',
#         'SAMW',
#         'SASW',
#         'SARW',
#         'SAAW'
#         ]
#
#label = ['SAST75',
#         'SAST75MW',
#         'SAST75SW',
#         'SAST75RW',
#         'SAST75AW'
#         ]
 
#label = ['SADSA',
#         'SADSAMW',
#         'SADSASW',
#         'SADSARW',
#         'SADSAAW'
#         ]
 
#label = ['SADSAC',
#         'SADSACMW',
#         'SADSACSW',
#         'SADSACRW',
#         'SADSACAW'
#         ]
 
#label = ['SAST95',
#         'SAST95MW',
#         'SAST95SW',
#         'SAST95RW',
#         'SAST95AW'
#         ]
#
#label = ['SAST75R',
#         'SAST75R\'',
#         'SAST75R\'\'',
#         'SAST75SR',
#         'SAST75SR\'',
#         'SAST75SR\'\''
#         ]
 
#label = ['SAST95R',
#         'SAST95R\'',
#         'SAST95R\'\'',
#         'SAST95SR',
#         'SAST95SR\'',
#         'SAST95SR\'\''
#         ]
label = [
        'SAST95R1',
         'SAST95R',
         'SAST95R\'',
         'SAST95SR1',
         'SAST95SR',
         'SAST95SR\''
         ]
 
#label = [
#        'AU',
#        'AU',
#        'AU'
#        ]
 
colors = [
        'r',
        'g',
        'b',
        'c',
        'y',
        'purple'
        ]
 
plt.style.use('fivethirtyeight')
 
fig1 = plt.figure()
#bax1 = brokenaxes(xlims=((325,485), (570, 1490)),hspace=.05)
bax1 = fig1.add_subplot(111)
#ax2 = fig1.add_subplot(212)   
 
bonds = []
 
y0 = []
rel_y0 = []
classe = []
text_file = 'EHD_AU_replicates_was.tex'
title_name = 'EHD AU replicates before and after Stachyose immersion'
 
#JUST BECAUSE OF THE SILICON PEAK
func_cut_off = 600
 
with open(text_file,'w',encoding="utf-8") as tf:
    tf.write("\\documentclass{article}\n\\usepackage{array}\n\\usepackage{booktabs}\n\\begin{document}\n\\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}")
 
######################################################################################
 
for k in range(len(file)):
        
    df_1=pd.read_csv(file[k],sep=';',header=None)
    
    n_row = sum(df_1.iloc[0,0] == df_1.iloc[:,0])
    n_parts=df_1.shape[0]//n_row
    
    a = []
    df_1_2 = pd.DataFrame()
    
    xData = []
    yData = []
       
    for i in range(n_row):
    
        xData.append(df_1.iloc[i*n_parts:(i+1)*n_parts,0].values)
        yData.append(df_1.iloc[i*n_parts:(i+1)*n_parts,1].values)
        y0.append(df_1.iloc[i*n_parts:(i+1)*n_parts,1].values)
    
    xData = pd.DataFrame(xData).mean()
 
    index = xData.astype(int) == func_cut_off
    index = [i for i, x in enumerate(index) if x][0]
    
    xData = xData[:index]
    
    yDataDes = pd.DataFrame(yData).std()
    
    yDataDes = yDataDes[:index]
    
    yData = pd.DataFrame(yData).mean()    
    
    yData = yData[:index]
    
        ######PLOT DATA-BASELINE########
    
    base = als_baseline(yData)
        
    #    # Signal smoothing 
    
    smoothed_signal1 = smooth2(smoo,  yData)
    smoothed_signal = smoothed_signal1 - base - min(base) - min(smoothed_signal1 - base - min(base))
 
          
    dftest = pd.read_csv('C:\\Users\\paulo\\Desktop\\birmingham_02\\Molecular Imprint surfaces\\MI\\experimental work\\RAMAN\MI completed\\data anaylisis with python\\raman peak sugars.csv',sep=';',header=None)
    dftest = dftest.dropna(axis='columns')
    dftest.columns = ['bond','range-','range+','average','intensity']
    
#    xnew = np.linspace(xData.min() , xData.max(),3000)
#    spl = make_interp_spline(sorted(xData), smoothed_signal[::-1])
#    power_smooth = spl(xnew)
#    
##    smoothed_signal = despike(power_smooth)
#    smoothed_signal = power_smooth
    
    
    ##############################################################################
    #########WRITE LABELS ON THE GRAPH WITH RESPECTIVE BONDS######################
    ##############################################################################
    
    rel_y = []
    
 
    for n in range(smoothed_signal.shape[0]):
#        rel_y.append( (smoothed_signal[n] - np.nanmin(smoothed_signal)) / (  np.nanmax(smoothed_signal) -np.nanmin(smoothed_signal) ))
#    
        rel_y.append( (smoothed_signal[n] - smoothed_signal.min() ) / ( smoothed_signal.max() -smoothed_signal.min()) )
    
 
    rel_y = pd.DataFrame(rel_y)
    rel_y = rel_y.iloc[:,0]
    
    peaksf, _ = find_peaks(rel_y,distance=10)  
 
    bax1.plot(xData, rel_y+k ,label = label[k], color=colors[k])
#    bax1.fill_between(xData,k+rel_y-yDataDes/(yDataDes.max()*10),k+rel_y+yDataDes/(yDataDes.max()*10),color=colors[k], alpha=0.3)
#    bax1.vlines(xData[peaksf], ymin=(rel_y+k).min(),ymax=(rel_y+k).max(),ls='dotted',linewidth=1)
    bax1.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize = 16)
    bax1.set_ylabel('Relative Intensity (a.u.)', fontsize = 16)
#    bax1.set_title('Raman Spectrum before and after stachyose on DSA + Complex substrate', fontsize = 20,y=1.02)
#    bax1.set_title('Raman Spectrum of different sugars over a Gold substrate', fontsize = 20,y=1.02)
    bax1.set_title('Raman Spectrum of '+title_name, fontsize = 20,y=1.02)
    bax1.legend(loc='upper left')
#    bax1.set_xlim(570,xnew.max())
    
 
    texts = []
    texts_aux = []
    
    peak_loc = xData[peaksf].reset_index(drop=True)
    
    for i in range(dftest.shape[0]):
        for j in range(peak_loc.shape[0]):
            if abs(peak_loc[j]-dftest.iloc[i,3])<tol:
                texts.append([peak_loc[j] , rel_y[peaksf].iloc[j] , dftest.iloc[i,0],dftest.iloc[i,4]])
                
    
    
    df_text = pd.DataFrame(texts)
    df_text.columns = ['peak' , 'int', 'label','stre']
    
    items = list(df_text.iloc[:,0])
    
    values = []
    
    values.append(items[0])
    
    for m in range(len(items)-1):
        if abs(items[m]-items[m+1])<5:
            values.append(np.nan)
        else:
            values.append(items[m+1])
    
      
    column_f = np.ndarray.tolist(df_text.iloc[:,0].values)
    center_f = np.ndarray.tolist(pd.DataFrame(values).fillna(method='ffill').iloc[:,0].values)
    height_f = np.ndarray.tolist(df_text.iloc[:,1].values)
    text_f = np.ndarray.tolist(df_text.iloc[:,2].values)
    strength_f = np.ndarray.tolist(df_text.iloc[:,3].values)
    
    #INCLUDE OR EXCLUDE TO SHOW OFF LABELS RELATED TO SUGAR PREAKS
#    if k !=0:
    for x, y in zip(center_f, height_f):
#        if x > 570:
        texts_aux.append(bax1.text(x, y +k, str(int(x)), fontsize=12, rotation='vertical'))
#            if x <500 or x>540:
#                texts_aux.append(bax1.text(x, y+k , str(int(x)), fontsize=16, rotation='vertical'))
 
 
##############################################################################
#########MAKE TABLE RELATING THE INTENSITY WITH THE LABEL STRENGTH############
##############################################################################
    
    
    a=[]
    
    
    for i in range(len(strength_f)):
        if strength_f[i] == 'vs' or strength_f[i] == 's':
            a.append([text_f[i] , round(100*(height_f[i]-min(height_f))/(max(height_f)-min(height_f)),1)])
        if strength_f[i] == 'm' or strength_f[i] == 'sm' or strength_f[i] == 'mw':
            a.append([text_f[i] , round(100*(height_f[i]-min(height_f))/(max(height_f)-min(height_f)),1)])
        if strength_f[i] == 'w' or strength_f[i] == 'vw':
            a.append([text_f[i] , round(100*(height_f[i]-min(height_f))/(max(height_f)-min(height_f)),1)])
     
    df_bond = pd.DataFrame(a)
    df_bond.columns = ['Bonds','Accuracy']
 
 
    all_df = pd.concat([df_bond['Bonds'],pd.DataFrame(column_f)[0].round(decimals=2),pd.DataFrame(height_f)[0].round(decimals=2)], axis=1)
    all_df.columns=[label[k]+' Bonds' , 'Center $(cm^{-1})$' , 'Relative Intensity (a.u.)']
    all_df = all_df.set_index(label[k]+' Bonds')
 
 
    with open(text_file,'a',encoding="utf-8") as tf:
        tf.write(all_df.to_latex()) 
        tf.write("\n\\newpage\n")
 
  
   
 
y0 = pd.DataFrame(y0)
 
y0 = y0.T[:index].T
 
y0_0 = []
 
for o in range(y0.shape[0]):
    base = als_baseline(y0.iloc[o,:])
    diff = abs(base-y0.iloc[o,:])
    y0_0.append(diff)    
    
y0 = pd.DataFrame(y0_0)
 
 
for p in range(y0.shape[0]):
    rel_y0.append((y0.iloc[p,:]-y0.iloc[p,:].min())/(y0.iloc[p,:].max()-y0.iloc[p,:].min()))
    classe.append(label[p*len(label)//y0.shape[0]])
#    for o in range(len(label)):
#        if classe[p] == label[o]:
#            classe[p] = o
            
#    [i for i, x in enumerate(index) if x][0]
 
    
 
rel_y0 = pd.DataFrame(rel_y0, columns = y0.columns)
classe = pd.DataFrame(classe)
 
#rel_y00 = StandardScaler().fit_transform(rel_y0)
 
##########################################################################################################
############################PCA###########################################################################
##########################################################################################################
############################LOADINGS######################################################################
##########################################################################################################
 
 
 
for i in range(len(label)):
    latex_table = []
    if i == 0:
        a = 1
    else:
        a = 0
    if i == 1:
        b = 1
    else:
        b = 0
    if i == 2:
        c = 1
    else:
        c = 0
    if i == 3:
        d = 1
    else:
        d=0
    if i == 4:
        e = 1
    else:
        e=0
    base = rel_y0[i*n_row:(i+1)*n_row]
    result1 = rel_y0[(i+1)%len(label)*n_row:(i+2)%(len(label)+e)*n_row]
    result2 = rel_y0[(i+2)%len(label)*n_row:(i+3)%(len(label)+d)*n_row]
    result3 = rel_y0[(i+3)%len(label)*n_row:(i+4)%(len(label)+c)*n_row]
    result4 = rel_y0[(i+4)%len(label)*n_row:(i+5)%(len(label)+b)*n_row]
    result5 = rel_y0[(i+5)%len(label)*n_row:(i+6)%(len(label)+a)*n_row]
#    result = [result1,result2]
    
    result = [result1,result2,result3,result4,result5]
#    result = [result1,result2,result3]
    
    fig = plt.figure()
    
    for j in range(len(result)):
        dataset = pd.concat([base,result[j]])
        n_comp = 2
        
        
        pca = PCA(n_components=n_comp)
        principalComponents = pca.fit_transform(dataset)
        columns = ['principal component '+str(i) for i in range(1,n_comp+1)]
        principalDf = pd.DataFrame(data = principalComponents , columns = columns)
        finalDf = pd.concat([principalDf, classe], axis = 1)
        finalDf = finalDf.dropna()
        z = np.abs(stats.zscore(finalDf.iloc[:,:2]))
        finalDf = finalDf[(z < 3).all(axis=1)]
        finalDf.reset_index(drop=True)
 
 
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        for k in range(n_comp):
            loadings_ = loadings[:,k]
#            smooth2(smoo, loadings_)
            smoothed = smooth2(smoo, loadings_)
            peaksf, _ = find_peaks(smoothed,distance=10) 
            
            plt.plot(xData,smoothed,label='L'+str(1+k)+' '+label[i]+' vs '+classe[0][dataset.index[-1]]+'(var: %d%%)' %(pca.explained_variance_ratio_[k]*100),color=colors[1:len(colors)][j],alpha=(n_comp-k)/(n_comp))
            plt.hlines(0, xmin=xData.min(),xmax=xData.max(),ls='dotted',linewidth=1)
            plt.xlabel('Raman Shift (cm$^{-1}$)', fontsize = 16)
            plt.ylabel('PCA Loadings', fontsize = 16)
            plt.title('Loadings of '+label[i]+' compared to the other replicates', fontsize = 20 , y = 1.02)
            
#            for x, y in zip(xData[peaksf],smoothed[peaksf]):
#                for l in range(dftest.iloc[:,3].shape[0]):
#                    if abs(x-dftest.iloc[l,3])<tol:
#                        plt.text(x, y, str(int(x)), fontsize=12, rotation='vertical')
                
#        latex_table.append([dftest.iloc[l,0],round(x,2),y.round(decimals=3)])
            
        plt.legend(loc='best')
        plt.show()
    colors = shift_right(colors)
 
 
#    latex_table = pd.DataFrame(latex_table)
#    latex_table.columns=[label[i]+' Bonds' , 'Center $(cm^{-1})$' , 'Relative Intensity (a.u.)']
#    latex_table = latex_table.set_index(label[i]+' Bonds')
 
#    with open(text_file,'a',encoding="utf-8") as tf:
#        tf.write(latex_table.to_latex())
#        tf.write("\n\\newpage\n")
 
 
with open(text_file,'a',encoding="utf-8") as tf:
    tf.write("\n\\end{document}")       
    
pca_dist = []
pca_dist_std = []
 
pca = PCA(n_components=len(label))
principalComponents = pca.fit_transform(rel_y0)
columns = ['principal component '+str(i) for i in range(1,len(label)+1)]
principalDf = pd.DataFrame(data = principalComponents , columns = columns)
finalDf = pd.concat([principalDf, classe], axis = 1)
z = np.abs(stats.zscore(finalDf.iloc[:,:2]))
finalDf = finalDf[(z < 3).all(axis=1)]
finalDf.reset_index(drop=True)
 
fig = plt.figure()
 
plt.xlabel('PC 1 (var: %d%%)' %(pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (var: %d%%)' %(pca.explained_variance_ratio_[1]*100))
plt.title('PCA of '+title_name)
 
for target, color in zip(label,colors):
    indicesToKeep = classe.iloc[:,0] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s =75
               ,alpha=0.7
               ,label=target
              )
    
    pca_dist.append([finalDf.loc[indicesToKeep, 'principal component 1'].mean(),finalDf.loc[indicesToKeep, 'principal component 2'].mean(),target])
    pca_dist_std.append([finalDf.loc[indicesToKeep, 'principal component 1'].std(),finalDf.loc[indicesToKeep, 'principal component 2'].std(),target])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(loc='best')
    
plt.show()
 
 
##########################################################################################################
############################LDA###########################################################################
##########################################################################################################
 
lda = LinearDiscriminantAnalysis(n_components=len(label)-1)
principalComponentslda = lda.fit(rel_y0,classe[0]).transform(rel_y0)
columns = ['principal component '+str(i) for i in range(1,len(label))]
principalDflda = pd.DataFrame(data = principalComponentslda , columns = columns)
finalDflda = pd.concat([principalDflda, classe], axis = 1)
z = np.abs(stats.zscore(finalDflda.iloc[:,:2]))
finalDflda = finalDflda[(z < 3).all(axis=1)]
finalDflda.reset_index(drop=True)
#
##loadings = lda.coef_.T * np.sqrt(lda.explained_variance_ratio_)
#loadings = lda.scalings_*np.sqrt(lda.explained_variance_ratio_)
#
#fig = plt.figure()
#
#latex_table = []
#
#for j in range(len(label)-1):
#    loadings_ = loadings[:,j]
##    loadings_ = np.zeros(len(loadings))
##    for i in range(len(loadings)):
##        loadings_[i] = ((loadings[i,j]-loadings[:,j].min())/(loadings[:,j].max()-loadings[:,j].min()))
#    
#    smoothed = smooth2(smoo, loadings_)
#    peaksf, _ = find_peaks(smoothed,distance=10) 
#    
#    plt.plot(xData,smoothed,label='L'+str(1+j),color='m',alpha=(len(label)-j)/len(label))
#
#    
##    plt.plot(xData,smoothed,label='L'+str(1+j)+'(var: %d%%)' %(lda.explained_variance_ratio_[j]*100),color='m',alpha=(len(label)-j)/len(label))
##    plt.vlines(xData[peaksf], ymin=(smoothed).min(),ymax=(smoothed).max(),ls='dotted',linewidth=1)
#    plt.hlines(0, xmin=xData.min(),xmax=xData.max(),ls='dotted',linewidth=1)
#    
##    plt.plot(xData,loadings_+j,label='L'+str(1+j)+'(var: %d%%)' %(pca.explained_variance_ratio_[j]*100),color='k',alpha=(len(label)-j)/len(label))
##    plt.vlines(xData[peaksf], ymin=(loadings_+j).min(),ymax=(loadings_+j).max(),ls='dotted',linewidth=1)
#    
#    for x, y in zip(xData[peaksf],smoothed[peaksf]):
##        if x > 570:
##        plt.text(x, y+j+0.1 , str(int(x)), fontsize=16, rotation='vertical')
#        for i in range(dftest.iloc[:,3].shape[0]):
#            if abs(x-dftest.iloc[i,3])<tol:
#                plt.text(x, y, str(int(x)), fontsize=12, rotation='vertical')
#                
#                latex_table.append([dftest.iloc[i,0],round(x,2),y.round(decimals=3)])
#
#    
#    plt.xlabel('Raman Shift (cm$^{-1}$)', fontsize = 16)
#    plt.ylabel('LDA Loadings', fontsize = 16)
#    plt.title('Loadings of '+title_name, fontsize = 20 , y = 1.02)
##    plt.xlim(570,xData.max())
#    
#plt.legend(loc='best')
#plt.show()
#
#
#
#
#
#
#
#
#
#
#
fig = plt.figure()
 
lda_dist = []
lda_dist_std = []
 
plt.xlabel('LD 1 (var: %d%%)' %(lda.explained_variance_ratio_[0]*100))
plt.ylabel('LD 2 (var: %d%%)' %(lda.explained_variance_ratio_[1]*100))
plt.title('LDA of '+title_name)
 
for target, color in zip(label,colors):
    indicesToKeep = classe.iloc[:,0] == target
    plt.scatter(finalDflda.loc[indicesToKeep, 'principal component 1']
               , finalDflda.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s =75,
               alpha=0.7,
               label=target
               )
    lda_dist.append([finalDflda.loc[indicesToKeep, 'principal component 1'].mean(),finalDflda.loc[indicesToKeep, 'principal component 2'].mean(),target])
    lda_dist_std.append([finalDflda.loc[indicesToKeep, 'principal component 1'].std(),finalDflda.loc[indicesToKeep, 'principal component 2'].std(),target])
 
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(loc='best')
 
 
plt.show()
 
 
 
 
lda_dist = pd.DataFrame(lda_dist)
lda_dist_std = pd.DataFrame(lda_dist_std)
pca_dist = pd.DataFrame(pca_dist)
pca_dist_std = pd.DataFrame(pca_dist_std)
 
#CODE FOR COMPARING BETWEEN DATA FROM 17/08
 
#distance_names = []
#for i in range(len(label)//2):
#    distance_names.append('|'+label[0]+'-'+label[2+i]+'|')
#
#distance = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
#distance_err = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
#cmap = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
#
#for i in range(len(label)//2):
#    distance.iloc[0,i] = np.sqrt((pca_dist.iloc[i,0]-pca_dist.iloc[i+2,0])**2+(pca_dist.iloc[i,1]-pca_dist.iloc[i+2,1])**2)
#    distance.iloc[1,i] = np.sqrt((lda_dist.iloc[i,0]-lda_dist.iloc[i+2,0])**2+(lda_dist.iloc[i,1]-lda_dist.iloc[i+2,1])**2)
#
#
#    distance_err.iloc[0,i] =  np.sqrt((((pca_dist.iloc[i,0]-pca_dist.iloc[i+2,0])**2*(pca_dist_std.iloc[i,0]-pca_dist_std.iloc[i+2,0])**2)+(pca_dist.iloc[i,1]-pca_dist.iloc[i+2,1])**2*(pca_dist_std.iloc[i,1]-pca_dist_std.iloc[i+2,1])**2)/((pca_dist.iloc[i,0]-pca_dist.iloc[i+2,0])**2+(pca_dist_std.iloc[i,0]-pca_dist_std.iloc[i+2,0])**2))
#    distance_err.iloc[1,i] =  np.sqrt((((lda_dist.iloc[i,0]-lda_dist.iloc[i+2,0])**2*(lda_dist_std.iloc[i,0]-lda_dist_std.iloc[i+2,0])**2)+(lda_dist.iloc[i,1]-lda_dist.iloc[i+2,1])**2*(lda_dist_std.iloc[i,1]-lda_dist_std.iloc[i+2,1])**2)/((lda_dist.iloc[i,0]-lda_dist.iloc[i+2,0])**2+(lda_dist_std.iloc[i,0]-lda_dist_std.iloc[i+2,0])**2))
#
#    cmap.iloc[0,i] = str(round(distance.iloc[0,i],2))+' +/- '+str(round(distance_err.iloc[0,i],2))
#    cmap.iloc[1,i] = str(round(distance.iloc[1,i],2))+' +/- '+str(round(distance_err.iloc[1,i],2))
#
#
#num = round(distance.max().max())
#if (num % 2) != 0:
#    num += 1
#    
#import seaborn as sns
#distance = distance[distance.columns].astype(float)
#plt.figure()
#ax = sns.heatmap(distance,annot=cmap.values,vmin=0,vmax=num,annot_kws={"size": 20}, fmt = '',cmap='YlGnBu_r',cbar=False)
#ax.set_title('Distance between cluster center of samples before and after sugar immersion', fontsize=18)
#ax.set_xlabel('Samples', fontsize=18)
#ax.set_ylabel('Data Analysis Method', fontsize=18)
#ax.set_ylim(0,len(cmap.index))
#ax.set_xlim(0,len(cmap.columns))
#plt.show()
 
 
 
#CODE FOR COMPARING BETWEEN DATA FROM 19/07
 
#distance_names = []
#for i in range(1,len(label)):
#    distance_names.append('|'+label[0]+'-'+label[i]+'|')
#
#distance = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
#distance_err = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
#cmap = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
#
#for i in range(1,len(label)):
#    distance.iloc[0,i-1] = np.sqrt((pca_dist.iloc[0,0]-pca_dist.iloc[i,0])**2+(pca_dist.iloc[0,1]-pca_dist.iloc[i,1])**2)
#    distance.iloc[1,i-1] = np.sqrt((lda_dist.iloc[0,0]-lda_dist.iloc[i,0])**2+(lda_dist.iloc[0,1]-lda_dist.iloc[i,1])**2)
#
#    distance_err.iloc[0,i-1] = np.sqrt((((pca_dist.iloc[0,0]-pca_dist.iloc[i,0])**2*(pca_dist_std.iloc[0,0]-pca_dist_std.iloc[i,0])**2)+(pca_dist.iloc[0,1]-pca_dist.iloc[i,1])**2*(pca_dist_std.iloc[0,1]-pca_dist_std.iloc[i,1])**2)/((pca_dist.iloc[0,0]-pca_dist.iloc[i,0])**2+(pca_dist.iloc[0,1]-pca_dist.iloc[i,1])**2))
#    distance_err.iloc[1,i-1] = np.sqrt((((lda_dist.iloc[0,0]-lda_dist.iloc[i,0])**2*(lda_dist_std.iloc[0,0]-lda_dist_std.iloc[i,0])**2)+(lda_dist.iloc[0,1]-lda_dist.iloc[i,1])**2*(lda_dist_std.iloc[0,1]-lda_dist_std.iloc[i,1])**2)/((lda_dist.iloc[0,0]-lda_dist.iloc[i,0])**2+(lda_dist.iloc[0,1]-lda_dist.iloc[i,1])**2))
#
#    cmap.iloc[0,i-1] = str(round(distance.iloc[0,i-1],2))+' +/- '+str(round(distance_err.iloc[0,i-1],2))
#    cmap.iloc[1,i-1] = str(round(distance.iloc[1,i-1],2))+' +/- '+str(round(distance_err.iloc[1,i-1],2))
#
#num = round(distance.max().max())
#if (num % 2) != 0:
#    num += 1
#    
#import seaborn as sns
#distance = distance[distance.columns].astype(float)
#plt.figure()
#ax = sns.heatmap(distance,annot=cmap.values,vmin=0,vmax=num,annot_kws={"size": 20}, fmt = '',cmap='YlGnBu_r',cbar=False)
#ax.set_title('Distance between cluster center of samples before and after sugar immersion', fontsize=18)
#ax.set_xlabel('Samples', fontsize=18)
#ax.set_ylabel('Data Analysis Method', fontsize=18)
#ax.set_ylim(0,len(cmap.index))
#ax.set_xlim(0,len(cmap.columns))
#ax.set_xticklabels(distance.columns,rotation=0)
#plt.show()
#
 
 
#CODE FOR COMPARING BETWEEN DATA FROM 31/08
 
distance_names = []
escala = len(label)//2
for i in range(escala):
    distance_names.append('|'+label[i]+'-'+label[i+escala]+'|')
 
distance = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
distance_err = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
cmap = pd.DataFrame(index = ['PCA','LDA'] , columns = distance_names)
 
for i in range(escala):
    distance.iloc[0,i] = np.sqrt((pca_dist.iloc[i,0]-pca_dist.iloc[i+escala,0])**2+(pca_dist.iloc[i,1]-pca_dist.iloc[i+escala,1])**2)
    distance.iloc[1,i] = np.sqrt((lda_dist.iloc[i,0]-lda_dist.iloc[i+escala,0])**2+(lda_dist.iloc[i,1]-lda_dist.iloc[i+escala,1])**2)
 
 
    distance_err.iloc[0,i] =  np.sqrt((((pca_dist.iloc[i,0]-pca_dist.iloc[i+escala,0])**2*(pca_dist_std.iloc[i,0]-pca_dist_std.iloc[i+escala,0])**2)+(pca_dist.iloc[i,1]-pca_dist.iloc[i+escala,1])**2*(pca_dist_std.iloc[i,1]-pca_dist_std.iloc[i+escala,1])**2)/((pca_dist.iloc[i,0]-pca_dist.iloc[i+escala,0])**2+(pca_dist_std.iloc[i,0]-pca_dist_std.iloc[i+escala,0])**2))
    distance_err.iloc[1,i] =  np.sqrt((((lda_dist.iloc[i,0]-lda_dist.iloc[i+escala,0])**2*(lda_dist_std.iloc[i,0]-lda_dist_std.iloc[i+escala,0])**2)+(lda_dist.iloc[i,1]-lda_dist.iloc[i+escala,1])**2*(lda_dist_std.iloc[i,1]-lda_dist_std.iloc[i+escala,1])**2)/((lda_dist.iloc[i,0]-lda_dist.iloc[i+escala,0])**2+(lda_dist_std.iloc[i,0]-lda_dist_std.iloc[i+escala,0])**2))
 
    cmap.iloc[0,i] = str(round(distance.iloc[0,i],2))+' +/- '+str(round(distance_err.iloc[0,i],2))
    cmap.iloc[1,i] = str(round(distance.iloc[1,i],2))+' +/- '+str(round(distance_err.iloc[1,i],2))
 
 
num = round(distance.max().max())
if (num % 2) != 0:
    num += 1
    
import seaborn as sns
distance = distance[distance.columns].astype(float)
plt.figure()
ax = sns.heatmap(distance,annot=cmap.values,vmin=0,vmax=num,annot_kws={"size": 20}, fmt = '',cmap='YlGnBu_r',cbar=False)
ax.set_title('Distance between cluster center of samples before and after sugar immersion', fontsize=18)
ax.set_xlabel('Samples', fontsize=18)
ax.set_ylabel('Data Analysis Method', fontsize=18)
ax.set_ylim(0,len(cmap.index))
ax.set_xlim(0,len(cmap.columns))
ax.set_xticklabels(distance.columns,rotation=0)
plt.show()
