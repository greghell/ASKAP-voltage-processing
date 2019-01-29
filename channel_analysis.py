import glob
import os
import matplotlib.pyplot as plt
import sys
import operator
import numpy as np

fname = sys.argv[1]
nChan = int(sys.argv[2])
nResol = int(sys.argv[3])

fh = open(fname, "r")
head_beg = str(fh.read(4096))
    
idx = head_beg.find("HDR_SIZE ")
idx2 = head_beg.find("# Header size")
fh.seek(9+idx, 0)
hdrsize = int(str(fh.read(idx2 - idx - 9)))
print("header size : "+str(hdrsize))

idx = head_beg.find("NSAMPS_REQUEST ")
idx2 = head_beg.find("# The number of samples requested")
fh.seek(15+idx, 0)
nSam = int(str(fh.read(idx2 - idx - 15)))
print("Sample size : "+str(nSam))

idx = head_beg.find("\nFREQS ")
idx2 = head_beg.find("# Comma separated list")
fh.seek(7+idx, 0)
ffreqs = fh.read(idx2 - idx - 7)
ffreqs = ffreqs.split(",")
ffreqs = map(int, ffreqs)
print("Center frequency : "+str(ffreqs[nChan])+" MHz")

idx = head_beg.find("SAMP_RATE ")
idx2 = head_beg.find("# Sample rate in samples per second")
fh.seek(10+idx, 0)
fs = float(fh.read(idx2 - idx - 10))
print("Sampling frequency : "+str(fs/1.e6)+" MHz")
print("Frequency coverage : "+str(ffreqs[nChan]-(fs/1.e6)/2.)+" - "+str(ffreqs[nChan]+(fs/1.e6)/2.)+" MHz")
fh.close()
    
numSamples = int(nSam/nResol)
    
data = np.zeros((nResol,numSamples))
lut = np.array([0.,1.,2.,3.,4.,5.,6.,7.,-8.,-7.,-6.,-5.,-4.,-3.,-2.,-1.])

fh = open(fname, "rb")
fh.seek(hdrsize, 0)
sams = np.fromfile(fh, dtype='int8',count=8*nSam)
fh.close()
                
sigreal = lut[np.array(np.bitwise_and(sams, 0x0f))]
sigimag = lut[np.array(np.bitwise_and(sams >> 4, 0x0f))]
                
sig = sigreal + 1j*sigimag

sig = np.reshape(sig,(4,2*nSam),order='F')
sig = sig[:,nChan::8]
sig = np.reshape(sig,(1,nSam),order='F')
sig = np.reshape(sig[0,:nResol*int(nSam/nResol)],(nResol,int(nSam/nResol)),order='F')

sig = np.abs(np.fft.fftshift(np.fft.fft(sig,axis=0),axes=0))**2

plt.figure()
plt.subplot(121)
plt.plot(np.linspace(ffreqs[nChan]-(fs/1.e6)/2.,ffreqs[nChan]+(fs/1.e6)/2.,nResol),10.*np.log10(np.mean(sig,axis=1)))
plt.grid()
plt.xlabel('frequency [MHz]')
plt.ylabel('PSD [dB]')
plt.subplot(122)
plt.imshow(10.*np.log10(sig),aspect='auto',extent=[0.,nSam/1e6,ffreqs[nChan]-(fs/1.e6)/2.,ffreqs[nChan]+(fs/1.e6)/2.])
plt.xlabel('time [s]')
plt.ylabel('frequency [MHz]')
plt.show()
