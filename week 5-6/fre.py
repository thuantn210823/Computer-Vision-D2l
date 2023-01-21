import numpy as np

def format_pic(gray, pad = (0,0)):
    if pad != (0,0):
        gray_pad = np.zeros(pad)
        gray_pad[:gray.shape[0], :gray.shape[1]] = gray
        return gray_pad
    else:
        return gray
    
def crop_pic(gray_pad, gray, kernel = None):
    if kernel is None:
        return gray_pad[:gray.shape[0],:gray.shape[1]]
    else:
        return gray_pad[(kernel.shape[0]-1)//2+1 : (kernel.shape[0]-1)//2+gray.shape[0], (kernel.shape[1]-1)//2+1 : (kernel.shape[1]-1)//2+gray.shape[1]]

class Frequency_Domain:
    """
    """
    def __init__(self, arr, format_pic = format_pic, Fre = True, kwargs = {'pad' : (0,0)}):
        if Fre == True:
            self.arr = arr
            self.fft2 = arr
            self.fft2_shift = np.fft.fftshift(self.fft2)
        else:
            if format_pic is None:
                format_pic_ = arr
            else:
                format_pic_ = format_pic(arr, **kwargs)
            self.arr = arr
            self.fft2 = np.fft.fft2(format_pic_)
            self.fft2_shift = np.fft.fftshift(self.fft2)
    def __str__(self):
        return str(self.fft2)
    def __mul__(self, kernel):
        return Frequency_Domain(self.fft2*kernel.fft2)
    def __getitem__(self, index):
        return self.fft2[index]
    def __len__(self):
        return len(self.fft2)
    def shape(self):
        return self.fft2.shape
    def fft2(self):
        return Frequency_Domain(self.fft2)
    def fftshift(self):
        return Frequency_Domain(self.fft2_shift)
    def ifft2(self):
        return Frequency_Domain(np.fft.ifft2(self.fft2))
    def iffshift(self):
        return Frequency_Domain(np.fft.ifftshift(self.fft2))
    def magnitude(self):
        return np.abs(self.fft2)
    def toNumpy(self):
        return self.fft2
    def toSpatial(self):
        return np.real(self.fft2)
    def filterF(self, kernel, kwargs, crop_pic, kershift = False, crop = True):
        arr = self.arr
        if kershift == False:
            mul = self.fftshift()*kernel.fftshift()
        else:
            mul = self.fftshift()*kernel
        result = mul.iffshift().ifft2()
        if crop == True:
            return crop_pic(result.toSpatial(), **kwargs)
        else:
            return result.toSpatial()