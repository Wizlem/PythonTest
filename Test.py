__author__ = 'Wizlem'

from skimage.color import xyz2rgb

from numpy import genfromtxt, array, zeros, transpose, tile, matrix, expand_dims

from scipy.integrate import trapz

from matplotlib.pyplot import plot, imshow, show


CIEDATA = genfromtxt('CIE_ALLDATA.csv', delimiter=',', names=True)
CIEDATA = CIEDATA[17:97]

cmf = array([CIEDATA['CIE1931x'], CIEDATA['CIE1931y'], CIEDATA['CIE1931z']])

wl = CIEDATA['wavelength']
D65 = CIEDATA['D65']
D50 = CIEDATA['D50']

colhdrs = CIEDATA.dtype.names[15:]

img = zeros((800, 1200, 3))

XYZn = trapz(cmf*D65.T, wl)
XYZnD50 = trapz(cmf*D50.T, wl)

a = 0.055

for i in range(4):
    for j in range(6):
        n = 6*i+j
        color = CIEDATA[colhdrs[n]]

        XYZ = trapz(cmf*(D65*color).T, wl)/XYZn[1]
        XYZ = transpose(expand_dims(matrix(XYZ), 3), (0, 2, 1))

        rgb = xyz2rgb(XYZ)

        img[200*i:200*(i+1), 200*j:200*(j+1), :] = tile(rgb, (200, 200, 1))

imshow(img)
show()