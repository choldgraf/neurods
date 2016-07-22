"""Functions for data visualization"""
import matplotlib.pyplot as plt
import numpy as np

def find_squarish_dimensions(n):
    '''Get row,column dimensions for n elememnts

    Returns (nearly) sqrt dimensions for a given number. e.g. for 23, will
    return [5,5] and for 26 it will return [6,5]. For creating displays of
    sets of images, mostly. Always sets x greater than y if they are not
    equal.

    Returns
    -------
    x : int
       larger dimension (if not equal)
    y : int
       smaller dimension (if not equal)
    '''
    sq = np.sqrt(n);
    if round(sq)==sq:
        # if this is a whole number - i.e. a perfect square
        x = sq;
        y = sq;
        return x, y
    # One: next larger square
    x = [np.ceil(sq)]
    y = [np.ceil(sq)]
    opt = [x[0]*y[0]];
    # Two: immediately surrounding numbers
    x += [np.ceil(sq)];
    y += [np.floor(sq)];
    opt += [x[1]*y[1]];
    Test = np.array([o-n for o in opt])
    Test[Test<0] = 1000; # make sure negative values will not be chosen as the minimum
    GoodOption = np.argmin(Test);
    x = x[GoodOption]
    y = y[GoodOption]
    return x, y

# Mark's slicing function. Perhaps fancier than we need. 
def slice_3d_matrix(volume, axis=2, figh=None, vmin=None, vmax=None, cmap=plt.cm.gray, nr=None, nc=None ):
    '''Slices 3D matrix along arbitrary axis

    Parameters
    ----------
    volume : array (3D)
    axis : int | 0,1,[2] (optional)
       axis along which to divide the matrix into slices

    Other Parameters
    ----------------
    vmin : float [max(volume)] (optional) 
       color axis minimum
    vmax : float [min(volume)] (optional)
       color axis maximum
    cmap : matplotlib colormap instance [plt.cm.gray] (optional)
    nr : int (optional)
       number of rows
    nc : int (optional)
       number of columns
    '''
    if figh is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figh)
    if nr is None or nc is None:
        nc,nr = find_squarish_dimensions(volume.shape[axis])
    if vmin is None:
        vmin = volume.min()
    if vmax is None:
        vmax = volume.max()
    ledges = np.linspace(0, 1, nc+1)[:-1]
    bedges = np.linspace(1, 0, nr+1)[1:]
    width = 1/float(nc)
    height = 1/float(nr)
    bottoms,lefts = zip(*list(itertools.product(bedges, ledges)))
    for ni,sl in enumerate(np.split(volume, volume.shape[axis],axis=axis)):
        #ax = fig.add_subplot(nr, nc, ni+1)
        ax = fig.add_axes((lefts[ni], bottoms[ni], width, height))
        ax.imshow(sl.squeeze(), vmin=vmin, vmax=vmax, interpolation="nearest",cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig