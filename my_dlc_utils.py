# Importing the toolbox (takes several seconds)
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def Histogram(vector, color, bins):
    dvector = np.diff(vector)
    dvector = dvector[np.isfinite(dvector)]
    plt.hist(dvector, color=color, histtype='step', bins=bins)


def PlottingResults(Dataframe, bodyparts2plot, alphavalue=.2, pcutoff=.5, colormap='jet', fs=(4, 3)):
    """ Plots poses vs time; pose x vs pose y; histogram of differences and likelihoods."""
    plt.figure(figsize=fs)
    colors = get_cmap(len(bodyparts2plot), name=colormap)

    for bpindex, bp in enumerate(bodyparts2plot):
        Index = Dataframe[bp,'likelihood'].values > pcutoff
        plt.plot(Dataframe[bp,'x'].values[Index], Dataframe[bp,'y'].values[Index], '.',
                 color=colors(bpindex), alpha=alphavalue)

    plt.gca().invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    # plt.savefig(os.path.join(tmpfolder,"trajectory"+suffix))
    plt.figure(figsize=fs)
    Time = np.arange(np.size(Dataframe[bodyparts2plot[0]]['x'].values))

    for bpindex, bp in enumerate(bodyparts2plot):
        Index = Dataframe[bp]['likelihood'].values > pcutoff
        plt.plot(Time[Index], Dataframe[bp,'x'].values[Index], '--', color=colors(bpindex), alpha=alphavalue)
        plt.plot(Time[Index], Dataframe[bp,'y'].values[Index], '-', color=colors(bpindex), alpha=alphavalue)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.xlabel('Frame index')
    plt.ylabel('X and y-position in pixels')
    # plt.savefig(os.path.join(tmpfolder,"plot"+suffix))

    plt.figure(figsize=fs)
    for bpindex, bp in enumerate(bodyparts2plot):
        Index = Dataframe[bp,'likelihood'].values > pcutoff
        plt.plot(Time, Dataframe[bp,'likelihood'].values, '-', color=colors(bpindex), alpha=alphavalue)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.xlabel('Frame index')
    plt.ylabel('likelihood')

    # plt.savefig(os.path.join(tmpfolder,"plot-likelihood"+suffix))

    plt.figure(figsize=fs)
    bins = np.linspace(0, np.amax(Dataframe.max()), 100)

    for bpindex, bp in enumerate(bodyparts2plot):
        Index = Dataframe[bp,'likelihood'].values < pcutoff
        X = Dataframe[bp,'x'].values
        X[Index] = np.nan
        Histogram(X, colors(bpindex), bins)
        Y = Dataframe[bp,'x'].values
        Y[Index] = np.nan
        Histogram(Y, colors(bpindex), bins)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.ylabel('Count')
    plt.xlabel('DeltaX and DeltaY')

    # plt.savefig(os.path.join(tmpfolder,"hist"+suffix))


def calc_distance_between_points_in_a_vector_2d(v1):
    """calc_distance_between_points_in_a_vector_2d [for each consecutive pair of points, p1-p2, in a vector, get euclidian distance]
    This function can be used to calculate the velocity in pixel/frame from tracking data (X,Y coordinates)

    Arguments:
        v1 {[np.array]} -- [2d array, X,Y position at various timepoints]

    Raises:
        ValueError

    Returns:
        [np.array] -- [1d array with distance at each timepoint]
    >> v1 = [0, 10, 25, 50, 100]
    >> d = calc_distance_between_points_in_a_vector_2d(v1)
    """
    # Check data format
    if isinstance(v1, dict) or not np.any(v1) or v1 is None:
        raise ValueError(
            'Feature not implemented: cant handle with data format passed to this function')

    # If pandas series were passed, try to get numpy arrays
    try:
        v1, v2 = v1.values, v2.values
    except:  # all good
        pass
    # loop over each pair of points and extract distances
    dist = []
    for n, pos in enumerate(v1):
        # Get a pair of points
        if n == 0:  # get the position at time 0, velocity is 0
            p0 = pos
            dist.append(0)
        else:
            p1 = pos  # get position at current frame

            # Calc distance
            dist.append(np.abs(distance.euclidean(p0, p1)))

            # Prepare for next iteration, current position becomes the old one and repeat
            p0 = p1

    return np.array(dist)