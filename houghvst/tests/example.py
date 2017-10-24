import timeit
import numpy as np
import tifffile
import houghvst.estimation.estimation as est
from houghvst.estimation.gat import compute_gat, compute_inverse_gat


def main():
    movie = tifffile.imread('../../images/demoMovie.tif')
    movie = movie.astype(np.float)

    # makes estimation numerically better:
    movie -= movie.mean()

    # use one every 200 frames
    temporal_stride = 200
    # use one every 8 patches (patches are 8x8 by default)
    spatial_stride = 8

    movie_train = movie[::temporal_stride]

    t = timeit.default_timer()
    estimation_res = est.estimate_vst_movie(movie_train, stride=spatial_stride)
    print('\tTime', timeit.default_timer() - t)

    alpha = estimation_res.alpha
    sigma_sq = estimation_res.sigma_sq

    movie_gat = compute_gat(movie, sigma_sq, alpha=alpha)
    # save movie_gat here
    movie_gat_inv = compute_inverse_gat(movie_gat, sigma_sq, alpha=alpha,
                                        method='asym')
    # save movie_gat_inv here


if __name__ == '__main__':
    main()
