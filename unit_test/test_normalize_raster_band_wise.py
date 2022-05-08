
from pathlib import Path
import numpy as np
import os

from pre_processing.utils_preprocessing import normalize_raster_band_wise

def test_normalize_raster_band_wise_values_are_correct():

    input_raster = np.zeros([6, 38, 38])
    number_bands = input_raster.shape[0]

    expected_mean = 0

    for i_band in range(number_bands):
        input_raster[i_band] = np.random.randint(0, 38, [38,38])
    # have one band with bigger values
    i_last_band = i_band
    input_raster[i_last_band] *= 10000

    normalized_raster = normalize_raster_band_wise(input_raster)
    std_last_band = normalized_raster[i_last_band].std()

    for i_band in range(number_bands):
        std_band = normalized_raster[i_band].std()

        assert (std_last_band >= std_band)


test_normalize_raster_band_wise_values_are_correct()

