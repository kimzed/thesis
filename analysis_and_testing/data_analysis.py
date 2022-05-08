# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 11:57:34 2021

@author: baron015
"""

import os
import numpy as np
import matplotlib.pyplot as plt

working_directory = "C:/Users/57834/Documents/thesis"
os.chdir(working_directory)

import utils as fun

folder_results = "maps_and_results/"
file_results = f"{folder_results}results_data_analysis.txt"
fun.write_to_txt_file(file_results, "Results of data analysis \n\n\n")

VISUALIZATION= False

print("""
      Visualization
      """)

      
import dataset as dataset


year = 2011
folder_nairobi_dataset = f"data/{year}/nairobi_negatives_dataset/"
folder_greenhouse_dataset = f"data/{year}/greenhouse_dataset/"

folder_gt = folder_greenhouse_dataset+"ground_truth_rasters"
folder_landsat = folder_greenhouse_dataset+"landsat_rasters"

dataset_grenhouses = dataset.GreenhouseDataset(folder_data=folder_landsat, folder_labels=folder_gt,
                               transform=True)



from random import randint

if VISUALIZATION:
    value = randint(0, len(dataset_grenhouses)-1)
    fun.visualize_sample(dataset_grenhouses, value)
    # get one sample
    value = randint(0, len(dataset_grenhouses))
    sample = dataset_grenhouses.__getitem__(value)[0]
    print(f"Mean is {sample.mean()}")
    print(f"Std is {sample.std()}")
    fun.visualize_sample(dataset_grenhouses, value)

print("""
      Spectral behaviour
      """)

import dataset
years = [2011,2014,2019]
for year in years:
    year = [year]
    folders_data, folders_labels = dataset.get_data_folders(year)

    dataset_full = dataset.merge_datasets(data_folders=folders_data, label_folders=folders_labels)


    imgs = [dataset_full.__getitem__(i)[0] for i in range(len(dataset_full))]
    masks = [dataset_full.__getitem__(i)[1].reshape(38,38) for i in range(len(dataset_full))]
    #imgs = [dataset_full.__getitem__(1)[0], dataset_full.__getitem__(2)[0]]
    #masks = [dataset_full.__getitem__(1)[1].reshape(38,38), dataset_full.__getitem__(2)[1].reshape(38,38)]

    data_imgs = np.stack(imgs, axis=0, out=None)
    data_mask = np.stack(masks, axis=0, out=None)[:,None,:,:]
    masks_stack = np.repeat(data_mask, 6, axis=1)

    stats_gh = {"mu_band":[], "std_band":[]}
    stats_nogh = {"mu_band":[], "std_band":[]}

    for i in range(6):



        # extracting values of a single band
        band_pxls = data_imgs[:,i,:,:][masks_stack[:,i,:,:] == 1]

        # storing the mu and std
        stats_gh["mu_band"].append(band_pxls.mean())
        stats_gh["std_band"].append(band_pxls.std())

        # same for other pixels
        band_pxls_non_gh = data_imgs[:,i,:,:][masks_stack[:,i,:,:] == 0]
        stats_nogh["mu_band"].append(band_pxls_non_gh.mean())
        stats_nogh["std_band"].append(band_pxls_non_gh.std())

    # converting to np arrays
    for key in stats_nogh:
        stats_nogh[key] = np.array(stats_nogh[key])
        stats_gh[key] = np.array(stats_gh[key])

    band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
    plt.plot(band_names, stats_nogh["mu_band"], label="Non-greenhouse")
    plt.fill_between(band_names, stats_nogh["mu_band"]-stats_nogh["std_band"],
                     stats_nogh["mu_band"]+stats_nogh["std_band"],
        alpha=1, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=0)
    plt.plot(band_names, stats_gh["mu_band"], label="Greenhouse")
    plt.fill_between(band_names, stats_gh["mu_band"]-stats_gh["std_band"],
                    stats_gh["mu_band"]+stats_gh["std_band"],
        alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848',
        linewidth=4, linestyle='dashdot', antialiased=True)

    plt.xlabel('Bands')
    plt.ylabel('Mean reflectance')
    plt.legend()
    file_plot = f'{folder_results}spectral_behaviour_{year}.png'
    plt.savefig(file_plot)
    #plt.show()


    print("""
    percentage of GH through time
    """)

    gh_pixels = data_imgs[masks_stack == 1]
    non_gh_pixels = data_imgs[masks_stack == 0]

    number_gh_pixels = len(gh_pixels)
    number_pixels_total = number_gh_pixels + len(non_gh_pixels)
    proportion_gh_pixels = number_gh_pixels / number_pixels_total


    proportion_result = f"for year {year}, {len(non_gh_pixels)} non-gh pixels, proportion gh pixels is {proportion_gh_pixels}, and there are {number_gh_pixels} gh pixels. \n\n\n"

    fun.write_to_txt_file(file_results, proportion_result)

    print("outliers using boxplots")

    random_indexes_gh = np.random.randint(number_gh_pixels, size=9000)
    random_indexes_non_gh = np.random.randint(len(non_gh_pixels), size=9000)

    data_box_plot = [gh_pixels[random_indexes_gh], non_gh_pixels[random_indexes_non_gh]]

    fig, ax = plt.subplots()
    labels = ['greenhouse pixels', 'non-greenhouse pixels']
    ax.boxplot(data_box_plot, labels=labels)
    file_box_plot = f"{folder_results}boxplot_{year}.png"
    plt.savefig(file_box_plot)

    plt.show()






print("""Analyse difference between satellites""")

stats_gh = {}

for year in years:


    stats_gh["mu_band" + str(year)] = []
    stats_gh["std_band" + str(year)] = []

    folders_data, folders_labels = dataset.get_data_folders([year])

    folders_data = [folder for folder in folders_data if "greenhouse_dataset" in folder]
    folders_labels = [folder for folder in folders_labels if "greenhouse_dataset" in folder]

    dataset_full = dataset.merge_datasets(data_folders=folders_data, label_folders=folders_labels)

    imgs = [dataset_full.__getitem__(i)[0] for i in range(len(dataset_full))]
    masks = [dataset_full.__getitem__(i)[1].reshape(38, 38) for i in range(len(dataset_full))]

    data_imgs = np.stack(imgs, axis=0, out=None)
    data_mask = np.stack(masks, axis=0, out=None)[:, None, :, :]

    masks_stack = np.repeat(data_mask, 6, axis=1)

    gh_pixels = data_imgs[masks_stack == 1]


    for i in range(6):

        # extracting values of a single band
        band_pxls = data_imgs[:,i,:,:][masks_stack[:,i,:,:] == 1]

        # storing the mu and std
        stats_gh["mu_band"+str(year)].append(band_pxls.mean())
        stats_gh["std_band"+str(year)].append(band_pxls.std())


    for key in stats_gh:
        stats_gh[key] = np.array(stats_gh[key])


    band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]

    plt.plot(band_names, stats_gh["mu_band"+str(year)], label="Greenhouse"+str(year))
    plt.fill_between(band_names, stats_gh["mu_band"+str(year)]-stats_gh["std_band"+str(year)],
                    stats_gh["mu_band"+str(year)]+stats_gh["std_band"+str(year)],
        alpha=0.2, # , edgecolor='#CC4F1B', facecolor='#FF9848'
        linewidth=4, linestyle='dashdot', antialiased=True)


    # to have similar axis scale for x
    plt.ylim(-1300, 2100)

    plt.xlabel('Bands')
    plt.ylabel('Mean reflectance')
    plt.legend()
    file_plot = f'{folder_results}GHspectral_behaviour_{year}.png'
    plt.savefig(file_plot)
    #plt.show()

    plt.clf()


def write_to_txt_file(file:str, text:str):

    with open(file, "a") as my_text:
        my_text.write(text)