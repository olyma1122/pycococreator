import datetime
import fiftyone.zoo as foz
import fnmatch
import glob
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import re
import shutil

from pathlib import Path
from PIL import Image
from pycococreator.pycococreatortools import pycococreatortools
from tqdm import tqdm


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


FORMAT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def download_objs(categories, subsets, path=str(Path.home().joinpath("data"))):
    """
    Download the specific categories from Google's Open Images Dataset(v6).

    Args:
        categories (list): a list of categories
        subsets (list): a list of subsets, including 'train', 'validation' and 'test' subset
        path (str): a folder that will be saved to

    """
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        classes=categories,
        splits=subsets,
        dataset_dir=str(path),
        dataset_name="".join(categories),
    )

    logging.info("Open Images categories: {} Downloaded".format(categories))


def compare_items(categories_id, subset, path):
    """
    Compare the inventories.

    Args:
        categories_id (list): a list of categories id
        subset (list): a subset, including 'train', 'validation' and 'test' subset
        path (str): a directory where the dataset will be saved

    Returns:
        corrected_img_files[pandas.DataFrame]: a dataframe of images
        corrected_mask_files[pandas.DataFrame]: a dataframe of masks

    """
    # Get Original label .csv file
    _table = pd.read_csv(str(path.joinpath(subset, "labels/segmentations.csv")))

    # Get Original files from folder
    _ori_files = np.array(glob.glob(str(path.joinpath(subset, "data/*.jpg"))))

    # Get Mask files from sub folders
    _mask_files = np.array([])

    for folder in path.joinpath(subset, "labels/masks").iterdir():
        _files = glob.glob(str(folder.joinpath("*.png")))
        _mask_files = np.append(_mask_files, np.array(_files))

    # Summary & Compare
    _table = _table.loc[_table["MaskPath"].str.contains('|'.join(categories_id))]

    # Summary & Compare: Mask
    _mask_files = pd.DataFrame(_mask_files)
    _mask_files = _mask_files.loc[_mask_files[0].str.contains('|'.join(categories_id))]

    corrected_mask_files = _mask_files.loc[_mask_files[0].str.contains('|'.join(_table["MaskPath"]))]
    dropped = _mask_files.loc[~_mask_files[0].str.contains('|'.join(_table["MaskPath"]))]

    logging.info("> In {}, mask: dropped/original = {}/{}, \n{}".format(subset, dropped.shape[0], _mask_files.shape[0], dropped.to_numpy()))

    # Summary & Compare: Image
    _ori_files = pd.DataFrame(_ori_files)
    corrected_img_files = _ori_files.loc[_ori_files[0].str.contains('|'.join(_table["ImageID"]))]
    dropped = _ori_files.loc[~_ori_files[0].str.contains('|'.join(_table["ImageID"]))]

    logging.info("> In {}, image: dropped/original = {}/{}, \n{}".format(subset, dropped.shape[0], _ori_files.shape[0], dropped.to_numpy()))

    return corrected_img_files, corrected_mask_files


def inventory_dataset(categories, categories_id, subsets, path):
    """
    Invent the available data from downloaded dataset.

    Args:
        categories (list): a list of categories
        categories_id (list): a list of categories id
        subsets (list): a list of subsets, including 'train', 'validation' and 'test' subset
        path (str): a directory where the dataset will be saved

    Returns:
        img[pandas.DataFrame]: a dataframe of images
        mask[pandas.DataFrame]: a dataframe of masks

    """

    img = pd.DataFrame()
    mask = pd.DataFrame()

    for subset in subsets:
        _img, _mask = compare_items(categories_id=categories_id, subset=subset, path=path)

        img = img.append(_img, ignore_index=True)
        mask = mask.append(_mask, ignore_index=True)

    return img, mask


def split_dataset(categories, categories_id, img, mask, path):
    """
    Split the dataset as Train and Validation.

    Args:
        categories (list): a list of categories
        categories_id (list): a list of categories id
        img (pandas.DataFrame): a dataframe of images
        mask (pandas.DataFrame): a dataframe of masks
        path (str): a directory where the dataset will be saved

    """

    # Random
    files_id = img[0].str.rsplit("/", n=1, expand=True)
    files_id = files_id[1].str.split(".", n=1, expand=True)
    random_files_id = files_id[0].copy().to_numpy()
    random.shuffle(random_files_id)

    # Create Folder
    train_ori_dir = path.joinpath("../coco_instance" + "_" + "".join(categories).lower(), "train_ori")
    train_ann_dir = path.joinpath("../coco_instance" + "_" + "".join(categories).lower(), "train_ann")
    val_ori_dir = path.joinpath("../coco_instance" + "_" + "".join(categories).lower(), "val_ori")
    val_ann_dir = path.joinpath("../coco_instance" + "_" + "".join(categories).lower(), "val_ann")
    os.makedirs(train_ori_dir, exist_ok=True)
    os.makedirs(train_ann_dir, exist_ok=True)
    os.makedirs(val_ori_dir, exist_ok=True)
    os.makedirs(val_ann_dir, exist_ok=True)

    # Copy Image to target folder
    train_ori_set = img.loc[img[0].str.contains("|".join(random_files_id[:int(random_files_id.shape[0] * 0.9)]))]
    val_ori_set = img.loc[img[0].str.contains("|".join(random_files_id[int(random_files_id.shape[0] * 0.9):]))]

    for ori in train_ori_set.to_numpy().flatten():
        shutil.copy(ori, train_ori_dir)
    for ori in val_ori_set.to_numpy().flatten():
        shutil.copy(ori, val_ori_dir)

    # Copy Mask to target folder
    train_mask_set = mask.loc[mask[0].str.contains("|".join(random_files_id[:int(random_files_id.shape[0] * 0.9)]))]
    val_mask_set = mask.loc[mask[0].str.contains("|".join(random_files_id[int(random_files_id.shape[0] * 0.9):]))]

    for ori in train_mask_set.to_numpy().flatten():
        shutil.copy(ori, train_ann_dir)
    for ori in val_mask_set.to_numpy().flatten():
        shutil.copy(ori, val_ann_dir)

    convert_cocoformat(categories_id=categories_id, img_dir=str(train_ori_dir), ann_dir=str(train_ann_dir))
    convert_cocoformat(categories_id=categories_id, img_dir=str(val_ori_dir), ann_dir=str(val_ann_dir))


"""
Title: shapes_to_coco.py
Author: waspinator and Hanna Rudakouskaya
Date: 2021
Code version: 2.1.0
Availability: https://github.com/waspinator/pycococreator
"""
def convert_cocoformat(categories_id, img_dir, ann_dir):
    """
    Convert images and masks to COCO format.

    Args:
        categories_id (list): a list of categories id
        img_dir (str): a directory of images
        ann_dir (str): a directory of masks

    """

    def filter_for_jpeg(root, files):
        file_types = ['*.jpeg', '*.jpg']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        return files

    def filter_for_annotations(root, files, image_filename):
        file_types = ['*.png']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
        file_name_prefix = basename_no_extension + '.*'
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
        return files

    _info = {
        "description": "Open Images Dataset",
        "url": "https://opensource.google/projects/open-images-dataset",
        "version": "6.0",
        "year": 2021,
        "contributor": "openimages",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    _lic = [
        {
            "id": 1,
            "name": "Attribution 4.0 International (CC BY 4.0)",
            "url": "https://creativecommons.org/licenses/by/4.0/"
        }
    ]

    _categories = []
    for idx, item in enumerate(categories_id):
        _c = {
            'id': idx + 1,
            'name': str(item),
            'supercategory': 'openimages',
        }
        _categories.append(_c)

    coco_output = {
        "info": _info,
        "licenses": _lic,
        "categories": _categories,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(img_dir):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ann_dir):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    # print(annotation_filename)
                    class_id = [x['id'] for x in _categories if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    json_dir = img_dir.rsplit("/", 1)[0]
    with open('{}/{}_instances.json'.format(json_dir, img_dir.split("/")[-1].split("_")[0]), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
