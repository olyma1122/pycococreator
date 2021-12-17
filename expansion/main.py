import oiv6_tools

from pathlib import Path


if __name__ == '__main__':
    # Config
    categories = ["Box"]
    categories_id = ["m025dyy"]
    subsets = ["train", "validation", "test"]
    path = Path.home().joinpath("oiv6/data")

    # Prepare dataset
    oiv6_tools.download_objs(categories=categories, subsets=subsets, path=path)
    img, mask = oiv6_tools.inventory_dataset(categories=categories, categories_id=categories_id, subsets=subsets, path=path)
    oiv6_tools.split_dataset(categories=categories, categories_id=categories_id, img=img, mask=mask, path=path)
