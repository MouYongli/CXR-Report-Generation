################################################################################
############             MICC-CXR dataset download script            ###########
# without using GCP, download via wget from MICC-CXR website is not optimal
# using GCP is fucking expense !!!
# we add multiprocessing to speed up the download process
#
# prerequisites:
#  - MICC-CXR login credentials
#  - all the index.html files are downloaded  
#  - save login credentials in .wgetrc file, and chmod +600 .wgetrc for security
#    reason, only the owner can read and write the file
# 
# This download script is tested on 24 Cores CPU 
# using dask for multiprocessing
# Jin Er, LfB RWTH Aachen, 2023
################################################################################

import os
from functools import partial
import glob
import argparse
from typing import List

from multiprocessing import Process
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re
import codecs

from sqlalchemy import all_
from sympy import Li 

def download_html_index(
    path: str
) -> None:
    os.system(
        f"wget -r -N -c https://physionet.org/files/mimic-cxr/2.0.0/files/p10/ -P {path}"
    )

def get_image_basename(
    path: str,
    extension: str = "jpg"
) -> list:
    """
    get all image basenames in a given path
    """
    with codecs.open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        # use regex to find all image basename between <a href=" and .dcm
        pattern = re.compile(r'<a href="(.+?).jpg">')
        basenames = re.findall(pattern, text)
        dirpath = os.path.dirname(path)
        basenames = [os.path.join(dirpath, basename +  f".{extension}") for basename in basenames]
        return basenames

def get_all_subdir(
    path: str,
    patient: int = 10,
) -> list:
    """
    get all subdirectories of a given path
    """
    # p********/s******/id******.dcm
    path_pattern = os.path.join(
        path, "physionet.org/files/mimic-cxr-jpg/2.0.0/files", f"p{patient}/*/*", "*.html")
    file_list = glob.glob(path_pattern, recursive=True)
    image_list = []
    for index in file_list:
        image_list += get_image_basename(index)
    return image_list

def _wget_download(
    dst_dir: str,
    pathes: List[str],
) -> None:
    """
    download file from url to path
    """
    path_components = [p.split(os.sep) for p in pathes]
    pathes = [os.path.join(
        "https://physionet.org/files/mimic-cxr/2.0.0/", *path_component[6:]
        ) for path_component in path_components]
    
    # path = " ".join(pathes)
    for p in pathes:
        os.system(
            f"wget --limit-rate=100m  -r -N -c -p {p} -P {dst_dir}"
        )

def fetch_multiple(
    img_path_list: list,
    dst_dir: str,
    num_workers: int = 10,
):
    """
    using dask for multiple wget download
    """
    def divide_chunks(l=img_path_list, n=num_workers): 
        """help function as generator"""
        # looping till length l 
        for i in range(0, len(l), len(l) // n):  
            yield l[i:i + len(l) // n]
    
    pl_generator = divide_chunks()
    
    for pl in pl_generator:
        func = partial(_wget_download, dst_dir=dst_dir, pathes=pl)
        tmp_p = Process(target=func)
        tmp_p.start()
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    executor = ThreadPoolExecutor(max_workers=5)
    download_tasks = []
    # Submit the download tasks
    for file_url in pl_generator:
        func = partial(_wget_download, dst_dir=dst_dir, pathes=file_url)
        download_task = executor.submit(func)
        download_tasks.append(download_task)

    # Use tqdm to create a progress bar
    with tqdm(total=len(img_path_list) // num_workers) as progress_bar:
        for completed_task in as_completed(download_tasks):
            result = completed_task.result()
            # Update the progress bar
            progress_bar.update(1)

def main(args):
    if args.download_index_html:
        print("start downloading index.html files")
        download_html_index(args.dst_dir)
        print("finished downloading index.html files")
    
    print("collecting all subdirectories and html index")
    all_subdir = get_all_subdir(args.dst_dir, args.patient)

    print("initializating dask and start downloading ....")
    fetch_multiple(
        all_subdir, 
        args.dst_dir, 
        num_workers=10
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Multi-threaded MICC-CXR dataset download script")
    parser.add_argument("--dst_dir", type=str, default="./test", help="destination directory")
    parser.add_argument("-d", "--download_index_html", type=bool, default=False, help="download index.html files, multithread preprocessing")
    parser.add_argument("-p", "--patient", type=int, default=10, help="10, 11 -- 19")
    args = parser.parse_args()
    main(args)