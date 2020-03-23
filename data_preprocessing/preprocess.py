import os
import copy
import sys
import random
import tarfile
import getopt
import logging
import pandas as pd
import urllib.request


def clear_dir(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


def download_and_unzip(url, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    logging.info("Downloading {} ".format(url))
    filename, headers = urllib.request.urlretrieve(url, target_path + '\data.tar')
    tr = tarfile.open(filename)
    logging.info("Extracting tar file ...")
    tr.extractall(path=target_path)
    tr.close()
    os.remove(filename)


def sample_categories(images, upper, lower, max_categories, num_machines, base_path, random_state):
    image_attrs = {}
    nunique = images.groupby("landmark_id")["id"].nunique()
    full_categories = nunique[nunique >= upper].sample(max_categories, random_state=random_state)
    full_categories = full_categories.apply(lambda x: random.randint(lower, upper))

    for c in full_categories.index:
        target_size = full_categories.loc[c]
        matched_images = images[images["landmark_id"] == c].sample(target_size, random_state=random_state)
        matched_images["machine"] = random.sample([i%num_machines for i in range(target_size)], target_size)
        for index, row in matched_images.iterrows():
            new_path = "{}\machine_{}\{}".format(base_path, row["machine"], row["landmark_id"])
            new_file_name = "{}_{}.png".format(row["id"], row["landmark_id"])
            image_attrs[row["id"]] = {"new_path": new_path, "new_file_name": new_file_name}
    return image_attrs


def allocate_images(image_attrs, tar_url, base_path):
    image_attrs_copy = copy.deepcopy(image_attrs)
    temp_path = base_path + "\\temp"
    total_images = len(image_attrs)
    downloaded_images = 0
    tar_index = 0

    while len(image_attrs_copy) > 0:
        logging.info("{} out of {} downloaded: {:.5f}%".format(downloaded_images, total_images, downloaded_images / total_images))
        download_and_unzip(tar_url[tar_index], temp_path)
        for subdir, dirs, files in os.walk(temp_path):
            if len(files) == 0:
                continue
            image_ids = [f.split(".")[0] for f in files]
            for image in image_ids:
                if image in image_attrs_copy:
                    if not os.path.exists(image_attrs[image]["new_path"]):
                        os.makedirs(image_attrs[image]["new_path"])
                    destination = "{}\{}".format(image_attrs[image]["new_path"], image_attrs[image]["new_file_name"])
                    os.rename("{}\{}.jpg".format(subdir, image), destination)

                    image_attrs_copy.pop(image, None)
                    downloaded_images += 1
        tar_index += 1


def main(data, data_set, path, clear, upper, lower, max_categories, num_machines, random_state):
    if clear:
        clear_dir(path)
        logging.warning("All files in the specified directory are cleared.")

    logging.info("Loading data file ...")
    images = pd.read_csv(data)

    base_tar_url = "https://s3.amazonaws.com/google-landmark/{}/images_{}.tar"
    tar_url = [base_tar_url.format(data_set, str(i).zfill(3)) for i in range(500)]

    logging.info("Random Sampling categories ...")
    image_attrs = sample_categories(images, upper, lower, max_categories, num_machines, path, random_state)

    logging.info("Allocating images ...")
    allocate_images(image_attrs, tar_url, path)


def start(argv):
    EXAMPLE_STRING = 'preprocess.py -a True/False -d department index -i input filename -o output filename'

    # default parameters
    data = ""
    data_set = "train"
    path = ""
    clear = False
    upper = 50
    lower = 30
    max_categories = 2000
    num_machines = 5
    random_state = 1234

    try:
        opts, args = getopt.getopt(argv,
                                   "d:s:p:c:u:l:m:n:r",
                                   ["data=", "data_set=", "path=", "clear=",
                                    "upper=", "lower=", "max_categories=", "num_machines=", "random_state="])
    except getopt.GetoptError:
        print(EXAMPLE_STRING)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--data"):
            if arg != "":
                data = arg
            else:
                print("The data file cannot be empty!!")
                sys.exit()

        elif opt in ("-s", "--data_set"):
            if arg != "":
                data_set = arg
            else:
                print("The data set name cannot be empty!!")
                sys.exit()

        elif opt in ("-p", "--path"):
            if arg != "":
                path = arg
            else:
                print("The path cannot be empty!!")
                sys.exit()

        elif opt in ("-c", "--clear"):
            if arg == "True":
                clear = True
            elif arg == "False":
                clear = False
            else:
                print("clear must be either 'True' or 'False'.")
                sys.exit()

        elif opt in ("-u", "--upper"):
            try:
                upper = int(arg)
            except Exception:
                print("upper must be an integer!")
                sys.exit()

        elif opt in ("-l", "--lower"):
            try:
                lower = int(arg)
            except Exception:
                print("lower must be an integer!")
                sys.exit()

        elif opt in ("-m", "--max_categories"):
            try:
                max_categories = int(arg)
            except Exception:
                print("maximum number of categories  must be an integer!")
                sys.exit()

        elif opt in ("-n", "--num_machines"):
            try:
                num_machines = int(arg)
            except Exception:
                print("number of machines must be an integer!")
                sys.exit()

        elif opt in ("-r", "--random_state"):
            try:
                random_state = int(arg)
            except Exception:
                print("random state must be an integer!")
                sys.exit()
    main(data, data_set, path, clear, upper, lower, max_categories, num_machines, random_state)


if __name__ == "__main__":
    # print(sys.argv)
    logging.basicConfig(level=logging.DEBUG)
    start(sys.argv[1:])
