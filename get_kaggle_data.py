# Imports
import os
import requests as rq

def get_kaggle_mnist(save_location=os.getcwd()):

    download_location = raw_input("Url where the that can be found: ")
    os.chdir(save_location)

    kaggle_username = raw_input("Kaggle username: ")
    kaggle_password = raw_input("Kaggle password: ")

    kaggle_credentials = {"UserName": kaggle_username, "Password": kaggle_password}

    download_chunks = 512 * 1024

    files = ["train.csv", "test.csv"]
    file_locations = [(download_location + f) for f in files]

    for l in file_locations:
        temp = rq.get(l)
        url = rq.post(temp.url, data=kaggle_credentials)

        for name in files:
            f = open(name, "w")
            for parts in url.iter_content(chunk_size=download_chunks):
                if parts:
                    f.write(parts)
            f.close()

        print("Finished downloading file {download}.".format(download=name))

    print("Finished downloading all files.")

get_access()


