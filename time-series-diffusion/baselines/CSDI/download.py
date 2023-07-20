import tarfile
import zipfile
import os
import wget
import requests
import pandas as pd
import pickle
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='pm25', \
                        help='dataset type, can either be physio or pm25')
    parser.add_argument('--save_path', type=str, default='../../datasets/', \
                        help='location to save datasets')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    if args.dataset_type == "physio":
        url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
        dest_loc = os.path.join(args.save_path, "set-a.tar.gz")
        if not os.path.exists(dest_loc):
            wget.download(url, out=args.save_path)
        with tarfile.open(dest_loc, "r:gz") as t:
            t.extractall(path=os.path.join(args.save_path, "physio"))

    elif args.dataset_type == "pm25":
        url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
        dest_loc = os.path.join(args.save_path, "STMVL-Release.zip")
        if not os.path.exists(dest_loc):
            wget.download(url, out=args.save_path)
        with zipfile.ZipFile(dest_loc) as z:
            z.extractall(path=os.path.join(args.save_path, "pm25"))

        def create_normalizer_pm25(): 
            '''
            storing the mean and the variance of all the columns - 36 columns in total 
            each column represents a single air quality tracking station
            '''
            df = pd.read_csv(
                os.path.join(args.save_path, "pm25/Code/STMVL/SampleData/pm25_ground.txt"),
                index_col="datetime",
                parse_dates=True,
            )
            test_month = [3, 6, 9, 12]
            for i in test_month:
                df = df[df.index.month != i]

            mean = df.describe().loc["mean"].values
            std = df.describe().loc["std"].values
            path = os.path.join(args.save_path, "pm25/pm25_meanstd.pk")
            with open(path, "wb") as f:
                pickle.dump([mean, std], f)
        create_normalizer_pm25()