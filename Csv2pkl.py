import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import encodingList


def MFH_encoding(dataset_path):
    dim=7
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    encoded_predict, encoded_on, encoded_off = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded, on_encoded, off_encoded = encodingList.MFH(row['on_seq'], row['off_seq'],
                                                                                     dim=dim)
        encoded_predict.append(on_off_encoded)
        encoded_on.append(on_encoded)
        encoded_off.append(off_encoded)

    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 24, 7))
    X_on = np.array(encoded_on, dtype=np.float32).reshape((len(encoded_on), 1, 24, 5))
    X_off = np.array(encoded_off, dtype=np.float32).reshape((len(encoded_off), 1, 24, 5))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = '../DataSets/EncodedDatasets'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{dataset_name}_sci3_dim{dim}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'X_on': X_on,
            'X_off': X_off,
            'labels': labels
        }, f)

    print(f"Encoded data saved for {dataset_name}")



def CrisprNet(dataset_path):
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    encoded_predict = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded = encodingList.crispr_net_coding(row['on_seq'], row['off_seq'])
        encoded_predict.append(on_off_encoded)


    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 24, 7))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = '../DataSets/EncodedDatasets/CrisprNet'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'labels': labels
        }, f)

    print(f"Encoded data saved for {dataset_name}")


def CrisprIp(dataset_path):
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    encoded_predict = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded = encodingList.cnn_predict(row['on_seq'], row['off_seq'])
        encoded_predict.append(on_off_encoded)


    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 24, 7))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = '../DataSets/EncodedDatasets/CrisprIp'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'labels': labels
        }, f)

    print(f"Encoded data saved for {dataset_name}")

def CnnStd(dataset_path):
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    encoded_predict = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded = encodingList.cnn_predict(row['on_seq'], row['off_seq'])
        encoded_predict.append(on_off_encoded)


    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 23, 4))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = '../DataSets/EncodedDatasets/CnnStd'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'labels': labels
        }, f)

    print(f"Encoded data saved for {dataset_name}")



def CrisprDnt_23_14(dataset_path):
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    encoded_predict = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded = encodingList.dnt_coding(row['on_seq'], row['off_seq'])
        encoded_predict.append(on_off_encoded)


    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 23, 14))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = '../DataSets/EncodedDatasets/CrisprDnt_23_14'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'labels': labels
        }, f)

    print(f"Encoded data saved for {dataset_name}")


def CrisprDnt_24_14(dataset_path):
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    encoded_predict = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded = encodingList.dnt_coding_24_14(row['on_seq'], row['off_seq'])
        encoded_predict.append(on_off_encoded)
    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 24, 14))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = '../DataSets/EncodedDatasets/CrisprDnt_24_14'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'labels': labels
        }, f)
    print(f"Encoded data saved for {dataset_name}")

def dl_offtarget(dataset_path):
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    encoded_predict = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded = encodingList.dl_offtarget(row['on_seq'], row['off_seq'])
        encoded_predict.append(on_off_encoded)
    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 23, 8))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = '../DataSets/EncodedDatasets/dl_offtarget'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'labels': labels
        }, f)
    print(f"Encoded data saved for {dataset_name}")

def dl_crispr(dataset_path):
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    encoded_predict = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded = encodingList.dl_crispr(row['on_seq'], row['off_seq'])
        encoded_predict.append(on_off_encoded)
    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 20, 20))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = '../DataSets/EncodedDatasets/dl_crispr'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'labels': labels
        }, f)
    print(f"Encoded data saved for {dataset_name}")





def get_dataset_paths():
    folders = {
        'Indel': '../DataSets/Indel',
        # 'Mismatch': '../DataSets/Mismatch'
    }
    datasets = {
        'Indel': ["CIRCLE_seq.csv", "GUIDE-Seq.csv"],
        # 'Mismatch': ["Hek293t.csv", "K562.csv", "K562Hek293.csv", "Kleinstiver.csv",
        #              "Haeussler.csv", "Listgarten.csv", "SITE.csv", "Tasi.csv", "Doench.csv"]
    }
    return [os.path.join(folders[folder], dataset)
            for folder, dataset_list in datasets.items()
            for dataset in dataset_list]
def main():
    dataset_paths = get_dataset_paths()

    for dataset_path in dataset_paths:
        MFH_encoding(dataset_path)


if __name__ == '__main__':
    main()