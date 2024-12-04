import os
import random
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score
)
import mymodel as model_module
import Data.DataEncoding.encodingList as encoding_list
import time
def set_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_predictions(dataset_name, model_name, true_labels, predicted_scores, fold, tag):
    result_dir = f"./Result/ModelData/{model_name}/{dataset_name}"
    os.makedirs(result_dir, exist_ok=True)
    result_df = pd.DataFrame({'Label': true_labels, 'Score': predicted_scores})
    result_df.to_csv(f"{result_dir}/fold_{fold}.csv", index=False)

def log_results(results_df, model_name):
    log_dir = "./Result/Log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_name}_log.txt")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{current_time} {model_name}\n")
        log_file.write(results_df.to_string(index=False))
        log_file.write("\n\n")

def create_callbacks(model_name, dataset_name, fold, batch_size):
    callbacks = [
        EarlyStopping(monitor='loss', min_delta=0.0001, patience=9, verbose=1, mode='auto'),
        ModelCheckpoint(
            filepath=f'./Result/Model/{model_name}/{dataset_name}/fold{fold}_best_model.h5',
            monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='min', min_lr=1e-7),
        # TensorBoard(log_dir=f"logs/{model_name}_{dataset_name}_{batch_size}_{fold}", histogram_freq=1, update_freq='epoch')
    ]
    return callbacks

def save_trained_model(dataset_name, model, model_name, fold, tag):
    model_dir = f"./Result/Model/{model_name}/{dataset_name}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f'{model_dir}/{model_name}-{fold}-{tag}.h5')

def get_dataset_paths():
    folders = {
        'Mismatch': './Data/DataSets/Mismatch',
        'Indel': './Data/DataSets/Indel'
    }
    datasets = {
        # 'Mismatch': ["Hek293t.csv", "K562.csv", "K562Hek293.csv", "Kleinstiver.csv",
        #              "Haeussler.csv", "Listgarten.csv", "SITE.csv", "Tasi.csv", "Doench.csv"],
        # 'Indel': ["CIRCLE_seq.csv", "GUIDE-Seq.csv"]
        'Mismatch': ["Hek293t.csv"],
        'Indel': []
    }
    return [os.path.join(folders[folder], dataset)
            for folder, dataset_list in datasets.items()
            for dataset in dataset_list]

def encode_sequences(dataframe):
    encoded_predict, encoded_on, encoded_off = [], [], []
    for _, row in dataframe.iterrows():
        on_off_encoded, on_encoded, off_encoded = encoding_list.MFH(row['on_seq'], row['off_seq'], dim=7)
        encoded_predict.append(on_off_encoded)
        encoded_on.append(on_encoded)
        encoded_off.append(off_encoded)
    return (np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 24, 7)),
            np.array(encoded_on, dtype=np.float32).reshape((len(encoded_on), 1, 24, 5)),
            np.array(encoded_off, dtype=np.float32).reshape((len(encoded_off), 1, 24, 5)))


def load_encoded_data(dataset_name,CODING_NAME):
    with open(f'./Data/DataSets/EncodedDatasets/{CODING_NAME}/{dataset_name}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['X_predict'], data['X_on'], data['X_off'], data['labels']

def timeprint():
    current_timestamp = time.time()
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_timestamp))
    print(current_time_str)


# Constants
RANDOM_SEED = 2024
NUM_CLASSES = 2
NUM_EPOCHS = 500
INITIAL_LEARNING_RATE = 0.001

def main():

    set_random_seeds(RANDOM_SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    bestbz = {
        'CIRCLE_seq': 8192,
        'Tasi': 1024,
        'SITE': 1024,
        'K562': 256,
        'Kleinstiver': 256,
        'Hek293t': 2048
    }

    param_combinations = [


        {
            'model_name': 'CRISPR-MFH',
            'coding_name': 'MFH',
            'model_tag': '',
            'dataset_list': ["K562"],
            'back': ["Hek293t", "K562", "K562Hek293", "Kleinstiver", "Haeussler", "Listgarten", "SITE", "Tasi",
                     "GUIDE-Seq", "CIRCLE_seq", "Doench"]
        }
    ]

    for params in param_combinations:

        MODEL_NAME = params['model_name']
        CODING_NAME = params['coding_name']
        MODEL_TAG = params['model_tag']


        for dataset_name in params['dataset_list']:
            if(bestbz[dataset_name]):
                BATCH_SIZE = bestbz[dataset_name]
            else:
                BATCH_SIZE = 512

            print(f"{timeprint()} Processing dataset: {dataset_name} with model: {MODEL_NAME}, best bz: {BATCH_SIZE}")


            X_predict, X_on, X_off, labels = load_encoded_data(dataset_name, CODING_NAME)

            results_df = pd.DataFrame(
                columns=['Fold', 'Accuracy', 'F1_score', 'Precision', 'Recall', 'ROC_AUC', 'PR_AUC'])
            stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

            for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(X_predict, labels), 1):
                print(f"Processing dataset: {dataset_name} with model: {MODEL_NAME} Processing fold {fold}")

                X_predict_train, X_predict_test = X_predict[train_indices], X_predict[test_indices]
                X_on_train, X_on_test = X_on[train_indices], X_on[test_indices]
                X_off_train, X_off_test = X_off[train_indices], X_off[test_indices]
                y_train, y_test = labels[train_indices], labels[test_indices]

                X_predict_test, X_predict_val, X_on_test, X_on_val, X_off_test, X_off_val, y_test, y_val = train_test_split(
                    X_predict_test, X_on_test, X_off_test, y_test, test_size=0.2, random_state=RANDOM_SEED
                )

                y_train, y_test, y_val = map(lambda x: to_categorical(x, NUM_CLASSES), [y_train, y_test, y_val])

                model_function = getattr(model_module, MODEL_NAME)
                model = model_function()
                model.compile(optimizer=tf.keras.optimizers.Adam(lr=INITIAL_LEARNING_RATE, amsgrad=False),
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=['accuracy', 'Precision', 'Recall'])

                callbacks = create_callbacks(MODEL_NAME, dataset_name, fold, BATCH_SIZE)

                model.fit(
                    [X_predict_train, X_on_train, X_off_train], y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=([X_predict_val, X_on_val, X_off_val], y_val),
                )

                save_trained_model(dataset_name, model, MODEL_NAME, fold, MODEL_TAG)

                predictions = model.predict([X_predict_test, X_on_test, X_off_test])
                predicted_labels = np.argmax(predictions, axis=1)
                true_labels = np.argmax(y_test, axis=1)
                predicted_scores = predictions[:, 1]

                save_predictions(dataset_name, MODEL_NAME, true_labels, predicted_scores, fold, MODEL_TAG)

                metrics = {
                    'Accuracy': accuracy_score,
                    'F1_score': f1_score,
                    'Precision': precision_score,
                    'Recall': recall_score,
                    'ROC_AUC': roc_auc_score,
                    'PR_AUC': average_precision_score
                }

                fold_results = {'Fold': fold}
                for metric_name, metric_func in metrics.items():
                    score = metric_func(true_labels, predicted_labels) if 'AUC' not in metric_name else metric_func(
                        true_labels, predicted_scores)
                    fold_results[metric_name] = round(score, 4)

                results_df = results_df.append(fold_results, ignore_index=True)

            # Calculate averages
            average_row = {'Fold': 'Average'}
            for metric in metrics.keys():
                average_row[metric] = round(results_df[metric].mean(), 4)
            results_df = results_df.append(average_row, ignore_index=True)

            # Add model information
            results_df = results_df.append({
                'Fold': f"{MODEL_NAME}",
                "Accuracy": f'{dataset_name}',
                "F1_score": f'ep:{NUM_EPOCHS}',
                "Precision": f'bz:{BATCH_SIZE}',
                "Recall": f'lr:{INITIAL_LEARNING_RATE}',
                "ROC_AUC": '',
                "PR_AUC": f'{MODEL_TAG}'
            }, ignore_index=True)

            log_results(results_df, MODEL_NAME)
            print(timeprint())
            print(results_df)


if __name__ == '__main__':
    main()