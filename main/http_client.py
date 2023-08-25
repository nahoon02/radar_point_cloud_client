import os
import sys

""" uncommnet this two lines when this file is executed in [windows] command line"""
#sys.path.insert(0, 'c:/python39_env/lib/site-packages')
#sys.path.append('d:/radar_point_cloud_client')

""" uncommnet this when this lines when this file is executed in [linux] command line"""
#sys.path.insert(0, os.path.expanduser('~/python39_env/lib/python3.9/site-packages'))
#sys.path.append(os.path.expanduser('~/radar_point_cloud_client'))

import requests
import pandas as pd
import time
import numpy as np
from util.time_conversion import convert_second_to_time
from util.multiclass import MultiClassMetric


def main(url_post, csv_file_path):

    if not os.path.isfile(csv_file_path):
        raise RuntimeError(f'dataset file = [{csv_file_path}] does not exists !!!')

    # make dataframe
    df = pd.read_csv(csv_file_path, low_memory=False)

    # get ['order'] list
    order_sr = df['order'].drop_duplicates()
    order_list = order_sr.tolist()

    success_count = 0
    fail_count = 0
    total_samples = len(order_list)
    error_count = 0
    confusion_matrix = np.zeros((21, 21), dtype=np.int32)
    for idx, order in enumerate(order_list):

        start_time = time.time()

        order_df = df[df['order'] == order]

        # get a ground truth
        label_list = order_df['label'].values.tolist()
        label = label_list[0]

        # get input list
        input_df = order_df[['frame', 'x', 'y', 'z', 'snr']]
        input_list = input_df.values.tolist()

        post_data = {
            'frame_chunk': input_list
        }

        """
            send a post request
        """
        response = requests.post(url_post, json=post_data)

        # change json to  dict format
        response_dict = response.json()

        # get prediction result
        prediction = response_dict['result']

        """
            measure the prediction result
        """
        if prediction == 'fail':
            error_count += 1
        else: # success case
            if label == prediction:
                success_count += 1
            else:
                fail_count += 1

            confusion_matrix[label][prediction] += 1

        accuracy = success_count / (success_count + fail_count)

        """
           compute remaining time 
        """
        elapsed_time = time.time() - start_time
        processed_samples = error_count + success_count + fail_count
        remaining_samples = total_samples - processed_samples
        processed_time = convert_second_to_time(processed_samples * elapsed_time)
        remaining_time = convert_second_to_time(remaining_samples * elapsed_time)

        print(f'[{idx+1}/{total_samples}] processed... success_count=[{success_count:3d}], fail_count=[{fail_count:3d}],'
              f' accuracy=[{accuracy*100:.1f}%]')
        print(f'{processed_time} elapsed, {remaining_time} remained')

    # end of for loop

    """
        summary performance
    """
    metric = MultiClassMetric(num_classes=21)
    metric.set_cm(confusion_matrix)

    total_samples, class_samples_dict = metric.get_samples()

    print(f'total samples = {total_samples}')
    print(f'class samples = {class_samples_dict}')

    accuracy, f1_score, class_accuracy_dict, class_precision_dict, class_recall_dict, \
        class_f1_dict, cm_list = metric.result(True)

    print(f'accuracy = {float(accuracy) * 100:.1f}%')
    print(f'f1 score = {float(f1_score) * 100:.1f}%')
    print(f'class_accuracy = {class_accuracy_dict}')
    print(f'class_precision = {class_precision_dict}')
    print(f'class_recall = {class_recall_dict}')
    print(f'class_f1 score = {class_f1_dict}')
    print(f'confusion matrix')
    print(confusion_matrix)


if __name__ == '__main__':

    # server [post] URL
    #url_post = 'http://localhost:8080/inference'
    url_post = 'http://192.168.219.204:8080/inference'

    # dataset file for linux
    #dataset_file_path = os.path.expanduser('~/projects/radar_point_cloud_client/dataset/test_460.csv')
    # dataset file for windows
    dataset_file_path = 'd:/projects/radar_point_cloud_client/dataset/test_460.csv'

    main(url_post, dataset_file_path)

