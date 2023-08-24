import os
import pandas as pd
import numpy as np


def remake_csv(csv_file_path, samples_per_class, num_class, output_file_path):
    # make dataframe
    df = pd.read_csv(csv_file_path, low_memory=False)

    class_df_list = []

    for idx in range(num_class):
        class_idx = idx + 1

        class_df = df[df['label'] == class_idx]

        order_sr = class_df['order'].drop_duplicates()
        order_list = order_sr.tolist()

        if samples_per_class > len(order_list):
            selected_order_list = order_list
        else:
            selected_order_list = order_list[:samples_per_class]

        selected_class_df = class_df[class_df['order'].isin(selected_order_list)]

        class_df_list.append(selected_class_df)

        print(f'class = [{class_idx}] selected_order_list = {selected_order_list}')

    merged_df = pd.concat(class_df_list, ignore_index=True)
    merged_df.to_csv(output_file_path, index=False)

    order_sr = merged_df['order'].drop_duplicates()
    order_list = order_sr.tolist()
    print(f'total samples = [{len(order_list)}] ')


if __name__ == '__main__':

    NUM_CLASS = 20

    input_csv_file = 'd:/projects/radar_point_cloud_client/dataset/valid.csv'
    samples_per_class = 23

    output_csv_file = 'd:/projects/radar_point_cloud_client/dataset/valid_460.csv'

    if not os.path.isfile(input_csv_file):
        raise RuntimeError(f'csv file does not exist ==> [{input_csv_file}]')

    remake_csv(input_csv_file, samples_per_class, NUM_CLASS, output_csv_file)