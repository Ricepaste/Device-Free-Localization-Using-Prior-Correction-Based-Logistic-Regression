from PCA import *
import os
import pandas as pd


def read_file_name() -> str:
    '''
    讀取所有csv檔名,並以list回傳
    '''
    file_list = os.listdir()
    file_name = []
    for file in file_list:
        if file.endswith('.csv'):
            file_name.append(file)
    return file_name


def read_data(file_name: str) -> np.array:
    '''
    讀取csv檔案,並以np.array回傳

    ### Parameters:
    file_name (str): 檔案名稱
    '''
    data = np.array(pd.read_csv(file_name, delimiter=',').values)

    return data


if __name__ == '__main__':
    file_name = read_file_name()
    data = read_data(file_name[0])
    print(f"Name of the current file: {file_name[0]}")
    print(f"Shape of the current data: {data.shape}")

    pca = PCA(n_components=10)
    pca.fit(data)
    X_projected = pca.project(data)

    print(f"Shape of X_projected: {X_projected.shape}")
    # print(f"X_projected: {X_projected}")
