import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 目標移動
# def trajectory(width, height, num_points=100):
#     # 生成軌跡1:矩形上的點
#     x = np.linspace(0, width, num_points)
#     y = np.linspace(0, height, num_points)
#     return np.column_stack((x, y))

# LOS數據生成


def calculate_LOS_loss(distance, frequency):
    if distance == 0:
        return 878787  # 當距離為零時返回負無窮或其他適當的值
    # 計算自由空間損耗
    Lbf = 32.5 + 20 * np.log10(frequency) + 20 * np.log10(distance/1000)
    return Lbf

# def generate_LOS_data(node_positions, frequency):
#     print("          LOS-------------------")
#     distance = np.linalg.norm(node_positions[0] - node_positions[1])
#     LOS_data = -calculate_LOS_loss(distance, frequency)  # 可能不對: 0-TelsoB 最大傳輸功率
#     print(LOS_data)
#     return LOS_data  # 返回單個值而不是列表

# LOS數據生成


def generate_LOS_data(node_positions, frequency):
    print("          LOS-------------------")
    noise = np.random.normal(0, 1)
    distance = np.linalg.norm(node_positions[0] - node_positions[1])
    if distance == 0:
        return 878787  # 當距離為零時返回負無窮或其他適當的值
    LOS_data = -calculate_LOS_loss(distance, frequency) - noise
    # print(LOS_data) # Test
    return LOS_data


# NLOS數據生成
def generate_NLOS_data(node_positions, frequency):
    print("          NLOS-------------------")
    noise = np.random.normal(0, 5)
    distance = np.linalg.norm(node_positions[0] - node_positions[1])
    if distance == 0:
        return 878787  # 當距離為零時返回負無窮或其他適當的值
    NLOS_data = -calculate_LOS_loss(distance, frequency) - noise
    # print(NLOS_data) # Test
    return NLOS_data

# 確定目標與節點關係


def Check_LOS(now_node_positions, node_positions, reference_position):
    node1 = now_node_positions[0]
    node2 = now_node_positions[1]
    for info in line_info:
        # print("info[0]=" + str(info[0]))
        if np.array_equal(node_positions[int(info[0])], node1) and np.array_equal(node_positions[int(info[1])], node2):
            m = info[2]
            # print("info[2]=" + str(info[2]))
            min_x = info[3]
            # print("info[3]=" + str(info[3]))
            max_x = info[4]
            slope = (reference_position[1] - node1[1]) / \
                (reference_position[0] - node1[0] + 1e-10)  # 避免除以零
            # print("slope=" + str(slope))
            # print("m=" + str(m))
            # 計算最小 x 範圍和最大 x 範圍
            # if (( slope >= 0 and m >= 0 ) or ( slope <=0 and m <= 0 )):
            if (np.isclose(slope, m, 1e-6) or np.isclose(slope, -m, 1e-3) or (abs(slope) >= 1e+4 and abs(m) >= 1e+4)):
                if reference_position[0] >= min_x and reference_position[0] <= max_x:
                    # print("OntheWay!!!!!")
                    return True
    return False

# 生成資料集


def construct_training_testing_dataset(node_positions, reference_positions, frequency):
    dataset = []
    for reference_position in reference_positions:
        print("ref -------------------")
        data_row = []
        for measure in range(0, 800):
            rss_data = []
            for i in range(len(node_positions)):
                for j in range(i+1, len(node_positions), 1):
                    node1 = node_positions[i]
                    node2 = node_positions[j]
                    # 根據 LOS 或 NLOS 計算 RSS 數據
                    if Check_LOS([node1, node2], node_positions, reference_position):
                        rss_data.append(generate_NLOS_data(
                            [node1, node2], frequency))
                    else:
                        rss_data.append(generate_LOS_data(
                            [node1, node2], frequency))
            data_row.append(rss_data)
            print(data_row)
        dataset.append(data_row)

    return np.array(dataset)


def generate_datasets(node_positions, reference_positions, frequency):
    for idx, reference_position in enumerate(reference_positions):
        dataset = construct_training_testing_dataset(
            node_positions, [reference_position], frequency)
        df = pd.DataFrame(dataset[0])
        df.to_csv(f'dataset_{idx}.csv', index=False)

# 以下為Standarize--------------------------------------------------------


def standardized(reference_positions, input_file_prefix):
    for idx, reference_position in enumerate(reference_positions):
        input_file = f'{input_file_prefix}_{idx}.csv'
        df = pd.read_csv(input_file, index_col=0)  # 加载数据集
        X = df.values  # 将数据集转换为NumPy数组
        scaler = StandardScaler()  # 初始化标准化器
        X_standardized = scaler.fit_transform(X)  # 标准化数据集
        # 这里可以将标准化后的数据集用于后续的处理
        output_file = f'standardized_dataset_{idx}.csv'
        df_standardized = pd.DataFrame(X_standardized)
        df_standardized.to_csv(output_file, index=False)
        print(
            f'Standardized dataset for reference point {reference_position} saved to {output_file}')


# 實際執行
# node_positions = np.array([[0.1, 4.9], [3, 4.9], [5.9, 4.9], [0.1, 2.5],
#                             [5.9, 2.5], [0.1, 0.1], [3, 0.1], [5.9, 0.1]])


node_positions = np.array([[0, 5], [3, 5], [5, 5], [0, 2.5],
                           [5, 2.5], [0, 0], [3, 0], [5, 0]])

# 查詢節點間的直線表
line_info = np.zeros((28, 5))  # 28-link, 5-每行包含起、終、斜率、最小 x 範圍、最大 x 範圍
counter = 0
for i in range(len(node_positions)):
    for j in range(i+1, len(node_positions), 1):
        node1 = node_positions[i]
        node2 = node_positions[j]
        # 計算斜率
        m = (node2[1] - node1[1]) / (node2[0] - node1[0] + 1e-10)  # 避免除以零
        # 計算最小 x 範圍和最大 x 範圍
        min_x = min(node1[0], node2[0])
        max_x = max(node1[0], node2[0])
        line_info[counter] = [i, j, m, min_x, max_x]
        # print(counter)
        counter = counter + 1

# 根据不同的参考点生成不同的数据集，并存入同一个CSV文件的不同分页中

# 以下為PCA--------------------------------------------------------


# print(line_info)
# width = 4  # x 位置
# height = 3  # y 位置
# target_positions = trajectory(width, height)
reference_positions = np.array(
    [[3, 1], [4, 1], [4, 2], [4, 2.5], [3, 4], [4, 4], [1, 2.5], [1, 3]])
frequency = 2400  # 頻率為 2.4 GHz
# dataset = construct_training_testing_dataset(node_positions, reference_positions, frequency)
# print(dataset)
# np.savetxt('dataset.txt', dataset)
generate_datasets(node_positions, reference_positions, frequency)
input_file = 'dataset'
standardized_datasets = standardized(reference_positions, input_file)
# # 將資料轉換為 DataFrame
# df = pd.DataFrame(dataset)

# # 寫入 Excel 檔案
# df.to_excel('dataset.xlsx', index=False)
