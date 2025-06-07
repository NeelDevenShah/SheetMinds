
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("AI GENERATED CODE EXECUTION STARTED")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

import pandas as pd
import numpy as np
import json
import traceback


# Load the data from the embedded JSON
data = [{'Monthly Statistical Report For Month Of July - 2023': 'S.No.', 'Unnamed: 1': 'States/UTs/PSUs', 'Unnamed: 2': 'No. Of Tenders', 'Unnamed: 3': 'Value of Tenders (Rs. in Crores)', 'Unnamed: 4': 'No. Of Tenders', 'Unnamed: 5': 'Value of Tenders (Rs. in Crores)', 'Unnamed: 6': 'No. Of Tenders', 'Unnamed: 7': 'Value of Tenders (Rs. in Crores)', 'Unnamed: 8': 'No. Of Tenders', 'Unnamed: 9': 'Value of Tenders (Rs. in Crores)', 'Unnamed: 10': 'No. Of Tenders', 'Unnamed: 11': 'Contract Value(Rs. in Crores)'}, {'Monthly Statistical Report For Month Of July - 2023': 1, 'Unnamed: 1': 'CPPP', 'Unnamed: 2': 12065, 'Unnamed: 3': 49321.59, 'Unnamed: 4': 15744, 'Unnamed: 5': 108071.85, 'Unnamed: 6': 53322, 'Unnamed: 7': 260892, 'Unnamed: 8': 1533765, 'Unnamed: 9': 6312195.28, 'Unnamed: 10': 14000, 'Unnamed: 11': 187445.87}, {'Monthly Statistical Report For Month Of July - 2023': 2, 'Unnamed: 1': 'Andaman and Nicobar', 'Unnamed: 2': 340, 'Unnamed: 3': 299.95, 'Unnamed: 4': 273, 'Unnamed: 5': 107.77, 'Unnamed: 6': 1141, 'Unnamed: 7': 783, 'Unnamed: 8': 11307, 'Unnamed: 9': 9333.46, 'Unnamed: 10': 70, 'Unnamed: 11': 31.11}, {'Monthly Statistical Report For Month Of July - 2023': 3, 'Unnamed: 1': 'Arunachal Pradesh', 'Unnamed: 2': 29, 'Unnamed: 3': 507.51, 'Unnamed: 4': 37, 'Unnamed: 5': 280.57, 'Unnamed: 6': 89, 'Unnamed: 7': 876, 'Unnamed: 8': 1181, 'Unnamed: 9': 14512.58, 'Unnamed: 10': 15, 'Unnamed: 11': 28.8}, {'Monthly Statistical Report For Month Of July - 2023': 4, 'Unnamed: 1': 'Assam', 'Unnamed: 2': 722, 'Unnamed: 3': 1723.48, 'Unnamed: 4': 548, 'Unnamed: 5': 2047.5, 'Unnamed: 6': 2383, 'Unnamed: 7': 7443, 'Unnamed: 8': 48217, 'Unnamed: 9': 205231.87, 'Unnamed: 10': 2560, 'Unnamed: 11': 6828.58}, {'Monthly Statistical Report For Month Of July - 2023': 5, 'Unnamed: 1': 'Chandigarh', 'Unnamed: 2': 826, 'Unnamed: 3': 614.74, 'Unnamed: 4': 945, 'Unnamed: 5': 458.05, 'Unnamed: 6': 3436, 'Unnamed: 7': 2732, 'Unnamed: 8': 90511, 'Unnamed: 9': 30752.72, 'Unnamed: 10': 1841, 'Unnamed: 11': 678.85}, {'Monthly Statistical Report For Month Of July - 2023': 6, 'Unnamed: 1': 'Dadra and Nagar Haveli', 'Unnamed: 2': 56, 'Unnamed: 3': 351.58, 'Unnamed: 4': 39, 'Unnamed: 5': 361.63, 'Unnamed: 6': 215, 'Unnamed: 7': 1203, 'Unnamed: 8': 6356, 'Unnamed: 9': 14116.94, 'Unnamed: 10': 32, 'Unnamed: 11': 128.53}, {'Monthly Statistical Report For Month Of July - 2023': 7, 'Unnamed: 1': 'Daman and Diu', 'Unnamed: 2': 79, 'Unnamed: 3': 358.4, 'Unnamed: 4': 71, 'Unnamed: 5': 207.06, 'Unnamed: 6': 255, 'Unnamed: 7': 765, 'Unnamed: 8': 2876, 'Unnamed: 9': 9401.83, 'Unnamed: 10': 10, 'Unnamed: 11': 16.24}, {'Monthly Statistical Report For Month Of July - 2023': 8, 'Unnamed: 1': 'Haryana', 'Unnamed: 2': 6354, 'Unnamed: 3': 4543.86, 'Unnamed: 4': 7831, 'Unnamed: 5': 6213.06, 'Unnamed: 6': 30963, 'Unnamed: 7': 24433, 'Unnamed: 8': 362575, 'Unnamed: 9': 273996.46, 'Unnamed: 10': 15164, 'Unnamed: 11': 9489.76}, {'Monthly Statistical Report For Month Of July - 2023': 9, 'Unnamed: 1': 'Himachal Pradesh', 'Unnamed: 2': 1633, 'Unnamed: 3': 945.77, 'Unnamed: 4': 1794, 'Unnamed: 5': 2499.64, 'Unnamed: 6': 6621, 'Unnamed: 7': 8998, 'Unnamed: 8': 108270, 'Unnamed: 9': 87652.02, 'Unnamed: 10': 2454, 'Unnamed: 11': 1559.37}, {'Monthly Statistical Report For Month Of July - 2023': 10, 'Unnamed: 1': 'Jammu and Kashmir', 'Unnamed: 2': 32179, 'Unnamed: 3': 2881.02, 'Unnamed: 4': 29259, 'Unnamed: 5': 3183.14, 'Unnamed: 6': 74844, 'Unnamed: 7': 10826, 'Unnamed: 8': 613529, 'Unnamed: 9': 216044.14, 'Unnamed: 10': 215, 'Unnamed: 11': 67.72}, {'Monthly Statistical Report For Month Of July - 2023': 11, 'Unnamed: 1': 'Jharkhand', 'Unnamed: 2': 1196, 'Unnamed: 3': 3804.04, 'Unnamed: 4': 1455, 'Unnamed: 5': 3198.47, 'Unnamed: 6': 6966, 'Unnamed: 7': 18805, 'Unnamed: 8': 93978, 'Unnamed: 9': 388298.76, 'Unnamed: 10': 150, 'Unnamed: 11': 3918.41}, {'Monthly Statistical Report For Month Of July - 2023': 12, 'Unnamed: 1': 'Kerala', 'Unnamed: 2': 10624, 'Unnamed: 3': 7735.16, 'Unnamed: 4': 8383, 'Unnamed: 5': 4705.54, 'Unnamed: 6': 31948, 'Unnamed: 7': 27435, 'Unnamed: 8': 823821, 'Unnamed: 9': 358702.9, 'Unnamed: 10': 24340, 'Unnamed: 11': 11567.9}, {'Monthly Statistical Report For Month Of July - 2023': 13, 'Unnamed: 1': 'Ladakh', 'Unnamed: 2': 673, 'Unnamed: 3': 248.44, 'Unnamed: 4': 631, 'Unnamed: 5': 382.18, 'Unnamed: 6': 2590, 'Unnamed: 7': 1483, 'Unnamed: 8': 24241, 'Unnamed: 9': 15143.49, 'Unnamed: 10': 107, 'Unnamed: 11': 26.53}, {'Monthly Statistical Report For Month Of July - 2023': 14, 'Unnamed: 1': 'Madhya Pradesh', 'Unnamed: 2': 8095, 'Unnamed: 3': 13029.45, 'Unnamed: 4': 8601, 'Unnamed: 5': 26962.57, 'Unnamed: 6': 34531, 'Unnamed: 7': 69598, 'Unnamed: 8': 327107, 'Unnamed: 9': 700219.58, 'Unnamed: 10': 22956, 'Unnamed: 11': 35902.64}, {'Monthly Statistical Report For Month Of July - 2023': 15, 'Unnamed: 1': 'Maharashtra', 'Unnamed: 2': 16486, 'Unnamed: 3': 16664.73, 'Unnamed: 4': 18124, 'Unnamed: 5': 17359.18, 'Unnamed: 6': 68377, 'Unnamed: 7': 81504, 'Unnamed: 8': 1547592, 'Unnamed: 9': 1124139.41, 'Unnamed: 10': 31148, 'Unnamed: 11': 51344.76}, {'Monthly Statistical Report For Month Of July - 2023': 16, 'Unnamed: 1': 'Manipur', 'Unnamed: 2': 20, 'Unnamed: 3': 296, 'Unnamed: 4': 17, 'Unnamed: 5': 332.63, 'Unnamed: 6': 67, 'Unnamed: 7': 1421, 'Unnamed: 8': 2285, 'Unnamed: 9': 33216.89, 'Unnamed: 10': 2, 'Unnamed: 11': 19.44}, {'Monthly Statistical Report For Month Of July - 2023': 17, 'Unnamed: 1': 'Meghalaya', 'Unnamed: 2': 15, 'Unnamed: 3': 79.91, 'Unnamed: 4': 6, 'Unnamed: 5': 11.63, 'Unnamed: 6': 32, 'Unnamed: 7': 578, 'Unnamed: 8': 1435, 'Unnamed: 9': 10244.59, 'Unnamed: 10': 8, 'Unnamed: 11': 50.51}, {'Monthly Statistical Report For Month Of July - 2023': 18, 'Unnamed: 1': 'Mizoram', 'Unnamed: 2': 7, 'Unnamed: 3': 67.29, 'Unnamed: 4': 3, 'Unnamed: 5': 25.92, 'Unnamed: 6': 15, 'Unnamed: 7': 147, 'Unnamed: 8': 339, 'Unnamed: 9': 3477.59, 'Unnamed: 10': 8, 'Unnamed: 11': 100.61}, {'Monthly Statistical Report For Month Of July - 2023': 19, 'Unnamed: 1': 'Nagaland', 'Unnamed: 2': 2, 'Unnamed: 3': 0, 'Unnamed: 4': 0, 'Unnamed: 5': 0, 'Unnamed: 6': 11, 'Unnamed: 7': 12, 'Unnamed: 8': 478, 'Unnamed: 9': 2895.63, 'Unnamed: 10': 1, 'Unnamed: 11': 1.77}, {'Monthly Statistical Report For Month Of July - 2023': 20, 'Unnamed: 1': 'Odisha', 'Unnamed: 2': 3276, 'Unnamed: 3': 7681.48, 'Unnamed: 4': 2669, 'Unnamed: 5': 5591.82, 'Unnamed: 6': 12612, 'Unnamed: 7': 20806, 'Unnamed: 8': 423801, 'Unnamed: 9': 414900.27, 'Unnamed: 10': 5238, 'Unnamed: 11': 6106.01}, {'Monthly Statistical Report For Month Of July - 2023': 21, 'Unnamed: 1': 'Punjab', 'Unnamed: 2': 2498, 'Unnamed: 3': 1599.67, 'Unnamed: 4': 2742, 'Unnamed: 5': 2239.61, 'Unnamed: 6': 11185, 'Unnamed: 7': 14604, 'Unnamed: 8': 211428, 'Unnamed: 9': 154425.81, 'Unnamed: 10': 6091, 'Unnamed: 11': 5261.47}, {'Monthly Statistical Report For Month Of July - 2023': 22, 'Unnamed: 1': 'Rajasthan', 'Unnamed: 2': 12867, 'Unnamed: 3': 27103.49, 'Unnamed: 4': 14763, 'Unnamed: 5': 32223.12, 'Unnamed: 6': 48262, 'Unnamed: 7': 118447, 'Unnamed: 8': 608047, 'Unnamed: 9': 1111196.53, 'Unnamed: 10': 1114, 'Unnamed: 11': 6055.49}, {'Monthly Statistical Report For Month Of July - 2023': 23, 'Unnamed: 1': 'Sikkim', 'Unnamed: 2': 7, 'Unnamed: 3': 327.84, 'Unnamed: 4': 8, 'Unnamed: 5': 203.78, 'Unnamed: 6': 38, 'Unnamed: 7': 1183, 'Unnamed: 8': 626, 'Unnamed: 9': 7498.53, 'Unnamed: 10': 0, 'Unnamed: 11': 0}, {'Monthly Statistical Report For Month Of July - 2023': 24, 'Unnamed: 1': 'GOA', 'Unnamed: 2': 483, 'Unnamed: 3': 416.74, 'Unnamed: 4': 644, 'Unnamed: 5': 1232.39, 'Unnamed: 6': 2081, 'Unnamed: 7': 2864, 'Unnamed: 8': 10116, 'Unnamed: 9': 14020.61, 'Unnamed: 10': 865, 'Unnamed: 11': 602.7}, {'Monthly Statistical Report For Month Of July - 2023': 25, 'Unnamed: 1': 'Tamil Nadu', 'Unnamed: 2': 23351, 'Unnamed: 3': 11307.21, 'Unnamed: 4': 14351, 'Unnamed: 5': 23387.09, 'Unnamed: 6': 63477, 'Unnamed: 7': 54148, 'Unnamed: 8': 398400, 'Unnamed: 9': 383963.08, 'Unnamed: 10': 6689, 'Unnamed: 11': 15315.59}, {'Monthly Statistical Report For Month Of July - 2023': 26, 'Unnamed: 1': 'Tripura', 'Unnamed: 2': 758, 'Unnamed: 3': 478.89, 'Unnamed: 4': 742, 'Unnamed: 5': 476.22, 'Unnamed: 6': 2961, 'Unnamed: 7': 1925, 'Unnamed: 8': 39328, 'Unnamed: 9': 31428.06, 'Unnamed: 10': 1906, 'Unnamed: 11': 692.34}, {'Monthly Statistical Report For Month Of July - 2023': 27, 'Unnamed: 1': 'Uttar Pradesh', 'Unnamed: 2': 13707, 'Unnamed: 3': 10222, 'Unnamed: 4': 16190, 'Unnamed: 5': 34596.65, 'Unnamed: 6': 62022, 'Unnamed: 7': 95224, 'Unnamed: 8': 1327159, 'Unnamed: 9': 1514605.88, 'Unnamed: 10': 15671, 'Unnamed: 11': 9259.11}, {'Monthly Statistical Report For Month Of July - 2023': 28, 'Unnamed: 1': 'Uttarkhand', 'Unnamed: 2': 1195, 'Unnamed: 3': 3440.45, 'Unnamed: 4': 1206, 'Unnamed: 5': 1001.52, 'Unnamed: 6': 4835, 'Unnamed: 7': 10096, 'Unnamed: 8': 70157, 'Unnamed: 9': 134297.91, 'Unnamed: 10': 2054, 'Unnamed: 11': 2539.19}, {'Monthly Statistical Report For Month Of July - 2023': 29, 'Unnamed: 1': 'West Bengal', 'Unnamed: 2': 16478, 'Unnamed: 3': 10211.7, 'Unnamed: 4': 16015, 'Unnamed: 5': 7190.16, 'Unnamed: 6': 113791, 'Unnamed: 7': 44850, 'Unnamed: 8': 1089557, 'Unnamed: 9': 568260.41, 'Unnamed: 10': 55125, 'Unnamed: 11': 52109.51}, {'Monthly Statistical Report For Month Of July - 2023': 30, 'Unnamed: 1': 'Union Territory of Lakshadweep', 'Unnamed: 2': 33, 'Unnamed: 3': 189.13, 'Unnamed: 4': 48, 'Unnamed: 5': 69.39, 'Unnamed: 6': 140, 'Unnamed: 7': 267, 'Unnamed: 8': 2424, 'Unnamed: 9': 3755.53, 'Unnamed: 10': 1, 'Unnamed: 11': 0}, {'Monthly Statistical Report For Month Of July - 2023': 31, 'Unnamed: 1': 'NCT of Delhi', 'Unnamed: 2': 1756, 'Unnamed: 3': 1304.4, 'Unnamed: 4': 2235, 'Unnamed: 5': 1093.08, 'Unnamed: 6': 7358, 'Unnamed: 7': 5408, 'Unnamed: 8': 310136, 'Unnamed: 9': 166572.79, 'Unnamed: 10': 211, 'Unnamed: 11': 101.61}, {'Monthly Statistical Report For Month Of July - 2023': 32, 'Unnamed: 1': 'Puducherry', 'Unnamed: 2': 235, 'Unnamed: 3': 124.74, 'Unnamed: 4': 307, 'Unnamed: 5': 177.45, 'Unnamed: 6': 917, 'Unnamed: 7': 503, 'Unnamed: 8': 12764, 'Unnamed: 9': 6417.48, 'Unnamed: 10': 370, 'Unnamed: 11': 274.96}, {'Monthly Statistical Report For Month Of July - 2023': 33, 'Unnamed: 1': 'Bharat Heavy Electricals Ltd', 'Unnamed: 2': 901, 'Unnamed: 3': 1799.39, 'Unnamed: 4': 998, 'Unnamed: 5': 778.49, 'Unnamed: 6': 3469, 'Unnamed: 7': 3665, 'Unnamed: 8': 26980, 'Unnamed: 9': 22464.56, 'Unnamed: 10': 97, 'Unnamed: 11': 140.56}, {'Monthly Statistical Report For Month Of July - 2023': 34, 'Unnamed: 1': 'Coal India Ltd', 'Unnamed: 2': 2016, 'Unnamed: 3': 2490.37, 'Unnamed: 4': 2192, 'Unnamed: 5': 639.95, 'Unnamed: 6': 7851, 'Unnamed: 7': 10743, 'Unnamed: 8': 301744, 'Unnamed: 9': 588077.13, 'Unnamed: 10': 8707, 'Unnamed: 11': 16120.48}, {'Monthly Statistical Report For Month Of July - 2023': 35, 'Unnamed: 1': 'Chennai Petroleum Corp Ltd', 'Unnamed: 2': 42, 'Unnamed: 3': 3.24, 'Unnamed: 4': 31, 'Unnamed: 5': 0, 'Unnamed: 6': 151, 'Unnamed: 7': 13, 'Unnamed: 8': 9250, 'Unnamed: 9': 634.47, 'Unnamed: 10': 12, 'Unnamed: 11': 128.57}, {'Monthly Statistical Report For Month Of July - 2023': 36, 'Unnamed: 1': 'Defence PSU', 'Unnamed: 2': 142, 'Unnamed: 3': 48.87, 'Unnamed: 4': 167, 'Unnamed: 5': 1487.26, 'Unnamed: 6': 635, 'Unnamed: 7': 2286, 'Unnamed: 8': 28622, 'Unnamed: 9': 19056.91, 'Unnamed: 10': 108, 'Unnamed: 11': 102.6}, {'Monthly Statistical Report For Month Of July - 2023': 37, 'Unnamed: 1': 'Indian Oil Corporation Limited', 'Unnamed: 2': 1222, 'Unnamed: 3': 5225.55, 'Unnamed: 4': 1168, 'Unnamed: 5': 7365.62, 'Unnamed: 6': 4319, 'Unnamed: 7': 27885, 'Unnamed: 8': 171006, 'Unnamed: 9': 407658.84, 'Unnamed: 10': 1790, 'Unnamed: 11': 9996.09}, {'Monthly Statistical Report For Month Of July - 2023': 38, 'Unnamed: 1': 'NTPC Ltd', 'Unnamed: 2': 705, 'Unnamed: 3': 175.32, 'Unnamed: 4': 740, 'Unnamed: 5': 499.63, 'Unnamed: 6': 3062, 'Unnamed: 7': 6406, 'Unnamed: 8': 77993, 'Unnamed: 9': 317119.72, 'Unnamed: 10': 310, 'Unnamed: 11': 754.42}, {'Monthly Statistical Report For Month Of July - 2023': 39, 'Unnamed: 1': 'Pradhan Mantri Gram Sadak Yojana', 'Unnamed: 2': 1194, 'Unnamed: 3': 4205.74, 'Unnamed: 4': 643, 'Unnamed: 5': 2458.88, 'Unnamed: 6': 3780, 'Unnamed: 7': 13673, 'Unnamed: 8': 193109, 'Unnamed: 9': 547747.87, 'Unnamed: 10': 554, 'Unnamed: 11': 4355.74}]
df = pd.DataFrame(data)

# User's analysis code
import pandas as pd
import numpy as np

# The data is already loaded in a variable called 'df'

# Display the first few rows of the DataFrame to understand its structure
# print(df.head())

# Display the data types of each column
# print(df.dtypes)

# Display summary statistics for numerical columns
# print(df.describe())

# Display information about the DataFrame, including data types and missing values
# print(df.info())

# Rename columns for easier access
df.columns = [
    'S_No', 'States_UTs_PSUs', 'No_Of_Tenders_Col2', 'Value_of_Tenders_Col3',
    'No_Of_Tenders_Col4', 'Value_of_Tenders_Col5', 'No_Of_Tenders_Col6',
    'Value_of_Tenders_Col7', 'No_Of_Tenders_Col8', 'Value_of_Tenders_Col9',
    'No_Of_Tenders_Col10', 'Contract_Value_Col11'
]

# Convert relevant columns to numeric, handling errors by coercing invalid values to NaN
cols_to_numeric = [
    'No_Of_Tenders_Col2', 'Value_of_Tenders_Col3', 'No_Of_Tenders_Col4',
    'Value_of_Tenders_Col5', 'No_Of_Tenders_Col6', 'Value_of_Tenders_Col7',
    'No_Of_Tenders_Col8', 'Value_of_Tenders_Col9', 'No_Of_Tenders_Col10',
    'Contract_Value_Col11'
]

for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with any NaN values after numeric conversion
df = df.dropna()

# Remove the first row which contains column descriptions
df = df.iloc[1:]

# Calculate the total number of tenders
total_tenders = df['No_Of_Tenders_Col2'].sum() + df['No_Of_Tenders_Col4'].sum() + df['No_Of_Tenders_Col6'].sum() + df['No_Of_Tenders_Col8'].sum() + df['No_Of_Tenders_Col10'].sum()

# Calculate the total value of tenders
total_value = df['Value_of_Tenders_Col3'].sum() + df['Value_of_Tenders_Col5'].sum() + df['Value_of_Tenders_Col7'].sum() + df['Value_of_Tenders_Col9'].sum() + df['Contract_Value_Col11'].sum()

# Create a summary DataFrame
summary_data = {'Total Number of Tenders': [total_tenders],
                'Total Value of Tenders (Rs. in Crores)': [total_value]}
summary_df = pd.DataFrame(summary_data)

# The result is the summary DataFrame
result = summary_df
    
# Ensure result is defined
if 'result' not in locals() and 'result' not in globals():
    result = df

# Convert result to a serializable format
if hasattr(result, 'to_dict'):
    if hasattr(result, 'index'):
        if hasattr(result, 'reset_index'):
            result = result.reset_index()
        if hasattr(result, 'to_dict'):
            output = result.to_dict(orient='records' if hasattr(result, 'columns') else None)
    else:
        output = result.to_dict()
elif isinstance(result, (np.ndarray, list, tuple)):
    output = result.tolist() if hasattr(result, 'tolist') else list(result)
elif isinstance(result, (str, int, float, bool)) or result is None:
    output = result
else:
    output = str(result)

# Ensure the output is JSON serializable
def make_serializable(obj):
    if isinstance(obj, (np.integer, int, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, float, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    return str(obj)

# Apply serialization
if output is not None:
    if isinstance(output, (dict, list, tuple)):
        output = make_serializable(output)

# Save the output
with open('result.json', 'w') as f:
    json.dump(output, f, default=str)

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('AI GENERATED CODE EXECUTION COMPLETED')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
