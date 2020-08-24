import pandas as pd
import numpy as np


def readcsv_to_df(path, num_rows=None):
    pd.set_option('display.max_colwidth', 120)

    # for debug
    # data = pd.read_csv(r'C:\Users\shwa01\nmtproject\github\joeynmt\models\sy_transformer_wmt17_ende_groundhog_3\test_50.n_csv.dev', nrows=num_rows)

    data = pd.read_csv(path, nrows=num_rows, index_col=[0])
    data.style.set_properties(**{'text-align': 'left'})
    return data


def export_to_csv(data) -> None:  # Write DataFrame to a comma-separated values (csv) file
    data.to_csv(r'C:\Users\shwa01\nmtproject\github\joeynmt\models\sy_transformer_wmt17_ende_groundhog_3\test_50.csv')


# TODO: remove <pad> tokens
def clear_pad(data):
    dt_array = data["Predictions"].to_numpy(dtype = str)  #numpy.ndarray
    for i,s in enumerate(dt_array):
        dt_array[i]=np.char.strip(np.char.replace(s,'<pad>',''))
    #print(dt_array)
    data["Predictions"] = dt_array
    return data


# TODO: TER scores



# run function
test_data = readcsv_to_df( r'C:\Users\shwa01\nmtproject\github\joeynmt\models\sy_transformer_wmt17_ende_groundhog_3\hyps_50.csv',60)
print(test_data)
#clean_data = clear_pad(test_data)
