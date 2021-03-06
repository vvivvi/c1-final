import numpy as np
import pandas as pd
import os.path
import gc

###

###


def load_feature_set(id_string, data_folder='.', datasets = 'all'):
    # read the specified feature set into memory from a disk file
    # return the feature vectors X and labels Y

    # to conserve memory, only subsets of the full feature set specified
    # with the argument 'datasets' are returned

    # additionally return the mapping between the test set indices and submission IDs
    # if test set features are included in the specified 'datasets'

    assert datasets in ['all', 'train_and_val','trainval','test']

    date_block_val = 23  # Dec 2014
    date_block_test = 35  # Dec 2015

    filename = os.path.join(data_folder,
                            'feature_set_{}.csv').format(id_string)
    print('reading from file {}'.format(filename))
    all_data = pd.read_csv(os.path.join(data_folder, filename))

    all_data['target'] = np.clip(all_data['target'], 0, 20)
    all_data['time_of_year'] = (all_data['date_block_num'] + 6) % 12

    dates = all_data['date_block_num']
    if datasets in ['all', 'train_and_val']:
        dates_train = (dates <= date_block_val - 2)

    if datasets in ['all', 'trainval']:
        dates_trainval = (dates <= date_block_test - 2)


# extract training, validation and test sets (labels and features)
    if datasets in ['all', 'train_and_val']:
        y_train = all_data.loc[dates_train, 'target']
        y_val = all_data.loc[dates == date_block_val, 'target']
    
    if datasets in ['all', 'trainval']:
        y_trainval = all_data.loc[dates_trainval, 'target']
    
    to_drop_cols = ['target']

    if datasets in ['all', 'train_and_val']:
        X_train = all_data.loc[dates_train].drop(to_drop_cols, axis=1)
        X_val = all_data.loc[dates == date_block_val].drop(to_drop_cols, axis=1)
    if datasets in ['all', 'trainval']:
        X_trainval = all_data.loc[dates_trainval].drop(to_drop_cols, axis=1)
    if datasets in ['all', 'test']:    
        X_test = all_data.loc[dates == date_block_test].drop(to_drop_cols,
                                                             axis=1)

        # determine how to permute test set predictions for submission generation

        test_spec = pd.read_csv(os.path.join(data_folder, 'test.csv'))

        shop_item2submissionid = {}
        for idx, row in test_spec.iterrows():
            shop_item2submissionid[str(row['shop_id']) + '_' +
                                   str(row['item_id'])] = row['ID']

        test_data = all_data.loc[dates == date_block_test,
                                 ['shop_id', 'item_id']]

        testidx2submissionidx = np.zeros(test_data.shape[0], dtype=np.int32)
        for idx in range(test_data.shape[0]):
            row = test_data.iloc[idx]
            testidx2submissionidx[idx] = shop_item2submissionid[
                str(row['shop_id']) + '_' + str(row['item_id'])]

    #invert the mapping
        submissionidx2testidx = np.zeros(test_data.shape[0], dtype=np.int32)
        for i in range(test_data.shape[0]):
            submissionidx2testidx[testidx2submissionidx[i]] = i

        del test_data
        gc.collect()

    if datasets == 'all':
        return X_train, y_train, X_trainval, y_trainval, X_val, y_val, X_test, submissionidx2testidx
    elif datasets == 'train_and_val':
        return X_train, y_train, X_val, y_val
    elif datasets == 'trainval':
        return X_trainval, y_trainval
    elif datasets == 'test':        
        return X_test, submissionidx2testidx

def downcast_dtypes(df):
    '''
        Changes column types in the dataframe df: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''

    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df


###


# a wrapper class to use pre-defined division to training and hold-out set
# as a cross-validation object
class HoldOut:
    """
    Hold-out cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.
    Here the split is given as a input parameter in the class initialisation
    Unlike in other cross-validation schemes, the hold-out
    consists of only one iteration.

    Parameters
    ----------
    train_indices, test_indices : the class just passes on these when yielding splits


    """
    def __init__(self, train_indices, test_indices):
        self.train_indices = train_indices
        self.test_indices = test_indices

    def __iter__(self):
        yield self.train_indices, self.test_indices


###


def write_predictions_by_array(array, filename, data_folder='.'):
    # writes a formatted submission file onto
    # disk using the specified array of predictions

    df = pd.DataFrame(np.clip(array, 0, 20))
    df.columns = ['item_cnt_month']
    df.to_csv(os.path.join(data_folder, filename), index_label='ID')


###


def clipped_rmse(gt, predicted, clip_min=0, clip_max=20):
    # the clipped root mean squared error metric for the C1 sales prediction competition
    target = np.minimum(np.maximum(gt, clip_min), clip_max)
    return np.sqrt((target - predicted)**2).mean()


###
def average_submissions(filenames_in, id_out, data_folder='.'):
    # forms unweighted mean submission by averaging all the submission files
    # given as inputs
    df_first = pd.read_csv(os.path.join(data_folder, filenames_in[0]))
    n_rows = df_first.shape[0]
    result = np.zeros((n_rows, 1))
    for fn in filenames_in:
        df = pd.read_csv(os.path.join(data_folder, fn))
        contrib = df['item_cnt_month'].to_numpy() / len(filenames_in)
        print(contrib.shape)
        result += contrib[..., np.newaxis]
    fn_out = 'submission-average-{}.csv'.format(id_out)
    write_predictions_by_array(result, os.path.join(data_folder, fn_out))


###


def weight_submissions(filenames_in, weights, id_out, data_folder='.'):
    # forms weighted average submission by averaging all the submission files
    # given as inputs with specified weights

    assert len(filenames_in) == len(weights)
    df_first = pd.read_csv(os.path.join(data_folder, filenames_in[0]))
    n_rows = df_first.shape[0]
    result = np.zeros((n_rows, 1))
    for fn, w in zip(filenames_in, weights):
        df = pd.read_csv(os.path.join(data_folder, fn))
        contrib = df['item_cnt_month'].to_numpy() * w / len(filenames_in)
        print(contrib.shape)
        result += contrib[..., np.newaxis]
    fn_out = 'submission-weighted-{}.csv'.format(id_out)
    write_predictions_by_array(result, os.path.join(data_folder, fn_out))


###
def safe_div(a, b):
    epsilon = 1e-10
    return (a) / (b + epsilon)


###
def combine_score_dicts(list_of_dicts):
    res_dict = {}
    for d in list_of_dicts:
        for key in d:
            res_dict[key] = d[key] + res_dict.get(key, 0)
    return res_dict
