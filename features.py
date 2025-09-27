import pandas as pd
from os import path, mkdir
import numpy as np
import torch
from torch.utils import data
from src import  setting
from sklearn.preprocessing import StandardScaler
from torch import save

class CustomDataLoader:
    pass
class SamplesDataLoader(CustomDataLoader):

    whole_df = None
    Y = None
    drug_features_lengths = []
    cellline_features_lengths = []
    F_cl = None
    var_filter = None
    raw_x = None
    combine_drug_multi_gene_express = None
    single_drug_response = None
    @classmethod
    def drug_features_prep(cls):
        drug_features = ['drug_target_profile']
        cls.genes = pd.read_csv("E:\Thesis\cell\combin_genes.csv", dtype={'entrez': np.int})
        cls.entrez_set = cls.genes['entrez']
        add_single_response_to_drug_target = True
        cls.simulated_drug_target=pd.read_csv("E:\Thesis\drug\pro_drug_target1.csv",index_col=0)
        cls.drug_a_features = None
        cls.drug_b_features = None
        cls.cellline_features = None
        cls.synergy_score = pd.read_csv("E:\Thesis\drug\synergy_score.csv")
        cls.single_drug_response = pd.read_csv("E:\Thesis\drug\single_response_features.csv")

        ### generate drugs features
        if cls.drug_a_features is None or cls.drug_b_features is None or cls.drug_features is None:
            cls.drug_features = []
            cls.drug_a_features = []
            cls.drug_b_features = []

            if 'drug_target_profile' in drug_features:

                drug_a_target_feature = cls.simulated_drug_target.loc[list(cls.synergy_score['drug_a_name']), :]
                drug_a_target_feature = pd.DataFrame(drug_a_target_feature).reset_index(drop=True)
                if add_single_response_to_drug_target:
                    # drug_a_single_response = cls.single_drug_response.loc[list(cls.synergy_score['drug_a_name']), :]

                    drug_a_single_response = cls.single_drug_response.merge(cls.synergy_score,how='right',
                                                                            # left_on=['drug', 'cell_line'],
                                                                            on=['drug_a_name', 'cell_line'])[
                        'pIC50'].values

                    drug_a_target_feature = drug_a_target_feature[:len(drug_a_single_response)]
                    # assert len(drug_a_single_response) == len(drug_a_target_feature), "single repsonse data didn't have same length with drug feature"
                    drug_a_target_feature['pIC50'] = drug_a_single_response

                drug_a_target_feature.fillna(0, inplace=True)
                # drug_a_target_feature.to_csv("feature.csv")
                cls.drug_a_features.append(drug_a_target_feature.values)
                cls.drug_features_lengths.append(drug_a_target_feature.shape[1])
                drug_b_target_feature = cls.simulated_drug_target.loc[list(cls.synergy_score['drug_b_name']), :]
                drug_b_target_feature = pd.DataFrame(drug_b_target_feature).reset_index(
                    drop=True)

                if setting.add_single_response_to_drug_target:
                    # drug_b_single_response = cls.single_drug_response.loc[list(cls.synergy_score['drug_b_name']), :]
                    drug_b_single_response = cls.single_drug_response.merge(cls.synergy_score,how='right',
                                                                            # left_on=['drug', 'cell_line'],
                                                                            on=['drug_b_name', 'cell_line'])[
                        'pIC50'].values
                    drug_b_target_feature = drug_b_target_feature[:len(drug_b_single_response)]
                    assert len(drug_b_single_response) == len(drug_b_target_feature), "single repsonse data didn't have same length with drug feature"

                    drug_b_target_feature['pIC50'] = drug_b_single_response
                drug_b_target_feature.fillna(0, inplace=True)
                cls.drug_b_features.append(drug_b_target_feature.values)

        return [cls.drug_a_features, cls.drug_b_features]

    @classmethod
    def cellline_features_prep(cls):

            cls.genes = pd.read_csv("E:\Thesis\cell\combin_genes.csv", dtype={'entrez': np.int})
            cls.entrez_set = cls.genes['entrez']
            cellline_features= ['gene_dependence','GSVA']
            cls.cellline_features = []
            cls.sel_dp=pd.read_csv("genedp35.csv",index_col=0).T
            # cls.expression_di=pd.read_csv("E:\Thesis\cell\process.csv",index_col=0)
            dp_features =[]
            cellline_express_features=[]
            ### generate cell lines features
            if 'gene_dependence' in cellline_features:

                dp_features = cls.sel_dp
                dp_features = pd.DataFrame(dp_features).reset_index(drop=True)
                dp_features.fillna(0, inplace=True)
                cls.cellline_features.append(dp_features.values)
                cls.cellline_features_lengths.append(dp_features.shape[1])

            if 'gene_expression' in cellline_features:
                cellline_express_features = cls.expression_df.T.loc[list(cls.synergy_score['cell_line']), :]
                cls.cellline_features.append(cellline_express_features.values)
                cls.cellline_features_lengths.append(cellline_express_features.shape[1])

            if setting.add_single_response_to_drug_target:


                for i in range(len(cls.cellline_features)):

                    cls.cellline_features[i] = np.concatenate([cls.cellline_features[i],
                                                        np.array([[0] * len(cls.cellline_features[i])]).reshape(-1,1)],
                                                       axis=1)
                    cls.cellline_features_lengths[i] += 1

            return cls.cellline_features

    @classmethod
    def construct_whole_raw_X(cls):

        ### return dataframe
        ###  first_half_drugs_features                first_half_cellline_features
        ###  switched_second_half_drugs_features      second_half_cellline_features
        if cls.whole_df is None:
            two_drugs_features_list = cls.drug_features_prep()
            cellline_features_list = cls.cellline_features_prep()
            first_half = np.concatenate(tuple(two_drugs_features_list[0] + two_drugs_features_list[1] +
                                              cellline_features_list), axis=0)
            second_half = np.concatenate(tuple(two_drugs_features_list[1] + two_drugs_features_list[0] +
                                               cellline_features_list), axis=0)
            cls.whole_df = np.concatenate(tuple([first_half, second_half]), axis=1)#.reset_index(drop=True)
        return cls.whole_df

    @classmethod
    def Raw_X_features_prep(cls, methods):

        ### Generate final raw features dataset
        ### return: ndarray (n_samples, n_type_features, feature_dim) if 'attn'
        ###         ndarray (n_samples, n_type_features * feature_dim) else
        raw_x = cls.construct_whole_raw_X()
        entrez_array = np.array(list(cls.entrez_set))
        if methods == 'attn':
            x = raw_x.reshape(-1, setting.n_feature_type, len(cls.entrez_set))
            filter_drug_features_len = filter_cl_features_len = x.shape[-1]
            drug_features_name = cl_features_name = cls.entrez_set

        elif methods == 'flexible_attn':

            return raw_x, cls.drug_features_lengths, cls.cellline_features_lengths

        else:
            drug_features_len = int(1 / setting.n_feature_type * raw_x.shape[1])
            cl_features_len = int(raw_x.shape[1] - 2 * drug_features_len)
            assert cl_features_len == int((1 - 2 / setting.n_feature_type) * raw_x.shape[1]), \
                "features len are calculated in wrong way"
            var_filter = raw_x.var(axis=0) > 0
            filter_drug_features_len = sum(var_filter[:drug_features_len])
            filter_cl_features_len = sum(var_filter[2*drug_features_len:])
            drug_features_name = entrez_array[var_filter[:drug_features_len]]
            cl_features_name = np.array(list(entrez_array) * (setting.n_feature_type - 2))[var_filter[2 * drug_features_len:]]
            x = raw_x[:, var_filter]
            assert filter_drug_features_len == len(drug_features_name) and filter_cl_features_len == len(cl_features_name), \
                                                                                  'features len and names do not match'
        return x, filter_drug_features_len, filter_cl_features_len, list(drug_features_name), list(cl_features_name)
    @classmethod
    def Y_features_prep(cls):

        ### Generate final y features in ndarray (-1, 1)
        #cls.__dataloader_initializer()
        Y_labels = cls.synergy_score.loc[:, 'synergy']
        Y_half = Y_labels.values.reshape(-1, 1)
        Y = np.concatenate((Y_half, Y_half), axis=0)
        return Y
    @classmethod
    def get_final_index(cls):
        cls.final_index=None
        if cls.final_index is None:
            cls.synergy_score = pd.read_csv("E:\Thesis\drug\synergy_score.csv")
            synergy_score = cls.synergy_score
            final_index_1 = synergy_score.reset_index().apply(
                lambda row: row['drug_a_name'] + '_' + row['drug_b_name'] + '_' +
                            row['cell_line'] + '_' + str(row['index']), axis=1)
            final_index_2 = synergy_score.reset_index().apply(
                lambda row: row['drug_b_name'] + '_' + row['drug_a_name'] + '_' +
                            row['cell_line'] + '_' + str(row['index']), axis=1)
            cls.final_index = pd.concat([final_index_1, final_index_2], axis=0).reset_index(drop=True)
        return cls.final_index

samples_loader = SamplesDataLoader()
samples_loader.get_final_index()
class DataPreprocessor:

    X = None
    Y = None
    drug_features_len = None
    cl_features_len = None
    synergy_score = None
    methods = None

    def __init__(self, methods):
        self.methods = methods
        pass

    @classmethod
    def __dataset_initializer(cls):

        if cls.X is None:
            cls.X, cls.drug_features_len, cls.cl_features_len, _, _ = SamplesDataLoader.Raw_X_features_prep(cls.methods)
        if cls.Y is None:
            cls.Y = SamplesDataLoader.Y_features_prep()
        if cls.synergy_score is None:
            cls.synergy_score = pd.read_csv("E:\Thesis\drug\synergy_score.csv")

    @classmethod
    def reg_train_eval_test_split(cls, fold = 'fold', test_fold = 0):
        index_in_literature = True
        if cls.synergy_score is None:
            cls.synergy_score = pd.read_csv("E:\Thesis\drug\synergy_score.csv")

        if setting.index_in_literature:
            evluation_fold = np.random.choice(list({0,1,2,3,4}-{test_fold}))
            evluation_fold = 0
            print(evluation_fold)
            test_index = np.array(cls.synergy_score[cls.synergy_score[fold] == test_fold].index)
            evaluation_index = np.array(cls.synergy_score[cls.synergy_score[fold] == evluation_fold].index)
            train_index = np.array(cls.synergy_score[(cls.synergy_score[fold] != test_fold) &
                                                     (cls.synergy_score[fold] != evluation_fold)].index)



        train_index = np.concatenate([train_index, train_index + cls.synergy_score.shape[0]])
        evaluation_index_2 = evaluation_index + cls.synergy_score.shape[0]
        test_index_2 = test_index + cls.synergy_score.shape[0]
        yield train_index, test_index, test_index_2, evaluation_index, evaluation_index_2

    @classmethod
    def cv_train_eval_test_split_generator(cls, fold = 'fold'):

        if cls.synergy_score is None:
            cls.synergy_score = pd.read_csv("E:\Thesis\drug\synergy_score.csv")

        assert setting.index_in_literature, "Cross validation is only available when index_in_literature is set to True"
        for evluation_fold in range(1,5):
            test_index = np.array(cls.synergy_score[cls.synergy_score[fold] == 0].index)
            evaluation_index = np.array(cls.synergy_score[cls.synergy_score[fold] == evluation_fold].index)
            train_index = np.array(cls.synergy_score[(cls.synergy_score[fold] != 0) &
                                                     (cls.synergy_score[fold] != evluation_fold)].index)
            train_index = np.concatenate([train_index + cls.synergy_score.shape[0], train_index])
            evaluation_index_2 = evaluation_index + cls.synergy_score.shape[0]
            test_index_2 = test_index + cls.synergy_score.shape[0]
            yield train_index, test_index, test_index_2, evaluation_index, evaluation_index_2

class MyDataset(data.Dataset):

    synergy_score = None
    drug_smile = None
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, prefix=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.prefix = prefix
        if MyDataset.synergy_score is None:
            print('prepare synergy score')
            MyDataset.synergy_score =  pd.read_csv("E:\Thesis\drug\synergy_score.csv")
            synergy_score_reverse = MyDataset.synergy_score.copy()
            synergy_score_reverse['drug_a_name'] = MyDataset.synergy_score['drug_b_name']
            synergy_score_reverse['drug_b_name'] = MyDataset.synergy_score['drug_a_name']
            MyDataset.synergy_score = pd.concat([MyDataset.synergy_score, synergy_score_reverse])
            MyDataset.synergy_score.reset_index(inplace=True)
        if MyDataset.drug_smile is None:
            print('prepare drug smile')
            name_smile_df = pd.read_csv("E:\Thesis\data\chemicals\inchi_merck.csv")
            MyDataset.drug_smile = {name: smile for name, smile in zip(name_smile_df['Name'], name_smile_df['SMILE'])}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.prefix is None:
            drug_combine_file = path.join(setting.data_folder, ID + '.pt')
        else:
            drug_combine_file = self.prefix + '_datas/' + ID + '.pt'
        # Load data and get label
        try:
            X = torch.load(drug_combine_file)
        except:
            raise
        y = self.labels[ID]
        drug_a = MyDataset.synergy_score.loc[index, 'drug_a_name']
        drug_a_smiles = MyDataset.drug_smile[drug_a]
        drug_b = MyDataset.synergy_score.loc[index, 'drug_b_name']
        drug_b_smiles = MyDataset.drug_smile[drug_b]

        return (X, drug_a_smiles, drug_b_smiles), y