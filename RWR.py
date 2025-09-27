import networkx as nx
import numpy as np
import pandas as pd
# 药物靶点矩阵
drug_target=pd.read_csv("E:\Thesis\drug\drug_target.csv",index_col=0)
# 蛋白质相似度得分矩阵
df1=pd.read_csv("E:\Thesis\cell\string_network_matrix.csv",index_col=0)
# print(df1)
# print(df1.index)
# print(df1.loc[148, :])


def get_max_probability(drug_target, network):

    # input drug_target matrix: columns = genes, index = drugs
    result_drug_target = pd.DataFrame(np.zeros(shape=drug_target.shape), columns=drug_target.columns,
                                      index=drug_target.index)
    for gene in result_drug_target.columns:
        for drug in result_drug_target.index:


            if drug_target.loc[drug, gene] == 1:
                result_drug_target.loc[drug, gene] = 1
            else:
                result_drug_target.loc[drug, gene] = (drug_target.loc[drug, :] * network.loc[int(gene), :]).max()

    result_drug_target.to_csv("E:\Thesis\drug\pro_drug_target1.csv")
    return result_drug_target
get_max_probability(drug_target,df1)
def reindex(df,df1):
    column_order = df1.columns.tolist()
    column_order = [int(num) for num in column_order]
    #df = df.reindex(columns=column_order)
    df = df.loc[:, list(column_order)]
    get_max_probability(df,df1)
    return df


def normalize_matrix(raw_matrix, axis):

    def __normalize(vector):

        denominator = sum(vector)
        if not denominator:
            return vector

        return vector/denominator

    # normalize each row
    normalized_matrix = pd.DataFrame(np.zeros(shape=raw_matrix.shape, dtype='float'), columns=raw_matrix.columns, index=raw_matrix.index)
    if axis == 0 or axis == 'index':

        for i in raw_matrix.index:
            normalized_matrix.loc[i, :] = __normalize(raw_matrix.loc[i, :])

    elif axis == 1 or axis == 'column':

        for i in raw_matrix.columns:
            normalized_matrix.loc[:, i] = __normalize(raw_matrix.loc[:, i])

    

    return normalized_matrix