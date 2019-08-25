from treeinterpreter import treeinterpreter as ti

import sys
sys.path.append('../src/')

from importlib import reload
import class_utilities_mod
import variable_utilities

reload(class_utilities_mod)
from class_utilities_mod import *
reload(variable_utilities)
from variable_utilities import *
import random
random.seed(1234)


clientName = 'ABC'
date = '21Aug2019'


# xgb_params = {
#     'eta': 0.04,
#     'max_depth': 5,
#     'subsample': 0.6,
#     'colsample_bytree':0.6,
#     'objective': 'binary:logistic',
#     'eval_metric': ['auc','logloss'],
#     'lambda': 0.8,   
#     'alpha': 0.4, 
#     'base_score': train_y.mean(),
#     'silent': 0}
xgb_params = {
    'eta': 0.01,
    'max_depth': 3,
    'subsample': 0.6,
    'colsample_bytree':0.3,
    'objective': 'binary:logistic',
    'eval_metric': ['auc','logloss'],
    'lambda': 0.8,   
    'alpha': 0.4, 
    'base_score': 0.5,
    'silent': 0}
# xgb_params = {
#     'eta': 0.015,
#     'max_depth': 3,
#     'subsample': 0.6,
#     'colsample_bytree':0.3,
#     'objective': 'binary:logistic',
#     'eval_metric': ['auc','logloss'],
#     'lambda': 0.8,   
#     'alpha': 0.4, 
#     'base_score': 0.5,
#     'silent': 0}
cut_ratio = 0.1


FM = FinalModel(model_params=xgb_params, n_iter=350)
FM.fit_model(train_X.values, train_y.values, dev_X.values, dev_y.values)
train_result = FM.custom_result(train_X.values, train_y.values, cut_ratio)
dev_result = FM.custom_result(dev_X.values, dev_y.values,cut_ratio)



finalModel_class_path = '../scoring_file/FM.pkl'

pickle.dump(FM, open(finalModel_class_path, 'wb'), pickle.HIGHEST_PROTOCOL)

train_result.to_csv('../output/Train_Result.csv', index = False)

dev_result.to_csv('../output/Dev_Scored.csv')


class Model_Process(object):
    def __init__(self,processed_file = None, unprocessed_file=None, train_result=None,):
        assert(type(unprocessed_file)!=type(None))
        self.unprocessed_file_ = unprocessed_file
        self.processed_file_ = processed_file
        self.train_result_ = train_result
#         if self.train_result_ is not None:
#             self.train_result_['ML Adj Level'] = self.train_result_.true_y.values
    def make_dataframe(self, filename, sheet_name = 0, low_memory = False):
        if(type(filename) is type(pd.DataFrame())):
            return filename
        try:
            extension = filename.split('.')[-1]
        except:
            print("Please enter a file with an extension")
            return -1
        df = pd.DataFrame()
        try:
            if extension == 'csv':
                df = pd.read_csv(filename, low_memory = low_memory)
            elif extension == 'xls' or extension == 'xlsx':
                df = pd.read_excel(filename, sheet_name = sheet_name)
            elif extension == 'pickle':
                df = pd.read_pickle(filename)
            else:
                print(extension + " files are not supported\n")
        except FileNotFoundError:
            print("Please enter a valid file name")
        return df

    def drop_collective(self, df_main, df_sub=None, getIndex=False):
        drop_indices = df_main.index.difference(df_main.dropna().index)
        df_main = df_main.dropna()
        if df_sub is not None:
            df_sub.drop(drop_indices)
        if getIndex:
            return (drop_indices, df_main, df_sub)
        return (df_main, df_sub)
    
    def transform(self, df, prepcs, textPrpcs, FE, FT, FM):
        '''
        Transforms unprocessed df using existing FT, FM, prepcs.
        '''
        org_df = df.copy()

        # create a sudo-unique-id if not
        df['Line Item Id by Me'] = [str(i) for i in range(df.shape[0])]

        # preprocess
#         print('sparsity of scoring data before filling missing is: %.3f'%desc.compute_sparsity(df))

        df = prepcs.map_colNames(df, colMap=scoring_col_map)
        print('column names standardized.')
        df = prepcs.convert_dtypes(df, schema=scoring_schema)
        print('column dtypes converted.')

        df = prepcs.create_missing_indicator(df, input_cols=missing_indicator_cols)
        print('created missing indicators.')

        df = prepcs.transform_missing(df)
        print('scoring data is of %d rows'%df.shape[0])

#         print('sparsity of scoring data after filling missing is: %.3f\n'%desc.compute_sparsity(df))

        # text preprocess
        dtm, df['clean_description'] = textPrpcs.transform_corpus2TFIDF(df=df, input_col='description', save_wordDist=False, return_clean=True)
        print('shape of dev dtm is %d by %d'%dtm.shape)
        
#         WEPadding = textPrpcs.transform_corpus2WE(df=df, input_col='description')
#         print('shape of scoring data WE padding is (%d, %d, %d)\n'%WEPadding.shape)
        
        # feature engineering
        df = FE.transform(df.copy(), input_TFIDFX=dtm, drop_cols=drop_cols)

#         print('sparsity of new scoring data is: %.3f\n'%desc.compute_sparsity(df))

        # feature transforming
        dum = FT.transform_one_hot(df)
        print('Aftering transforming to dummy variables, scoring set is of %d cols'%dum.shape[1])

        X = dum[FT.selected_features_]
        print('shape of scoring X is %d,%d\n'%X.shape)
        
        from custom_cutoff import cutoff
        global scored_result
        scored_result = FM.score_result(org_df, X.values, cutoff=cutoff)
        print('%d lines are flagged by AI in the invoices\n'%scored_result['Recommend Adjustment'].sum())
        scored_result['adjusted'] = np.where(scored_result['AdjReason'].isnull(),0,1)
        scored_result.rename(index=str, columns={"ML Score": "predict_y", "Recommend Adjustment": "ai_adjusted", "adjusted" : "true_y"},inplace=True)
        scored_result = scored_result.drop('true_y',1)
        scored_result.rename(index=str, columns={"predict_y": "ML Score", "ai_adjusted": "Recommend Adjustment"},inplace=True)
        return X
    
    def preprocess_transform(self, filename):
        df = self.make_dataframe(filename)
        
        add_string_date = '' 
        preprocess_class_path = '../scoring_file/prepcs'+add_string_date+'.pkl'
#         describe_class_path = '../scoring_file/desc'+add_string_date+'.pkl'
        textPreprocess_class_path = '../scoring_file/textPrpcs'+add_string_date+'.pkl'
        featureEngineering_class_path = '../scoring_file/FE'+add_string_date+'.pkl'
        featureTransform_class_path = '../scoring_file/FT'+add_string_date+'.pkl'
        finalModel_class_path = '../scoring_file/FM'+add_string_date+'.pkl'
        
        prepcs = pickle.load(open(preprocess_class_path, 'rb'))
#         desc = pickle.load(open(describe_class_path, 'rb'))
        textPrpcs = pickle.load(open(textPreprocess_class_path, "rb" ))
        FE = pickle.load(open(featureEngineering_class_path, "rb" ))
        FT = pickle.load(open(featureTransform_class_path, "rb" ))
        FM = pickle.load(open(finalModel_class_path, "rb"))
        return self.transform(df, prepcs, textPrpcs, FE, FT, FM)


    def process(self, drop_col_vague=None):
        if self.processed_file_ is None:
            self.train_X_= self.preprocess_transform(self.unprocessed_file_)
        else:
            self.train_X_ = self.make_dataframe(self.processed_file_)
        self.train_unprocessed_ = self.make_dataframe(self.unprocessed_file_)
        assert(self.train_X_.shape[0]==self.train_unprocessed_.shape[0])
        
        self.train_X_, self.train_unprocessed_ = self.drop_collective(self.train_X_, self.train_unprocessed_) 
        if self.processed_file_ is not None:
            div_train_Y_int = self.train_result_["true_y"].values

        # Removing columns which cannot be explained
        if(drop_col_vague is not None):
            self.train_X_.drop(columns=np.array(drop_col_vague), inplace=True)
        if self.processed_file_ is not None:
            return (self.train_X_, self.train_unprocessed_, div_train_Y_int)
        else:
            return (self.train_X_, self.train_unprocessed_)


class Interpret():
    def __init__(self, estimators=100, jobs=-1, unprocessed_df=None, model_name=None):
        self.model_name_ = model_name
        if self.model_name_ is None:
            self.rf = RandomForestClassifier(n_estimators=estimators,n_jobs=jobs)
        else:
            try:
                extension = self.model_name_.split('.')[-1]
            except:
                print("Please enter a file with an extension")
                return -1
            try:
                if extension == 'pickle' or extension == 'pkl':
                    self.rf = pickle.load(open(model_name, 'rb'))
                else:
                    print(extension + " files are not supported\n")
            except FileNotFoundError:
                print("Please enter a valid file name")
        self.unprocessed_df_ = unprocessed_df
    def get_model(self):
        return self.rf
    def fit(self, train_X, train_Y=None):
        self.train_X_ = train_X
        self.train_Y_ = train_Y
        if self.model_name_ is None:
            self.rf.fit(train_X, train_Y)
        return
    def interpret(self, df):
        self.interpret_input_ = df
        self.prediction_, self.bias_, self.contributions_ = ti.predict(self.rf,self.interpret_input_)
        self.prediction_.reshape(1, -1), self.bias_.reshape(1, -1), self.contributions_.reshape(1, -1)
        return self.prediction_, self.bias_, self.contributions_
    
    def display_feature_contributions(self, top=5):
        '''
        Prints the features and their respective contributions.
        '''
        for i in self.interpret_input_.index:
            a,b,x = self.prediction_, self.bias_, self.contributions_
            ac=[]
            for c, feature in sorted(zip(self.contributions_[i],self.train_X_.columns),key=lambda x:~abs(x[0].any())):
                ac.append((feature, np.round(c, 3)))
                ac = sorted(ac, key = lambda x: (x[-1][-1]), reverse=True)
            print('Invoice Line Item ID:' + str(self.unprocessed_df_['Invoice Line Item Id'][i]))
            ac = pd.Series(ac)
            print(ac[:top])
            print("-"*20)
        return
    def get_contributions_datafile(self, filename, column_name = 'Model Explained', top=5):
        '''
        Load the final dataframe into a file with appropriate extension and filename
        extension : provide the valid extension of the file
        filename : provide appropriate file name (without the extension)
        '''
        try:
            extension = filename.split('.')[-1]
        except:
            print("Please enter a file with an extension")
            return -1
        self.unprocessed_df_[column_name] = ['']*len(self.unprocessed_df_)
        for i in self.interpret_input_.index:
            a,b,x = self.prediction_, self.bias_, self.contributions_
            ac=[]
            for c, feature in sorted(zip(self.contributions_[i],self.train_X_.columns),key=lambda x:~abs(x[0].any())):
                ac.append((feature, np.round(c, 3)))
                ac = sorted(ac, key = lambda x: (x[-1][-1]), reverse=True)
            ac = pd.Series(ac)
            dic = {key: value for (key, value) in ac[:top] if value[1]!=0}
            self.unprocessed_df_.loc[i, column_name] = ", ".join(dic.keys())
        try:
            if extension == 'csv':
                df = self.unprocessed_df_
                df = df.merge(scored_result[[keyCol, 'ML Score', 'Recommend Adjustment']], how = 'left', on = keyCol)
                df.to_csv(filename)
            elif extension == 'xls' or extension == 'xlsx':
                df = self.unprocessed_df_
                df = df.merge(scored_result[[keyCol, 'ML Score', 'Recommend Adjustment']], how = 'left', on = keyCol)
                df.to_csv(filename)
            elif extension == 'pickle':
                df = self.unprocessed_df_
                df = df.merge(scored_result[[keyCol, 'ML Score', 'Recommend Adjustment']], how = 'left', on = keyCol)
                df.to_csv(filename)
        except:
            print(extension + " files are not supported\n")
        return df

M_test = Model_Process(unprocessed_file='../data/Century_Link_poc_scoring_data.csv')
test_X_m, test_unprocessed_m = M_test.process()

TI = Interpret(model_name="../scoring_file/{}_RF_ML_Explain_100T_true_y.pickle".format(clientName), unprocessed_df=test_unprocessed_m)
TI.fit(test_X_m)

%time p_test,b_test,c_test=TI.interpret(test_X_m)

scored_df = TI.get_contributions_datafile(filename='../output/scored_invoice_ML_explain.csv')