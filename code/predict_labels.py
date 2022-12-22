#!/usr/bin/env python

"""
Script for labeling spans in unseen BAR titles.

"""

## TODO: sort out imported libraries

import sys
import os
import argparse
import configparser
import joblib
import logging
import unicodedata

import pandas as pd

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

sys.path.append('lib')
from bar_preprocessors import BARTokenizer, FeatureExtractor, FeatureCollector, BIOGenerator, entity_spans
from bar_tools import Annotation, AHVChecker, DateChecker, OffsetParser, prob_diff_sec, get_doc_labels, censoring_labels

logging.info('Starting...')


class BARModels(object):
    def __init__(self, models_dir=None, models_dict=None, relabeler=True):

        self.labels = ['j', 'n', 'p', 'd', 'O', 'wd', 'a']

        if models_dir == models_dict == None:
            logging.exception("Either models directory or models dictionary needs to be defined.")
            raise ValueError("Either models_dir or models_dict must be True.")

        if relabeler is True:
            self.relabeler = Relabeler()
        else:
            self.relabeler = None

        if models_dir:
            vectorizer_path = os.path.join(models_dir, 'vectorizer.joblib')
            logging.info(f'Loading vectorizer from {vectorizer_path}')
            self.vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.joblib'))
            self.models_dict = self.load_models(models_dir)
        else:
            models_dict_copy = models_dict.copy()
            self.vectorizer = models_dict_copy.pop('vectorizer')
            self.models_dict = models_dict_copy

    def load_models(self, models_dir):
        logging.info('Loading models...')

        models_dict = dict()

        for model_fn in os.listdir(models_dir):
            if model_fn == 'vectorizer.joblib' or model_fn.endswith('encoder.joblib') or model_fn.startswith('.'):
                continue

            model_path = os.path.join(models_dir, model_fn)

            if model_fn.endswith('joblib'):
                models_dict[model_fn] = joblib.load(model_path)

        return models_dict

    def get_clf_probsdict_df(self, probs_dictlist, clf_label=''):
        
        le = {0: 'O', 1: 'a', 2: 'd', 3: 'j', 4: 'n', 5: 'p', 6: 'wd'}
        #{0: 'O', 1: 'a', 2: 'd', 3: 'j', 4: 'n', 5: 'p', 6: 'wd'}

        probs_preds_df = pd.DataFrame(probs_dictlist)

        if set(probs_preds_df.columns) == set(list(le.values())):
            probs_preds_df.columns = [f'{c}-{clf_label}' for c in probs_preds_df.columns]
        else:
            probs_preds_df.columns = [f'{le[c]}-{clf_label}' for c in probs_preds_df.columns]
        
        return probs_preds_df
    
    
    def apply_models(self, doc_feat_df, tok_feat_df):

        preds_dfs_list = list()
        
        feat_matrix = self.vectorizer.transform(tok_feat_df.tok_features)
        feat_matrix = feat_matrix.toarray()

        logging.debug(f'Shape of features (document level): {doc_feat_df.doc_features.shape}')
        logging.debug(f'Shape of feature matrix (token level): {feat_matrix.shape}')

        for model_fn, clf in self.models_dict.items():

            if 'crf' in model_fn.lower():

                preds_probas = clf.predict_marginals(doc_feat_df.doc_features)

                probs_dictlist = [item for sublist in preds_probas for item in sublist]
                probsdict_df = self.get_clf_probsdict_df(probs_dictlist, clf_label='crf')
                preds_dfs_list.append(probsdict_df)


            else:
                preds_probas = clf.predict_proba(feat_matrix)
                
                if 'mlp' in model_fn.lower():
                    probsdict_df = self.get_clf_probsdict_df(preds_probas, clf_label='mlp')
                elif 'svm' in model_fn.lower():
                    probsdict_df = self.get_clf_probsdict_df(preds_probas, clf_label='svm')
                else:
                    raise Exception(f'Error: Model not recognized: {model_fn}')
                preds_dfs_list.append(probsdict_df)

        combined_probs = pd.concat(preds_dfs_list, axis=1)
        return combined_probs

    @staticmethod
    def find_label_weighted_meanthresh(prob_dict, n_thresh=0.2, j_thresh=0.15, weight_dict={'crf':2, 'svm':1, 'mlp':1}):
        if sum([prob_dict[n]*weight_dict[n[-3:]] for n in [k for k in prob_dict.keys() if k.startswith('n-')]])/sum(weight_dict.values()) > n_thresh :
            return 'n'
        if sum([prob_dict[j]*weight_dict[j[-3:]] for j in [k for k in prob_dict.keys() if k.startswith('j-')]])/sum(weight_dict.values()) > j_thresh :
            return 'j'
        else:
            max_key = max(prob_dict, key=prob_dict.get) 
            return max_key[:2].rstrip('-')


    def predict_labels(self, doc_feat_df, tok_feat_df, id_field='IDENTIFIER', blackout_labels=['n', 'd', 'j', 'a'], n_thresh=0.2, j_thresh=0.15, weight_dict={'crf':2, 'svm':1, 'mlp':1}):
        logging.info('Applying models...')

        combined_probs = self.apply_models(doc_feat_df, tok_feat_df)

        logging.info('Ensembling...')
        prob_dicts = combined_probs[[c for c in combined_probs.columns if any([c.startswith(f'{i}-') for i in self.labels])]].apply(lambda x: x.to_dict(), axis=1)

        weighted_meanthresh = prob_dicts.apply(lambda x: BARModels.find_label_weighted_meanthresh(x, n_thresh=n_thresh, j_thresh=j_thresh, weight_dict=weight_dict))

        labeled_df = tok_feat_df[[id_field, 'td-idx']].reset_index(drop=True)
        labeled_df['token'] = tok_feat_df['tok_features'].apply(lambda x: x['token']).reset_index(drop=True)
        labeled_df['weighted_meanthresh'] = weighted_meanthresh

        ## Re-labeling according to rules
        if self.relabeler:
            logging.info('Applying relabeling...')
            preds = self.relabeler.relabel_all(labeled_df, prob_dicts, id_field=id_field)
        else:
            preds = labeled_df['weighted_meanthresh']

        blackout_labels = preds.apply(lambda x: x in blackout_labels)
        labeled_df['blackout_label'] = blackout_labels
        labeled_df.columns = [id_field, 'td-idx', 'token', 'prediction', 'blackout_label']

        return labeled_df


class BarAnonymizer(object):
    def __init__(self, dictionary_paths, models_dir, context_window=4, j_thresh=0.1, n_thresh=0.17):
        self.dictionary_paths = dictionary_paths
        self.models_dir = models_dir
        self.context_window = context_window
        self.j_thresh = j_thresh
        self.n_thresh = n_thresh
        self.blackout_labels = ['n', 'd', 'j', 'a']
        self.weight_dict = {'crf': 2, 'svm': 1, 'mlp': 1}
        self.tokenizer = BARTokenizer()
        self.feature_collector = FeatureCollector(self.tokenizer, self.dictionary_paths, context_window=self.context_window)
        self.models = BARModels(models_dir=models_dir, relabeler=True)

    def blackout_format(self, labeled_df, id_field='IDENTIFIER', black_char=chr(9608)):

        labeled_df['labeled_tokens'] = labeled_df[['token', 'blackout_label']].apply(lambda x: black_char * 3 if str(x[1]) == 'True' else x[0], axis=1)

        pred_df_entries = labeled_df.groupby(id_field).agg(lambda x: ' '.join([str(i) for i in list(x)])).reset_index()
        return pred_df_entries

    def labels_format(self, labeled_df, id_field='IDENTIFIER'):
        labeled_df['labeled_tokens'] = labeled_df[['token', 'ensemble_label']].apply(lambda x: f"<anonym type='{x[1]}'>{x[0]}</anonym>" if x[1] in self.blackout_labels else x[0], axis=1)
        pred_df_entries = labeled_df.groupby(id_field).agg(lambda x: ' '.join([str(i) for i in list(x)])).reset_index()
        return pred_df_entries


    @staticmethod
    def get_probs_weighted_meanthresh(prob_dict, n_thresh=0.2, j_thresh=0.15, weight_dict={'crf':2, 'svm':1, 'mlp':1}):

        n_score = sum([prob_dict[n]*weight_dict[n[-3:]] for n in [k for k in prob_dict.keys() if k.startswith('n-')]])/sum(weight_dict.values())
        j_score = sum([prob_dict[j]*weight_dict[j[-3:]] for j in [k for k in prob_dict.keys() if k.startswith('j-')]])/sum(weight_dict.values())
      
        if n_score > n_thresh :
            return {'label':'n', 'prob': n_score}
        if j_score > j_thresh :
            return {'label':'j', 'prob': j_score}
        else:
            max_key = max(prob_dict, key=prob_dict.get)
            return {'label': max_key[:2].rstrip('-'), 'prob': prob_dict[max_key]}

    @staticmethod
    def get_prob_diff(prob_dicts):
        probs_df = pd.DataFrame(prob_dicts.tolist())
        probs_matrix = probs_df.values
        all_prob_diffs = prob_diff_sec(probs_matrix, average=False)
        return all_prob_diffs


    def anonymize_df(self, in_df, anon_field='TITEL', id_field='IDENTIFIER', context_field='HIERARCHIE', anon_style='blacked', get_scores=False):
        '''Input: a pandas dataframe.'''

        # normalize non-breaking whitespaces (e.g. '\xa0')
        in_df = in_df.copy()
        in_df['titel_fixed'] = in_df[anon_field].astype('str').apply(lambda x: unicodedata.normalize("NFKD", x))
    
        #feat_df = self.feature_collector.add_features(in_df, self.dictionary_dir, context_field=context_field)
        feat_df = self.feature_collector.add_features(in_df, anon_field=anon_field, context_field=context_field, id_field=id_field)

        feature_extractor = FeatureExtractor(feat_df, context_window=self.context_window)
        doc_ids = pd.Series(in_df[id_field].unique())

        logging.info('Running feature extraction...')
        doc_features = doc_ids.apply(lambda x: feature_extractor.doc2features(x, id_r=id_field))  # series of dictionaries

        doc_feat_df = pd.concat((doc_features, doc_ids), axis=1)
        doc_feat_df.columns = ['doc_features', id_field]

        # convert to token level
        tok_feat_df = doc_feat_df.explode('doc_features')
        tok_feat_df = tok_feat_df.reset_index(drop=True).reset_index()
        tok_feat_df.columns = ['td-idx', 'tok_features', id_field]

        labeled_df = self.models.predict_labels(doc_feat_df, tok_feat_df, id_field=id_field, blackout_labels=self.blackout_labels, n_thresh=self.n_thresh, j_thresh=self.j_thresh, weight_dict=self.weight_dict)

        logging.info('Producing output format...')

        if anon_style == 'blacked':
            blackout_df = self.blackout_format(labeled_df, id_field=id_field, black_char=chr(9608))
            blackout_df = blackout_df[[id_field, 'token', 'blackout_label', 'labeled_tokens']].copy()
            return blackout_df
        elif anon_style == 'tags':
            labels_df = self.labels_format(labeled_df, id_field=id_field)
            return labels_df
        else:
            raise Exception(f'Unknown style for anonymization: {anon_style}. Valid options for anon_style are blacked or tags.')


    def json2df(self, json_data, id_field='IDENTIFIER'):
        column_names = [id_field, 'HIERARCHIE', 'SIGNATUR', 'TITEL', 'DARIN', 'ZUSATZKOMPONENTE', 'LAND', 'ENTSTEHUNGSZEITRAUM', 'Annot_Sprache', 'Annot_Bemmerkung', 'Liste', 'QS-Korrektur']
        if isinstance(json_data, list):
            json_data_df = pd.DataFrame(json_data)
            json_data_split = json_data_df.text.apply(lambda x: x.split('\t'))  # TODO: verify if number of fields corresponds to number of columns in column_names
            lengths = json_data_split.apply(lambda x: len(x)).unique()
            if len(lengths) == 1 and lengths[0] != len(column_names):
                raise Exception(f'Wrong field length of input. Expected: {len(column_names)}, received {lengths}.')
            in_df = pd.DataFrame(json_data_split.to_list(), columns=column_names)
        elif isinstance(json_data, dict):
            json_fields = json_data['text'].split('\t')
            in_df = pd.DataFrame(json_fields).T
            in_df.columns = column_names
        else:
            raise Exception('Wrong data format in Json input.')
        return in_df

    def process_file_input(self, file_path):
        # process input formats
        if file_path.endswith('.tsv'):
            in_df = pd.read_csv(file_path, sep='\t', dtype='str')
        elif file_path.endswith('.xlsx'):
            in_df = pd.read_excel(file_path, dtype='str')
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as infile:
                json_data = json.loads(infile)
                raise NotImplementedError
                # TODO: process json input as file!
        else:
            raise Exception('Input file format not recognized.')

        return in_df

    def anonymize_file_input(self, in_path, anon_style='blacked', anon_field_name='TITEL', id_field='IDENTIFIER', context_field='HIERARCHIE'):
        in_df = self.process_file_input(in_path)
        logging.debug(f'Columns in input dataframe: {in_df.columns}')
        blackout_df = self.anonymize_df(in_df, anon_field=anon_field_name, id_field=id_field, context_field=context_field, anon_style=anon_style)
        return blackout_df


    def json2df_newspec(self, json_data):
        if isinstance(json_data, dict):
            json_values = json_data['values']
            json_df = pd.DataFrame(json_values, index=[0])
            json_df['context'] = json_data['context']
            json_df['anon_style'] = json_data['options']['style']
            json_df['anon_fields'] = '|'.join(list(json_data['values'].keys()))
            json_df['entry_id'] = list(range(json_df.shape[0]))

        else:
            raise Exception('Wrong data format in Json input.')
        return json_df

    def anonymize_json(self, input):
        in_df = self.json2df_newspec(input)
        anonfields_df = pd.DataFrame()
        anon_style = in_df['anon_style'][0]
        for anon_field_name in in_df['anon_fields'][0].split('|'):
            blackout_df = self.anonymize_df(in_df, anon_field=anon_field_name, id_field=id_field, context_field='context', anon_style=anon_style)

            anonfields_df[anon_field_name] = blackout_df['labeled_tokens']

        if anonfields_df.shape[0] != 1:
            raise Exception(f'Json output has unusual shape: {anonfields_df.shape}')
        else:
            anonfields_df_normalized = anonfields_df.apply(lambda x: x.str.normalize('NFKC'), axis=1)
            json_out = anonfields_df_normalized.iloc[0].to_dict()
            return json_out

    def blacken(self, input, id_field='IDENTIFIER'):
        '''
        Input: one dictionary or a list of dictionaries.
        Output: an anonymized string or list of anonymized strings depending on type of input.
        '''
        in_df = self.json2df_newspec(input)
        blackout_df = self.anonymize_df(in_df, anon_ ='blacked', id_field=id_field)
        blackout_list = blackout_df.labeled_tokens.tolist()
        if len(blackout_list) == 1:
            return blackout_list[0]
        return blackout_list

    def get_labels(self, input, id_field='IDENTIFIER'):
        in_df = self.json2df_newspec(input)
        anonlabels_df = self.anonymize_df(in_df, anon_style='tags', id_field=id_field)
       
        anonlabels_list = anonlabels_df.labeled_tokens.tolist()
        if len(anonlabels_list) == 1:
            return anonlabels_list[0]
        return anonlabels_list


def run_anonymize(in_path, out_path, models_dir, dictionary_paths, id_field='IDENTIFIER'):

    anonymizer = BarAnonymizer(dictionary_paths, models_dir)

    blackout_df = anonymizer.anonymize_file_input(in_path, id_field=id_field)

    logging.info(f'Writing output to {out_path}.')
    blackout_df.to_csv(out_path, sep='\t', header=True, index=None)


class Relabeler(object):
    '''Relabeling according to rules'''
    def __init__(self):
        self.pers_titles = ['Dr.', 'Prof.', 'Herr', 'Frau', 'Hr.', 'Fr.', 'Mr.', 'Ms.', 'Mme.', 'Familie', 'Famille', 'Famiglia']
        self.numtok_after_titles = 3
        self.n_thresh_after_titles = 0.001

        self.n_thresh_no_nj = 0.01
        self.j_thresh_no_nj = 0.01

    def _no_nj(self, label_list):
        if not set(label_list).intersection(['j', 'n']):
            return True
        return False

    def relabel_after_title(self, labels_df, prob_dicts):
        # detect tokens which are titles and get their indices
        is_title = labels_df['token'].isin(self.pers_titles)
        title_indices = is_title[is_title == True].index
        after_title_indices = pd.Series(title_indices).apply(lambda x: [x+i for i in range(1,self.numtok_after_titles+1)]).explode().unique()

        after_title_indices = set(labels_df.index).intersection(after_title_indices)
        after_title_indices = pd.Index(after_title_indices)
        after_title_relabels = prob_dicts[after_title_indices].apply(lambda x: BARModels.find_label_weighted_meanthresh(x, n_thresh=self.n_thresh_after_titles))
        after_title_relabels = after_title_relabels.reindex(labels_df.index)

        return after_title_relabels

    def relabel_no_nj(self, labels_df, prob_dicts, id_field='IDENTIFIER'):
        grouped_labels = labels_df.groupby(id_field).agg(list).reset_index()
        no_nj = grouped_labels['weighted_meanthresh'].apply(lambda x: self._no_nj(x))
        no_nj_veids = grouped_labels[no_nj][id_field].to_list()

        no_nj_relabel = labels_df[id_field].apply(lambda x: x in no_nj_veids)

        no_nj_relabeled = prob_dicts[no_nj_relabel].apply(lambda x: BARModels.find_label_weighted_meanthresh(x, n_thresh=self.n_thresh_no_nj, j_thresh=self.j_thresh_no_nj))

        no_nj_relabeled = no_nj_relabeled.reindex(labels_df.index)

        return no_nj_relabeled

    def relabel_all(self, labels_df, prob_dicts, id_field='IDENTIFIER'):
        
        try:
            relabels_after_title = self.relabel_after_title(labels_df, prob_dicts)
        except Exception as e:
            print(e)

        relabels_no_nj = self.relabel_no_nj(labels_df, prob_dicts, id_field=id_field)

        relabeled_joined = relabels_after_title.fillna(relabels_no_nj).fillna(labels_df['weighted_meanthresh'])

        return relabeled_joined

    
def main():
    """
    Invoke this module as a script
    """

    description = "An NLP based anonymizer for archival documents metadata. This script is for data anonymization and needs previously trained models. Use train_models.py for model training."
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('-l', '--logfile', dest='logfilename',
                        help='write log to FILE', metavar='FILE')

    argparser.add_argument('--in_path', metavar='in_path', type=str, required=True,
                        help='Path to tsv or excel file containing entries to be labeled.')
    argparser.add_argument('--out_path', metavar='out_path', type=str, required=True,
                        help='Path to output file (tsv with labels included).')
    argparser.add_argument('--models_dir', metavar='models_dir', type=str, required=True,
                        help='Path to directory containing the machine learning models.')
    argparser.add_argument('--config_file', metavar='config_file', type=str, required=True,
                        help='Path to settings file containing default settings.')

    args = argparser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)

    config_args = config['config_args']

    # dictionary of dictionary filenames (for lookup features)
    dictionary_paths = {k:v for k,v in dict(config.items('dictionary_paths')).items() if k.endswith('path')}
    argparser.add_argument('--dictionary_paths', default=dictionary_paths)

    log_level_str = config['logger_args']['logger_level']

    argparser.add_argument('--logger_level', default=log_level_str, dest='log_level_str')
    argparser.add_argument('--id_field', default=config_args['id_field'])

    args = argparser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(log_level_str)

    if args.logfilename:  # write log to file
        file_output_handler = logging.FileHandler(args.logfilename, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        file_output_handler.setFormatter(formatter)
        logger.addHandler(file_output_handler)
        logger.info('Writing to Logfile')
        
    logging.info(f'Logging level:{logger.level}:{log_level_str}')
    
    logging.debug(f'Args: {args}')
    
    #TODO: take id_field from config file
    run_anonymize(args.in_path, args.out_path, args.models_dir, dictionary_paths, id_field=args.id_field)

    logging.info('Exiting...')
    sys.exit(0)  # Everything went ok!
    
    
if __name__ == '__main__':
    main()
