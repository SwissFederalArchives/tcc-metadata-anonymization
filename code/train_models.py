#!/usr/bin/env python

"""
Script for training machine learning models for token-level anonymization of BAR titles.

"""

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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import sklearn_crfsuite

sys.path.append('lib')
from bar_preprocessors import BARTokenizer, FeatureExtractor, FeatureCollector, BIOGenerator, entity_spans
from bar_tools import Annotation, AHVChecker, DateChecker, OffsetParser

logging.info('Starting...')


def get_annotations(in_string):
    '''Get annotations as dictionary (needed by displacy)'''
    offset_parser = OffsetParser()
    offset_parser.get_offsets(in_string)
    
    annotations_dict = {'text':offset_parser.text_string,
                'ents':[{'start':tag.start, 'end':tag.end, 'label':tag.name}
              for tag in offset_parser.tags]}
    
    return annotations_dict  


class TrainPreprocessor(object):
    '''Pre-processing and Feature Extraction.'''
    def __init__(self, dictionary_paths, anno_field='annotation', context_field='context', id_field='VE_ID', context_window=4):
        self.anno_field = anno_field
        self.context_field = context_field
        self.id_field = id_field
        self.context_window = context_window

        self.tokenizer = BARTokenizer()
        self.bio_generator = BIOGenerator()
        self.feature_collector = FeatureCollector(self.tokenizer, dictionary_paths, context_window=self.context_window)

    def get_bio(self, in_df, feat_df):
        '''
        anno_field: column header of annotated text field; contains annotations in the format of html tags.
        id_field: column header of id column
        '''
        # normalize non-breaking whitespaces (e.g. '\xa0')
        titel_fixed = in_df[self.anno_field].apply(lambda x: unicodedata.normalize("NFKD", x))


        annotations = titel_fixed.apply(lambda x: get_annotations(x))
        annotations_list = list(annotations)

        spacy_docs = annotations.apply(lambda x: entity_spans(x, self.tokenizer.nlp))

        id_df = pd.concat([spacy_docs, in_df[self.id_field].apply(lambda x: str(x))], axis=1)
        id_df.apply(lambda x: self.bio_generator.add_entry(x[0], identifier=x[1], identifier_name=self.id_field), axis=1)

        # token-level table with BIO annotations
        bio_df = self.bio_generator.df

        # add 0 and p labels
        bio_df['annotation'] = bio_df['annotation'].apply(lambda x: 'O' if x == '' else x)
        labels_puncts = pd.concat([bio_df['annotation'].reset_index(drop=True), feat_df['is_punct'].reset_index(drop=True)], axis=1)
        recoded_labels = labels_puncts.apply(lambda x: 'p' if str(x[1]) == 'True' and x[0] == 'O' else x[0], axis=1)
        recoded_labels = recoded_labels[~recoded_labels.apply(lambda x: str(x).lower()=='nan')]

        bio_df['annotation'] = recoded_labels.tolist()
        return bio_df       


    def build_feature_overview(self, in_df):
        annotations = in_df[self.anno_field].apply(lambda x: get_annotations(x))
        annotations_list = list(annotations)

        anno_df = pd.DataFrame(annotations_list)
        anno_df[self.id_field] = in_df[self.id_field]
        anno_df[self.context_field] = in_df[self.context_field]
        anno_df['annotations'] = annotations_list

        feat_df = self.feature_collector.add_features(anno_df, anon_field='text', context_field=self.context_field, id_field=self.id_field)

        # get manual annotations in BIO schema
        bio_df = self.get_bio(in_df, feat_df)
        tok_anno = bio_df[[self.id_field, 'annotation']].copy() # token-level annotations
        doc_anno = bio_df.groupby(self.id_field, sort=False).agg(list)['annotation'].reset_index()  # document-level annotations

        feature_extractor = FeatureExtractor(feat_df, context_window=self.context_window)

        # preparing document-level feature overview
        doc_ids = pd.Series(in_df[self.id_field].unique())
        doc_features = doc_ids.apply(lambda x: feature_extractor.doc2features(x, id_r=self.id_field))  # series of dictionaries
        doc_feat_df = pd.concat((doc_features, doc_ids), axis=1)
        doc_feat_df.columns = ['doc_features', self.id_field]

        # convert to token level feature overview
        tok_feat_df = doc_feat_df.explode('doc_features')
        tok_feat_df = tok_feat_df.reset_index(drop=True).reset_index()
        tok_feat_df.columns = ['td-idx', 'tok_features', self.id_field]  # add token indices/offsets for tokens in entry
        tok_feat_df = tok_feat_df[[self.id_field, 'td-idx', 'tok_features']]

        tok_anno = tok_anno.reset_index(drop=True)
        tok_anno['td-idx'] = tok_feat_df['td-idx']  # add token indices/offsets to token annotations
        tok_anno = tok_anno[[self.id_field, 'td-idx', 'annotation']]

        # setting id field as index on document feature overview and document-level annotations
        doc_feat_df = doc_feat_df[[self.id_field, 'doc_features']]
        doc_feat_df[self.id_field] = doc_feat_df[self.id_field].astype('str')
        #doc_feat_df = doc_feat_df.set_index(self.id_field)
        #doc_anno = doc_anno.set_index(self.id_field)
        doc_feat_df = doc_feat_df.reset_index(drop=True)
        doc_anno = doc_anno.reset_index(drop=True)

        return ((tok_feat_df, tok_anno), (doc_feat_df, doc_anno))


def train_crf(doc_feat_df, doc_anno, c1=0.03436075827766486, c2=1.9135506711641375e-05):

    X = doc_feat_df['doc_features']
    y = doc_anno['annotation']

    logging.debug(f'Training data shape (document level): {X.shape}')
    logging.info('Training CRF classifier...')

    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=c1,
    c2=c2,
    max_iterations=100,
    all_possible_transitions=True)

    length_diffs = X[(y.apply(lambda x: len(x)))!=(X.apply(lambda x: len(x)))].apply(lambda x: [y['lemma'] for y in x]),  y[(y.apply(lambda x: len(x)))!=(X.apply(lambda x: len(x)))]

    crf.fit(X, y)
    logging.info(f'CRF model: training completed.')
    return crf

def train_mlp_svm(tok_feat_df, tok_anno, label_set={'n', 'j', 'w', 'wd', 'a', 'd', 'O', 'p'}):

    le = {label:enc for enc, label in enumerate(sorted([str(i) for i in list(label_set)]))}
    logging.debug(f'Encoding labels to the following numerical representation: {le}')
    
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(tok_feat_df['tok_features'])
    X = X.toarray()
    logging.debug(f'Training data shape (token level): {X.shape}')

    try:
        y = tok_anno['annotation'].apply(lambda x: le[x])
    except KeyError as e:
        logging.error(f'Unknown label found: {e}. Please fix training data and try again.')
        raise e

    logging.info('Training MLP Classifier...')
    mlp_clf = MLPClassifier()
    mlp_clf.fit(X, y)
    logging.info('MLP model: training completed.')

    logging.info('Training SVM Classifier...')
    base_clf_svm = LinearSVC(max_iter=10000)
    svm_clf = CalibratedClassifierCV(base_estimator=base_clf_svm, cv=3)
    svm_clf.fit(X, y)
    logging.info('SVM model: training completed')
    return (mlp_clf, svm_clf, vectorizer)

def train_classifiers(doc_feat_df, doc_anno, tok_feat_df, tok_anno, label_set={'n', 'j', 'w', 'wd', 'a', 'd', 'O', 'p'}):
    crf_model = train_crf(doc_feat_df, doc_anno)  # TODO: add c1, c2 arguments
    mlp_model, svm_model, vectorizer = train_mlp_svm(tok_feat_df, tok_anno)
    logging.info('Training of ensemble subsystems completed.')
    return (crf_model, mlp_model, svm_model, vectorizer)

def prepare_training(in_df, dictionary_paths, anno_field='annotation', context_field='context', id_field='doc_id'):
    in_df[anno_field] = in_df[anno_field].apply(lambda x: unicodedata.normalize("NFKD", x))
    logging.info(f'Number of lines in input data: {in_df.shape[0]}')
    train_preprocessor = TrainPreprocessor(dictionary_paths, anno_field=anno_field, context_field=context_field, id_field=id_field)
    logging.info('Building feature overview... (This may take some time)')
    ((tok_feat_df, tok_anno), (doc_feat_df, doc_anno)) = train_preprocessor.build_feature_overview(in_df)
    return ((tok_feat_df, tok_anno), (doc_feat_df, doc_anno))

def save_classifiers(models_dir, crf_model, mlp_model, svm_model, vectorizer):
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)
    joblib.dump(crf_model, os.path.join(models_dir, 'crf-model.joblib'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer.joblib'))
    joblib.dump(mlp_model, os.path.join(models_dir, 'mlp-model.joblib'))
    joblib.dump(svm_model, os.path.join(models_dir, 'svm-model.joblib'))
    logging.info('All models successfully saved.')

def run_training(in_path, models_dir, dictionary_paths, anno_field='annotation', context_field='context', id_field='doc_id'):
    label_set = {'n', 'j', 'w', 'wd', 'a', 'd', 'O', 'p'}
    in_df = pd.read_csv(in_path, sep='\t')
    ((tok_feat_df, tok_anno), (doc_feat_df, doc_anno)) = prepare_training(in_df, dictionary_paths, anno_field=anno_field, context_field=context_field, id_field=id_field)
    crf_model, mlp_model, svm_model, vectorizer = train_classifiers(doc_feat_df, doc_anno, tok_feat_df, tok_anno, label_set=label_set)
    save_classifiers(models_dir, crf_model, mlp_model, svm_model, vectorizer)

    
def main():
    """
    Invoke this module as a script
    """

    description = "An NLP based anonymizer for archival documents metadata. This script is for training the feature-based ensemble system. Use predict_labels.py for data anonymization following model training."
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('-l', '--logfile', dest='logfilename',
                        help='write log to FILE', metavar='FILE')
    argparser.add_argument('--in_path', metavar='in_path', type=str, required=True,
                        help='Path to tsv file containing annotations in the format of html tags.')
    argparser.add_argument('--models_dir', metavar='models_dir', type=str, required=True,
                        help='Path to directory where the machine learning models are to be stored.')
    argparser.add_argument('--config_file', metavar='config_file', type=str, required=True,
                        help='Path to settings file containing default settings.')

    
    args = argparser.parse_args()

    print(args)

    config = configparser.ConfigParser()
    config.read(args.config_file)
    config_args = config['config_args']
    
    # dictionary of dictionary filenames (for lookup features)
    dictionary_paths = {k:v for k,v in dict(config.items('dictionary_paths')).items() if k.endswith('path')}
    print(dictionary_paths)

    # add arguments from config.ini  ## TODO: make these also work as command line arguments
    argparser.add_argument('--anno_field', default=config_args['anno_field'])
    argparser.add_argument('--context_field', default=config_args['context_field'])
    argparser.add_argument('--id_field', default=config_args['id_field'])
    argparser.add_argument('--context_window', default=config_args['context_window'])
    argparser.add_argument('--dictionary_paths', default=dictionary_paths)

    log_level_str = config['logger_args']['logger_level']

    argparser.add_argument('--logger_level', default=log_level_str, dest='log_level_str')

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
    
    run_training(args.in_path, args.models_dir, args.dictionary_paths, anno_field=args.anno_field, context_field=args.context_field, id_field=args.id_field)

    logging.info('Exiting...')
    sys.exit(0)  # Everything went ok!
    
    
if __name__ == '__main__':
    main()
