#!/usr/bin/env python3.7

"""
Different tools for pre-processing BAR data

"""

import sys
import os
import re

import spacy

from spacy.tokens import Span

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS, PUNCT
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.lang.de.punctuation import TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES, TOKENIZER_INFIXES, BASE_TOKENIZER_PREFIXES

import collections

import string

import logging

import pandas as pd
import numpy as np

from datetime import date
from babel import Locale

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

sys.path.append('lib')

from stopwords import stopwords
from bar_tools import Annotation, AHVChecker, DateChecker


def entity_spans(annotations_dict, nlp, mode='manual_annotations'):
    '''
    Generate a spacy doc with added (manual) entity annotations (in SpaCy format).
    Mode decides if matches which cannot be mapped to tokens should be logged as warnings.
    Only for 'manual_annotations' errors are logged but not for 'regex_matches' such as identification numbers. These are dropped without a warning.
    '''

    text = annotations_dict['text']
    doc = nlp(text)

    annotations = [Annotation(start=d['start'], end=d['end'], label=d['label']) for d in annotations_dict['ents']]
    
    token_offset_dict = {tok.idx:tok.i for tok in doc}  # maps character offset to token offset

    try:
        annotations = [a.set_token_offset(token_offset_dict) for a in annotations]

    except IndexError as e:  # match cannot be assigned to one spacy token / most likely corresponds to subtoken
        annotations = list()

        if mode == 'manual_annotations':
            logging.warning(f'A match in the following snipped cannot be assigned to full token and will therefore be skipped: {text}')
            logging.warning(f'Annotation Dict: {annotations_dict}')
    
    annotation_spans = [Span(doc, a.token_start, a.token_end, label=a.label) for a in annotations]
    
    try:
        doc.ents = annotation_spans  # deletes spacy entities from doc
    except ValueError as ve:
        logging.error(f'ValueError for entry: {text} . Possible solution: change regex in tokenizer to keep tokens together.')
        raise ve
    
    return doc



class BARTokenizer(object):
    '''Custom Tokenizer for processing database entries build on top of SpaCy.'''
    def __init__(self):
        logging.info('Tokenizer initiated')
        self.nlp = spacy.load('de_core_news_lg')
        self.prefixes = TOKENIZER_PREFIXES + [r'''^[\.\-]''']  #add full stop to prefixes

        # modify tokenizer suffix patterns
        self.suffixes = TOKENIZER_SUFFIXES  #  no modification needed

        # modify tokenizer infix patterns
        self.infixes = TOKENIZER_INFIXES + [
            r"(?<=[{a}0-9\.])(?:{h})(?=[{a}0-9])".format(a=ALPHA, h=HYPHENS),  # rule to split at hyphens bw. letters
            r"(?<=[{a}0-9\.\)])(?:[{p}\.\/\+])(?=[{a}0-9\,])".format(a=ALPHA, p=PUNCT),  # rule to split at all punct bw. letters
            r"(?<=[{a}])(?:\.)(?=[0-9])".format(a=ALPHA),  # rule to split at full stop between letter and digit
            ]

        self._update_rules()

    def _update_rules(self):
        # compile patterns and add to tokenizer
        prefix_re = spacy.util.compile_prefix_regex(self.prefixes)
        self.nlp.tokenizer.prefix_search = prefix_re.search

        suffix_re = spacy.util.compile_suffix_regex(self.suffixes)
        self.nlp.tokenizer.suffix_search = suffix_re.search

        infix_re = spacy.util.compile_infix_regex(self.infixes)
        self.nlp.tokenizer.infix_finditer = infix_re.finditer

    def spacy_tokenize(self, input):
        doc = self.nlp(input)
        return doc


class FeatureExtractor(object):
    '''Extracts features from text but does not build a feature matrix. A new FeatureExtractor is needed for each processed dataframe.'''
    def __init__(self, df, context_window=4):
        self.df = df.reset_index(drop=True)
        self.context_window = context_window
        
    def token_feature_dict(self, r):
        t_dict = r[['token', 'pos_fine',
           'lemma',
           'lower',
           'shape',
           'is_alpha',
           'is_digit',
           'is_punct',
           'is_left_punct',
           'is_right_punct',
           'like_num',
           'is_stop',
           'datecheck',
            'ahvcheck',
            'hierarchie',
           'in_personnames',
           'in_swissplaces',
           'in_register',
           'common_words'
                 ]].to_dict()
        
        # character n-gram features
        t_lower_l = t_dict['lower']

        t_dict['length'] = len(t_lower_l)

        if 'register_prob' in r:
            t_dict['register_prob'] = round(float(r['register_prob']), 3)
        
        n = 3
        t_ngrams = [''.join(ngram) for ngram in list(zip(*[t_lower_l[i:] for i in range(n)]))]
        c = collections.Counter(t_ngrams)
        ngram_feat = {f'c_ngram:{i}':1 for i in c.items()}
        
        n_shapes = [2,3]
        shape_ngrams = list()
        shape = t_dict['shape']
        for n_shape in n_shapes:
            ngram_list = [''.join(ngram) for ngram in list(zip(*[shape[i:] for i in range(n_shape)]))]
            shape_ngrams.extend(ngram_list)
        shape_ngram_feat = {f's_ngram:{i[0]}':True for i in shape_ngrams}
    
        hier_feat = {f'in_hier:{i}':True for i in r['hier_list'].split('|')}
        
        td = {**t_dict, **ngram_feat, **hier_feat, **shape_ngram_feat}

        return td
    
    def context_feature_dict(self, idx, doc_df):
        cfd = dict()

        for i in range(1, self.context_window+1):
            no_before = False
            no_after = False
            
            before_idx = idx-i
            after_idx = idx+i
            
            if before_idx in doc_df.index and no_before is False:
                r_before = doc_df.loc[before_idx]
                cfd[f'lower-{i}'] = r_before['lower']
                cfd[f'pos-{i}'] = r_before['pos_fine']
                cfd[f'is_punct-{i}'] = r_before['is_punct']
                cfd[f'shape-{i}'] = r_before['shape']
            else:
                cfd[f'lower-{i}'] = 'BOS' # start of sentence
                no_before = True
            
            if after_idx in doc_df.index and no_after is False:
                r_after = doc_df.loc[after_idx]
                
                cfd[f'lower+{i}'] = r_after['lower']
                cfd[f'pos+{i}'] = r_after['pos_fine']
                cfd[f'is_punct+{i}'] = r_after['is_punct']
                cfd[f'shape+{i}'] = r_after['shape']
            else:
                cfd[f'lower+{i}'] = 'EOS'  # end of sentence
                no_after = True
            
        return cfd        
            
    def tokenrow2features(self, idx, doc_df):
        r = self.df.loc[idx]
        tfd = self.token_feature_dict(r)
        cfd = self.context_feature_dict(idx, doc_df)
        
        fd = {**tfd, **cfd}

        # convert all feature values to strings 
        fd = {k:str(v) for k,v in fd.items()}
        
        return fd
    
    def doc2features(self, doc_id, id_r='IDENTIFIER'):
        doc_df = self.df[self.df[id_r] == doc_id]
        doc_idx = doc_df.index.to_series()
        doc_features = doc_idx.apply(lambda x: self.tokenrow2features(x, doc_df))
        return doc_features.tolist()
        


class BIOGenerator(object):
    '''Generate BIO (beginning-inside-outside) format for annotations.'''
    def __init__(self):
        self.df = pd.DataFrame(columns=['token','annotation','BIO', 'annotation-BIO'])
        
    def add_entry(self, spacy_doc, identifier=None, identifier_name='ID'):
        entity_frame = pd.DataFrame([(token, token.ent_type_, token.ent_iob_) for token in spacy_doc])
        entity_frame.columns = ['token','annotation','BIO']
        entity_frame['annotation-BIO'] = entity_frame[['annotation','BIO']].apply(lambda x: f'{x[0]}-{x[1]}' if x[0] else 'O', axis=1)
        if identifier:
            entity_frame[identifier_name] = identifier
        #self.df = self.df.append(entity_frame)
        self.df = pd.concat((self.df, entity_frame))


class FeatureCollector(object):
    def __init__(self, tokenizer, dictionary_paths, context_window=4):
        self.tokenizer = tokenizer
        self.dictionary_paths = dictionary_paths
        self.ahv_checker = AHVChecker()
        self.ahv_bio_generator = BIOGenerator()
        self.dchecker = DateChecker(
                     sep=['.', ' '],
                     day_numbers=False,
                     month_numbers=False,
                     year_numbers_long=True,
                     year_numbers_short=False,
                     ydm=True,
                     mdy=True)

        # Load common words file
        words_file = self.dictionary_paths['commonwords_path']
        common_df = pd.read_csv(words_file, sep='\t', header=None)
        self.common_words = common_df[0]

        ## Load personname list
        personname_file = self.dictionary_paths['personnames_path']
        person_df = pd.read_csv(personname_file, sep='\t')
        self.personnames = person_df.personname

        ## Load firm register list (with or without) token probabilities
        creg_file = self.dictionary_paths['firmnames_path']
        creg_df = pd.read_csv(creg_file, sep='\t', dtype='str')
        self.creg_list = list(creg_df['unigram_lower'])

        if 'register_prob' in creg_df:  # only if there is a pre-computed probability value in the company names dictionary
            self.creg_token_prob_dict = pd.Series(creg_df['register_prob'].values,index=creg_df.unigram).to_dict()

        ## Load placenames list
        placenames_file = self.dictionary_paths['placenames_path']
        self.placenames_df = pd.read_csv(placenames_file)

    def process_hierarchie(self, context_field_data, remove_title=True):
        '''
        Use regex to match words in the archive plan context/hierarchie
        Optionally remove the title from the context (assumes that the last part of the content is the same as the title)
        '''
        if remove_title is True:
            # processing: remove last part which corresponds to the title of the entry
            hier_entry_processed = ' '.join([i.rstrip('\r') for i in context_field_data.split('\n') if i][:-1])
        else:
            hier_entry_processed = ' '.join([i.rstrip('\r') for i in context_field_data.split('\n') if i])
        hier_words = re.findall("[^\W\d_]+", hier_entry_processed, re.UNICODE)
        return(list(set(hier_words)))


    def add_features(self, in_df, anon_field='titel_fixed', context_field='context', id_field='IDENTIFIER'):
        '''
        anon_field: text field to be anonymized
        '''

        text_field = anon_field
        
        # apply spacy preprocessing and tokenization using the bar tokenizer
        spacy_docs = in_df[text_field].apply(lambda x: self.tokenizer.nlp(x))

        # Build the feature overview
        id_df_s = pd.concat([in_df[id_field], spacy_docs], axis=1)
        id_df_s.columns = [id_field, 'spacy_docs']

        ## Add AHV Info
        in_df['matched_ahv'] = in_df[text_field].apply(lambda x: self.ahv_checker.return_ahv_offs(x))
        in_df['ahv_match_dict'] = in_df['matched_ahv'].apply(lambda x: self.ahv_checker.convert_match(x))

        # add hierarchie feature: first 5 characters
        id_df_s['hierarchie'] = in_df[context_field].apply(lambda x: x[:5])

        # add hierarchie feature: words in the hierarchie
        id_df_s['hier_list'] = in_df[context_field].apply(lambda x: '|'.join(self.process_hierarchie(x, remove_title=True)))
        id_df_s['hier_list'] = id_df_s['hier_list'].astype('str')
        
        id_df_s['spacy_features'] = id_df_s['spacy_docs'].apply(lambda x: [(t.text,
                                                                       t.ent_type_,
                                                                       t.ent_iob_,
                                                                       t.pos_,
                                                                       t.tag_,
                                                                       t.lemma_,
                                                                       t.norm_,
                                                                       t.lower_,
                                                                       t.shape_,
                                                                       t.is_alpha,
                                                                       t.is_digit,
                                                                       t.is_punct,
                                                                       t.is_left_punct,
                                                                       t.is_right_punct,
                                                                       t.like_num,
                                                                       t.is_stop) for t in x])
        
        ## EXPAND representation to token level                                                              
        id_df_s_expanded = id_df_s.explode('spacy_features').copy().reset_index(drop=True)
        logging.debug(id_df_s_expanded.columns)

        id_df_s_expanded[['token',
                      'ent_type',
                      'ent_iob',
                      'pos_coarse',
                      'pos_fine',
                      'lemma',
                      'norm',
                      'lower',
                      'shape',
                      'is_alpha',
                      'is_digit',
                      'is_punct',
                      'is_left_punct',
                      'is_right_punct',
                      'like_num',
                      'is_stop'
                     ]] = pd.DataFrame(id_df_s_expanded['spacy_features'].tolist(), index=id_df_s_expanded.index)

        id_df_s_expanded = id_df_s_expanded.drop(['spacy_docs', 'spacy_features'], axis=1)
        
        
        feat_df = id_df_s_expanded.copy().reset_index(drop=True)

        ahv_annotations = in_df[[text_field, 'ahv_match_dict']].apply(lambda x: {'text':x[0], 'ents':x[1]}, axis=1)
        spacy_docs_ahv = ahv_annotations.apply(lambda x: entity_spans(x, self.tokenizer.nlp, mode='regex_matches'))

        id_df_ahv = pd.concat([spacy_docs_ahv.reset_index(drop=True), in_df[id_field].astype('str').reset_index(drop=True)], axis=1)

        id_df_ahv.columns = ['annotation', id_field]
        id_df_ahv.apply(lambda x: self.ahv_bio_generator.add_entry(x[0], identifier=x[1], identifier_name=id_field), axis=1)
        ahv_bio_df = self.ahv_bio_generator.df.reset_index(drop=True)

        feat_df['ahvcheck'] = ahv_bio_df['annotation'].apply(lambda x: True if x == 'a' else False)
        
        # Use datechecker to check if the token looks like a date (or part of a date)
        feat_df['datecheck'] = feat_df.token.apply(lambda x: self.dchecker.check_date(x))
        
        ### Add dictionary features ###

        ## Common words feature
        feat_df['common_words'] = feat_df.lower.isin(self.common_words)
        
        # compare lower-cased version
        feat_df['in_personnames'] = feat_df.lower.isin(self.personnames)
        
        places = self.placenames_df['Ortschaftsname']
        places_split = places.apply(lambda x: x.split())

        # compare parts of placenames if they are not in the stopwords list
        stop_words = stopwords['german']
        places_nostops = places_split.apply(lambda x: [i for i in x if i not in stop_words])
        places_unique = pd.Series(places_nostops.explode().unique())
        places_unique_lower = places_unique.str.lower().str.strip(string.punctuation).unique()
        places_series = pd.Series(places_unique_lower)
        feat_df['in_swissplaces'] = feat_df.lower.isin(places_series)

        # Check if token is in registry
        feat_df['in_register'] = feat_df.token.str.lower().isin(self.creg_list)

        if 'register_prob' in feat_df:  # only if there is a probability value in the registry (company names) dictionary
            feat_df['register_prob'] = feat_df.token.apply(lambda x: self.creg_token_prob_dict.get(x,  0.0))

        feat_df = feat_df[[id_field] + [c for c in feat_df.columns if not c == id_field]]
        
        return feat_df

