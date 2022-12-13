#!/usr/bin/env python3.7

"""
Different tools for processing BAR data

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

import pandas as pd
import numpy as np

from datetime import date
from babel import Locale

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

sys.path.append('lib')


class DateChecker(object):
    def __init__(self,
                 locale='de',
                 year_span=(1850, 'today'),
                 sep=['.', ' '],
                 day_numbers=True,
                 month_numbers=True,
                 year_numbers_long=True,
                 year_numbers_short=True,
                 ydm=True,
                 mdy=True):
        
        '''
        Tool to check a string to see if it is a date
        
        Define what will be recognized:
        day_numbers :: numeric tokens from 1 to 31
        month_numbers :: numeric tokens from 1 to 12
        year_numbers_long :: numeric tokens with four digits in the defined time span (e.g. 1984)
        year_numbers_short :: numberic tokens with two digits in the defined time span (e.g. 84)
        ydm :: date tokens of the format YY.dd.mm
        mdy :: date tokens of the format mm.dd.YY
        '''
        
        self.locale = Locale(locale)
        self.months_short = [m for m in self.locale.months['format']['abbreviated'].values()]
        self.months_long = [m.rstrip('.') for m in self.locale.months['format']['wide'].values()]
        
        self.sep_list = sep
        
        if year_span[1] == 'today':
            self.top_year = date.today().year
        else:
            self.top_year = int(year_span[1])
        self.years = [str(i) for i in range(year_span[0], self.top_year+1)]
        self.years_short = [i[2:] for i in self.years]
        
        self.dmy_pattern, self.ydm_pattern, self.mdy_pattern, self.day_numbers_list, self.month_numbers_list = self._build_pattern()
        
        # to be recognized
        self.day_numbers = day_numbers
        self.month_numbers = month_numbers
        self.year_numbers_long = year_numbers_long
        self.year_numbers_short = year_numbers_short
        self.ydm = ydm
        self.mdy = mdy
        
        
    def _build_pattern(self):

        month_pattern_str = '|'.join(self.months_long + self.months_short + [i.rstrip('.') for i in self.months_short])
        month_pattern_num1 = [str(i) for i in range(1, 13)]
        month_pattern_num2 = [f'0{i}' if len(str(i)) == 1 else f'{i}' for i in range(1, 13)]
        month_numbers = month_pattern_num1 + month_pattern_num2
        month_pattern_num = '|'.join(month_numbers)
        month_pattern = '|'.join([month_pattern_num, month_pattern_str])
        
        year_pattern1 = '|'.join(self.years)
        year_pattern2 = '|'.join(self.years_short)
        year_pattern = '|'.join([year_pattern1, year_pattern2])
        
        day_pattern1 = [str(i) for i in range(1, 32)]
        day_pattern2 = [f'0{i}' if len(str(i)) == 1 else f'{i}' for i in range(1, 32)]
        day_numbers = day_pattern1 + day_pattern2
        day_pattern = '|'.join(day_numbers)
        
        separators = [re.escape(i) for i in self.sep_list]
        sep_pattern = '|'.join(separators)
        
        dmy_pattern = re.compile(f'({day_pattern})({sep_pattern}) ?({month_pattern})({sep_pattern})? ?({year_pattern})?')
        ydm_pattern = re.compile(f'({year_pattern})?({sep_pattern}) ?({day_pattern})({sep_pattern}) ?({month_pattern})')
        mdy_pattern = re.compile(f'({month_pattern})({sep_pattern}) ?({day_pattern})({sep_pattern})? ?({year_pattern})?')
        
        return(dmy_pattern, ydm_pattern, mdy_pattern, day_numbers, month_numbers)
        

    def check_date(self, token):
        token = str(token).rstrip(string.punctuation)
        if self.year_numbers_long is True and token in self.years:
            return True
        if token in self.years_short and self.year_numbers_short is True:
            return True
        if bool(self.dmy_pattern.match(token)) is True:
            return True
        if self.ydm is True and bool(self.ydm_pattern.match(token)) is True:
            return True
        if self.mdy is True and bool(self.mdy_pattern.match(token)) is True:
            return True
        if self.day_numbers is True and token in self.day_numbers_list:
            return True
        if self.month_numbers is True and token in self.month_numbers_list:
            return True
        if token in self.months_long:
            return True
        if token in self.months_short:
            return True
        else:
            return False


class AHVChecker(object):
    def __init__(self):
        
        '''
        Tool to check a string to see if it is an AHV Number
        '''
         
        self.patterns = self._build_patterns()
        
        
    def _build_patterns(self):
        pattern_dict = dict()
        pattern_dict['pattern1'] = re.compile('[0-9]{3}\.[0-9]{2}\.[0-9]{3}\.?([0-9]{3})?') # 123.12.123 or 123.12.123.123
        #pattern_dict['pattern2'] = re.compile('[0-9]{2,4}\-[0-9]{3,4}(\-[0-9]{1,4})') # 123-123-12 123-1234 
        #pattern_dict['pattern3'] = re.compile('\([0-9]{2,4}((:|\.)[0-9])?\)\s?[0-9]{2,3}(\.[0-9])?\/[0-9]{2,3}')
        #(123:0) 123/12 - (123)123.0/123 - (1234:0)123/123 - (1234:0) 12/123 - (123.0)123.1/123
        #pattern_dict['pattern4'] = re.compile('(A|N){1}\s[0-9\s]{3,6}')
        pattern_dict['pattern5'] = re.compile('ordipro nr.\s?([0-9]+)', re.I)
        return pattern_dict

    def is_ahv(self, token):
        token = str(token).rstrip(string.punctuation)
        
        for pn, p in self.patterns.items():
            if bool(p.match(token)) is True:
                return True
        return False
    
    def has_ahv(self, titel):
        for pn, p in self.patterns.items():
            if len(re.findall(p, titel)) != 0:
                return True
        return False
        
    def return_ahv_offs(self, titel):
        titel_matches = list()
        for pn, p in self.patterns.items():
            ahv_matches = re.finditer(p, titel)
            
            if pn == 'pattern5':
                matches_list = [(m.group(1), m.span(1)) for m in ahv_matches]
            else:
                matches_list = [(m.group(), m.span()) for m in ahv_matches]
                
            if len(matches_list) > 0:
                titel_matches.append(matches_list)
        return [i for sublist in titel_matches for i in sublist]
    
    def convert_match(self, match_list):
        match_dict_list = [{'start': v[0], 'end': v[1], 'label': 'a'} for k, v in match_list]
        return match_dict_list
              
    
def get_doc_labels(doc_id, df, id_r='doc_id', label_c='annotation-BIO'):  #TODO: is this still used?
    doc_df = df[df[id_r] == doc_id]
    doc_df = doc_df.fillna('O')
    doc_labels = doc_df[label_c].tolist()
    return doc_labels


def plot_confusion_matrix(true_labels, pred_labels, labels):
    cm = sklearn.metrics.confusion_matrix(true_labels, pred_labels, labels=labels)

    sns.diverging_palette(150, 275, s=80, l=55, n=9)

    ax = plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap="YlGnBu", norm=LogNorm()); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)


def censoring_labels(x):
    if x not in ['p', 'O']:
        return 1  # to be censored
    if x == 'O':
        return 0  # to be kept uncensored
    else:
        return 2  # punctuation


def prob_diff_sec(prob_matrix, average=False):
    '''Calculates the difference between the first and second most likely label as vector or as average score.'''
    # get indices of second largest probability scores
    sec_l_idx = np.argsort(prob_matrix)[:, -2]
    # get scores of second largest probability scores
    sec_l = prob_matrix[np.arange(len(prob_matrix)), sec_l_idx]
    # get scores of first largest probability scores
    first_l = prob_matrix.max(axis=1)
    diff_sec = first_l - sec_l
    if average is False:
        return diff_sec
    else:
        return diff_sec.mean()
        
      

class Tag(object):
    def __init__(self, start=-1, end=-1, string='', name=''):
        self.start = start  # start character in text
        self.end = end  # end character in text
        self.name = name  # tag name
        self.c = self.start  # initialization counter
        self.string = string
        
          

class OffsetParser(object):
    '''OffsetParser parses annotations in html format and obtains offsets for each annotation as needed by other annotation formats (e.g. Spacy)'''
    def __init__(self):
        self.all_c = -1  # overall offset counter
        self.opening_tag = False  # track if inside of opening html tag
        self.closing_tag = False  # track if inside of closing html tag
        
        self.tag_open = False  # track if a tag is open
        self.html_string = ''  # string: html tags included
        self.text_string = ''  # string: text only
        
        self.tags = list()
        
    def get_offsets(self, in_string):
        '''Parse in_string and record offsets of annotations.'''
        self.html_string = in_string
        
        for ch in in_string:  # go through all characters
            
            if ch == '<' and self.tag_open is False:
                # opening tag
                self.tag_open = True
                self.opening_tag = True
                tag = Tag(start=self.all_c+1)
                continue
                
            if ch == '>' and self.tag_open is True:
                self.opening_tag = False
                continue
                
            if ch == '<' and self.tag_open is True:
                # first character of closing tag
                self.tag_open = False
                self.closing_tag = True
                tag.end = self.all_c + 1
                continue
                
            if ch == '/' and self.closing_tag is True:
                continue
                
            if ch == '>' and self.closing_tag is True:
                self.tags.append(tag)
                self.closing_tag = False
                continue
                
            if self.opening_tag is True:
                # read tag name
                if tag.name is None:
                    tag.name = ch
                else:
                    tag.name = tag.name + ch
                continue
                
            if self.closing_tag is True:
                # sanity check: is tag name (first letter) in closing tag the same? (only with one letter tag names)
                if len(tag.name) == 1:
                    try:
                        assert ch == tag.name
                    except Exception as e:
                        logging.debug(tag.name)
                        logging.debug(ch)
                        logging.debug(self.html_string)
                        raise e                    
                    
            else:
                self.all_c += 1
                self.text_string += ch
                
                if self.tag_open is True:
                    tag.string += ch

class Annotation(object):
    '''Object representing a Manual Annotation'''
    def __init__(self, start=-1, end=-1, label=None):
        self.start = start  # character-level start offset
        self.end = end  # character-level end offset
        self.label = label # label of the current annotation
        self.char_range = list(range(self.start, self.end))  # character range for the current annotation
        self.token_start = int() # token-level start offset
        self.token_end = int() # token-level end offset
        
    def set_token_offset(self, token_offset_dict):
        '''Adds token offset (start and end) values to the annotation as obtained by SpaCy tokenization.'''
        
        spanning_tokens = {k:v for k,v in token_offset_dict.items() if k in self.char_range}
        spanning_tokens_offsets = sorted(spanning_tokens.values())

        try:
        
            setattr(self, 'token_start', spanning_tokens_offsets[0])  # start token
            setattr(self, 'token_end', spanning_tokens_offsets[-1]+1)  # end token
        except IndexError as e:
            logging.debug('Spanning Tokens Error.')
            raise e
        
        return self
        