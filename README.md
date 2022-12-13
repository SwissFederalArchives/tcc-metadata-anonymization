# tcc-metadata-anonymization

An NLP based anonymizer for archival documents metadata.

This repository contains code written for the redaction of sensitive strings from the entries of the Swiss Federal Archive (Schweizerisches Bundesarchiv - BAR).

For purposes of illustration synthetic example data (in German) is included. 

## Annotations

The synthetic example data contains the following set of (manually applied) labels in the format of html markup:

```
• <n> natural person: usually a first or family name
• <j> legal entity: companies, associations; not explicitly including product names
• <d> birth date: full day, month, year
• <a> social security number, military registration number or other identifier, such as the foreigners register (Ordipro) maintained by the Federal Department of Foreign Affairs.
```

## Task
The task is to recognize and blacken all strings of text that correspond to the redactable labels ```<n>```, ```<j>```, ```<d>``` and ```<a>``` in unseen data.

**Example:**

**Input:**   « Strafverfahren gegen *Muster*, *Max*, *14.10.1967* wegen Betrug »

**Output:** « Strafverfahren gegen ███, ███, ███ wegen Betrug »


## Method
The system combines three different types of classifiers into a feature-based ensemble system: Conditional Random Fields (CRFs), Support Vector Machines (SVMs) and Multilayer Perceptrons (MLP) based on their implementation in [Scikit-learn]().

Optional relabeling according to a defined set of rules is applied within a post-processing step to reach a final prediction. 

### Overview of features

We distinguish between token-based and context-based types of features. 

**Token-based features:**

• The token itself (lower-cased)

• The lemma (base form) of the token

• The part-of-speech (pos) tag of the token

• The shape of the token (e.g., Xxxx oder dd.dd.dd)

• If the token is alphabetic/numeric/a stopword/a punctuation character

• Character 2-grams of the token

• If the token looks like a date according to regular expressions

• If the token looks like an identifier (label `<a>`) according to regular expressions

• If the token is found in the dictionary of personnames

• If the token is found in the dictionary of swiss place names


**Context-based features:**

Features within a context window of n (default: 4) preceding and subsequent tokens is taken into account.

• The (lower-cased) context tokens

• The part-of-speech (pos-tags) of the context tokens

• If the context tokens are punctuation characters

• The shape of the context tokens 

*We use tokens, lemmas, pos-tags, token shape, information regarding token nature (alphabetic/numeric/a stopword/a punctuation character) as provided by [SpaCy](https://spacy.io/)s tokenization (adapted [de\_core\_news\_lg](https://spacy.io/models/de) pipeline).*


## Code

### Installation

Installation of the necessary environment:
 
```shell
conda create --name tcc-bar-anonym
conda activate tcc-bar-anonym
conda install python=3.10
conda install ipykernel
conda install joblib
conda install pandas
conda install -c conda-forge scikit-learn
conda install spacy
python -m spacy download de_core_news_lg
conda install babel
conda install seaborn
conda install python-crfsuite
pip install sklearn-crfsuite
```

### Set up
Use the configuration file (config.ini) to set up the following paths and parameters:


### Usage

The code can be used from the command line or by importing it as a package (see included jupyter notebook [cross_validate.ipynb]()).

#### Command line

Call from within the code directory.

##### Training
```shell
python3 train_models.py --in_path ../data_syn/annotated_data_example.tsv --models_dir ../models_syn_new --config_file ../config_syn.ini
```
##### Prediction
```shell
python3 predict_labels.py --in_path ../data_syn/annotated_data_example.tsv --out_path ../data_syn/output_anon.tsv --models_dir ../models_syn_new --config_file ../config_syn.ini
```

## Contact

If you have questions or comments, please do not hesitate to contact us under the following address:

**tcc@cl.uzh.ch**



