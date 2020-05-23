# w2v-pos-tagger

#### Single Token Part-of-Speech Tagging using Support Vector Machines and Word Embedding Features

## Content

1) [Introduction](#introduction)
2) [Setup](#2-setup)
3) [Documentation](#3--documentation)
   1) [Corpus Analysis and Normalization](#3.1-corpus-analysis-&-normalization)
   2) [Baselines](#3.2-baselines)
   3) [SVM-Tagger](#3.3-svm-tagger)

## Introduction

**w2v-pos-tagger** is a project submission to an NLP & IR class in *2017*. The task description
was as follows:

<img align="right" width="480" height="360"
     src="img/embedding_size_train_size__test_time__f1_micro.png" style="margin:10px"
     alt="embedding size comparison">

<p align="left">

 * Train a part-of-speech tagger for the German language.
 * Use the [TIGER corpus](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger/)
   as training set.
 * Use the [HDT corpus](https://corpora.uni-hamburg.de/hzsk/de/islandora/object/treebank:hdt)
   as test set.
 * Map the [STTS](http://www.sfs.uni-tuebingen.de/resources/stts-1999.pdf) annotations to the
   [Universal tagset](https://universaldependencies.org/u/pos/) and use the latter as labels.
 * Train a POS-tagger model on an
   [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
   using an RBF kernel.
 * Write Precision, Recall and F<sub>1</sub> score metrics.
 * Compare the results to the pre-trained tagger of [spaCy](https://spacy.io/) 1.x and an
   [NLTK](https://www.nltk.org/) based tagger trained with the
   [ClassifierBasedGermanTagger](https://github.com/ptnplanet/NLTK-Contributions).
 * Describe the results in a [course paper](paper/pos_paper_funke.pdf).

A few choices of the task requirements were questionable, for example the demand of an SVM
classifier and the mandatory usage for a nonlinear kernel, also the requirement for different 
train/test corpora and the size of the test set itself (full corpus). However, this project stays
within the bounds of these constrains.

Defining a design for the feature engineering was part of the challenge. I decided for unigram word
vectors as input features for the SVM and applied a comprehensive hyperparameter search for 
embedding types, vector sizes and SVM hyperparameters. This simple, yet effective, approach of 
learning a hyperplane separation through the (RBF-transformed) embedding space gave an 
F<sub>1</sub> score of 0.919 for the best models. The approach demonstrates how well static unigram
word vectors can represent syntactic language features. The project is described in:

* **A. Funke**: *Single Token Part-of-Speech Tagging using Support Vector Machines and
  Word Embedding Features*. Course paper for "Natural Language Processing and Information
  Retrieval" (HHU 2017)  
  [paper/pos_paper_funke.pdf](paper/pos_paper_funke.pdf)

  **Abstract:**  
  *Part-of-speech tagging (POS) is a common technique in Natural Language Processing pipelines.
  It enriches a corpus with grammatical information which can be exploited not only for syntactic
  but also for semantic evaluation. In this paper we propose an SVM based POS-tagger trained
  on the TIGER corpus using word embeddings of small dimensionality as input data. The feature-set
  is based only on the token itself without context of surrounding words. A maximum F<sub>1</sub>
  micro score of 0.919 when tested on the HDT corpus could be achieved.*

</P>

## 2 — Setup

At first, set up a conda environment:

```bash
conda env create -f environment.yml
conda activate w2vpos
```

For convenience install the project as a package:

```bash
python setup.py develop
```

Download the required part-of-speech annotated corpora:

```bash
# TIGER
echo "Please view and accept the license agreement for the TIGER corpus."
xdg-open https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/license/htmlicense.html
curl https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/download/tigercorpus-2.2.conll09.tar.gz | tar xvz -C corpora/tiger-conll

# HDT
curl -SL "https://corpora.uni-hamburg.de:8443/fedora/objects/file:hdt_hdt-conll/datastreams/hdt-conll-tar-xz/content?asOfDateTime=2016-02-17T15:38:47.643Z&download=true" | tar xvfJ - -C corpora
```

In order to train an NLTK based reference model we need one more dependency:

```bash
git clone git@github.com:ptnplanet/NLTK-Contributions.git lib/NLTK-Contributions
cp lib/NLTK-Contributions/ClassifierBasedGermanTagger/ClassifierBasedGermanTagger.py src/w2v-pos-tagger/
```


## 3 — Documentation


### 3.0 TL;DR

```bash
# describe the copora and their labels
w2vpos-corpora-analyser

# preprocess the corpora
w2vpos-data-loader

# train an NLTK baseline model
w2vpos-nltk-tiger-trainer

# tag the corpora with spaCy and NLTK
w2vpos-baseline-pos-tagger

# evaluate the reference inference
w2vpos-baseline-pos-tagger-evaluator

# learn distributed word vectors as features for our classifier
w2vpos-embedding-builder

# train a SVM classifier levearaging those features
w2vpos-svm-tagger-train
```



### 3.1 Corpus Analysis and Normalization

To analyse the tagset of both corpora, run

```bash
w2vpos-corpora-analyser
```

To normalize and fix a few issues in the corpora and to persist the results run

```bash
w2vpos-data-loader
```

This will cache the pre-processing as csv files in `corpora/out/`.

Additionally, the script will map STTS to the Universal Tagset according to
this mapping:
https://github.com/slavpetrov/universal-pos-tags/blob/master/de-tiger.map




### 3.2 Baselines
 
#### 3.2.1 Tagging with spaCy and NLTK

Before training our own SVM model we will evaluate the POS-tagging efficiency of common
NLP frameworks such as `spaCy` and `NLTK`. In order to provide a comparable NLTK based
tagger we will train it first.

```bash
w2vpos-nltk-tiger-trainer
```

The newly trained NLTK tagger will be saved to `corpora/out/nltk_german_classifier_data.pickle`.

To apply the spaCy and NLTK part-of-speech tagging run

```bash
w2vpos-baseline-pos-tagger
```

#### 3.2.2 Baseline Evaluation

```bash
w2vpos-baseline-pos-tagger-evaluator
```

will measure the performance of the newly tagged corpora against the ground truth
on several related metrics:

* accuracy
* precision (weighted)
* recall (weighted)
* F<sub>1</sub> measure (weighted)

The evaluation results are saved to `corpora/out/evaluation/`.

The script expects that `w2vpos-baseline-pos-tagger` has already been run.


### 3.3 SVM-Tagger

After these preparations we should have a good understanding of the dataset
and the performance of comparable approaches. We can now build our own SVM classifier.

#### 3.3.1 Learning Custom Word Vectors

Since we will be using word vectors as features for the SVM we will learn an embedding
space from the combined TIGER and HDT corpus by applying good old word2vec.

```bash
w2vpos-embedding-builder
```

will generate 16 (default) different word embeddings using the following hyperparmeter sets:

| Architecture | Case folding | Dimensionality |
|--------------|--------------|----------------|
| cbow         | none         |       12       |
| cbow         | none         |       25       |
| cbow         | none         |       50       |
| cbow         | none         |       100      |
| cbow         | lowercasing  |       12       |
| cbow         | lowercasing  |       25       |
| cbow         | lowercasing  |       50       |
| cbow         | lowercasing  |       100      |
| skip-gram    | none         |       12       |
| skip-gram    | none         |       25       |
| skip-gram    | none         |       50       |
| skip-gram    | none         |       100      |
| skip-gram    | lowercasing  |       12       |
| skip-gram    | lowercasing  |       25       |
| skip-gram    | lowercasing  |       50       |
| skip-gram    | lowercasing  |       100      |

The hyperparameters can be customized. For details:

```bash
w2vpos-embedding-builder --help
```

The embeddings are saved to `corpora/out/embeddings/`.

The original project trained the embedding for 5 epochs. You may want to increase
the number of iterations over the corpus for better performance:

```bash
w2vpos-embedding-builder --epochs 30
```

#### 3.3.2 Train a Support Vector Classifier

Finally, let's train our model.


```bash
w2vpos-svm-tagger-train
```

The hyperparameters of the SVM can be partly customized. You can chose one of the 
pretrained words embeddings and other training parameters. Start with the defaults
and then find additional information in the script's help.

```bash
w2vpos-svm-tagger-train --help

# optional arguments:
#  -h, --help            show this help message and exit
#  --verbose
#  --no-verbose
#  -a {cb,sg}, --architecture {cb,sg}
#  -d DIMENSIONALITY, --dimensionality DIMENSIONALITY
#  --lowercase
#  --no-lowercase
#  --train_size TRAIN_SIZE
#                        Train only on a slice of the trainset with length
#                        `train_size`.
#  --shrinking
#  --no-shrinking
#  --scale               Normalize the feature vectors.
#  --no-scale
#  --C C                 Soft-margin parameter.
#  --cache-size CACHE_SIZE
#                        Specify the size of the kernel cache (in MB).
#  --max-iter MAX_ITER   Limit the number of iterations.
#  --kernel {linear,poly,rbf}
```
