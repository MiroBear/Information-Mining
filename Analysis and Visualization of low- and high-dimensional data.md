# Introduction, Purpose

This is a description of general methods, techniques and steps required
to analyse and visualize low- and high-dimensional data for classification
purposes. Described are numerous parametric classification methods
excluding deep learning.

Data onto which this can be applied are publicly available as described
in section data sources (e.g. Kaggle).

## Part 1 - Pre-processing, feature extraction and data visualization

- Extraction of features
- Alignment of data sets to common spaces
- Removal of artifacts
- Handling of missing data
- Visualization techniques

## Part 2 - Feature engineering

Feature selection: Pearson Correlation, ...

## Part 3 - Classification methods for low-dimensional data

- (multi-nomial / multi-class) Logistic Regression (LR)
- Elastic-net Logistic Regression (ENLR)
- SVM
- Linear discriminant analysis (LDA)
- Fisher discriminant analysis (FDA)

## Part 4 - Classification methods for high-dimensional data (p >> N)

High-dimensional features spaces (compared to number of samples)

## Part 5 - Combination of multiple data sources

- Combination in one or multiple classifiers
- Unify correlated and complementary information

- PCA
- LDA
- Tensor-approaches
- Generalizations of LDA (used also for face detection)
- Generalized Singular Value Decomposition (GSVD)
- Graph Embedding (GE)
  - Locality Preserving Projections (LPP)
  - Marginal Fisher Analysis (MFA)
  - Locally Linear Embedding (LLE)
  - Graph-based Fisher Analysis (GbFA)
  - Tensor-based Marginal Fisher Analysis (TMFA)
  - Correlational Tensor Analysis (CTA)

## Part 7 - General topics of classification

Use of features for the classification tasks, evaluation of different approaches to classification with estimation of their accuracy and robustness, ....

# Correspondences between geometric objects

Geometric objects subdivided into sub-parts.
Benefit: no common space required to which registration might be complicated.
Population specific correspondences.
Goals:
- Detect individual anatomical differences within a population.
- Serve as improved extraction of relevant features.

Method to find correspondences:
- Spectral matching which allows to match two or more geometric objects coming from a group.

Difficulties:
- Differences in the size and extend of sub-parts -> normalization required.
- Differences in the anatomy which lead to different divisions of the objects.
- Similarity measures to define a model which allows to compare patches with each other.

# Open data sets

- Kaggle:
  -
- ...

# Recommended Books

- The Elements of Statistical Learning. Data Mining, Inference, and Prediction. Trevor Hastie, Robert Tibshirani, Jerome Friedman. 2nd Edition, 2008.
- An Introduction to Statistical Learning with Applications in R. Gareth James, Daniela Witten,  Trevor Hastie, Robert Tibshirani. 2013.
- Pattern Recognition and Machine Learning, Bishop. 2006.
- Pattern Classification. Duda. 2000.
- Artificial Intelligence - A Modern Approach. Stuart Russell, Peter Norvig. 3rd Edition. 2016.
