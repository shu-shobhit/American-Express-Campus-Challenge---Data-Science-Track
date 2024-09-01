# American-Express-Campus-Challenge---Data-Science-Track
Predict outcomes of T20 cricket matches

# AmEx T20 Match Prediction Challenge

## Table of Contents
1. [Introduction](#introduction)
2. [Challenge Overview](#challenge-overview)
3. [Data Description](#data-description)
4. [Evaluation Criteria](#evaluation-criteria)
5. [Our Approach](#our-approach)
   - [Objective Function](#objective-function)
   - [Sampling](#sampling)
   - [Feature Engineering](#feature-engineering)
   - [Modeling Technique](#modeling-technique)
   - [Model Performance](#model-performance)
6. [Observations & Inferences](#observations--inferences)
7. [Feature Engineering & Selection](#feature-engineering--selection)
8. [Possible Improvements](#possible-improvements)

## Introduction

This repository contains our team's solution for the American Express T20 Match Prediction Challenge, part of the 2024 Amex Campus Super Bowl Competition. Our team, Trigonal Bipyramidal, consists of Shobhit Kumar Shukla, Aarya Joshi, and Sayan Das from the Indian Institute of Technology, Kharagpur.

## Challenge Overview

The challenge involves building the best Machine Learning model using boosting algorithms to accurately predict the winning team for a T20 cricket match. The competition is structured in multiple rounds:

1. **Round 1**: Initial model submission and evaluation
2. **Round 2**: Model re-evaluation on new data (no retraining allowed)
3. **Final Round**: Presentation of approach and Q&A session

**We successfully advanced to the pre-final rounds, and was among the top 10% out of 423 participating teams**

## Data Description

The challenge provides several datasets:

1. **Train Data**: Primary dataset with game-level information (948 rows, 23 columns)
2. **Match Level Data**: Additional dataset with all games' scorecards (1689 rows, 30 columns)
3. **Batsman Level Data**: Detailed batsman scorecard (24483 rows, 21 columns)
4. **Bowler Level Data**: Detailed bowler scorecard (18539 rows, 18 columns)
5. **Round 1 Submission Data**: Data for Round 1 predictions (271 rows, 21 columns)
6. **Round 2 Submission Data**: Data for Round 2 predictions (details to be provided later)

The datasets cover T20 games from the past two years, across domestic and international tournaments.

## Evaluation Criteria

The primary evaluation metric is Accuracy, calculated as:

```
Accuracy = Number of correct predictions / Total number of games to predict
```

## Our Approach

### Objective Function

We chose the binary cross-entropy loss as our objective function. This is standard for classification tasks and helps the model learn to distinguish between winning and losing teams by minimizing the difference between predicted and actual outcomes.

### Sampling

To improve the generalizability of our model, we employed Stratified Sampling. This technique ensures that each subset of the dataset accurately represents the distribution of key variables, maintaining the statistical properties of the original data throughout the sampling process.

### Feature Engineering

Feature engineering was a crucial part of our approach. We created a total of 58 features, of which 10 were selected for the final model. Our feature engineering process was thorough and involved several stages:

1. **Basic Statistical Features**:
   - Average of Total Fours scored by the team in the last 10 matches
   - Average of Total Sixes scored by the team in the last 10 matches
   - Average of total wickets taken by the team in the last 10 matches
   - Total Bowling Strike Rate of the team in the last 10 matches

2. **Advanced Performance Metrics**:
   We developed more complex features based on cricket analytics research:
   - Team Adjusted Combined Bowling Performance (ACBR): This feature quantifies a team's bowling performance by taking into account traditional bowling statistics such as economy rate, bowling average, and strike rate. Lower ACBR values indicate better bowling performance.
   - Team Batting Performance (BP): This evaluates a team's batting effectiveness by considering the runs scored, the strike rate, and overall match strike rate. It adjusts individual performance by considering the match conditions.
   - Team Strength: A composite measure that combines a team's BP and ACBR to provide an overall performance metric.

3. **Form-based Features**:
   - Team's Current Form (CF): This feature quantifies the recent performance of a team based on the outcomes of its last five matches. Higher CF values indicate better current form.
   - Team's Past Margins of Victory: Normalized values of the difference between the number of runs or the number of wickets between winning & losing team for a team in the last n games of victory.

4. **Player-specific Features**:
   - Team's Bat Score: Following ICC guidelines, we calculated average batsmen scores for each team over the last 10 matches, based on runs, sixes, fours, strike rate, and ducks.
   - Team's Bowling Score: A measure of bowling performance based on wickets, maiden overs, runs conceded, economy, and runs per over following ICC guidelines, for past n matches.

5. **Venue-specific Features**:
   - We calculated several "on ground" features, such as Combined Average Strike Rate of Team on that ground, to capture a team's performance on a match's specific ground from past data.

6. **Interaction Features**:
   - We created features that capture the interaction between teams, such as Team1's win percentage against Team2 in the last 15 games.

7. **Time-based Features**:
   - We incorporated temporal aspects by creating features that look at performance over different time windows (e.g., last 5 games, last 10 games, last 15 games).

Our feature engineering process was iterative. We started with basic features and progressively added more complex ones, always validating their impact on model performance. We also paid close attention to potential data leakage, ensuring that our features only used information that would have been available before each match.

#### Feature Selection

To select the most impactful features, we employed several methods:

1. Recursive Feature Elimination
2. Analysis of Variance (ANOVA)
3. Mutual Information Method
4. Chi-Square Method

We also created intuitive feature sets capturing bowling, batting, and overall team performance. The final set of features was chosen based on a combination of statistical importance and domain knowledge.

### Modeling Technique

We experimented with several boosting algorithms:

1. Gradient Boosting Machine (GBM)
2. LightGBM
3. XGBoost
4. CatBoost

After extensive testing, we selected GBM (Gradient Boosting Classifier) as our final model due to its superior performance on the R1 test data.

### Model Performance

Here's a summary of our model performance:

| Technique          | Mean CV Scores | Test scores | Training Set scores |
|--------------------|----------------|-------------|---------------------|
| CatBoost (Set1)    | 0.6128         | 0.5876      | 0.87                |
| CatBoost (Set2)    | 0.5917         | 0.5901      | 0.75                |
| GBM                | 0.5412         | 0.6273      | 0.635               |

## Observations & Inferences

1. The CatBoost Classifier with set-1 performed poorly on the Test set despite high mean CV scores. This suggests that the model may not have generalized well due to the non-representative nature of the training data.

2. The complexity of the features in set-1 led to overfitting on the training data (train data score = 0.87).

3. GBM did not perform as well on Cross Validation but still performed best on the test set. This is likely due to less complex features that do not overfit the training data.

### Other Techniques

In addition to our primary modeling approach, we experimented with several advanced techniques to potentially improve our model's performance:

#### Principal Component Analysis (PCA)

We applied PCA as a dimensionality reduction technique to our feature set. Our process was as follows:

1. We reduced the dimensionality to a total of 5 dimensions, which represented 95% of the data variance.
2. However, we observed that this approach led to underfitting of the model.
3. While PCA didn't improve our final model, it provided valuable insights into the feature space and potential redundancies in our engineered features.

#### Meta-Modeling Approach

We also explored a meta-modeling technique to create new, potentially more predictive features:

1. We used a CatBoost Regressor as a meta-model to create five new features from the dataset.
2. These new meta-features were then used as inputs into various classifier models.
3. However, we found that the models using these meta-features exhibited overfitting on the training data.
4. Despite not being included in our final solution, this experiment provided insights into complex feature interactions and the potential risks of overly sophisticated feature engineering.

These advanced techniques, while not part of our final solution, demonstrate our team's commitment to exploring innovative approaches to improve model performance. They also highlight the importance of balancing model complexity with generalizability, especially given the constraints of the available training data.


## Possible Improvements

1. Acquiring a larger past dataset to create better features, as many times past matches data were insufficient or missing.
2. Obtaining match data at every over rather than just innings-level data, as well as real-time match data.
3. Experimenting with player ranking features based on overall performance.
4. Incorporating partnership data of batsmen.
5. Further experiments with Dimensionality Reduction techniques:
   - Explore alternative PCA implementations or other techniques like t-SNE or UMAP.
   - Investigate the optimal number of components to balance information retention and model simplicity.

6. Refinement of the Meta-Modeling Approach:
   - Experiment with different algorithms for the meta-model.
   - Implement regularization techniques to prevent overfitting in the meta-model.
   - Explore feature selection methods specifically for meta-features.

7. More experiments with Meta Model Approaches.
