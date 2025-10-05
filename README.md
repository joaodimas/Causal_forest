# CATE Estimation with Causal Forest

Author: João Dimas (joaohdimas@gmail.com)

This notebook provides a practical comparison of several methods for estimating Conditional Average Treatment Effects (CATE) using simulated data.  

We explore how different models perform under various data-generating processes, from simple linear relationships to complex scenarios with unobserved heterogeneity. 

This is particularly relevant for applications like personalized marketing, dynamic pricing, or customized user interventions where understanding *"for whom"* an action is effective is critical.

---
## 1. Introduction to CATE Estimation

In many business and research settings, the goal is not just to measure the Average Treatment Effect (ATE) of an intervention, but to understand how this effect varies across individuals with different characteristics.  
This heterogeneous effect is the **Conditional Average Treatment Effect**, or **CATE**, denoted as $\tau(x)$.

Mathematically, it is defined as the expected difference in potential outcomes for an individual with covariates $X = x$:

$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
$$

where $Y(1)$ is the potential outcome if treated and $Y(0)$ is the potential outcome if not treated.  
Our goal is to use data to build a model that can accurately estimate $\tau(x)$ for any given individual.

---
## 2. CATE Estimation Methods

We will compare a set of models ranging from traditional econometric methods to modern machine learning estimators.

### 2.1. Linear Model (with Regularization)

A flexible linear model serves as a strong baseline. To capture non-linearities and interactions, we first pre-process the features.  
Continuous variables are binned, and high-cardinality categorical variables are one-hot encoded.  
We then fit a linear model that includes interaction terms between the treatment and these processed features.

The model is specified as:

$$
Y = \beta_0 + f(X)\beta_1 + T\delta + (T \cdot f(X))\gamma + \epsilon
$$

where $f(X)$ represents the feature transformation (binning and encoding).  
The CATE is then reconstructed from the fitted coefficients:

$$
\hat{\tau}(x) = \hat{\delta} + f(x)\hat{\gamma}
$$

We use `LassoCV` to handle the high dimensionality of $f(X)$ and perform automatic feature selection via L1 regularization.

### 2.2. K-Nearest Neighbors (KNN) Regression

K-Nearest Neighbors (KNN) regression is a classic non-parametric method that estimates outcomes locally.  
Instead of fitting a global model, it makes predictions for a new data point by looking at the $k$ most similar data points (neighbors) from the training set in the feature space.

We implement a T-Learner version of this approach:

1. Split the data into treated ($T = 1$) and control ($T = 0$) groups.  
2. Fit two separate KNN regression models to predict the outcome in each group:

   $$\hat{\mu}_1(x) = \text{Average}(Y_i \mid i \in k\text{-NN}(x), T_i = 1)
   \quad \text{and} \quad
   \hat{\mu}_0(x) = \text{Average}(Y_i \mid i \in k\text{-NN}(x), T_i = 0)$$

3. The CATE is the difference between the predictions of these two models:

   $$\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

The prediction for a point $x$ is typically a weighted average of the outcomes of its neighbors,  
where closer neighbors are given more weight (e.g., `weights='distance'`).  

### 2.3. Causal Forest

The Causal Forest, developed by Athey and Wager, is a non-parametric method based on an ensemble of "honest" trees (use one subsample to choose splits and a disjoint subsample to estimate leaf effects; this reduces bias and enables valid inference).
Causal forests (or Generalized Random Forests, GRF) grow honest trees on orthogonalized data and choose splits that maximize heterogeneity in treatment effects, yielding consistent CATE under overlap.

**Intuition for splitting:** First remove what $X$ explains from both outcome and treatment:
$$Y-\hat m(X),\quad T-\hat e(X).$$
Their scaled product
$$U=\frac{(Y-\hat m(X))(T-\hat e(X))}{\hat e(X)[1-\hat e(X)]}$$
acts like a local “effect signal.” Positive $U$: treated did better than expected given $X$. Negative $U$: treated did worse.

A good split groups similar signals together. If one child has mostly positive $U$ and the other mostly negative, heterogeneity is high. A simple teaching proxy is
$$S=\frac{n_L n_R}{n}\big(\bar U_L-\bar U_R\big)^2,$$
which is large when child means of $U$ are far apart and both children are sizable. Equal means give $S\approx0$. In practice GRF optimizes a closely related orthogonalized criterion rather than a literal regression tree on $U$.

**Setup:** Estimate nuisance functions $m(x)=\mathbb E[Y\mid X=x]$ and $e(x)=\mathbb P(T=1\mid X=x)$ (in an RCT with assignment rate $p$, set $e(x)=p$).

**Pseudo-outcome:** Using $U_i$ defined above,
$$\mathbb E[U_i\mid X_i=x]=\tau(x).$$

**Prediction:** For a query $x$, aggregate leaf estimates across trees to get $\hat\tau(x)$. Forest weights act as adaptive nearest neighbors.

**Assumptions:** SUTVA; overlap $0<e(X)<1$; randomization or unconfoundedness $Y(t)\perp T\mid X$. Consistency holds with accurate nuisance functions and honesty.

## 3. Evaluation Metrics

To evaluate our models, we use two complementary metrics:

- **CATE Mean Squared Error (MSE)**: This measures the overall accuracy of the CATE predictions against the known ground truth from the simulation.  
  It provides a single score for a model’s predictive power.

- **Uplift Decile Chart**: This is a business-oriented metric that evaluates a model’s ability to rank users from most to least responsive.  
  It directly answers the question: “If I target the top 10% of users as ranked by my model, do they actually have the highest treatment effect?”  
  A good model will show a strong upward trend, indicating that its ranking is meaningful.

---
## 4. Simulation Scenarios

We will test our models under three distinct data-generating processes (DGPs).

### Scenario 1: Linear Heterogeneity

This is the simplest case. The true CATE is a linear function of the observed features.  
This scenario should favor the linear models.

### Scenario 2: Non-linear Heterogeneity

This is a more realistic and challenging scenario. The true CATE is a complex function of the features,  
involving sharp conditional logic (`if/else`), non-linear transformations (`sin`), and deep interactions.  
This is the type of structure that tree-based, non-parametric models are designed to uncover.

### Scenario 3: Unobserved Heterogeneity (Boomerang Effect)

This is the most difficult scenario and a critical test of model diagnostics.  
The treatment effect is primarily driven by an unobserved variable (`user_latent_preference`).  
Furthermore, this hidden variable flips the sign of the effect for half the population, creating a **"boomerang effect"** where the treatment is beneficial for some and harmful for others.

In this case, the Average Treatment Effect (ATE) will be close to zero. Under randomization an A/B test gives an unbiased ATE; a near-zero ATE can still mask offsetting subgroup effects.
We expect all CATE models to fail in terms of predictive accuracy (high MSE),  
but the Uplift Decile Chart becomes a crucial diagnostic tool.  
A flat uplift chart will signal that the model has no ranking power, indicating that a key driver of heterogeneity is missing from the data.

---

## 5. Code

[View Jupyter Notebook](CATE_with_causal_forest.ipynb)

---

## 6. Results

### Scenario 1: Linear Heterogeneity
    
![png](CATE_with_causal_forest_files/CATE_with_causal_forest_13_0.png)

**Comments:** 

Under linear heterogeneity the signal is smooth and mostly additive, so all three learners recover it: the causal forest approximates the line with many honest splits, the penalized linear model fits it directly (with some bias from binning/L1), and KNN averages locally. That’s why CATE MSE is low for all, with CF best, linear close, and KNN worst. More important for targeting, each model ranks users correctly (the uplift quartiles rise monotonically) so any of them would prioritize high-effect users well in this DGP. 

In short: linear heterogeneity $\implies$ broad model success.

---
### Scenario 2: Non-Linear Heterogeneity

![png](CATE_with_causal_forest_files/CATE_with_causal_forest_16_0.png)

**Comments:**

Non-linear heterogeneity favors flexible models. In this scenario, we have many non-linearities in CATE (age > 45 changes slope; geo between 200–600 adds a sine term; the high-engagement×high-geo rule adds a step) plus rectification at 0. The linear+binned Lasso is misspecified, so it saturates and underestimates extremes and shows the weakest quartile slope. KNN locally averages, so it tracks smooth parts but blurs across boundaries and bands; uplift quartiles are cleanly monotone. Causal forest captures thresholds and interactions via splits (age gate, geo band, engagement×geo rule), so it fits best and yields the steepest uplift curve. 

In short: under non-linear heterogeneity, CF > KNN > linear; all still rank reasonably, with CF strongest.

---
### Scenario 3: Unobserved Heterogeneity
    
![png](CATE_with_causal_forest_files/CATE_with_causal_forest_19_0.png)

**Comments:**

In this scenario the driver of treatment response is unobserved and flips the sign for half the users, so X contains almost no signal about $\tau(x)$. All three learners collapse toward zero predictions and MSE explodes because the true effects swing widely. The uplift quartiles hover near zero without a monotone pattern, indicating no ranking power; any apparent bumps are noise. An RCT would still estimate ATE unbiasedly, but that masks offsetting subgroup harms/benefits. 

In short: without features that proxy the latent preference, or a redesigned experiment to expose it, targeting is indistinguishable from random.


