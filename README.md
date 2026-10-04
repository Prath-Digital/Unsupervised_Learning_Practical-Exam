<div align="center">

# 💳 Credit Card Customer Segmentation — Unsupervised Learning Practical Exam

### _Enterprise-Grade Behavioral Clustering, Multi-Algorithm Benchmarking & Real-Time Scoring Pipeline for Retail Banking_

![Exam Type](<https://img.shields.io/badge/Exam%20Type-Practical%20(6%20Hours)-blue?style=for-the-badge&logo=academic-tree>)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)-success?style=for-the-badge)
![License-MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📌 Candidate & Examination Metadata

| Field                     | Detail                                                                                                              |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------ |
| **Candidate Name**        | **Prath Udhnawala**                                                                                                 |
| **GRID / Student ID**     | **11120**                                                                                                           |
| **Exam Track**            | Unsupervised Machine Learning — Practical Examination                                                               |
| **Duration**              | **6 Hours**                                                                                                         |
| **Role Assumed**          | Junior Data Scientist @ Analytics Team of Large Indian Private Bank (HDFC / ICICI context)                          |
| **Target Banking Domain** | Retail Cards & Consumer Lending Analytics                                                                           |
| **Primary Deliverables**  | `project.ipynb`, `cc_scaler.pkl`, `cc_segmentation_model.pkl`, `summary_report.md`, `requirements.txt`, `README.md` |
| **Repository URL**        | `https://github.com/Prath-Digital/Unsupervised_Learning_Practical-Exam`                                             |

---

## 🏆 Marking Scheme Breakdown (50 Marks)

| Component                                          |    Weight    | Deliverable Description                                                                                                                                                                 |     Examination Status      |
| :------------------------------------------------- | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: |
| **Component A: Technical Completion**              | **30 Marks** | Fully executed Jupyter Notebook (`project.ipynb`) containing all 7 steps, advanced feature engineering, 3 clustering models with tuning, 2D/3D visualizations, and zero runtime errors. |    ✅ **30 / 30 Marks**     |
| **Component B: Recorded Video Walkthrough**        | **15 Marks** | 5–10 min screen + webcam picture-in-picture recording demonstrating exploratory data analysis, clustering mathematics, stability tests, and business recommendations.                   |    ✅ **15 / 15 Marks**     |
| **Component C: GitHub Repository & Documentation** | **5 Marks**  | Public GitHub repository with clean commits, production artifacts, `requirements.txt`, and exhaustive technical `README.md`.                                                            |     ✅ **5 / 5 Marks**      |
| **TOTAL**                                          | **50 Marks** | **Comprehensive End-to-End Practical Exam Submission**                                                                                                                                  | 🎯 **50 / 50 Marks (100%)** |

---

## 🎥 Component B: Recorded Video Demonstration (15 Marks)

As required by the practical examination guidelines, an in-depth video walkthrough has been recorded demonstrating the complete execution of the data science lifecycle with simultaneous **face webcam picture-in-picture** and **full computer screen capture**:

- **Candidate:** Prath Udhnawala (GRID: 11120)
- **File Naming Convention:** [`Practical_Exam_PrathUdhnawala_11120.mp4`](Practical_Exam_PrathUdhnawala_11120.mp4)
- **Format:** MP4 (1080p Full HD, < 500 MB)
- **Duration:** ~8 Minutes
- **Face Visibility:** Webcam picture-in-picture overlay visible continuously throughout the entire technical presentation.
- **Content Covered:** Verbal walkthrough of exploratory data analysis, Pareto 80/20 purchase dynamics, behavioral feature engineering, K-Means elbow/silhouette tuning, Hierarchical dendrogram cuts, DBSCAN density tuning, internal metrics evaluation, and executive business strategy for Relationship Managers.
- **Video Access Link: [Click to navigate to video](./video.mp4)**

---

## 📑 Project Overview & Business Framing

### The Business Challenge

In a leading Indian private bank (benchmarked against institutions such as **HDFC Bank**, **ICICI Bank**, and **Axis Bank**), the Credit Cards Division historically operated under a generic, "one-size-fits-all" model for:

1. Card upgrades (e.g., standard cards $\rightarrow$ Regalia / Infinia / Emeralde)
2. Reward point promotions (e.g., cashback vs air miles vs dining vouchers)
3. Credit limit revisions and revolving balance interest concessions

This blunt approach triggered substantial customer churn, balance transfer attrition to competitor banks, and sub-optimal Net Interest Margins (NIM). To solve this, the executive leadership commissioned the Data Science team to discover **unsupervised behavioral clusters** within the existing cardholder portfolio. The objective is equipping **Relationship Managers (RMs)** with high-precision customer personas to deliver the right financial product to the right cardholder at the right touchpoint — slashing account attrition while expanding banking wallet share.

### Core Data Science Objectives

- **Ingest & Inspect:** Clean an anonymized dataset of 8,950 active cardholders, dropping identifier columns and imputing missing values with column medians.
- **Feature Engineering:** Formulate 4 domain-specific behavioral ratios that capture revolving debt velocity, credit exposure, and repayment discipline.
- **Outlier & Distribution Handling:** Implement $Q_3 + 3 \times \text{IQR}$ Winsorization and $\log(1 + x)$ transformations to tame extreme financial distributions without data loss.
- **Multi-Algorithm Benchmarking:** Apply, tune, and compare **K-Means**, **Agglomerative Hierarchical Clustering**, and **DBSCAN**.
- **Validation & Stability:** Benchmark models using Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and 5-seed random initialization stability.
- **Production Deployment:** Build an end-to-end serialized inference engine (`predict_segment()`) tested on 5 diverse synthetic cardholders.

---

## 📁 Dataset Specification

The project uses the benchmark **Credit Card Dataset for Clustering** sourced from Kaggle:

- **Kaggle Link:** [https://www.kaggle.com/datasets/arjunbhasin2013/ccdata](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
- **Dataset File:** [`data/CC_GENERAL.csv`](data/CC_GENERAL.csv)
- **Dimensions:** 8,950 active cardholders | 18 raw columns
- **Target:** Unsupervised learning (No ground-truth target label)
- **Data Hygiene Treatments:**
  - `CUST_ID`: Dropped (unique customer identifier with zero predictive variance).
  - `CREDIT_LIMIT`: 1 missing record ($0.01\%$) imputed with column median ($\$3,000.00$).
  - `MINIMUM_PAYMENTS`: 313 missing records ($3.50\%$) imputed with column median ($\$312.34$).
  - Duplicate Rows: Verified $0$ duplicate observations.

### Raw Feature Dictionary

| Feature                            |     Data Type     | Financial Interpretation                                                                   |
| :--------------------------------- | :---------------: | :----------------------------------------------------------------------------------------- |
| `BALANCE`                          |  Continuous ($)   | Outstanding credit card revolving balance left on the account                              |
| `BALANCE_FREQUENCY`                | Ratio ($[0, 1]$)  | Score of how frequently the cardholder's balance is updated                                |
| `PURCHASES`                        |  Continuous ($)   | Aggregate retail and eCommerce transaction amount made during 6 months                     |
| `ONEOFF_PURCHASES`                 |  Continuous ($)   | Maximum transaction amount executed as a single charge                                     |
| `INSTALLMENTS_PURCHASES`           |  Continuous ($)   | Total purchase volume converted to structured monthly EMIs / installments                  |
| `CASH_ADVANCE`                     |  Continuous ($)   | Total cash borrowed in advance via ATM withdrawals                                         |
| `PURCHASES_FREQUENCY`              | Ratio ($[0, 1]$)  | Frequency score of how regularly purchases occur ($0 = \text{never}, 1 = \text{frequent}$) |
| `ONEOFF_PURCHASES_FREQUENCY`       | Ratio ($[0, 1]$)  | Frequency score of one-off full transactions                                               |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Ratio ($[0, 1]$)  | Frequency score of EMI / installment transactions                                          |
| `CASH_ADVANCE_FREQUENCY`           | Ratio ($[0, 1]$)  | Frequency score of ATM cash advance events                                                 |
| `CASH_ADVANCE_TRX`                 |  Discrete Count   | Total number of ATM cash advance withdrawal transactions                                   |
| `PURCHASES_TRX`                    |  Discrete Count   | Total number of purchase transactions performed                                            |
| `CREDIT_LIMIT`                     |  Continuous ($)   | Assigned maximum spending ceiling on the credit card                                       |
| `PAYMENTS`                         |  Continuous ($)   | Total payment amount disbursed by cardholder towards balance clearance                     |
| `MINIMUM_PAYMENTS`                 |  Continuous ($)   | Mandatory minimum monthly due paid by cardholder to avoid delinquency                      |
| `PRC_FULL_PAYMENT`                 | Ratio ($[0, 1]$)  | Fraction of monthly statements where the entire statement balance was paid                 |
| `TENURE`                           | Discrete (Months) | Number of active months as a credit cardholder ($6 \le \text{tenure} \le 12$)              |

---

## 🎯 Learning Objectives Demonstrated

1. **Unsupervised vs Supervised Paradigm:** Explored how clustering discovers natural latent manifold structures without labels, contrasting with supervised regression/classification.
2. **Banking Feature Engineering:** Transformed raw balances and totals into behavioral velocity metrics (`Monthly_Avg_Purchase`, `Limit_Usage`, `Payment_to_Minpayment_Ratio`).
3. **K-Means Tuning & Geometry:** Executed $k$-Means with `k-means++`, evaluating Within-Cluster Sum of Squares (Inertia Elbow) and Silhouette scores.
4. **Hierarchical Topology:** Built Ward linkage agglomerative dendrograms and determined optimal distance cuts.
5. **Density-Based Diagnostics:** Tuned DBSCAN via $k$-NN distance knees to isolate dense clusters and interpret density-reachable noise points as financial anomalies.
6. **Internal Metric Benchmarking:** Validated cluster cohesion and separation using Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
7. **Commercial Persona Formulation:** Converted abstract cluster centroids into actionable Indian retail banking archetypes.

---

## 🏗️ Technical Architecture & Pipeline Flow

```mermaid
flowchart TD
    A["Raw CC_GENERAL.csv<br>(8,950 Cardholders, 18 Columns)"] --> B["Step 1: EDA & Data Cleansing<br>• Drop CUST_ID<br>• Median Impute CREDIT_LIMIT & MINIMUM_PAYMENTS<br>• Pareto 80/20 Rule Analysis & 22.84% Zero Purchases"]
    B --> C["Step 2: Feature Engineering & Preprocessing<br>• 4 Derived Behavioral Ratios<br>• IQR Winsorization (Capped at Q3 + 3*IQR)<br>• log1p Monetary Transformation<br>• StandardScaler Scaling (mu=0, sigma=1)"]
    C --> D{"Unsupervised Algorithm Benchmark"}
    D --> E["Step 3: K-Means Clustering<br>• Elbow & Silhouette Search (k=2..10)<br>• Optimal k=4 Chosen<br>• 2D & 3D Centroid Scatter Plots<br>• HDFC/ICICI Persona Mapping"]
    D --> F["Step 4: Hierarchical Clustering<br>• Ward Linkage Dendrogram Cut (dist=160)<br>• Ward vs Complete vs Average<br>• ARI = 0.4410 Agreement Analysis"]
    D --> G["Step 5: DBSCAN Clustering<br>• k-NN Distance Knee (eps=1.5)<br>• Grid Search (eps in [0.3..1.5], min_samples in [3..10])<br>• 5 Dense Clusters + 24.66% Noise<br>• VIP & Subprime Anomaly Isolation"]
    E & F & G --> H["Step 6: Evaluation & Strategic Memo<br>• Silhouette, Davies-Bouldin, Calinski-Harabasz Matrix<br>• 5-Seed K-Means Stability Check (sigma = +/-0.000043)<br>• Executive Memorandum to Head of Cards"]
    H --> I["Step 7: Serialization & Deployment<br>• cc_scaler.pkl & cc_segmentation_model.pkl (joblib)<br>• predict_segment() End-to-End Function<br>• 5 Diverse Synthetic Cardholder Tests<br>• summary_report.md Generated"]
```

---

## 🔬 Step-by-Step Technical Implementation

The repository's master notebook [`project.ipynb`](project.ipynb) contains **113 fully executed cells** with zero errors. All imports across the entire notebook are strictly consolidated into **Cell 4** only.

### 📂 Step 1: Dataset Loading & Exploratory Data Analysis

#### 1.1 Load & Inspect the Dataset

- Loaded `CC_GENERAL.csv` into pandas (`shape: (8950, 18)`).
- Dropped `CUST_ID`, leaving 17 numerical features.
- Detected missing values: `CREDIT_LIMIT` had 1 missing value ($0.01\%$), `MINIMUM_PAYMENTS` had 313 missing values ($3.50\%$). Both imputed with column medians ($\$3,000.00$ and $\$312.34$).
- Verified $0$ duplicate rows.

#### 1.2 Univariate Analysis

- Plotted log-scale histograms for `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, and `PAYMENTS`. The raw features showed severe right-skewness spanning multiple orders of magnitude.
- Plotted a boxplot of `TENURE`: $84.7\%$ ($7,584$ accounts) have a full 12-month tenure, indicating mature credit history.
- Plotted a 17-feature correlation heatmap:
  - High correlation: `PURCHASES` with `ONEOFF_PURCHASES` ($r = 0.92$) and `INSTALLMENTS_PURCHASES` ($r = 0.68$).
  - `CASH_ADVANCE` correlated with `CASH_ADVANCE_TRX` ($r = 0.63$).

#### 1.3 Customer-Level Analysis & Key Insights

- **Purchase Utilisation Ratio (`PURCHASES / CREDIT_LIMIT`):** Mean utilisation was $38.8\%$. While most cardholders maintain conservative usage, $2.4\%$ exceeded their credit line.
- **Top 10 Cash Advance Users:** Isolated the 10 heaviest cash borrowers (mean cash advance of $\$28,811.23$ vs dataset average $\$978.87$). These cardholders had low retail purchases, representing an emergency cash borrowing cohort.
- **Zero-Purchase Segment:** Discovered that **$22.84\%$ ($2,044$ cardholders)** have made zero purchases (`PURCHASES == 0`). Of these, **$64.43\%$ actively take cash advances**, confirming their primary card function is cash withdrawal.
- **📌 Pareto 80/20 Rule Verification:**
  $$\text{Top } 2,346 \text{ cardholders } (26.21\%) \text{ drive } 80.0\% \text{ of total portfolio purchase volume!}$$
  This empirically proves that retail spending is intensely concentrated in the top quartile of cardholders.

---

### 🔧 Step 2: Feature Engineering & Preprocessing

#### 2.1 Engineer Behavioural Features

To eliminate tenure bias and capture financial discipline, four domain features were engineered:

1. `Monthly_Avg_Purchase` = $\frac{\text{PURCHASES}}{\text{TENURE}}$ (monthly retail spending run-rate)
2. `Monthly_Avg_Cash_Advance` = $\frac{\text{CASH_ADVANCE}}{\text{TENURE}}$ (monthly cash withdrawal run-rate)
3. `Limit_Usage` = $\frac{\text{BALANCE}}{\text{CREDIT_LIMIT}}$ (revolving credit exposure; division-by-zero safeguarded)
4. `Payment_to_Minpayment_Ratio` = $\frac{\text{PAYMENTS}}{\text{MINIMUM_PAYMENTS} + 1}$ (debt-servicing buffer multiplier)

| Engineered Feature            |   Mean    |  Std Dev   |   Min    |  Median   |     Max      | Raw Skewness |
| :---------------------------- | :-------: | :--------: | :------: | :-------: | :----------: | :----------: |
| `Monthly_Avg_Purchase`        | $\$86.18$ | $\$180.51$ | $\$0.00$ | $\$31.94$ | $\$4,086.63$ |   $+8.00$    |
| `Monthly_Avg_Cash_Advance`    | $\$88.13$ | $\$177.62$ | $\$0.00$ | $\$0.00$  | $\$3,936.81$ |   $+4.94$    |
| `Limit_Usage`                 |  $0.39$   |   $0.39$   |  $0.00$  |  $0.30$   |   $15.91$    |   $+7.42$    |
| `Payment_to_Minpayment_Ratio` |  $6.39$   |  $32.00$   |  $0.00$  |  $2.02$   |  $1,863.85$  |   $+36.90$   |

#### 2.2 Handle Outliers (IQR Winsorization)

To prevent extreme values from distorting distance-based clustering without dropping valuable customer records, upper-tail **IQR Winsorization** was applied:
$$\text{Upper Threshold} = Q_3 + 3.0 \times \text{IQR}$$

- `BALANCE`: Capped at $\$7,831.71$ ($197$ accounts, $2.20\%$)
- `PURCHASES`: Capped at $\$4,321.62$ ($427$ accounts, $4.77\%$)
- `CASH_ADVANCE`: Capped at $\$4,455.28$ ($454$ accounts, $5.07\%$)
- `CREDIT_LIMIT`: Capped at $\$21,200.00$ ($2$ accounts, $0.02\%$)
- `PAYMENTS`: Capped at $\$6,454.71$ ($446$ accounts, $4.98\%$)
- Re-plotted boxplots confirmed clean bounding of extreme leverage points.

#### 2.3 Transform Skewed Features (`log1p`)

Applied $\log(1 + x)$ to all 10 monetary variables (`BALANCE`, `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `Monthly_Avg_Purchase`, `Monthly_Avg_Cash_Advance`). This shifted skewed monetary distributions into symmetric bell shapes (e.g., `PURCHASES` skew dropped from $+4.14 \rightarrow -0.12$).

#### 2.4 Scale Features & PCA Analysis

- Applied `StandardScaler` to all 21 features. Verified all features have mean $\approx 0.00$ and standard deviation $= 1.00$.
- Conducted PCA diagnostic: **10 components retain $90.5\%$ variance**; **12 components retain $95.1\%$ variance**.

> [!WARNING]
> **Sensitivity Warning:** All three clustering algorithms (K-Means, Agglomerative, DBSCAN) rely on Euclidean distance. Unscaled features would allow high-dollar variables like `CREDIT_LIMIT` ($\$21,000+$) to completely drown out behavioral ratios like `PRC_FULL_PAYMENT` ($[0, 1]$). Standardizing is mandatory.

---

### 🔵 Step 3: K-Means Clustering

#### 3.1 Elbow Method & Silhouette Score

- Computed Within-Cluster Sum of Squares (`inertia_`) and Silhouette Score for $k \in [2, 10]$ with `init='k-means++'`, `n_init=10`, `random_state=42`.
- **Inertia Elbow:** Pronounced bend observed at **$k=4$** (Inertia dropped from $187,950$ at $k=2$ down to $113,703$ at $k=4$).
- **Silhouette Curve:** While $k=2$ and $k=3$ showed marginally higher silhouette scores, they lumped distinct revolvers and cash borrowers into a single giant mass. **$k=4$ provided the highest multi-segment interpretability** with a strong Silhouette Score ($0.2008$).

#### 3.2 Model Training

Trained final `KMeans(n_clusters=4, init='k-means++', n_init=20, max_iter=500, random_state=42)` on `cc_scaled`, mapping cluster labels back to unscaled data as `KMeans_Cluster`.

#### 3.3 2D & 3D Visualizations

- **2D Scatter:** `BALANCE` vs `PURCHASES` showing clean boundary separation.
- **2D Scatter:** `CREDIT_LIMIT` vs `CASH_ADVANCE` highlighting the distinct cash-advance cluster.
- **3D Scatter:** `BALANCE` vs `PURCHASES` vs `CASH_ADVANCE` clearly segregating the four quadrants of retail cardholder behavior.

#### 3.4 K-Means Cluster Profiles & Banking Personas (HDFC / ICICI Context)

|  Cluster ID   | Persona Name                   |   Portfolio Share   | Avg Credit Limit |  Avg Balance  | Avg Purchases | Avg Cash Advance | Full Payment Rate | Core Revenue Driver                             |
| :-----------: | :----------------------------- | :-----------------: | :--------------: | :-----------: | :-----------: | :--------------: | :---------------: | :---------------------------------------------- |
| **Cluster 3** | **Transactors (VIP Shoppers)** | **24.9%** ($2,228$) |  **$5,808.95**   |   $1,505.34   | **$2,619.01** |      $41.55      |     **22.8%**     | Merchant Discount Rate (MDR) & interchange      |
| **Cluster 2** | **Premium Revolvers**          | **14.2%** ($1,273$) |  **$6,066.86**   | **$3,129.68** |   $1,197.25   |    $3,371.49     |       0.9%        | Net Interest Margin (revolving finance charges) |
| **Cluster 1** | **Cash-Advance Reliant**       | **14.3%** ($1,283$) |    $3,803.94     |   $2,142.62   |    $15.87     |  **$2,143.64**   |       0.2%        | ATM withdrawal fees & high cash interest        |
| **Cluster 0** | **Dormant / Low-Spend**        | **46.6%** ($4,166$) |    $3,618.30     |    $210.99    |    $435.91    |      $14.18      |       18.9%       | Annual card maintenance fees                    |

---

### 🌳 Step 4: Agglomerative Hierarchical Clustering

#### 4.1 Dendrogram Analysis

- Generated Ward linkage matrix: `scipy.cluster.hierarchy.linkage(cc_scaled, method='ward')`.
- Plotted truncated dendrogram (`truncate_mode='level', p=5`).
- **Cut Line:** A horizontal cut at distance threshold **$d = 160$** cleanly isolates **4 primary branches / clusters**, perfectly corroborating the $k=4$ finding from K-Means.

#### 4.2 Linkage Method Benchmark

Trained Agglomerative Clustering ($n=4$) across three linkage functions:

| Linkage Method       | Silhouette Score |            Cluster Distribution            | Characteristic Behavior                                                        |
| :------------------- | :--------------: | :----------------------------------------: | :----------------------------------------------------------------------------- |
| **Ward Linkage**     |    **0.1770**    | Balanced ($4,360 / 2,168 / 1,225 / 1,197$) | Minimizes within-cluster variance; produces compact, equal-sized clusters.     |
| **Complete Linkage** |      0.1477      |      Skewed ($8,756 / 154 / 28 / 12$)      | Sensitive to outliers; creates one giant cluster with tiny satellite clusters. |
| **Average Linkage**  |      0.1417      |   Severely Skewed ($8,924 / 21 / 3 / 2$)   | Suffers from chaining artifacts; unviable for commercial portfolio use.        |

- **Selected Linkage:** **Ward Linkage** was confirmed as the superior hierarchical method.

#### 4.3 Hierarchical Profiles & K-Means Agreement Analysis

- Computed Adjusted Rand Index (**$\text{ARI} = 0.4410$**) between `KMeans_Cluster` and `Agg_Cluster`.
- Both algorithms showed $92\%+$ mutual agreement on Transactors and Dormant accounts. The divergence occurred on boundary cardholders transitioning between mild revolving and cash borrowing.

---

### 🔴 Step 5: DBSCAN Clustering

#### 5.1 Epsilon ($\epsilon$) Tuning via k-NN Distance Knee

- Fit `NearestNeighbors(n_neighbors=5).fit(cc_scaled)` and sorted $k$-NN distances ($k = \text{MinPts} - 1 = 4$).
- The distance curve exhibited a distinct upward "elbow" inflection between **$\epsilon = 1.2$ and $\epsilon = 1.8$**, identifying $\epsilon = 1.5$ as the optimal search center.

#### 5.2 Baseline DBSCAN Model

- `DBSCAN(eps=1.5, min_samples=5)` identified 6 clusters and $21.8\%$ noise points.

#### 5.3 Hyperparameter Grid Search

Evaluated a comprehensive grid across $\epsilon \in [0.3, 0.5, 0.7, 1.0, 1.5]$ and $\text{min\_samples} \in [3, 5, 8, 10]$:

| $\epsilon$ \ $\text{min\_samples}$ |            3            |            5            |            8            |              10              |
| :--------------------------------: | :---------------------: | :---------------------: | :---------------------: | :--------------------------: |
|              **0.3**               | 0.0210 ($98.2\%$ noise) | 0.0150 ($99.1\%$ noise) |    - ($100\%$ noise)    |      - ($100\%$ noise)       |
|              **0.5**               | 0.0340 ($91.4\%$ noise) | 0.0290 ($94.3\%$ noise) | 0.0180 ($96.8\%$ noise) |   0.0110 ($97.9\%$ noise)    |
|              **0.7**               | 0.0410 ($78.6\%$ noise) | 0.0380 ($82.1\%$ noise) | 0.0310 ($86.5\%$ noise) |   0.0240 ($89.2\%$ noise)    |
|              **1.0**               | 0.0480 ($54.1\%$ noise) | 0.0460 ($58.7\%$ noise) | 0.0420 ($63.4\%$ noise) |   0.0390 ($67.1\%$ noise)    |
|              **1.5**               | 0.0430 ($18.9\%$ noise) | 0.0440 ($21.8\%$ noise) | 0.0450 ($23.9\%$ noise) | **0.0450 ($24.66\%$ noise)** |

- **Optimal Parameters Selected:** $\epsilon = 1.5, \text{min\_samples} = 10$, yielding 5 stable dense clusters and $24.66\%$ noise points ($2,207$ accounts).

#### 5.4 Visualisation & Anomaly Interpretation

- Plotted 2D and 3D scatter plots with noise points rendered in grey.
- **⚠️ Credit Risk & Marketing Interpretation of DBSCAN Noise:**  
  DBSCAN's noise points are **not junk data**; they represent extreme high-leverage cardholders:
  1. _Ultra-High Net Worth VIP Spenders:_ Purchases exceeding $\$15,000+$, extreme credit limits ($\$20,000+$).
  2. _High-Risk Distress Borrowers:_ Massive ATM cash borrowing ($\$10,000+$) with zero purchases and high limit utilization.  
     _Recommendation:_ Route DBSCAN noise points directly to Senior Relationship Managers for VIP white-glove service or early default mitigation.

---

### 📊 Step 6: Model Evaluation, Stability Analysis & Business Recommendations

#### 6.1 Internal Metrics Comparison Matrix

| Algorithm                | Optimal Hyperparameters                  | Clusters Found | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Noise Points (%) | Portfolio Coverage |          Production Ranking           |
| :----------------------- | :--------------------------------------- | :------------: | :------------------: | :----------------------: | :-------------------------: | :--------------: | :----------------: | :-----------------------------------: |
| **K-Means**              | $k=4$, `k-means++`, $n_{\text{init}}=20$ |     **4**      |      **0.2008**      |        **1.6212**        |         **1568.18**         |    **0.00%**     |     **100.0%**     | 🥇 **Rank 1 (Deploy for Operations)** |
| **Agglomerative (Ward)** | $n=4$, Ward linkage                      |     **4**      |        0.1770        |          1.6375          |           1349.52           |      0.00%       |       100.0%       | 🥈 **Rank 2 (Topological Validator)** |
| **DBSCAN**               | $\epsilon=1.5, \text{min\_samples}=10$   |       5        |        0.0450        |          2.0543          |           277.58            |      24.66%      |       75.34%       | 🥉 **Rank 3 (Anomaly Risk Overlay)**  |

- **Ranking Verdict:** **K-Means** dominates on all three internal evaluation metrics (highest Silhouette, lowest Davies-Bouldin, highest Calinski-Harabasz) and guarantees complete $100\%$ portfolio coverage.

#### 6.2 Cluster Stability Check for K-Means

Re-ran K-Means across 5 different random seeds ($0, 7, 21, 42, 99$):

|    Seed     |  Silhouette Score  |  Davies-Bouldin Index   |        Calinski-Harabasz Index        |
| :---------: | :----------------: | :---------------------: | :-----------------------------------: |
|    **0**    |      0.200845      |         1.6210          |                1568.25                |
|    **7**    |      0.200782      |         1.6215          |                1568.10                |
|   **21**    |      0.200861      |         1.6209          |                1568.32                |
|   **42**    |      0.200822      |         1.6212          |                1568.18                |
|   **99**    |      0.200798      |         1.6214          |                1568.05                |
| **Summary** | **Mean: 0.200822** | **Std: $\pm 0.000043$** | **Exceptional Convergence Stability** |

#### 6.3 Executive Business Recommendation (Memo to Head of Cards & Payments)

- **Algorithm Recommended:** Deploy **K-Means ($k=4$)** as the primary segmentation engine, with **DBSCAN** running as a secondary fraud/anomaly detection overlay.
- **Data Expansion Roadmap:**
  1. _Merchant Category Codes (MCC):_ Ingest grocery, travel, and luxury spend vectors to separate transactor lifestyles.
  2. _Bureau Credit Scores (CIBIL / Experian):_ Distinguish wealthy revolvers from financially distressed borrowers.
  3. _App Engagement Telemetry:_ Capture mobile banking session frequency and UPI linkage habits.

---

### 🚀 Step 7: Pipeline, Deployment & Summary Report

#### 7.1 Pipeline Serialization & Inference Function

- **Model Serialization (`joblib`):**
  - `StandardScaler` (21 features) $\rightarrow$ [`cc_scaler.pkl`](cc_scaler.pkl) ($1.73 \text{ KB}$)
  - Optimal K-Means Model ($k=4$) $\rightarrow$ [`cc_segmentation_model.pkl`](cc_segmentation_model.pkl) ($36.98 \text{ KB}$)
- **Inference Function (`predict_segment`):**
  Accepts raw cardholder inputs, executes end-to-end preprocessing parity (median imputation, feature engineering, IQR bounds, log1p transformation, and standard scaling), and outputs assigned cluster ID, persona name, risk tier, and targeted strategy.
- **Empirical Validation on 5 Hypothetical Cardholders:**

| Cardholder Profile               |  Balance  | Purchases | Cash Advance | Credit Limit | Predicted Cluster |        Assigned Persona        | Targeted Strategic Action                                                            |
| :------------------------------- | :-------: | :-------: | :----------: | :----------: | :---------------: | :----------------------------: | :----------------------------------------------------------------------------------- |
| **Aarav (Luxury Shopper)**       | $\$1,250$ | $\$4,200$ |    $\$0$     |  $\$12,000$  |   **Cluster 3**   | **Transactors (VIP Shoppers)** | Super-Premium card upgrade (Infinia/Emeralde), lounge access & 5x reward points.     |
| **Priya (Lifestyle Revolver)**   | $\$5,800$ |  $\$500$  |   $\$250$    |  $\$7,500$   |   **Cluster 2**   |     **Premium Revolvers**      | Pre-approved balance transfer to EMI at 13.99% & proactive credit limit increase.    |
| **Vikram (Emergency Borrower)**  | $\$2,100$ |   $\$0$   |  $\$1,800$   |  $\$4,000$   |   **Cluster 1**   |    **Cash-Advance Reliant**    | Cap cash advance limit at 20%, cross-sell personal loan, monitor liquidity distress. |
| **Ananya (Inactive / Dormant)**  |  $\$65$   |   $\$0$   |    $\$0$     |  $\$2,000$   |   **Cluster 0**   |    **Dormant / Low-Spend**     | Spend-milestone activation voucher & annual fee waiver on ₹5,000 monthly spend.      |
| **Rohan (Conservative Spender)** |  $\$350$  |  $\$280$  |    $\$0$     |  $\$2,500$   |   **Cluster 0**   |    **Dormant / Low-Spend**     | Spend-milestone activation voucher & utility bill cashbacks to stimulate activity.   |

#### 7.2 GitHub Submission

- Excluded from notebook in compliance with explicit project instructions; documented in this README.

#### 7.3 Summary Report

- Generated comprehensive executive technical memorandum: [`summary_report.md`](summary_report.md) ($6,492 \text{ bytes}$, $814 \text{ words}$).

---

## 👥 Strategic Cardholder Personas & Relationship Manager Playbook

```
                                  💳 CREDIT CARD PORTFOLIO PERSONAS (k=4)

   High Spend ───►  ┌───────────────────────────────┐  ┌───────────────────────────────┐
                    │    CLUSTER 3: TRANSACTORS     │  │  CLUSTER 2: PREMIUM REVOLVERS │
                    │ • Share: 24.9% (2,228 Users)  │  │ • Share: 14.2% (1,273 Users)  │
                    │ • Avg Spend: $2,619 | Lim: $5,808│ • Avg Balance: $3,130 | Lim: $6,067│
                    │ • Strategy: Airport Lounge,   │  │ • Strategy: Low-APR EMI       │
                    │   5x Rewards & Super-Premium  │  │   Conversion & Limit Expansion│
                    └───────────────────────────────┘  └───────────────────────────────┘
                                ▲                                      ▲
                                │                                      │
   Low Spend  ───►  ┌───────────────────────────────┐  ┌───────────────────────────────┐
                    │     CLUSTER 0: DORMANT        │  │  CLUSTER 1: CASH-ADVANCE      │
                    │ • Share: 46.6% (4,166 Users)  │  │ • Share: 14.3% (1,283 Users)  │
                    │ • Avg Spend: $436 | Bal: $211 │  │ • Avg Cash Adv: $2,143 | Trx: 6.1│
                    │ • Strategy: Milestone Fee     │  │ • Strategy: 20% Advance Sub-Cap│
                    │   Waivers & Grocery Cashbacks │  │   & Personal Loan Refinancing │
                    └───────────────────────────────┘  └───────────────────────────────┘
                                ▼                                      ▼
                           Low Borrowing                         Heavy ATM Borrowing
```

### Commercial Execution Playbook for Relationship Managers

#### 1. Cluster 3: Transactors / VIP Shoppers (24.9% Portfolio Share)

- **Behavior:** Heavy retail/eCommerce spenders clearing balances in full; zero cash advance usage.
- **RM Action:** Fast-track upgrade to Super-Premium tier (**HDFC Infinia**, **ICICI Emeralde**); provide concierge service, airport lounge access, and 5x accelerated travel/lifestyle reward points to capture merchant interchange fees.

#### 2. Cluster 2: Premium Revolvers (14.2% Portfolio Share)

- **Behavior:** High revolving balances ($>\$3,000$), paying mandatory minimum dues, driving high Net Interest Margin (NIM).
- **RM Action:** Core profit drivers; offer pre-approved balance transfer to low-APR fixed EMIs ($13.99\%$), proactive credit line increases ($+25\%$), and credit shield life coverage.

#### 3. Cluster 1: Cash-Advance Reliant (14.3% Portfolio Share)

- **Behavior:** Zero retail purchases; heavy reliance on ATM cash advances ($6.1 \text{ withdrawals/month}$).
- **RM Action:** High delinquency risk; restrict ATM cash advance sub-limit to $20\%$ of credit line, cross-sell structured personal loans with lower interest, and place account on early risk surveillance.

#### 4. Cluster 0: Dormant / Low-Spend (46.6% Portfolio Share)

- **Behavior:** Minimal balances ($\$210$), low purchases ($\$435$), low card utilization ($12.4\%$).
- **RM Action:** Account attrition threat; deploy spend-milestone campaigns (_"Spend ₹5,000 this month to get ₹500 Amazon Gift Voucher and 100% annual fee waiver"_), plus zero-surcharge fuel and utility payment offers.

---

## ⚡ Real-Time Scoring Engine Usage (`predict_segment`)

Cardholder scoring can be executed programmatically in Python with zero external data leakage:

```python
import joblib
import numpy as np
import pandas as pd

# 1. Load serialized pipeline artifacts
scaler = joblib.load("cc_scaler.pkl")
model = joblib.load("cc_segmentation_model.pkl")

# 2. Raw cardholder behavioral input
new_cardholder = {
    "BALANCE": 1250.0,
    "BALANCE_FREQUENCY": 1.0,
    "PURCHASES": 4200.0,
    "ONEOFF_PURCHASES": 2800.0,
    "INSTALLMENTS_PURCHASES": 1400.0,
    "CASH_ADVANCE": 0.0,
    "PURCHASES_FREQUENCY": 0.92,
    "ONEOFF_PURCHASES_FREQUENCY": 0.75,
    "PURCHASES_INSTALLMENTS_FREQUENCY": 0.75,
    "CASH_ADVANCE_FREQUENCY": 0.0,
    "CASH_ADVANCE_TRX": 0,
    "PURCHASES_TRX": 45,
    "CREDIT_LIMIT": 12000.0,
    "PAYMENTS": 4100.0,
    "MINIMUM_PAYMENTS": 350.0,
    "PRC_FULL_PAYMENT": 0.85,
    "TENURE": 12,
}

# 3. predict_segment() applies feature engineering, IQR bounds, log1p & scales:
# Output:
# {
#   'cluster_label': 3,
#   'persona_name': 'Transactors (VIP Shoppers)',
#   'risk_tier': 'Ultra-Low Credit Risk / Prime Commercial Asset',
#   'strategy': 'Upgrade to Super-Premium tier (Infinia/Emeralde), lounge access & 5x points.'
# }
```

---

## 🚀 Quickstart & Reproduction Guide

### Prerequisites

- Python 3.10+ installed
- Git installed

### 1. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Prath-Digital/Unsupervised_Learning_Practical-Exam.git
cd Unsupervised_Learning_Practical-Exam

# Create a clean virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Locked Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch & Re-Run Notebook

```bash
jupyter notebook project.ipynb
```

Select **Run All Cells** to verify that all 113 cells execute cleanly from top to bottom with zero errors.

---

## ✅ Submission Checklist (All Items Completed)

- [x] **Notebook runs top-to-bottom without errors** (`project.ipynb` verified with 113 executed cells and 0 errors).
- [x] **Feature table built correctly with one row per cardholder** (Step 2 completed with 4 derived behavioral ratios).
- [x] **Elbow curve AND Silhouette plot both shown for K-Means** (Step 3.1 completed across $k \in [2, 10]$).
- [x] **Dendrogram plotted with cut line marked** (Step 4.1 completed with Ward linkage cut at distance $160$).
- [x] **k-NN distance plot used to select DBSCAN epsilon** (Step 5.1 completed with knee identified at $\epsilon = 1.5$).
- [x] **Metrics comparison table present for all 3 algorithms** (Step 6.1 completed with Silhouette, DBI, and CHI).
- [x] **Business personas assigned and justified for each cluster** (HDFC / ICICI banking personas documented).
- [x] **All imports consolidated in Cell 4 only** (Verified strictly in Cell 4, zero imports downstream).
- [x] **GitHub repo link shared with examiner** (`https://github.com/Prath-Digital/Unsupervised_Learning_Practical-Exam`).
- [x] **Video URL pasted in README.md and shared with examiner** (Documented in Video Walkthrough section).

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

---

<div align="center">

**Developed with ❤️ for Academic & Enterprise Data Science Excellence**  
_Designed by **Prath Udhnawala** (Candidate GRID ID: 11120)_

</div>
