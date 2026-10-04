# 📊 Credit Card Customer Segmentation — Executive Summary Report

**Author:** Prath Udhnawala  
**Role:** Senior Data Science Consultant  
**Target Audience:** Head of Cards & Consumer Lending (Retail Banking)  
**Dataset:** `CC_GENERAL.csv` (8,950 Active Credit Cardholder Accounts)  

---

### 1. Business Problem & Dataset Overview
Modern retail banks face significant challenges in maximizing credit card portfolio profitability while prudently managing default risk. Blanket marketing strategies and uniform credit limits fail to accommodate diverse consumer behaviours. This project formulates an unsupervised machine learning segmentation engine across **8,950 active cardholders** using 17 core financial attributes (capturing revolving balances, purchase modalities, ATM cash borrowings, credit lines, and repayment frequencies). The core business objective is identifying granular, mutually exclusive customer personas to drive personalized commercial interventions, optimize interchange revenues, and proactively mitigate subprime delinquency.

### 2. Feature Engineering & Preprocessing Strategy
Raw cardholder balances often conceal behavioral intensity due to varying credit tenures and credit line magnitudes. To surface true financial patterns, four domain-specific behavioral metrics were engineered:
1. **`Monthly_Avg_Purchase` & `Monthly_Avg_Cash_Advance`:** Normalized by account `TENURE` to eliminate tenure-related scale distortions.
2. **`Limit_Usage` (`BALANCE / CREDIT_LIMIT`):** Measures exposure and leverage against assigned credit limits.
3. **`Payment_to_Minpayment_Ratio`:** Measures debt repayment discipline above minimum mandatory dues.

**Preprocessing Justification:**
* **Missing Value Imputation:** Missing values in `CREDIT_LIMIT` (1 account) and `MINIMUM_PAYMENTS` (313 accounts) were imputed using column medians, preventing data loss while insulating the feature space from extreme skew.
* **IQR Winsorization:** Extreme financial outliers exceeding $Q_3 + 3 	imes 	ext{IQR}$ in `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, and `PAYMENTS` were capped at that boundary, neutralizing leverage effects without discarding valuable customer accounts.
* **Logarithmic Transformation (`log1p`):** Applied $\log(1 + x)$ to 10 highly right-skewed monetary variables, stabilizing variance and producing substantially more symmetric, bell-shaped distributions.
* **Feature Standardization:** Scaled all 21 transformed features to zero mean and unit variance via `StandardScaler`, ensuring spherical Euclidean distance calculations without dominance by high-magnitude credit lines.

### 3. Algorithm Comparison: Internal Metrics vs. Business Intuition
Three distinct unsupervised learning algorithms were systematically benchmarked:

| Algorithm | Optimal Hyperparameters | Clusters Found | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Noise Points (%) | Portfolio Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | $k=4$, `k-means++`, $n_{	ext{init}}=10$ | **4** | **0.2008** | **1.6212** | **1568.18** | **0.00%** | **100.0%** |
| **Agglomerative (Ward)** | $n=4$, Ward linkage | **4** | 0.1770 | 1.6375 | 1349.52 | 0.00% | 100.0% |
| **DBSCAN** | $\epsilon=1.5, 	ext{min\_samples}=10$ | 5 | 0.0450 | 2.0543 | 277.58 | 24.66% | 75.34% |

**Evaluation & Intuition Alignment:**
* **Metric Superiority:** K-Means with $k=4$ achieved the highest Silhouette Score (0.2008), lowest Davies-Bouldin Index (1.6212), and highest Calinski-Harabasz Index (1568.18). Re-running K-Means across 5 independent random seeds confirmed exceptional stability ($\mu = 0.200822, \sigma = \pm 0.000043$).
* **Business Intuition Alignment:** While DBSCAN effectively isolated severe anomalies (flagging 24.66% of the portfolio as noise, including ultra-wealthy purchasers and emergency borrowers), leaving nearly a quarter of cardholders unassigned is operationally unacceptable for commercial portfolio management. K-Means provides complete, well-separated 100% portfolio partitioning, aligning perfectly with operational business requirements.

### 4. Cardholder Segment Profiles & Targeted Interventions
1. **Cluster 3 — Transactors / VIP Shoppers (24.9% of portfolio):** High purchases ($2,619.01 avg), negligible cash advances ($41.55), high full repayment rate (22.8%).  
   * *Targeted Action:* Upgrade to premium travel/lifestyle cards (HDFC Infinia/Regalia, ICICI Emeralde); provide airport lounge privileges and accelerated dining reward points to maximize merchant interchange revenue.
2. **Cluster 2 — Premium Revolvers (14.2% of portfolio):** High revolving balances ($3,129.68), moderate purchases ($1,197.25), steady minimum payments, highest credit limits ($6,066.86).  
   * *Targeted Action:* Core interest income drivers; extend pre-approved balance transfer offers to lower-rate fixed EMIs (13.99% APR) and proactive credit limit increases to capture full wallet share.
3. **Cluster 1 — Cash-Advance Reliant (14.3% of portfolio):** Heavy ATM cash borrowing ($2,142.62 balance), zero purchases ($15.87 avg), high transaction frequencies.  
   * *Targeted Action:* Restrict cash-advance sub-limit to 20% of credit line, cross-sell structured personal loans to amortize high-interest debt, and initiate early credit-risk monitoring.
4. **Cluster 0 — Dormant / Low-Spend (46.6% of portfolio):** Minimal balances ($210.99), low purchases ($435.91), low card utilization.  
   * *Targeted Action:* Launch milestone activation campaigns (*"Spend ₹5,000 this month to earn a ₹500 voucher and 100% annual fee waiver"*), supplemented by zero-surcharge fuel and utility bill offers.

### 5. Future Roadmap & Production Next Steps
1. **Real-Time Scoring API:** Deploy `cc_scaler.pkl` and `cc_segmentation_model.pkl` in a containerized FastAPI microservice with sub-10ms response times for dynamic web/mobile onboarding segmentation.
2. **Merchant Category Codes (MCC):** Ingest itemized transaction vectors (Travel, Groceries, Dining, Luxury) to differentiate lifestyle preferences within transactors.
3. **Credit Bureau Integration:** Merge external CIBIL / Experian scores to differentiate wealthy revolvers from financially distressed borrowers.
4. **Automated Drift Monitoring:** Track Population Stability Index (PSI) to detect macroeconomic credit distribution shifts and trigger quarterly model recalibration.