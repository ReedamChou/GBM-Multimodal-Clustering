# GBM Multi-Modal Clustering Pipeline

**Title:**  
Multi-modal clustering combining MRI sub-region ratios, brain lobe location, and clinical variables identifies a high-risk GBM subtype characterised by temporal lobe dominance, high NC/EN ratio, and short survival :contentReference[oaicite:0]{index=0}

---

## Step 1: Load and Audit UCSF-PDGM

- Load the CSV dataset using `pandas`.
- Check each column for missing values.
- Handle missing values appropriately:
  - Numerical → median imputation
  - Categorical → mode or "unknown"
- Process NIfTI images as required.
- Perform general preprocessing:
  - Remove inconsistencies
  - Ensure data quality

---

## Step 2: Extract Imaging Features from NIfTI

For each patient, extract:

- Dominant brain lobe:
  - Frontal
  - Temporal
  - Parietal
  - Occipital
- Per-lobe ratios:
  - ED/EN
  - NC/EN  
  *(Total: 12 ratio columns across 4 lobes)*
- Global metrics:
  - NC/EN ratio
  - ED/EN ratio
  - ED total ratio
- Tumor Burden Index (TBI)

📌 Save output as a CSV (one row per patient).  
👉 Run once and reuse.

---

## Step 3: Build Master Feature Table

### 1. Merge Data
- Merge imaging CSV with clinical CSV using **patient ID**

### 2. Handle Missing Values
- Continuous → median
- Categorical → mode or "unknown"
- Avoid dropping patients

### 3. Encode Categorical Variables
- Dominant lobe → One-hot encoding
- Sex, MGMT, IDH → Binary (0/1)

### 4. Standardisation
- Apply **z-score normalization**
- Formula:

(x - mean) / std

- ⚠️ Do NOT include:
- OS (Overall Survival)
- Censoring flag

📌 Save scaler for later use.

---

## Step 4: Train-Test Split (70% / 30%)

- Use **stratified splitting** based on:
- MGMT status
- OS distribution
- Fit scaler on **train only**
- Transform both:
- Train
- Test

⚠️ Never fit on test data

---

## Step 5: Spectral Clustering (Train Set)

- Use: `sklearn.cluster.SpectralClustering`
- Try:

k = 2, 3, 4, 5, 6

- Compute silhouette score:

sklearn.metrics.silhouette_score

- Select best k

📊 Plot:
- Silhouette score vs k (elbow plot)

📌 Save:
- Cluster labels
- Cluster centroids

---

## Step 6: Cluster Characterisation

For each cluster compute:

- Median OS + IQR
- % MGMT methylated
- % IDH mutant
- Mean NC/EN ratio
- Mean ED/EN ratio
- Mean TBI
- Dominant lobe
- Mean age

### Statistical Tests

1. **Survival Analysis**
 - Kaplan-Meier curves
 - Log-rank test (p < 0.05)

2. **Continuous Features**
 - Kruskal-Wallis test

3. **Categorical Features**
 - Chi-square test

📌 Apply:
- Benjamini-Hochberg correction

### Identify High-Risk Cluster

Characteristics:
- Temporal lobe dominance
- High NC/EN ratio
- Low OS
- Low MGMT methylation

👉 Assign meaningful label  
Example:"Temporally-dominant high-necrosis subtype"


---

## Step 7: Validation on Test Set

- Apply saved scaler
- Assign cluster using:
  - Euclidean distance to centroids

### Validate:

1. Cluster proportions consistency
2. Survival replication:
   - Kaplan-Meier curves
   - Log-rank test

📌 Check:
- Temporal dominance
- NC/EN ratio consistency

⚠️ If survival separation disappears → overfitting

---

## Step 8: Reporting & Visualization

### Required Figures:

1. **Feature Heatmap**
   - Rows: features
   - Columns: clusters
   - Use: `seaborn.heatmap`

2. **Kaplan-Meier Curves**
   - Train set
   - Test set
   - Include confidence intervals

3. **Lobe Distribution Chart**
   - Bar plot (stacked/grouped)

4. **Silhouette Elbow Plot**
   - k vs score

5. **Cluster Summary Table**
   - OS
   - MGMT %
   - NC/EN
   - Dominant lobe
   - Cluster size

---

## Final Outcome

A validated GBM subtype defined by:
- Temporal lobe dominance
- High necrosis (NC/EN)
- Poor survival

👉 This is your **publishable insight**