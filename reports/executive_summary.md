# ✈️ Executive Summary: Flight Complexity & Operational Risk Prediction System
### Data-Driven Operational Intelligence for Commercial Aviation Ground Operations
**Developed for the United Airlines Skyhack Challenge | Upgraded to Production Enterprise Grade**

---

## 1. Executive Business Framing & Problem Context
In commercial airline operations, flight turnarounds at major hubs represent the most critical operational bottleneck. A single delayed turnaround can propagate downline throughout the entire aircraft routing sequence, resulting in missed passenger connections, mishandled transfer baggage, crew timeout violations, and FAA tarmac delay penalties.

Frontline teams (station controllers, ramp agents, gate supervisors, and dispatchers) coordinate hundreds of departures daily. However, not all flights carry equal operational friction. Flights characterized by:
- **Compressed Ground Turnaround Time** (tight scheduled turn vs minimum equipment turn time)
- **High Connecting Baggage Volumes** (especially "hot-transfers" with $<45$ minute connection windows)
- **High Special Service Request (SSR) Density** (airport wheelchairs, electric aisle chairs, unaccompanied minors)
- **Peak Hub Station Congestion**

traditionally went undetected until gate agents or ramp crews experienced delays during boarding or baggage loading.

### The Initial Limitation
The initial hackathon iteration relied on a post-hoc, descriptive, hand-coded formula:
$$\text{Overall Complexity} = 0.25 \times \text{Delay Score} + 0.50 \times \text{Passenger Score} + 0.25 \times \text{Bag Score}$$
Categorized by arbitrary thresholds ($<0.40$ Easy, $0.40-0.70$ Medium, $\ge 0.70$ Hard).

**Interview & Operational Gaps Identified:**
1. **Arbitrary Hand-Picked Weights**: The $25/50/25$ weighting was subjective intuition, unsupported by empirical learning.
2. **Post-Mortem Descriptive Scoring**: Using actual departure and arrival delay to score flight complexity meant the flight had *already departed or landed late*—providing zero predictive value for frontline intervention.
3. **Unused Operational Assets**: Rich operational data (such as 51,000+ Special Service Request remarks and international airport registries) remained unlinked.
4. **Lack of Statistical Rigor**: No formal hypothesis testing had evaluated whether observed carrier or route differences were statistically significant.

---

## 2. Enterprise System Architecture & Upgrades Implemented

```
+-----------------------------------------------------------------------------------+
|                           MULTI-SOURCE DATA INGESTION                            |
|  Flight Level Schedules  |  Airports Registry  |  PNR Remarks (SSR)  |  Baggage  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        ADVANCED FEATURE STORE ENGINEERING                         |
|  - Turnaround Tightness & Buffer Physics    - Special Service Request (SSR) Density |
|  - Baggage Transfer & Hot-Transfer Ratios   - Hub Station Congestion Indices        |
|  - Pre-Departure T-2h Feature Isolation    - FAA Delay Ground Truth (>15 min)      |
+-------------------+-------------------------------------+-------------------------+
                    |                                     |
                    v                                     v
+-----------------------------------+   +-------------------------------------------+
|    SUPERVISED CLASSIFICATION      |   |        PRE-DEPARTURE PREDICTIVE MODEL     |
|  LightGBM | XGBoost | RF | LogReg |   |  T-2h Early Warning Alert System (XGBoost)|
|  - Stratified 5-Fold CV           |   |  - ROC-AUC: 0.643 | PR-AUC: 0.603         |
|  - Weighted F1: 98.5% | Macro: 97%|   |  - Threshold Optimization for High Recall |
+-------------------+---------------+   +---------------------+---------------------+
                    |                                         |
                    v                                         v
+-----------------------------------+   +-------------------------------------------+
|   TreeSHAP MODEL EXPLAINABILITY   |   |   STATISTICAL VALIDATION & CLUSTERING     |
|  - Empirical Weight Validation    |   |  - One-Way ANOVA (Carrier F=662.4, p<0.001) |
|  - Top Operational Complexity Drivers |  - Chi-Square (Carrier vs Class p<0.001)   |
|  - Local Waterfall Flight Audits  |   |  - Unsupervised K-Means Route Archetypes  |
+-------------------+---------------+   +---------------------+---------------------+
                    |                                         |
                    +--------------------+--------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                INTERACTIVE SKYOP AI STREAMLIT OPERATIONS DASHBOARD                |
|  - Executive Fleet KPI Monitoring           - Pre-Departure Risk Simulator        |
|  - Dynamic SHAP Waterfall Diagnostics       - Tactical Frontline Action Playbooks |
+-----------------------------------------------------------------------------------+
```

---

## 3. Quantitative Machine Learning Benchmarks

### 3.1 Multi-Class Supervised Complexity Classification (8,099 Flights)
Evaluated across Stratified 5-Fold Cross-Validation and a held-out test split ($20\%$ holdout):

| Model Architecture | 5-Fold CV Accuracy | 5-Fold CV F1 (Weighted) | Test Set Accuracy | Test Set F1 (Weighted) | Test Set F1 (Macro) | Test Set ROC-AUC (OVR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Selected Top)** | **0.9802 ± 0.002** | **0.9803 ± 0.002** | **0.9852** | **0.9851** | **0.9713** | **0.9991** |
| **Logistic Regression (Baseline)** | 0.9802 ± 0.003 | 0.9807 ± 0.003 | 0.9846 | 0.9849 | 0.9649 | 0.9998 |
| **XGBoost Classifier** | 0.9798 ± 0.004 | 0.9797 ± 0.004 | 0.9796 | 0.9794 | 0.9586 | 0.9986 |
| **Random Forest Classifier** | 0.9406 ± 0.005 | 0.9392 ± 0.005 | 0.9284 | 0.9264 | 0.8581 | 0.9880 |

**Holdout Error Breakdown for Hard / Critical Flights:**
- Precision on "Hard" Flights: **97.0%**
- Recall on "Hard" Flights: **91.0%**
- F1-Score on "Hard" Flights: **0.940**

---

### 3.2 Pre-Departure Delay & Turnaround Risk Prediction (Strictly Pre-Flight T-2h)
Predicting whether a flight will experience severe operational delay ($>15$ minutes) *before departure*, using only booked passenger counts, baggage counts, aircraft seats, buffer time, and station congestion:

| Pre-Departure Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Pre-Flight)** | **0.5877** | **0.5974** | **0.5877** | **0.5841** | **0.6429** |
| **Random Forest (Pre-Flight)** | 0.5883 | 0.6036 | 0.5883 | 0.5812 | 0.6406 |
| **LightGBM (Pre-Flight)** | 0.5981 | 0.6043 | 0.5981 | 0.5968 | 0.6339 |
| **Logistic Regression (Pre-Flight)** | 0.5698 | 0.5747 | 0.5698 | 0.5686 | 0.6089 |

#### Operational Decision Threshold Tuning (Triage Matrix):
In operational dispatch, missing a flight delay (False Negative) has an order of magnitude higher operational cost than a precautionary ground alert (False Positive).

| Decision Threshold | Detection Recall (Delayed Flights) | Alert Precision | High-Risk F1 | Strategic Deployment Role |
| :---: | :---: | :---: | :---: | :--- |
| **0.30** | **95.9%** | 53.6% | 0.688 | **High-Sensitivity Safety Net** (Morning hub bank peak) |
| **0.40 (Recommended)** | **79.7%** | **57.8%** | **0.670** | **Balanced Dispatch Operating Point** |
| **0.50** | 49.2% | 63.7% | 0.555 | Conservative Alerting |
| **0.60** | 28.1% | 73.0% | 0.406 | Critical Escalation Only |

---

## 4. TreeSHAP Explainability & Empirical Weight Validation

### 4.1 Challenging the Heuristic 25/50/25 Weighting
Using TreeSHAP on 8,099 flights, we computed the empirical relative contribution of each operational dimension:

| Operational Dimension | Heuristic Hand-Picked Weight (%) | SHAP Empirical Learned Weight (%) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Delay & Turnaround Buffer** | 25.0% | **9.2%** | Delay is an *outcome*, while buffer is the primary shock absorber. |
| **Passenger & SSR Density** | 50.0% | **69.1%** | Passenger load, children, and wheelchair SSR dominate boarding time variance. |
| **Baggage Logistics** | 25.0% | **21.7%** | Hot-transfer baggage volume is critical on tight-connection turnarounds. |

### 4.2 Top Empirical Operational Drivers
1. **`passenger_complexity_score`** (Mean |SHAP| = 2.77)
2. **`bag_complexity_score`** (Mean |SHAP| = 0.86)
3. **`delay_complexity_score`** (Mean |SHAP| = 0.24)
4. **`child_ratio`** (Mean |SHAP| = 0.115)
5. **`stroller_ratio`** (Mean |SHAP| = 0.045)
6. **`transfer_ratio`** (Mean |SHAP| = 0.044)
7. **`ssr_total`** (Special Service Requests: Wheelchairs & Minors) (Mean |SHAP| = 0.032)
8. **`turnaround_tightness_ratio`** (Mean |SHAP| = 0.027)

---

## 5. Statistical Rigor & Unsupervised Network Archetypes

### 5.1 Hypothesis Testing Results
- **One-Way ANOVA across Airline Carriers**:
  $$F = 662.39, \quad p < 10^{-15}, \quad \eta^2 = 0.451$$
  *Conclusion*: Statistically proves that operational complexity is heavily dependent on carrier operating models (Express regional partners face significantly tighter turns and smaller buffers than Mainline aircraft).
- **Chi-Square Test of Independence (Carrier vs Complexity Tier)**:
  $$\chi^2 = 1292.98, \quad p = 3.58 \times 10^{-276}, \quad \text{Cramér's } V = 0.282$$
  *Conclusion*: Confirms that high-complexity flights are clustered non-randomly within specific airline fleets.

### 5.2 Unsupervised Route Archetypes (K-Means, $k=3$, Silhouette = 0.239)
1. **Cluster 0: "Turnaround-Squeezed Hub Feeders"**: Short turnaround times, moderate baggage transfer ratios, high frequency connecting into major hubs (e.g., ORD-ALB, ORD-ABQ).
2. **Cluster 1: "High-Risk Transfer Bottlenecks"**: High hot-transfer baggage volume, tight connection windows, international customs processing requirements.
3. **Cluster 2: "Standard Low-Stress Routes"**: Generous buffer times ($>45$ mins), primarily origin/destination point-to-point traffic.

---

## 6. Business Impact & Estimated Financial ROI

According to the **FAA / Airlines for America (A4A)** aviation economic index:
- Direct aircraft operating cost of delay: **$47.20 / minute** (crew duty overtime, fuel burn while taxiing or holding, auxiliary power unit usage).
- Passenger value of lost time: **$54.20 / minute**.
- Total societal & operational economic impact: **$101.40 / minute of delay**.

### Operational Cost Savings Model
- **Fleet Scope**: 8,099 scheduled flights.
- **Observed Delayed Flights ($>15$ mins)**: $52.3\%$ ($4,236$ flights).
- **Average Delay on Delayed Flights**: $38$ minutes.
- **Annualized Delay Minutes**: $4,236 \times 38 = 160,968$ minutes.

If the **Pre-Departure Risk Prediction System** and frontline mitigation playbooks prevent just **$5\%$ of turnaround delays** through proactive ramp staffing, prioritized baggage transfer tugs, and early SSR wheelchair staging:
$$\text{Delay Minutes Prevented} = 160,968 \times 0.05 = 8,048 \text{ minutes}$$
$$\text{Direct Airline Cost Savings} = 8,048 \times \$47.20 = \mathbf{\$379,865}$$
$$\text{Total Economic Impact} = 8,048 \times \$101.40 = \mathbf{\$816,067}$$

**Additional Ancillary Savings:**
- **Mishandled Baggage Claims**: Average claim cost is $\$300 - \$500$ per delayed bag. Preventing hot-transfer connection misses across 500 bags saves an additional **$\$150,000 - \$250,000$**.
- **Crew Timeout Avoidance**: Preventing downstream flight cancellations due to FAA flight duty period limits avoids cancellation penalties of **$\$10,000 - \$25,000$ per flight**.

---

## 7. Frontline Operations Playbook (Standard Operating Procedures)

### Tier 1: Easy / Low Risk (Probability $< 0.35$)
- Normal gate agent staffing (1 agent).
- Standard baggage loading sequence.
- Standard pushback sequencing.

### Tier 2: Medium Risk (Probability $0.35 - 0.60$)
- Alert gate crew 30 minutes prior to inbound arrival.
- Assign dedicated baggage tug runner for connecting bags.
- Pre-board passengers with children and strollers 5 minutes early.

### Tier 3: Hard / Critical Risk (Probability $\ge 0.60$)
- **Dual-Agent Gate Staffing**: One dedicated exclusively to pre-boarding wheelchair SSR passengers.
- **Pre-Staged Hot Baggage Cart**: Transfer tug staged at gate prior to aircraft block-in.
- **Ramp Supervisor Presence**: Active ground lead monitoring turnaround critical path.
- **Airport Station Control Coordination**: Priority pushback clearance locked with ATC.
