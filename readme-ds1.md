# DS Project 1 — Customer Clustering with K-Means Machine Learning

## Problem Statement
Manual customer segmentation is subjective and relies on human assumptions. This project uses K-Means Clustering to discover customer groupings that are natural and entirely data-driven.

## What We Did
Calculated RFM features from transaction data, normalized with StandardScaler, determined the optimal number of clusters using the Elbow Method and Silhouette Score, then ran K-Means with k=3. Each cluster was interpreted from a business perspective to produce actionable retention strategies.

## Tools Used
- Python (Pandas, Scikit-Learn, Matplotlib, Seaborn)
- Jupyter Notebook / Google Colab

## Dataset
- **Source:** UCI Online Retail Dataset
- **Size:** 541,909 transactions

## Key Results
| Metric | Value |
|--------|-------|
| Optimal Clusters | 3 |
| Algorithm | K-Means |
| Full Report | 40 pages |

## Cluster Breakdown
| Cluster | % of Customers | Avg Monetary | Profile |
|---------|---------------|--------------|---------|
| Cluster 1 — MVP | 19.40% | $5,599 | High value, frequent buyers |
| Cluster 2 — Mid Value | 26.40% | $1,131 | Regular buyers |
| Cluster 3 — Low Value | 24.20% | $189 | Inactive / one-time buyers |

## So What? — Actionable Insights

**Cluster 1 — MVP Customers**
- Introduce reward programs or membership tiers
- Provide VIP exclusive perks — early access, special discounts
- Use personalized recommendations based on purchase history
- Promote higher-value or premium products

**Cluster 2 — Mid Value Customers**
- Upselling: promote higher-value versions of products they already buy
- Cross-selling: recommend complementary products
- Offer bundles to encourage larger purchases
- Introduce loyalty rewards for frequent purchases

**Cluster 3 — Low Value Customers**
- Encourage second purchase through welcome-back promotions
- Offer discounts specifically for second purchase
- Use digital retargeting ads to bring them back

## 4-Week Action Plan
| Week | Focus | Action |
|------|-------|--------|
| Week 1 | Preparation | Segment customers, prepare campaigns, design promotions |
| Week 2 | Retain MVP | Launch VIP programs and exclusive promotions |
| Week 3 | Grow Mid-Value | Implement upselling and loyalty incentives |
| Week 4 | Reactivate Inactive | Run reactivation campaigns, offer comeback discounts |

> **Expected Outcome:** Protect 75% of revenue from high-value customers, increase mid-value spending, reactivate dormant buyers.

## Author
**Anthony Djiady Djie** — DS39+ Dibimbing.id
[ADJ Business Consulting](https://adjbusinessconsulting.github.io/adj-consulting)
