# 🍭 Factory Reallocation & Shipping Optimization
### Nassau Candy Distributor — Decision Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

> Built by **Bhawna Singh** as part of the Unified Mentor Internship Program

---

## 📌 Project Overview

Nassau Candy Distributor assigns 15 candy products to 5 manufacturing facilities using static legacy rules — leading to suboptimal shipping distances, high lead times, and margin erosion.

This project builds a **data-driven decision intelligence system** that recommends optimal factory-product reassignments using Transportation Linear Programming, geospatial distance modelling, and demand forecasting.

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| Orders Analysed | 10,194 |
| Products Optimised | 15 |
| Reassignments Recommended | 12 |
| Total Distance Saving | 11,597 km |
| Transport Cost Saving | 1,815,762 km·units |
| LP Solver Status | ✅ Optimal |

---

## 🗂️ Project Structure

```
Nassau-Candy-Optimization/
│
├── App.py                        # Main Streamlit application
├── Nassau_Candy_Distributor.csv  # Source dataset (10,194 orders)
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 📊 Overview & EDA | KPIs, lead time analysis, profit margins, ML honesty report |
| 🌍 Geospatial Distance | Haversine distance matrix, factory & region map |
| ⚙️ LP Optimization | Transportation LP results, reassignment table, capacity check |
| 🔮 Demand Forecasting | 6-month linear trend forecast per product |
| 🏆 Recommendations & Risk | Final reassignments, Sankey diagram, risk scorecard |

---

## 🔬 Analytical Methodology

- **Geospatial Modelling** — Haversine formula for factory-to-region distances
- **Transportation LP** — `scipy.optimize.linprog` with HiGHS solver
- **Machine Learning** — Linear Regression & Gradient Boosting (evaluated honestly)
- **Demand Forecasting** — Linear trend model on real monthly order data
- **Risk Scoring** — Based on observed margin variance and lead time consistency

---

## 🏭 The 5 Factories

| Factory | Products |
|---|---|
| Southwest Nut Processing Unit | Wonka Bar nut varieties |
| Southeast Chocolate Plant | Wonka Bar chocolate varieties |
| Northern Confectionery Facility | Laffy Taffy, SweeTARTS, Nerds, Fun Dip, Fizzy Lifting Drinks |
| Midwest Manufacturing Hub | Everlasting Gobstopper, Lickable Wallpaper, Wonka Gum |
| Central Distribution Centre | Hair Toffee, Kazookles |

> Factory names and product assignments are sourced from the official project specification — not from the raw CSV dataset.

---

## ⚙️ Installation & Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/Bhawna-Analytics/Nassau-Candy-Optimization.git
cd Nassau-Candy-Optimization

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run App.py
```

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
scipy>=1.11.0
```

---

## 🚀 Live Demo

👉 **[Open Live App on Streamlit Cloud](https://your-app-url.streamlit.app)**

---

## 👩‍💻 Author

**Bhawna Singh**
Unified Mentor Internship Program — June 2026
