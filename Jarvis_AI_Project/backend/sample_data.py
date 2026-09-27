"""
JARVIS AI - Sample datasets for instant demo/onboarding, no upload required.
"""
import numpy as np
import pandas as pd


def sales_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 250
    regions = ["North", "South", "East", "West"]
    products = ["Widget A", "Widget B", "Widget C", "Gadget X", "Gadget Y"]
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    units_sold = rng.integers(20, 200, n)
    unit_price = np.round(rng.uniform(10, 100, n), 2)
    revenue = np.round(units_sold * unit_price, 2)
    satisfaction = np.round(rng.uniform(2.5, 5.0, n), 1)

    df = pd.DataFrame({
        "month": rng.choice(months, n),
        "region": rng.choice(regions, n),
        "product": rng.choice(products, n),
        "units_sold": units_sold,
        "unit_price": unit_price,
        "revenue": revenue,
        "customer_satisfaction": satisfaction,
    })
    # sprinkle some missing values + a couple of extreme outliers, like real data
    miss_idx = rng.choice(n, 8, replace=False)
    df.loc[miss_idx, "customer_satisfaction"] = np.nan
    df.loc[3, "revenue"] = 25000
    df.loc[17, "revenue"] = 21000
    return df


def hr_sample() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 220
    depts = ["Engineering", "Sales", "Marketing", "Support", "HR"]

    df = pd.DataFrame({
        "employee_id": np.arange(1000, 1000 + n),
        "department": rng.choice(depts, n),
        "age": rng.integers(22, 62, n),
        "tenure_years": np.round(rng.uniform(0, 15, n), 1),
        "salary": (rng.uniform(35000, 130000, n) // 100 * 100).astype(int),
        "performance_score": np.round(rng.uniform(1, 5, n), 1),
        "attrition": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
    })
    df.loc[9, "salary"] = 350000
    df.loc[40, "salary"] = 310000
    miss_idx = rng.choice(n, 6, replace=False)
    df.loc[miss_idx, "performance_score"] = np.nan
    return df
