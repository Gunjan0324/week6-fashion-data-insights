"""End-to-end pipeline: cleaning -> EDA -> forecasting model -> returns classifier -> evaluation."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
PAL = ["#1F3A5F", "#C8553D", "#588B8B", "#F2A541", "#6D597A", "#8D99AE"]
M = {}

# ---------- 1. Cleaning ----------
raw = pd.read_csv("data/raw_sales.csv", parse_dates=["week"])
M["raw_rows"] = len(raw)
df = raw.drop_duplicates().copy(); M["duplicates_removed"] = M["raw_rows"] - len(df)
df["region"] = df["region"].str.title()
M["missing_marketing"] = int(df["marketing_spend"].isna().sum())
df["marketing_spend"] = df.groupby(["category", "channel"])["marketing_spend"].transform(lambda s: s.fillna(s.median()))
df["revenue"] = df["units_sold"] * df["unit_price"]
df["return_rate"] = df["returns"] / df["units_sold"].replace(0, np.nan)
df["net_revenue"] = (df["units_sold"] - df["returns"]) * df["unit_price"]
df["month"] = df["week"].dt.month; df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)
df = df.sort_values("week").reset_index(drop=True)
df.to_csv("data/clean_sales.csv", index=False)
M["clean_rows"] = len(df)
M["total_revenue"] = float(df["revenue"].sum()); M["net_revenue"] = float(df["net_revenue"].sum())
M["overall_return_rate"] = float(df["returns"].sum() / df["units_sold"].sum())

# ---------- 2. EDA / visuals ----------
cat_rev = df.groupby("category")["revenue"].sum().sort_values()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.barh(cat_rev.index, cat_rev.values/1e6, color=PAL[0]); ax.set_xlabel("Revenue (millions)")
ax.set_title("Revenue by category (2 years)", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig1_revenue_by_category.png"); plt.close()
M["rev_by_category"] = {k: float(v) for k, v in cat_rev[::-1].items()}

mon = df.pivot_table(index="month", columns="category", values="units_sold", aggfunc="sum")
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for i, c in enumerate(["Outerwear", "Dresses", "Activewear", "Tops"]):
    ax.plot(mon.index, mon[c], label=c, color=PAL[i], lw=2)
ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylabel("Units sold (2 yrs)")
ax.set_title("Seasonality by category", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=4, fontsize=8); plt.tight_layout()
plt.savefig("figures/fig2_seasonality.png"); plt.close()
M["peak_month"] = {c: int(mon[c].idxmax()) for c in mon.columns}

dd = df.groupby("discount").agg(units=("units_sold", "mean"), ret=("return_rate", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(6.5, 3.4)); ax2 = ax.twinx()
ax.bar(dd["discount"]*100, dd["units"], width=6, color=PAL[2]); ax2.plot(dd["discount"]*100, dd["ret"]*100, color=PAL[1], marker="o", lw=2)
ax.set_xlabel("Discount (%)"); ax.set_ylabel("Avg units per row", color=PAL[2]); ax2.set_ylabel("Return rate (%)", color=PAL[1])
ax2.spines["right"].set_visible(True)
ax.set_title("Discount vs volume and return rate", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig3_discount_effect.png"); plt.close()
M["discount_table"] = {f"{int(round(r.discount*100))}%": {"avg_units": round(float(r.units), 1), "return_rate_pct": round(float(r.ret)*100, 2)} for r in dd.itertuples()}

ch = df.groupby("channel").agg(rev=("revenue", "sum"), rr=("returns", "sum"), u=("units_sold", "sum"))
M["channel"] = {k: {"revenue": float(v.rev), "return_rate_pct": float(v.rr/v.u*100)} for k, v in ch.iterrows()}
cr = df.groupby("category").apply(lambda g: g["returns"].sum()/g["units_sold"].sum()*100, include_groups=False).sort_values()
M["return_rate_by_category"] = {k: float(v) for k, v in cr[::-1].items()}

# ---------- 3. Forecasting model (units_sold) ----------
feat_num = ["list_price", "discount", "marketing_spend", "holiday_week", "month", "weekofyear"]
cats_ = ["category", "region", "channel"]
X = pd.get_dummies(df[feat_num + cats_], columns=cats_, drop_first=True).astype(float)
y = df["units_sold"]
cut = np.sort(df["week"].unique())[80]
tr, te = df["week"] < cut, df["week"] >= cut
M["train_rows"], M["test_rows"] = int(tr.sum()), int(te.sum())

def reg_metrics(yt, yp):
    return {"MAE": float(mean_absolute_error(yt, yp)), "RMSE": float(mean_squared_error(yt, yp)**0.5),
            "R2": float(r2_score(yt, yp)), "WAPE_pct": float(np.abs(yt-yp).sum()/yt.sum()*100)}
models = {"Baseline (mean)": DummyRegressor(), "Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(300, min_samples_leaf=3, random_state=42, n_jobs=-1),
          "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
res, preds = {}, {}
for n, mdl in models.items():
    mdl.fit(X[tr], y[tr]); p = mdl.predict(X[te]); preds[n] = p; res[n] = reg_metrics(y[te], p)
M["regression_results"] = res
best = min([k for k in res if k != "Baseline (mean)"], key=lambda k: res[k]["RMSE"]); M["best_model"] = best

cv = []
for a, b in TimeSeriesSplit(5).split(X):
    m_ = type(models[best])(**models[best].get_params()).fit(X.iloc[a], y.iloc[a])
    cv.append(r2_score(y.iloc[b], m_.predict(X.iloc[b])))
M["cv_r2_folds"] = [float(v) for v in cv]; M["cv_r2_mean"] = float(np.mean(cv)); M["cv_r2_std"] = float(np.std(cv))

fig, ax = plt.subplots(figsize=(6.5, 3.4)); names = list(res)
ax.bar(names, [res[n]["RMSE"] for n in names], color=[PAL[5], PAL[2], PAL[0], PAL[1]])
ax.set_ylabel("RMSE (units, lower is better)"); plt.xticks(rotation=12, fontsize=8)
ax.set_title("Model comparison on held-out weeks", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig4_model_comparison.png"); plt.close()

wk = df[te].assign(pred=preds[best]).groupby("week")[["units_sold", "pred"]].sum()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(wk.index, wk["units_sold"], color=PAL[0], lw=2, label="Actual"); ax.plot(wk.index, wk["pred"], color=PAL[1], lw=2, ls="--", label="Forecast")
ax.set_ylabel("Total weekly units"); ax.legend(frameon=False)
ax.set_title(f"{best}: forecast vs actual (test period)", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig5_actual_vs_forecast.png"); plt.close()
M["weekly_agg_wape_pct"] = float(np.abs(wk["units_sold"]-wk["pred"]).sum()/wk["units_sold"].sum()*100)

imp = pd.Series(models[best].feature_importances_, index=X.columns).sort_values()[-8:]
fig, ax = plt.subplots(figsize=(6.5, 3.4)); ax.barh(imp.index, imp.values, color=PAL[2])
ax.set_title("Top demand drivers (feature importance)", loc="left", fontweight="bold"); plt.tight_layout()
plt.savefig("figures/fig6_feature_importance.png"); plt.close()
M["top_features"] = {k: float(v) for k, v in imp[::-1].items()}

# ---------- 4. Returns classifier ----------
dfc = df.dropna(subset=["return_rate"]).copy()
dfc["high_return"] = (dfc["return_rate"] > dfc["return_rate"].median()).astype(int)
Xc = pd.get_dummies(dfc[feat_num + cats_], columns=cats_, drop_first=True).astype(float)
yc = dfc["high_return"]; trc, tec = dfc["week"] < cut, dfc["week"] >= cut
clfs = {"Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(300, min_samples_leaf=3, random_state=42, n_jobs=-1)}
cres = {}
for n, c in clfs.items():
    c.fit(Xc[trc], yc[trc]); p = c.predict(Xc[tec])
    auc = 0.5 if n.startswith("Baseline") else roc_auc_score(yc[tec], c.predict_proba(Xc[tec])[:, 1])
    cres[n] = {"Accuracy": float(accuracy_score(yc[tec], p)), "Precision": float(precision_score(yc[tec], p, zero_division=0)),
               "Recall": float(recall_score(yc[tec], p, zero_division=0)), "F1": float(f1_score(yc[tec], p, zero_division=0)),
               "ROC_AUC": float(auc)}
M["classification_results"] = cres
cm = confusion_matrix(yc[tec], clfs["Random Forest"].predict(Xc[tec])); M["confusion_matrix_rf"] = cm.tolist()
fig, ax = plt.subplots(figsize=(3.8, 3.2)); ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max()/2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Low", "High"]); ax.set_yticklabels(["Low", "High"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Return-risk confusion matrix", fontsize=10, fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig7_confusion_matrix.png"); plt.close()

json.dump(M, open("reports/metrics.json", "w"), indent=2)
print(json.dumps(M, indent=1))
