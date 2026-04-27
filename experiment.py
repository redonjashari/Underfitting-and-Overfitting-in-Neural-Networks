import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def make_pipeline(hidden, lr=0.01, alpha=0.0001):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            max_iter=5000,
            random_state=SEED,
            learning_rate_init=lr,
            alpha=alpha,
        ))
    ])

MODEL_SPECS = {
    "Underfitting\n(1×2)":      dict(hidden=(2,),     lr=0.01,  alpha=0.0001),
    "Good Fit\n(2×16)":         dict(hidden=(16, 16), lr=0.01,  alpha=0.0001),
    "Overfitting\n(5×64)":      dict(hidden=(64,)*5,  lr=0.001, alpha=0.0001),
    "Regularised\n(5×64 + L2)": dict(hidden=(64,)*5,  lr=0.001, alpha=0.1),
}
COLORS       = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
TITLES_SHORT = ["(a) Underfitting", "(b) Good Fit",
                "(c) Overfitting",  "(d) Regularised"]
SHORT_LABELS = ["Underfitting", "Good Fit", "Overfitting", "Regularised"]

def train_models(X_train, X_test, y_train, y_test):
    results = {}
    for (name, spec), color in zip(MODEL_SPECS.items(), COLORS):
        pipe = make_pipeline(**spec)
        pipe.fit(X_train, y_train)
        pred_train = pipe.predict(X_train)
        pred_test  = pipe.predict(X_test)
        results[name] = {
            "pipe":       pipe,
            "train_mse":  mean_squared_error(y_train, pred_train),
            "test_mse":   mean_squared_error(y_test,  pred_test),
            "pred_train": pred_train,
            "pred_test":  pred_test,
            "color":      color,
        }
    return results

def run_cv(X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    cv_out = {}
    for (name, spec), lbl in zip(MODEL_SPECS.items(), SHORT_LABELS):
        ftr, fts = [], []
        for tr, va in kf.split(X):
            p = make_pipeline(**spec)
            p.fit(X[tr], y[tr])
            ftr.append(mean_squared_error(y[tr], p.predict(X[tr])))
            fts.append(mean_squared_error(y[va], p.predict(X[va])))
        cv_out[lbl] = dict(train_mean=np.mean(ftr), train_std=np.std(ftr),
                           test_mean=np.mean(fts),  test_std=np.std(fts))
    return cv_out

# ═════════════════════════════════════════════════════════════════════════════
# 1. AUTO MPG DATASET
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("AUTO MPG")
print("=" * 60)

url_mpg = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/auto-mpg/auto-mpg.data"
)
col_names = ["mpg","cylinders","displacement","horsepower",
             "weight","acceleration","model_year","origin","car_name"]
df_mpg = pd.read_csv(url_mpg, names=col_names, sep=r"\s+", na_values="?")
df_mpg = df_mpg.dropna().drop(columns=["car_name"]).reset_index(drop=True)
print(f"Auto MPG shape: {df_mpg.shape}")

MPG_FEATURES = ["cylinders","displacement","horsepower",
                "weight","acceleration","model_year","origin"]
X_mpg = df_mpg[MPG_FEATURES].values.astype(float)
y_mpg = df_mpg["mpg"].values.astype(float)

X_mpg_tr, X_mpg_te, y_mpg_tr, y_mpg_te = train_test_split(
    X_mpg, y_mpg, test_size=0.2, random_state=SEED)

# ── EDA ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_mpg, bins=20, color="#5dade2", edgecolor="white", lw=0.6)
axes[0].set_xlabel("MPG", fontsize=11); axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title("(a) MPG Distribution", fontsize=12)
axes[0].grid(axis="y", alpha=0.3)

corr_mpg = df_mpg[MPG_FEATURES + ["mpg"]].corr()
im = axes[1].imshow(corr_mpg.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
lbls = ["cyl","disp","hp","wt","acc","yr","org","mpg"]
axes[1].set_xticks(range(8)); axes[1].set_xticklabels(lbls, fontsize=8, rotation=45)
axes[1].set_yticks(range(8)); axes[1].set_yticklabels(lbls, fontsize=8)
axes[1].set_title("(b) Correlation Matrix", fontsize=12)
for i in range(8):
    for j in range(8):
        axes[1].text(j, i, f"{corr_mpg.values[i,j]:.2f}", ha="center", va="center",
                     fontsize=6, color="white" if abs(corr_mpg.values[i,j]) > 0.6 else "black")
plt.tight_layout()
plt.savefig("plot_mpg_eda.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_mpg_eda.png")

# ── Feature importance ────────────────────────────────────────────────────────
pearson_mpg = df_mpg[MPG_FEATURES].corrwith(df_mpg["mpg"]).abs().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.barh(pearson_mpg.index, pearson_mpg.values, color="#e59866", edgecolor="white")
for bar, val in zip(bars, pearson_mpg.values):
    ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
ax.set_xlabel("|Pearson r| with MPG", fontsize=11)
ax.set_title("Feature Importance — Auto MPG", fontsize=12)
ax.set_xlim(0, 1.1); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_mpg_feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_mpg_feature_importance.png")

# ── Train & evaluate ──────────────────────────────────────────────────────────
res_mpg = train_models(X_mpg_tr, X_mpg_te, y_mpg_tr, y_mpg_te)
for lbl, res in zip(SHORT_LABELS, res_mpg.values()):
    print(f"  {lbl:<15} train={res['train_mse']:.3f}  test={res['test_mse']:.3f}")

# ── Predictions scatter ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True, sharex=True)
for ax, (name, res), title in zip(axes, res_mpg.items(), TITLES_SHORT):
    ax.scatter(y_mpg_tr, res["pred_train"], s=14, alpha=0.45, color="gray", label="Train")
    ax.scatter(y_mpg_te, res["pred_test"],  s=14, alpha=0.7,  color="black", marker="^", label="Test")
    lims = [y_mpg.min()-2, y_mpg.max()+2]
    ax.plot(lims, lims, "r--", lw=1.2, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Actual MPG", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("Predicted MPG", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_mpg_predictions.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_mpg_predictions.png")

# ── Loss curves ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, res), title, color in zip(axes, res_mpg.items(), TITLES_SHORT, COLORS):
    ax.plot(res["pipe"]["mlp"].loss_curve_, color=color, lw=1.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Epoch", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("Training Loss (MSE)", fontsize=9)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_mpg_loss_curves.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_mpg_loss_curves.png")

# ── MSE bar chart ─────────────────────────────────────────────────────────────
tr_mse = [r["train_mse"] for r in res_mpg.values()]
te_mse = [r["test_mse"]  for r in res_mpg.values()]
x = np.arange(4); w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
b1 = ax.bar(x-w/2, tr_mse, w, label="Train MSE", color="#5dade2", edgecolor="white")
b2 = ax.bar(x+w/2, te_mse, w, label="Test MSE",  color="#e59866", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("MSE (MPG²)"); ax.set_title("Auto MPG — Train vs. Test MSE", fontsize=12)
ax.legend(); ax.grid(axis="y", alpha=0.3)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("plot_mpg_mse.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_mpg_mse.png")

# ── Cross-validation ──────────────────────────────────────────────────────────
cv_mpg = run_cv(X_mpg, y_mpg)
x = np.arange(4); w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
cv_tr = [cv_mpg[l]["train_mean"] for l in SHORT_LABELS]
cv_te = [cv_mpg[l]["test_mean"]  for l in SHORT_LABELS]
cv_te_std = [cv_mpg[l]["test_std"] for l in SHORT_LABELS]
ax.bar(x-w/2, cv_tr, w, label="CV Train MSE", color="#5dade2", edgecolor="white")
ax.bar(x+w/2, cv_te, w, label="CV Test MSE",  color="#e59866", edgecolor="white",
       yerr=cv_te_std, capsize=4, error_kw={"elinewidth":1.2})
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("MSE (MPG²)"); ax.set_title("Auto MPG — 10-Fold CV MSE (mean ± std)", fontsize=12)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_mpg_crossval.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_mpg_crossval.png")

# ── Learning curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, (name, spec), color, title in zip(axes, MODEL_SPECS.items(), COLORS, TITLES_SHORT):
    pipe = make_pipeline(**spec)
    tsz, tsc, vsc = learning_curve(pipe, X_mpg, y_mpg,
                                   train_sizes=np.linspace(0.1,1.0,8),
                                   cv=5, scoring="neg_mean_squared_error",
                                   random_state=SEED, n_jobs=-1)
    tr_lc = -tsc.mean(axis=1); va_lc = -vsc.mean(axis=1); va_std = vsc.std(axis=1)
    ax.plot(tsz, tr_lc, "o-", color=color,   lw=1.8, label="Train MSE")
    ax.plot(tsz, va_lc, "s--", color="black", lw=1.8, label="Val MSE")
    ax.fill_between(tsz, va_lc-va_std, va_lc+va_std, alpha=0.15, color="black")
    ax.set_title(title, fontsize=10); ax.set_xlabel("Training size", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("MSE (MPG²)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_mpg_learning_curves.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_mpg_learning_curves.png")

# ── Hyperparameter sensitivity ────────────────────────────────────────────────
unit_counts = [2, 4, 8, 16, 32, 64, 128]
hp_tr_mpg, hp_te_mpg = [], []
for n in unit_counts:
    p = make_pipeline(hidden=(n, n))
    p.fit(X_mpg_tr, y_mpg_tr)
    hp_tr_mpg.append(mean_squared_error(y_mpg_tr, p.predict(X_mpg_tr)))
    hp_te_mpg.append(mean_squared_error(y_mpg_te, p.predict(X_mpg_te)))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(unit_counts, hp_tr_mpg, "o-", color="#5dade2", lw=2, label="Train MSE")
ax.plot(unit_counts, hp_te_mpg, "s-", color="#e59866", lw=2, label="Test MSE")
ax.set_xlabel("Units per hidden layer", fontsize=11)
ax.set_ylabel("MSE (MPG²)", fontsize=11)
ax.set_title("Hyperparameter Sensitivity — Auto MPG", fontsize=12)
ax.set_xticks(unit_counts); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_mpg_hyperparam.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_mpg_hyperparam.png")


# ═════════════════════════════════════════════════════════════════════════════
# 2. NYC TAXI DATASET  (NYC TLC Green Taxi, Jan 2023)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("NYC TAXI")
print("=" * 60)

url_taxi = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data"
    "/green_tripdata_2023-01.parquet"
)
print("Downloading NYC Taxi data (this may take a moment)...")
df_taxi = pd.read_parquet(url_taxi)
print(f"Raw taxi shape: {df_taxi.shape}")

# ── Feature engineering ───────────────────────────────────────────────────────
df_taxi = df_taxi.copy()
df_taxi["duration_min"] = (
    df_taxi["lpep_dropoff_datetime"] - df_taxi["lpep_pickup_datetime"]
).dt.total_seconds() / 60
df_taxi["pickup_hour"]      = df_taxi["lpep_pickup_datetime"].dt.hour
df_taxi["pickup_dayofweek"] = df_taxi["lpep_pickup_datetime"].dt.dayofweek

TAXI_FEATURES = ["trip_distance", "passenger_count", "PULocationID",
                 "DOLocationID", "pickup_hour", "pickup_dayofweek", "RatecodeID"]
TAXI_TARGET   = "duration_min"

df_taxi = (df_taxi[TAXI_FEATURES + [TAXI_TARGET]]
           .dropna()
           .query("1 < duration_min < 120 and trip_distance > 0 and passenger_count > 0")
           .reset_index(drop=True))

# Sample to keep training fast; increase N_SAMPLE for more thorough results
N_SAMPLE = 15000
if len(df_taxi) > N_SAMPLE:
    df_taxi = df_taxi.sample(N_SAMPLE, random_state=SEED).reset_index(drop=True)
print(f"Taxi shape after cleaning & sampling: {df_taxi.shape}")

X_taxi = df_taxi[TAXI_FEATURES].values.astype(float)
y_taxi = df_taxi[TAXI_TARGET].values.astype(float)

X_taxi_tr, X_taxi_te, y_taxi_tr, y_taxi_te = train_test_split(
    X_taxi, y_taxi, test_size=0.2, random_state=SEED)

# ── EDA ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_taxi, bins=40, color="#5dade2", edgecolor="white", lw=0.6)
axes[0].set_xlabel("Trip Duration (min)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title("(a) Duration Distribution", fontsize=12)
axes[0].grid(axis="y", alpha=0.3)

corr_taxi = df_taxi[TAXI_FEATURES + [TAXI_TARGET]].corr()
feat_labels = ["dist","pax","PU","DO","hr","dow","rate","dur"]
im = axes[1].imshow(corr_taxi.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
axes[1].set_xticks(range(8)); axes[1].set_xticklabels(feat_labels, fontsize=8, rotation=45)
axes[1].set_yticks(range(8)); axes[1].set_yticklabels(feat_labels, fontsize=8)
axes[1].set_title("(b) Correlation Matrix", fontsize=12)
for i in range(8):
    for j in range(8):
        axes[1].text(j, i, f"{corr_taxi.values[i,j]:.2f}", ha="center", va="center",
                     fontsize=6, color="white" if abs(corr_taxi.values[i,j]) > 0.6 else "black")
plt.tight_layout()
plt.savefig("plot_taxi_eda.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_taxi_eda.png")

# ── Feature importance ────────────────────────────────────────────────────────
pearson_taxi = df_taxi[TAXI_FEATURES].corrwith(df_taxi[TAXI_TARGET]).abs().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.barh(pearson_taxi.index, pearson_taxi.values, color="#5dade2", edgecolor="white")
for bar, val in zip(bars, pearson_taxi.values):
    ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
ax.set_xlabel("|Pearson r| with Duration", fontsize=11)
ax.set_title("Feature Importance — NYC Taxi", fontsize=12)
ax.set_xlim(0, 0.75); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_taxi_feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_taxi_feature_importance.png")

# ── Train & evaluate ──────────────────────────────────────────────────────────
res_taxi = train_models(X_taxi_tr, X_taxi_te, y_taxi_tr, y_taxi_te)
for lbl, res in zip(SHORT_LABELS, res_taxi.values()):
    print(f"  {lbl:<15} train={res['train_mse']:.3f}  test={res['test_mse']:.3f}")

# ── Predictions scatter ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True, sharex=True)
for ax, (name, res), title in zip(axes, res_taxi.items(), TITLES_SHORT):
    ax.scatter(y_taxi_tr, res["pred_train"], s=6, alpha=0.25, color="gray",  label="Train")
    ax.scatter(y_taxi_te, res["pred_test"],  s=6, alpha=0.4,  color="black", marker="^", label="Test")
    lims = [y_taxi.min()-2, y_taxi.max()+2]
    ax.plot(lims, lims, "r--", lw=1.2, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Actual (min)", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("Predicted (min)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_taxi_predictions.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_taxi_predictions.png")

# ── Loss curves ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, res), title, color in zip(axes, res_taxi.items(), TITLES_SHORT, COLORS):
    ax.plot(res["pipe"]["mlp"].loss_curve_, color=color, lw=1.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Epoch", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("Training Loss (MSE)", fontsize=9)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_taxi_loss_curves.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_taxi_loss_curves.png")

# ── MSE bar chart ─────────────────────────────────────────────────────────────
tr_mse_t = [r["train_mse"] for r in res_taxi.values()]
te_mse_t = [r["test_mse"]  for r in res_taxi.values()]
x = np.arange(4); w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
b1 = ax.bar(x-w/2, tr_mse_t, w, label="Train MSE", color="#5dade2", edgecolor="white")
b2 = ax.bar(x+w/2, te_mse_t, w, label="Test MSE",  color="#e59866", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("MSE (min²)"); ax.set_title("NYC Taxi — Train vs. Test MSE", fontsize=12)
ax.legend(); ax.grid(axis="y", alpha=0.3)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("plot_taxi_mse.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_taxi_mse.png")

# ── Cross-validation ──────────────────────────────────────────────────────────
cv_taxi = run_cv(X_taxi, y_taxi)
x = np.arange(4); w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
cv_tr_t    = [cv_taxi[l]["train_mean"] for l in SHORT_LABELS]
cv_te_t    = [cv_taxi[l]["test_mean"]  for l in SHORT_LABELS]
cv_te_std_t= [cv_taxi[l]["test_std"]   for l in SHORT_LABELS]
ax.bar(x-w/2, cv_tr_t, w, label="CV Train MSE", color="#5dade2", edgecolor="white")
ax.bar(x+w/2, cv_te_t, w, label="CV Test MSE",  color="#e59866", edgecolor="white",
       yerr=cv_te_std_t, capsize=4, error_kw={"elinewidth":1.2})
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("MSE (min²)"); ax.set_title("NYC Taxi — 10-Fold CV MSE (mean ± std)", fontsize=12)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_taxi_crossval.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_taxi_crossval.png")

# ── Learning curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, (name, spec), color, title in zip(axes, MODEL_SPECS.items(), COLORS, TITLES_SHORT):
    pipe = make_pipeline(**spec)
    tsz, tsc, vsc = learning_curve(pipe, X_taxi, y_taxi,
                                   train_sizes=np.linspace(0.1,1.0,8),
                                   cv=5, scoring="neg_mean_squared_error",
                                   random_state=SEED, n_jobs=-1)
    tr_lc = -tsc.mean(axis=1); va_lc = -vsc.mean(axis=1); va_std = vsc.std(axis=1)
    ax.plot(tsz, tr_lc, "o-", color=color,   lw=1.8, label="Train MSE")
    ax.plot(tsz, va_lc, "s--", color="black", lw=1.8, label="Val MSE")
    ax.fill_between(tsz, va_lc-va_std, va_lc+va_std, alpha=0.15, color="black")
    ax.set_title(title, fontsize=10); ax.set_xlabel("Training size", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("MSE (min²)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_taxi_learning_curves.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_taxi_learning_curves.png")

# ── Hyperparameter sensitivity ────────────────────────────────────────────────
hp_tr_taxi, hp_te_taxi = [], []
for n in unit_counts:
    p = make_pipeline(hidden=(n, n))
    p.fit(X_taxi_tr, y_taxi_tr)
    hp_tr_taxi.append(mean_squared_error(y_taxi_tr, p.predict(X_taxi_tr)))
    hp_te_taxi.append(mean_squared_error(y_taxi_te, p.predict(X_taxi_te)))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(unit_counts, hp_tr_taxi, "o-", color="#5dade2", lw=2, label="Train MSE")
ax.plot(unit_counts, hp_te_taxi, "s-", color="#e59866", lw=2, label="Test MSE")
ax.set_xlabel("Units per hidden layer", fontsize=11)
ax.set_ylabel("MSE (min²)", fontsize=11)
ax.set_title("Hyperparameter Sensitivity — NYC Taxi", fontsize=12)
ax.set_xticks(unit_counts); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_taxi_hyperparam.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_taxi_hyperparam.png")


# ═════════════════════════════════════════════════════════════════════════════
# 3. COMPARISON PLOT  (train-test MSE gap side by side)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Normalise MSE by the variance of the target for fair comparison
var_mpg  = np.var(y_mpg)
var_taxi = np.var(y_taxi)

gap_mpg  = [(res["test_mse"]  - res["train_mse"]) / var_mpg
            for res in res_mpg.values()]
gap_taxi = [(res["test_mse"]  - res["train_mse"]) / var_taxi
            for res in res_taxi.values()]

x = np.arange(4); w = 0.35

for ax, gaps, title, unit in zip(
        axes,
        [gap_mpg, gap_taxi],
        ["Auto MPG", "NYC Taxi"],
        ["(relative to target variance)", "(relative to target variance)"]):
    colors_bar = ["#e74c3c" if g > 0 else "#2ecc71" for g in gaps]
    ax.bar(x, gaps, color=colors_bar, edgecolor="white", width=0.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=9)
    ax.set_ylabel("Normalised (Test − Train) MSE", fontsize=9)
    ax.set_title(f"{title} — Generalisation Gap", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot_comparison_gap.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_comparison_gap.png")

# ── Hyperparameter comparison side-by-side ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (hp_tr, hp_te), title, ylabel in zip(
        axes,
        [(hp_tr_mpg, hp_te_mpg), (hp_tr_taxi, hp_te_taxi)],
        ["Auto MPG", "NYC Taxi"],
        ["MSE (MPG²)", "MSE (min²)"]):
    ax.plot(unit_counts, hp_tr, "o-", color="#5dade2", lw=2, label="Train MSE")
    ax.plot(unit_counts, hp_te, "s-", color="#e59866", lw=2, label="Test MSE")
    ax.set_xlabel("Units per hidden layer", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"Hyperparameter Sensitivity — {title}", fontsize=11)
    ax.set_xticks(unit_counts); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_comparison_hyperparam.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved plot_comparison_hyperparam.png")


# ═════════════════════════════════════════════════════════════════════════════
# 4. TABLE VALUES
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("=" * 60)
print("\n--- Auto MPG single-split MSE ---")
for (name, res), lbl in zip(res_mpg.items(), SHORT_LABELS):
    print(f"  {lbl:<15} Train: {res['train_mse']:.3f}   Test: {res['test_mse']:.3f}")

print("\n--- Auto MPG 10-fold CV MSE ---")
for lbl in SHORT_LABELS:
    r = cv_mpg[lbl]
    print(f"  {lbl:<15} Train: {r['train_mean']:.3f}±{r['train_std']:.3f}  "
          f"Test: {r['test_mean']:.3f}±{r['test_std']:.3f}")

print("\n--- NYC Taxi single-split MSE ---")
for (name, res), lbl in zip(res_taxi.items(), SHORT_LABELS):
    print(f"  {lbl:<15} Train: {res['train_mse']:.3f}   Test: {res['test_mse']:.3f}")

print("\n--- NYC Taxi 10-fold CV MSE ---")
for lbl in SHORT_LABELS:
    r = cv_taxi[lbl]
    print(f"  {lbl:<15} Train: {r['train_mean']:.3f}±{r['train_std']:.3f}  "
          f"Test: {r['test_mean']:.3f}±{r['test_std']:.3f}")
