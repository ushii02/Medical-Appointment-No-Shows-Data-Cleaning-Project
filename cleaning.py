import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# ══════════════════════════════════════════════════════════════
# STEP 1 — Load & First Look
# ══════════════════════════════════════════════════════════════

df = pd.read_csv("KaggleV2-May-2016.csv")  
print("─" * 50)
print("STEP 1: RAW DATA OVERVIEW")
print("─" * 50)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nColumn names (raw):")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())


# ══════════════════════════════════════════════════════════════
# STEP 2 — Fix Column Names
# Problem: Inconsistent casing, typos (e.g. 'Hipertension')
# ══════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("STEP 2: STANDARDIZING COLUMN NAMES")
print("─" * 50)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace("-", "_")
    .str.replace(" ", "_")
)

# Rename specific columns for clarity
df.rename(columns={
    "patientid":        "patient_id",
    "appointmentid":    "appointment_id",
    "scheduledday":     "scheduled_date",
    "appointmentday":   "appointment_date",
    "hipertension":     "hypertension",      # fix typo
    "handcap":          "handicap",          # fix typo
    "no-show":          "no_show",
    "sms_received":     "sms_received",
}, inplace=True)

print("Cleaned column names:")
print(df.columns.tolist())


# ══════════════════════════════════════════════════════════════
# STEP 3 — Fix Data Types
# Problem: Date columns are stored as plain strings
# ══════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("STEP 3: FIXING DATA TYPES")
print("─" * 50)

# Convert date columns
df["scheduled_date"]  = pd.to_datetime(df["scheduled_date"],  errors="coerce")
df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")

# Extract useful features from dates
df["scheduled_day_of_week"]    = df["scheduled_date"].dt.day_name()
df["appointment_day_of_week"]  = df["appointment_date"].dt.day_name()
df["scheduled_month"]          = df["scheduled_date"].dt.month
df["waiting_days"] = (
    df["appointment_date"] - df["scheduled_date"]
).dt.days

print("Date columns converted ✓")
print(f"\nWaiting days sample:\n{df['waiting_days'].describe().round(1)}")


# ══════════════════════════════════════════════════════════════
# STEP 4 — Fix the No-Show Column
# Problem: 'No' means the patient DID show up — confusing!
# We'll create a clearer 'showed_up' column (1 = yes, 0 = no)
# ══════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("STEP 4: FIXING THE NO-SHOW COLUMN")
print("─" * 50)

print("Original no_show values:")
print(df["no_show"].value_counts())

# Reverse the logic: showed_up = 1 if no_show == 'No'
df["showed_up"] = (df["no_show"] == "No").astype(int)

print("\nNew showed_up column (1 = attended, 0 = no-show):")
print(df["showed_up"].value_counts())
print(f"\nNo-show rate: {(1 - df['showed_up'].mean()) * 100:.1f}%")


# ══════════════════════════════════════════════════════════════
# STEP 5 — Handle Outliers
# Problem: Age column has negative values and extreme outliers
# ══════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("STEP 5: HANDLING OUTLIERS IN AGE")
print("─" * 50)

print("Age stats (before):")
print(df["age"].describe().round(1))

# Flag invalid ages
invalid_ages = df[df["age"] < 0]
print(f"\nNegative ages found: {len(invalid_ages)} rows")
print(invalid_ages[["patient_id", "age"]])

# Plot before
plt.figure()
sns.histplot(df["age"], bins=40, color="#E53935")
plt.title("Age Distribution (Before Cleaning)")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("clean1_age_before.png", dpi=150)
plt.show()

# Remove rows with negative or unrealistic ages
df = df[(df["age"] >= 0) & (df["age"] <= 115)]

print(f"\nRows removed: {len(invalid_ages)}")
print("Age stats (after):")
print(df["age"].describe().round(1))

# Plot after
plt.figure()
sns.histplot(df["age"], bins=40, color="#43A047")
plt.title("Age Distribution (After Cleaning)")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("clean2_age_after.png", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════
# STEP 6 — Handle Negative Waiting Days
# Problem: Some appointments were scheduled AFTER the date
# ══════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("STEP 6: FIXING NEGATIVE WAITING DAYS")
print("─" * 50)

negative_wait = df[df["waiting_days"] < 0]
print(f"Rows with negative waiting days: {len(negative_wait)}")

# Remove — these are data entry errors
df = df[df["waiting_days"] >= 0]
print(f"Remaining rows: {len(df):,}")


# ══════════════════════════════════════════════════════════════
# STEP 7 — Remove Duplicates
# ══════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("STEP 7: REMOVING DUPLICATES")
print("─" * 50)

dupes = df.duplicated(subset=["appointment_id"]).sum()
print(f"Duplicate appointment IDs: {dupes}")

df = df.drop_duplicates(subset=["appointment_id"])
print(f"Rows after deduplication: {len(df):,}")


# ══════════════════════════════════════════════════════════════
# STEP 8 — Standardize Categorical Columns
# ══════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("STEP 8: STANDARDIZING CATEGORICAL VALUES")
print("─" * 50)

# Neighbourhood — standardize casing
df["neighbourhood"] = df["neighbourhood"].str.strip().str.title()

print("Gender values:", df["gender"].unique())
print("Sample neighbourhoods:", df["neighbourhood"].unique()[:5])


# ══════════════════════════════════════════════════════════════
# STEP 9 — Final Exploratory Visuals (on clean data)
# ══════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("STEP 9: VISUALIZATIONS ON CLEAN DATA")
print("─" * 50)

# No-show rate by day of week
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
noshow_by_day = df.groupby("appointment_day_of_week")["showed_up"].mean().reindex(dow_order)

plt.figure()
(1 - noshow_by_day).plot(kind="bar", color="#E53935", edgecolor="none")
plt.title("No-Show Rate by Day of Week")
plt.ylabel("No-Show Rate")
plt.xlabel("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("clean3_noshow_by_day.png", dpi=150)
plt.show()

# No-show rate by SMS received
sms_group = df.groupby("sms_received")["showed_up"].mean()
plt.figure()
sms_group.plot(kind="bar", color=["#EF9A9A", "#43A047"], edgecolor="none")
plt.title("Show-Up Rate: SMS Received vs Not")
plt.ylabel("Show-Up Rate")
plt.xticks([0, 1], ["No SMS", "SMS Received"], rotation=0)
plt.tight_layout()
plt.savefig("clean4_sms_effect.png", dpi=150)
plt.show()

# Waiting days vs no-show
plt.figure()
sns.boxplot(data=df, x="showed_up", y="waiting_days",
            palette=["#EF9A9A", "#A5D6A7"])
plt.title("Waiting Days vs Show-Up")
plt.xticks([0, 1], ["No-Show", "Showed Up"])
plt.ylabel("Waiting Days")
plt.tight_layout()
plt.savefig("clean5_waiting_days.png", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════
# STEP 10 — Save Clean Dataset
# ══════════════════════════════════════════════════════════════

df.to_csv("noshow_clean.csv", index=False)

print("\n" + "═" * 50)
print("CLEANING COMPLETE!")
print("═" * 50)
print(f"Final dataset: {len(df):,} rows × {df.shape[1]} columns")
print("Saved as: noshow_clean.csv")
print("\nCharts saved:")
for i, name in enumerate([
    "clean1_age_before.png",
    "clean2_age_after.png",
    "clean3_noshow_by_day.png",
    "clean4_sms_effect.png",
    "clean5_waiting_days.png"
], 1):
    print(f"  {i}. {name}")