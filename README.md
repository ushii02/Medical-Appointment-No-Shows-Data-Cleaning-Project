# 🏥 Medical Appointment No-Shows — Data Cleaning Project

A systematic data cleaning project documenting 10 cleaning steps on a messy real-world dataset — the kind of work that makes up 80% of a data scientist's actual job.

---

## 📌 What This Project Covers

- 10 systematic cleaning steps on a real-world medical dataset
- Handling missing values, outliers, and inconsistent data types
- Extracting insights on patient attendance patterns
- Visualizing the impact of key factors on no-show rates

---

## 📊 Visualizations

| Chart | Insight |
|-------|---------|
| clean1_age_before.png | Age distribution before cleaning |
| clean2_age_after.png | Age distribution after removing outliers |
| clean3_noshow_by_day.png | No-show rate by day of week |
| clean4_sms_effect.png | Impact of SMS reminders on attendance |
| clean5_waiting_days.png | Waiting days vs show-up rate |

---

## 💡 Key Findings

- Overall no-show rate is around **20%**
- Patients who waited longer were more likely to miss appointments
- SMS reminders had a surprisingly small effect on attendance
- Saturday appointments had the lowest no-show rate

---

## 🛠️ Tools & Libraries

- Python 3.14
- pandas
- numpy
- matplotlib
- seaborn

---

## 📁 Dataset Source

**Medical Appointment No Shows — Kaggle**

The dataset contains ~110,000 medical appointments in Brazil with patient demographics and attendance records.

---

## ▶️ How to Run

1. Clone this repo
2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

3. Place `KaggleV2-May-2016.csv` in the same folder
4. Run the script:

```bash
python noshow_cleaning.py
```

5. Find the clean dataset saved as `noshow_clean.csv`
