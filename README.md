# 🩺 AI-Powered Smart Wound Scanner

## 📌 Project Overview

This project is an **AI-based system** designed to automatically detect the **healing stage of wounds** using image processing and machine learning.

Instead of relying on human judgment, the system converts a wound image into **measurable data** and makes a **consistent, objective decision**.

👉 In simple terms:
**Image → Numbers → Decision**

---

## 🎯 Problem Statement

Traditional wound assessment is:

* Subjective (depends on doctor experience)
* Inconsistent (different doctors give different results)
* Manual and time-consuming

This project solves that by providing:
✔ Automated analysis
✔ Consistent results
✔ Fast diagnosis

---

## ⚙️ System Pipeline

The system works in 6 steps:

1. **Input Image**

   * Wound image captured using a smartphone

2. **Preprocessing**

   * Resize image (256×256)
   * Normalize pixel values
   * Reduce noise

3. **K-Means Clustering (Segmentation)**

   * Separates wound from healthy skin
   * Uses LAB color space

4. **Feature Extraction**

   * Wound Area
   * Red Tissue % (healing)
   * Yellow Tissue % (damage)

5. **Classification (SVM + Ensemble)**

   * Uses SVM with RBF kernel
   * Supported by Random Forest & Gradient Boosting

6. **Web Dashboard Output**

   * Displays:

     * Healing Stage
     * Wound Measurements
     * Clinical Suggestions

---

## 🧠 Healing Stages Classified

The model predicts:

* 🔴 **Inflammation** (early stage)
* 🟢 **Proliferation** (healing stage)
* ⚪ **Maturation** (final stage)

---

## 📊 Performance

| Metric    | Value     |
| --------- | --------- |
| Accuracy  | **97.5%** |
| Precision | 97.53%    |
| Recall    | 97.50%    |
| F1-Score  | 97.49%    |

✔ High accuracy
✔ Balanced performance
✔ Reliable predictions

---

## 🧪 Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Flask

---

## 🏗️ Project Structure

```
AI-Wound-Scanner/
│
├── dataset/
├── ml/
│   ├── preprocessing.py
│   ├── kmeans.py
│   ├── feature_extraction.py
│   ├── svm.py
│
├── app/
│   ├── app.py
│   ├── templates/
│   ├── static/
│
├── results/
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/AI-Wound-Scanner.git
cd AI-Wound-Scanner
```

### Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 3: Run Application

```bash
python app.py
```

### Step 4: Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧩 Key Features

✔ Automated wound detection
✔ Explainable AI (not black-box)
✔ Fast processing (~0.3 sec)
✔ Works on CPU (no GPU required)
✔ Suitable for rural healthcare

---

## ⚠️ Limitations

* Works on 2D images only
* Proliferation stage is harder to classify
* Depends on image quality

---

## 🔮 Future Scope

* 3D wound measurement
* Mobile app (Android/iOS)
* IoT-based smart bandages
* Cloud deployment
* Larger dataset training

---

## 🏥 Real-World Impact

* Reduces human error
* Improves diagnosis speed
* Supports doctors in decision-making
* Enables remote healthcare

---

## 👨‍💻 Author

**K. Manoj Kumar Reddy**
B.Tech – Electronics & Communication Engineering

---

## 📌 Final Thought

This project shows that:
👉 **Simple + Explainable AI > Complex Black Box (for healthcare)**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
