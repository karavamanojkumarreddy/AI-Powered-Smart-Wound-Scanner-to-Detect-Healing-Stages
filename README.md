# 🩺 AI-Powered Smart Wound Scanner

> Built by a ECE student who got tired of watching wound assessments depend entirely on which doctor walked into the room.

---

## What is this?

A machine learning system that looks at a wound photo and tells you what healing stage it's in — **automatically, consistently, and without needing a specialist in the room.**

No deep learning black box. No GPU server. No internet required.  
Just a photo → some math → a clear answer.

---

## Why does this exist?

Here's a frustrating reality in wound care:

Two doctors look at the same wound. They give different assessments. Both are experienced. Both are confident. Both are wrong about the other.

This inconsistency isn't a character flaw — it's a structural problem. Human assessment is inherently subjective, especially under time pressure, across experience levels, and in under-resourced settings.

This project doesn't replace doctors. It gives them a second opinion that never has a bad day.

---

## How it works

No black magic. The whole pipeline is explainable:

```
Smartphone Photo
      ↓
Resize + Normalize
      ↓
K-Means Clustering  ←  separates wound from healthy skin using LAB color space
      ↓
Feature Extraction  ←  wound area, red tissue %, yellow tissue %
      ↓
SVM + Ensemble      ←  classifies healing stage
      ↓
Web Dashboard       ←  shows result + clinical suggestion
```

The model sees what we tell it to look for. There are no hidden layers guessing for us.

---

## The three healing stages it detects

| Stage | What it means | Visual signal |
|---|---|---|
| 🔴 Inflammation | Early stage — body fighting infection | Redness, swelling |
| 🟢 Proliferation | Active healing — new tissue forming | Pink/red granulation tissue |
| ⚪ Maturation | Final stage — scar tissue forming | Pale, closed wound |

---

## Performance

Tested on a labelled wound image dataset:

| Metric | Score |
|---|---|
| Accuracy | 97.5% |
| Precision | 97.53% |
| Recall | 97.50% |
| F1-Score | 97.49% |

The model is balanced across all three classes — it's not just getting easy cases right and fumbling the harder ones.

---

## Tech stack

Nothing exotic. Runs on a basic laptop.

- **Python** — core language
- **OpenCV** — image processing
- **NumPy** — numerical operations
- **Scikit-learn** — SVM, Random Forest, Gradient Boosting
- **Flask** — web dashboard

No PyTorch. No TensorFlow. No CUDA. Intentionally.

---

## Project structure

```
AI-Wound-Scanner/
│
├── dataset/                   # wound images + labels
│
├── ml/
│   ├── preprocessing.py       # resize, normalize
│   ├── kmeans.py              # color-based segmentation
│   ├── feature_extraction.py  # area, tissue ratios
│   └── svm.py                 # classifier + ensemble
│
├── app/
│   ├── app.py                 # Flask backend
│   ├── templates/             # HTML pages
│   └── static/                # CSS, JS
│
├── results/                   # output logs, metrics
├── README.md
└── requirements.txt
```

---

## Run it yourself

**Step 1 — Clone**
```bash
git clone https://github.com/yourusername/AI-Wound-Scanner.git
cd AI-Wound-Scanner
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Launch**
```bash
python app.py
```

**Step 4 — Open browser**
```
http://127.0.0.1:5000
```

Upload a wound image. Get a result in under a second.

---

## What it's good at

- ✅ Works on a standard laptop CPU — no GPU needed
- ✅ Processing time ~0.3 seconds per image
- ✅ Explainable output — you can trace every decision
- ✅ Useful in low-resource settings (rural clinics, fieldwork)
- ✅ Consistent — gives the same output for the same input, every time

---

## Where it falls short

Being honest about this matters more than looking impressive:

- **2D images only** — can't measure wound depth
- **Image quality dependency** — blurry or poorly lit photos degrade results
- **Proliferation stage** is the hardest to classify — slight performance dip there
- **Dataset size** — performance needs validation on larger, more diverse datasets before clinical use

---

## What's next

Things worth building if this gets traction:

- [ ] 3D wound volume estimation
- [ ] Android/iOS mobile app
- [ ] Cloud deployment for remote access
- [ ] IoT integration with smart bandages
- [ ] Training on larger, more diverse clinical datasets

---

## The design philosophy

This project leans into one idea:

> **A system you can explain and trust beats a system that's accurate but opaque.**

In healthcare, "I don't know why it said that" is not acceptable. Every feature this model uses — tissue color ratios, wound area, cluster separation — has a clinical reason to exist. If it's wrong, you can find out why.

---

## About

**K. Manoj Kumar Reddy**  
B.Tech — Electronics & Communication Engineering

This was a final year project. The goal was never to build something flashy. It was to build something that could actually be handed to a nurse in a rural clinic and trusted.

---

## Contributing

Found a bug? Have a better feature extraction idea? Want to test it on a new dataset?  
Open an issue or submit a PR. Serious contributions welcome.

---

*If this helped you or gave you ideas for your own work — a star goes a long way. ⭐*
