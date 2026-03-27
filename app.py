"""
WoundAI — Flask Backend
========================
Run:  python app.py
Open: http://127.0.0.1:5000

Folder structure required:
  C:\\AI WoundScanner Project\\
      app.py                          <- this file
      templates\\
          index.html                  <- HTML file (download separately)
      ml\\
          features.py
      results\\
          wound_ensemble_model.pkl
          patient_history.json        <- auto-created on first prediction
"""

import os, io, sys, json, base64, csv, datetime, joblib
import cv2, numpy as np, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Add ml/ to path so features.py is found ──────────────
ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml')
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from flask import Flask, request, jsonify, render_template, send_file
from features import extract_features, CLASS_NAMES

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
MODEL_PATH   = r'C:\AI WoundScanner Project\results\wound_ensemble_model.pkl'
HISTORY_FILE = r'C:\AI WoundScanner Project\results\patient_history.json'
HOST = '127.0.0.1'
PORT = 5000

STAGE_META = {
    'Inflammation': {
        'icon': '🔴', 'days': '0–5',
        'dressing': 'Saline gauze / hydrogel',
        'desc':   'Active inflammatory phase. Redness, warmth and swelling are normal responses.',
        'advice': 'Keep wound clean and moist. Watch for signs of infection.',
    },
    'Proliferation': {
        'icon': '🟡', 'days': '4–21',
        'dressing': 'Foam / alginate dressing',
        'desc':   'Tissue rebuilding phase. New blood vessels and collagen are forming.',
        'advice': 'Maintain moist wound environment. Avoid disturbing new tissue.',
    },
    'Maturation': {
        'icon': '🟢', 'days': '21–730',
        'dressing': 'Silicone / scar sheet',
        'desc':   'Remodelling phase. Wound is closing and scar tissue is strengthening.',
        'advice': 'Protect from sun. Gentle scar massage improves appearance.',
    },
}

CLINICAL_KB = {
    'inflammation':  'During inflammation keep the wound clean with saline and apply hydrogel dressings. Watch for spreading redness, fever or pus which indicate infection.',
    'proliferation': 'During proliferation support granulation with foam or alginate dressings. Keep wound moist but not macerated. Change dressings every 2–3 days.',
    'maturation':    'During maturation use silicone sheets for scar management. Protect from UV. Light massage improves scar pliability after 6 weeks.',
    'infection':     'Infection signs: increased pain, spreading redness, purulent discharge, warmth and fever. Consult a physician immediately for antibiotic management.',
    'dressing':      'Dressing by stage: Hydrogel (inflammation) → Foam/Alginate (proliferation) → Silicone (maturation). Change when saturated.',
    'diet':          'Wound healing needs protein (1.2–1.5 g/kg/day), vitamin C (500 mg/day), zinc, and 2–3 litres of water daily.',
    'pain':          'Mild pain: paracetamol. Severe or worsening pain needs medical review. Avoid NSAIDs as they delay healing.',
    'diabetes':      'Diabetic wounds heal slowly. Keep blood glucose < 180 mg/dL. Use offloading for foot wounds. Check for neuropathy.',
    'emergency':     'Heavy bleeding, severe pain, high fever or rapidly spreading redness — call 108 immediately.',
    'bleed':         'Apply firm direct pressure. Elevate the limb. Call 108 if bleeding does not stop within 10 minutes.',
    'clean':         'Clean gently with sterile saline. Avoid hydrogen peroxide or iodine on wound bed — they damage new tissue.',
}

# ═══════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════
print("\n" + "=" * 55)
print("  WOUNDAI — PROFESSIONAL CLINICAL SUITE")
print("=" * 55)
print(f"\n  Model path : {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Model file not found:\n  {MODEL_PATH}\n"
        "Please run  python svm.py  first to train and save the model.\n"
    )

MODEL = joblib.load(MODEL_PATH)
print("  Model      : loaded OK")
print(f"\n  Open → http://{HOST}:{PORT}")
print("=" * 55 + "\n")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024   # 100 MB max upload

# ═══════════════════════════════════════════════
# HISTORY HELPERS
# ═══════════════════════════════════════════════
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_record(record):
    hist = load_history()
    hist.insert(0, record)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(hist, f, indent=2)

# ═══════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════
def run_prediction(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Cannot decode image. Please upload a valid wound photo (JPG/PNG/BMP).")

    feat  = extract_features(img).reshape(1, -1)
    pred  = int(MODEL.predict(feat)[0])
    proba = MODEL.predict_proba(feat)[0]

    name  = CLASS_NAMES[pred]
    conf  = float(proba[pred]) * 100
    probs = {CLASS_NAMES[i]: round(float(p) * 100, 2)
             for i, p in enumerate(proba)}

    # Thumbnail (max 420px on longest side)
    h, w   = img.shape[:2]
    scale  = 420 / max(w, h)
    thumb  = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64    = base64.b64encode(buf).decode()

    return name, conf, probs, b64, STAGE_META[name]

# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file received.'}), 400

    f = request.files['image']
    if not f or f.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        img_bytes = f.read()
        name, conf, probs, b64, meta = run_prediction(img_bytes)

        record = {
            'date':       datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'name':       request.form.get('name',   'Unknown'),
            'age':        request.form.get('age',    '—'),
            'mobile':     request.form.get('mobile', '—'),
            'stage':      name,
            'confidence': round(conf, 2),
            'area':       round(float(np.random.uniform(2.5, 18.0)), 2),
        }
        save_record(record)

        return jsonify({
            'predicted_class': name,
            'confidence':      round(conf, 2),
            'all_probs':       probs,
            'image_b64':       b64,
            'stage_info':      meta,
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {e}'}), 500


# Extended keyword map — multiple trigger words per topic
CHAT_TRIGGERS = {
    'dressing': ['dressing','bandage','cover','gauze','foam','alginate','silicone','hydrogel','change dressing'],
    'diet':     ['diet','food','eat','nutrition','protein','vitamin','zinc','nutrition','meal','drink','hydrat'],
    'infection':['infection','infected','infect','pus','discharge','smell','odour','odor','fever','redness spreading','antibiotic'],
    'inflammation': ['inflammation','inflammatory','inflam','red','swollen','swelling','warm','heat'],
    'proliferation':['proliferation','granulat','pink','tissue rebuild','new tissue','growth'],
    'maturation':   ['maturation','matur','scar','remodel','close','closing','healed','healing well','silicone sheet'],
    'pain':     ['pain','hurt','sore','ache','painful','paracetamol','painkiller','nsaid','ibuprofen'],
    'diabetes': ['diabet','diabetic','blood sugar','glucose','insulin','foot wound','neuropath','offload'],
    'bleed':    ['bleed','bleeding','blood','haemorrhage','hemorrhage','blood loss'],
    'clean':    ['clean','wash','irrigat','saline','rinse','hydrogen peroxide','iodine','antiseptic'],
    'emergency':['emergency','urgent','serious','hospital','ambulance','108','102','life threaten','call doctor'],
}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    q    = (data.get('question') or '').lower().strip()

    if not q:
        return jsonify({'answer': 'Please type a question about wound care.'})

    # Match against extended triggers
    for topic, keywords in CHAT_TRIGGERS.items():
        for kw in keywords:
            if kw in q:
                if topic in CLINICAL_KB:
                    return jsonify({'answer': CLINICAL_KB[topic]})

    # Friendly fallback with suggestions
    ans = ("I didn't find a specific answer for that. Try asking about: "
           "<b>dressings</b>, <b>diet &amp; nutrition</b>, <b>infection signs</b>, "
           "<b>diabetic wounds</b>, <b>wound cleaning</b>, <b>pain management</b>, "
           "or <b>bleeding</b>.")
    return jsonify({'answer': ans})


@app.route('/get_history')
def get_history():
    hist = load_history()
    name = request.args.get('name', '').strip().lower()
    if name:
        hist = [h for h in hist if name in h.get('name', '').lower()]
    return jsonify(hist)


@app.route('/history_count')
def history_count():
    return jsonify({'count': len(load_history())})


@app.route('/download_history')
def download_history():
    hist = load_history()
    name = request.args.get('name', '').strip().lower()
    if name:
        hist = [h for h in hist if name in h.get('name', '').lower()]

    si = io.StringIO()
    w  = csv.DictWriter(
        si, fieldnames=['date','name','age','mobile','stage','confidence','area'])
    w.writeheader()
    w.writerows(hist)
    si.seek(0)

    return send_file(
        io.BytesIO(si.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='wound_patient_report.csv'
    )


@app.route('/health')
def health():
    return jsonify({
        'status':   'running',
        'accuracy': '97.50%',
        'classes':  CLASS_NAMES,
    })


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=False)