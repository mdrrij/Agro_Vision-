import hashlib, json, re, secrets, uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pymysql
import tensorflow as tf
from PIL import Image
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, text

BASE = Path(__file__).parent
IMG = (128, 128)          # must match training
MIN_CONF = 80             # below this the result is "not recognized"
GREEN_MIN = 5             # minimum % of green pixels in a leaf photo (lower it if brown leaves get rejected)
LANGS = ("en", "ne", "hi")

# XAMPP defaults: user root, empty password
DB_USER, DB_PASS, DB_HOST, DB_NAME = "root", "", "localhost", "agrovision"

UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- knowledge
# (name en, ne, hi, advice en, ne, hi)
PLANTS = {
    "pepper": ("Pepper", "भेडे खुर्सानी", "शिमला मिर्च"),
    "potato": ("Potato", "आलु", "आलू"),
    "tomato": ("Tomato", "गोलभेडा", "टमाटर"),
}
D = {
    "healthy": ("Healthy", "स्वस्थ", "स्वस्थ",
        "The leaf looks healthy. Water regularly and check the plants often.",
        "पात स्वस्थ देखिन्छ। नियमित सिँचाइ गर्नुहोस् र बिरुवा बारम्बार जाँच्नुहोस्।",
        "पत्ती स्वस्थ दिख रही है। नियमित सिंचाई करें और पौधों की बार-बार जाँच करें।"),
    "earlyblight": ("Early blight", "अगौटे डढुवा रोग", "अगेती झुलसा रोग",
        "Remove infected leaves, avoid wetting leaves, rotate crops, and use a recommended fungicide if it spreads.",
        "रोगी पातहरू हटाउनुहोस्, पातमा पानी नपर्ने गरी सिँचाइ गर्नुहोस्, बाली चक्र अपनाउनुहोस्, फैलिए सिफारिस गरिएको ढुसीनाशक प्रयोग गर्नुहोस्।",
        "रोगग्रस्त पत्तियाँ हटाएँ, पत्तियों पर पानी न डालें, फसल चक्र अपनाएँ, फैलने पर अनुशंसित फफूंदनाशक का प्रयोग करें।"),
    "lateblight": ("Late blight", "पछौटे डढुवा रोग", "पछेती झुलसा रोग",
        "Spreads fast in cool, wet weather. Remove and destroy infected plants, keep leaves dry, and contact the local agriculture office quickly.",
        "चिसो र ओसिलो मौसममा छिट्टै फैलिन्छ। रोगी बिरुवा हटाई नष्ट गर्नुहोस्, पात सुख्खा राख्नुहोस् र तुरुन्त नजिकको कृषि कार्यालयमा सम्पर्क गर्नुहोस्।",
        "ठंडे और नम मौसम में तेजी से फैलता है। रोगी पौधे हटाकर नष्ट करें, पत्तियाँ सूखी रखें और तुरंत नजदीकी कृषि कार्यालय से संपर्क करें।"),
    "bacterialspot": ("Bacterial spot", "ब्याक्टेरियाजन्य थोप्ले रोग", "जीवाणु धब्बा रोग",
        "Use clean seed, do not handle wet plants, rotate crops, and spray copper-based medicine.",
        "सफा बीउ प्रयोग गर्नुहोस्, भिजेका बिरुवा नछुनुहोस्, बाली चक्र अपनाउनुहोस् र कपरयुक्त औषधि छर्कनुहोस्।",
        "साफ बीज का उपयोग करें, गीले पौधों को न छुएँ, फसल चक्र अपनाएँ और कॉपर आधारित दवा का छिड़काव करें।"),
    "leafmold": ("Leaf mold", "पात ढुसी रोग", "पत्ती फफूँद रोग",
        "Improve air flow, reduce humidity, avoid wetting leaves, and remove infected leaves.",
        "हावा खेल्ने व्यवस्था मिलाउनुहोस्, ओसिलोपन घटाउनुहोस्, पात भिज्न नदिनुहोस् र रोगी पात हटाउनुहोस्।",
        "हवा का आवागमन बढ़ाएँ, नमी कम करें, पत्तियाँ गीली न करें और रोगग्रस्त पत्तियाँ हटाएँ।"),
    "septoria": ("Septoria leaf spot", "सेप्टोरिया पात थोप्ले रोग", "सेप्टोरिया पत्ती धब्बा रोग",
        "Remove infected lower leaves, mulch the base, rotate crops, and use fungicide if needed.",
        "तल्ला रोगी पात हटाउनुहोस्, फेदमा मल्चिङ गर्नुहोस्, बाली चक्र अपनाउनुहोस् र आवश्यक परे ढुसीनाशक प्रयोग गर्नुहोस्।",
        "नीचे की रोगग्रस्त पत्तियाँ हटाएँ, जड़ के पास मल्चिंग करें, फसल चक्र अपनाएँ और जरूरत हो तो फफूंदनाशक का प्रयोग करें।"),
    "spidermite": ("Spider mites", "माकुरे किरा (स्पाइडर माइट)", "मकड़ी कीट (स्पाइडर माइट)",
        "Spray water under the leaves, use insecticidal soap or miticide, and avoid dry, dusty conditions.",
        "पातको मुनि पानी छर्कनुहोस्, कीटनाशक साबुन वा माइटनाशक प्रयोग गर्नुहोस् र सुक्खा धुलो वातावरणबाट जोगाउनुहोस्।",
        "पत्तियों के नीचे पानी छिड़कें, कीटनाशक साबुन या माइटनाशक का प्रयोग करें और सूखे, धूल भरे वातावरण से बचाएँ।"),
    "targetspot": ("Target spot", "टार्गेट स्पट रोग", "टारगेट स्पॉट रोग",
        "Improve air flow, clear plant waste, avoid overhead watering, and use fungicide if it spreads.",
        "हावा खेल्ने व्यवस्था मिलाउनुहोस्, बिरुवाका अवशेष हटाउनुहोस्, माथिबाट सिँचाइ नगर्नुहोस् र फैलिए ढुसीनाशक प्रयोग गर्नुहोस्।",
        "हवा का आवागमन बढ़ाएँ, पौधों के अवशेष हटाएँ, ऊपर से सिंचाई न करें और फैलने पर फफूंदनाशक का प्रयोग करें।"),
    "yellowleaf": ("Yellow leaf curl virus", "पहेँलो पात खुम्चिने भाइरस", "पीला पत्ती मोड़ वायरस",
        "No cure. Control whiteflies, remove infected plants, and plant resistant varieties.",
        "उपचार छैन। सेतो झिँगा नियन्त्रण गर्नुहोस्, रोगी बिरुवा हटाउनुहोस् र रोग प्रतिरोधी जात लगाउनुहोस्।",
        "इलाज नहीं है। सफेद मक्खी पर नियंत्रण करें, रोगी पौधे हटाएँ और रोग-प्रतिरोधी किस्में लगाएँ।"),
    "mosaic": ("Mosaic virus", "मोजेक भाइरस", "मोज़ेक वायरस",
        "No cure. Remove infected plants, clean tools, wash hands, and use resistant varieties.",
        "उपचार छैन। रोगी बिरुवा हटाउनुहोस्, औजार सफा गर्नुहोस्, हात धुनुहोस् र प्रतिरोधी जात प्रयोग गर्नुहोस्।",
        "इलाज नहीं है। रोगी पौधे हटाएँ, औज़ार साफ़ करें, हाथ धोएँ और प्रतिरोधी किस्में उपयोग करें।"),
}


def parse_class(name):
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    plant = next((p for p in PLANTS if key.startswith(p)), None)
    dk = next((k for k in D if k in key), None)
    return plant, dk


# ----------------------------------------------------------------- database
_c = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, autocommit=True)
_c.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4")
_c.close()

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4", pool_pre_ping=True)

DDL = [
    """CREATE TABLE IF NOT EXISTS users (
        user_id INT AUTO_INCREMENT PRIMARY KEY,
        full_name VARCHAR(100) NOT NULL,
        email VARCHAR(120) NOT NULL UNIQUE,
        password_hash VARCHAR(200) NOT NULL,
        language CHAR(2) NOT NULL DEFAULT 'en',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS sessions (
        token VARCHAR(64) PRIMARY KEY,
        user_id INT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE)""",
    """CREATE TABLE IF NOT EXISTS plants (
        plant_id INT AUTO_INCREMENT PRIMARY KEY,
        name_en VARCHAR(50) NOT NULL UNIQUE,
        name_ne VARCHAR(80) NOT NULL,
        name_hi VARCHAR(80) NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS diseases (
        disease_id INT AUTO_INCREMENT PRIMARY KEY,
        plant_id INT NOT NULL,
        class_label VARCHAR(120) NOT NULL UNIQUE,
        is_healthy BOOLEAN NOT NULL DEFAULT FALSE,
        name_en VARCHAR(100), name_ne VARCHAR(150), name_hi VARCHAR(150),
        tip_en TEXT, tip_ne TEXT, tip_hi TEXT,
        FOREIGN KEY (plant_id) REFERENCES plants(plant_id))""",
    """CREATE TABLE IF NOT EXISTS images (
        image_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        file_name VARCHAR(100) NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE)""",
    """CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INT AUTO_INCREMENT PRIMARY KEY,
        image_id INT NOT NULL,
        disease_id INT NULL,
        confidence DECIMAL(5,2) NOT NULL,
        recognized BOOLEAN NOT NULL DEFAULT TRUE,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
        FOREIGN KEY (disease_id) REFERENCES diseases(disease_id))""",
]

model = tf.keras.models.load_model(BASE / "agrovision_cnn.keras")
class_names = json.load(open(BASE / "class_names.json", encoding="utf-8"))

with engine.begin() as c:
    for q in DDL:
        c.execute(text(q))
    for pk, (e, n, h) in PLANTS.items():
        c.execute(text("INSERT IGNORE INTO plants (name_en,name_ne,name_hi) VALUES (:e,:n,:h)"),
                  {"e": e, "n": n, "h": h})
    for cls in class_names:
        pk, dk = parse_class(cls)
        if not pk or not dk:
            continue            # e.g. the "Unknown" class
        pid = c.execute(text("SELECT plant_id FROM plants WHERE name_en=:e"),
                        {"e": PLANTS[pk][0]}).scalar()
        v = D[dk]
        c.execute(text("""INSERT INTO diseases (plant_id,class_label,is_healthy,name_en,name_ne,name_hi,tip_en,tip_ne,tip_hi)
            VALUES (:p,:c,:h,:a,:b,:d,:x,:y,:z)
            ON DUPLICATE KEY UPDATE name_en=VALUES(name_en),name_ne=VALUES(name_ne),name_hi=VALUES(name_hi),
            tip_en=VALUES(tip_en),tip_ne=VALUES(tip_ne),tip_hi=VALUES(tip_hi)"""),
                  {"p": pid, "c": cls, "h": dk == "healthy", "a": v[0], "b": v[1], "d": v[2],
                   "x": v[3], "y": v[4], "z": v[5]})

# class label -> multilingual info, loaded once from the database
INFO = {}
with engine.connect() as c:
    for r in c.execute(text("""SELECT d.*, p.name_en pe, p.name_ne pn, p.name_hi ph
                               FROM diseases d JOIN plants p ON p.plant_id=d.plant_id""")).mappings():
        INFO[r["class_label"]] = {
            "id": r["disease_id"], "healthy": bool(r["is_healthy"]),
            "plant": {"en": r["pe"], "ne": r["pn"], "hi": r["ph"]},
            "disease": {l: r[f"name_{l}"] for l in LANGS},
            "tip": {l: r[f"tip_{l}"] for l in LANGS},
        }

# --------------------------------------------------------------------- auth
def hash_pw(pw, salt=None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()
    return f"{salt}${h}"


def check_pw(pw, stored):
    salt, h = stored.split("$")
    return secrets.compare_digest(hash_pw(pw, salt).split("$")[1], h)


def new_session(uid):
    token = secrets.token_hex(32)
    with engine.begin() as c:
        c.execute(text("INSERT INTO sessions (token,user_id) VALUES (:t,:u)"), {"t": token, "u": uid})
    return token


def current_user(authorization: str = Header(None)):
    token = (authorization or "").replace("Bearer ", "")
    with engine.connect() as c:
        u = c.execute(text("""SELECT u.user_id, u.full_name AS name, u.email, u.language
            FROM sessions s JOIN users u ON u.user_id=s.user_id
            WHERE s.token=:t AND s.created_at > NOW() - INTERVAL 7 DAY"""), {"t": token}).mappings().first()
    if not u:
        raise HTTPException(401, "Not logged in")
    return dict(u)


class Register(BaseModel):
    name: str
    email: str
    password: str
    language: str = "en"


class Login(BaseModel):
    email: str
    password: str


class Lang(BaseModel):
    language: str


# ---------------------------------------------------------------------- app
app = FastAPI(title="AgroVision AI")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/api/register")
def register(b: Register):
    email = b.email.strip().lower()
    if not b.name.strip() or "@" not in email or len(b.password) < 6 or b.language not in LANGS:
        raise HTTPException(400, "Invalid details")
    with engine.begin() as c:
        if c.execute(text("SELECT 1 FROM users WHERE email=:e"), {"e": email}).first():
            raise HTTPException(409, "Email already registered")
        uid = c.execute(text("INSERT INTO users (full_name,email,password_hash,language) VALUES (:n,:e,:p,:l)"),
                        {"n": b.name.strip(), "e": email, "p": hash_pw(b.password), "l": b.language}).lastrowid
    return {"token": new_session(uid),
            "user": {"user_id": uid, "name": b.name.strip(), "email": email, "language": b.language}}


@app.post("/api/login")
def login(b: Login):
    with engine.connect() as c:
        u = c.execute(text("SELECT user_id, full_name AS name, email, language, password_hash FROM users WHERE email=:e"),
                      {"e": b.email.strip().lower()}).mappings().first()
    if not u or not check_pw(b.password, u["password_hash"]):
        raise HTTPException(401, "Wrong email or password")
    user = {k: u[k] for k in ("user_id", "name", "email", "language")}
    return {"token": new_session(u["user_id"]), "user": user}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


@app.put("/api/language")
def set_language(b: Lang, user=Depends(current_user)):
    if b.language not in LANGS:
        raise HTTPException(400, "Unsupported language")
    with engine.begin() as c:
        c.execute(text("UPDATE users SET language=:l WHERE user_id=:u"), {"l": b.language, "u": user["user_id"]})
    return {"ok": True}


@app.post("/api/predict")
def predict(file: UploadFile = File(...), user=Depends(current_user)):
    try:
        img = Image.open(BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    name = f"{uuid.uuid4().hex}.jpg"
    thumb = img.copy()
    thumb.thumbnail((600, 600))
    thumb.save(UPLOADS / name, quality=85)

    arr = np.expand_dims(np.array(img.resize(IMG)), axis=0)   # model rescales inside
    probs = model.predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:4]
    best = class_names[top[0]]
    conf = round(float(probs[top[0]]) * 100, 2)
    info = INFO.get(best)
    # quick sanity check: a real crop leaf photo has some green in it (paper, walls, screens do not)
    px = arr[0].astype(int)
    green = float(((px[..., 1] > px[..., 0] + 8) & (px[..., 1] > px[..., 2] + 8)).mean() * 100)
    ok = info is not None and conf >= MIN_CONF and green >= GREEN_MIN

    with engine.begin() as c:
        iid = c.execute(text("INSERT INTO images (user_id,file_name) VALUES (:u,:f)"),
                        {"u": user["user_id"], "f": name}).lastrowid
        c.execute(text("INSERT INTO predictions (image_id,disease_id,confidence,recognized) VALUES (:i,:d,:c,:r)"),
                  {"i": iid, "d": info["id"] if ok else None, "c": conf, "r": ok})

    out = {"recognized": ok, "confidence": conf, "image_url": f"/uploads/{name}"}
    if ok:
        out.update(plant=info["plant"], disease=info["disease"], tip=info["tip"], healthy=info["healthy"],
                   top_3=[{"plant": INFO[class_names[i]]["plant"], "disease": INFO[class_names[i]]["disease"],
                           "confidence": round(float(probs[i]) * 100, 2)}
                          for i in top if class_names[i] in INFO][:3])
    return out


@app.get("/api/history")
def history(user=Depends(current_user)):
    with engine.connect() as c:
        rows = c.execute(text("""SELECT p.prediction_id AS id, i.file_name, p.confidence, p.recognized,
                p.predicted_at, d.class_label
            FROM predictions p JOIN images i ON i.image_id=p.image_id
            LEFT JOIN diseases d ON d.disease_id=p.disease_id
            WHERE i.user_id=:u ORDER BY p.prediction_id DESC LIMIT 30"""), {"u": user["user_id"]}).mappings().all()
    out = []
    for r in rows:
        info = INFO.get(r["class_label"]) if r["recognized"] else None
        out.append({"id": r["id"], "image_url": f"/uploads/{r['file_name']}", "confidence": float(r["confidence"]),
                    "recognized": bool(r["recognized"]), "date": r["predicted_at"].strftime("%d %b %Y, %H:%M"),
                    "plant": info["plant"] if info else None, "disease": info["disease"] if info else None,
                    "healthy": info["healthy"] if info else None})
    return out


@app.delete("/api/history/{pid}")
def delete_scan(pid: int, user=Depends(current_user)):
    with engine.begin() as c:
        r = c.execute(text("""SELECT i.image_id, i.file_name FROM predictions p
            JOIN images i ON i.image_id=p.image_id WHERE p.prediction_id=:p AND i.user_id=:u"""),
                      {"p": pid, "u": user["user_id"]}).first()
        if not r:
            raise HTTPException(404, "Not found")
        c.execute(text("DELETE FROM predictions WHERE prediction_id=:p"), {"p": pid})
        c.execute(text("DELETE FROM images WHERE image_id=:i"), {"i": r[0]})
    (UPLOADS / r[1]).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/logout")
def logout(authorization: str = Header(None)):
    with engine.begin() as c:
        c.execute(text("DELETE FROM sessions WHERE token=:t"), {"t": (authorization or "").replace("Bearer ", "")})
    return {"ok": True}