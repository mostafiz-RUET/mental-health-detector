import joblib
import numpy as np
import streamlit as st
from scipy.sparse import hstack
import re
import random
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Paths (deploy-safe) ----------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

# ---------------- Page config ----------------
st.set_page_config(page_title="Mental Health Detector", layout="centered")

# ---------------- Calm Night Theme CSS (single theme) ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color:#E5E7EB; }

    .stApp{
      background: radial-gradient(1200px 600px at 12% 10%, rgba(124, 58, 237, 0.25), transparent 60%),
                  radial-gradient(900px 520px at 88% 18%, rgba(45, 212, 191, 0.20), transparent 55%),
                  radial-gradient(1000px 700px at 50% 92%, rgba(56, 189, 248, 0.12), transparent 60%),
                  linear-gradient(180deg, #0B1220 0%, #0B1220 100%);
    }

    .block-container { padding-top: 2.0rem; max-width: 900px; }

    h1,h2,h3 { color:#E5E7EB; letter-spacing:-0.02em; }

    .quote-card{
      background: rgba(15, 23, 42, 0.55);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 16px;
      padding: 14px 16px;
      margin: 10px 0 18px 0;
      box-shadow: 0 18px 60px rgba(0,0,0,0.45);
      color:#E5E7EB;
      font-weight: 700;
    }

    .glass-card{
      background: rgba(15, 23, 42, 0.55);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 22px 80px rgba(0,0,0,0.55);
      margin-bottom: 14px;
    }

    textarea{
      border-radius: 14px !important;
      padding: 12px !important;
      border: 1px solid rgba(148, 163, 184, 0.25) !important;
      background: rgba(2, 6, 23, 0.55) !important;
      color: #E5E7EB !important;
    }

    .stButton > button{
      border-radius: 999px;
      padding: 0.55rem 1.05rem;
      border: 1px solid rgba(45, 212, 191, 0.35);
      background: linear-gradient(90deg, rgba(45,212,191,0.95), rgba(56,189,248,0.95));
      color: #041018;
      font-weight: 800;
      transition: all 0.18s ease-in-out;
      box-shadow: 0 14px 30px rgba(45, 212, 191, 0.18);
      width: 100%;
    }
    .stButton > button:hover{
      transform: translateY(-1px);
      filter: brightness(1.03);
      box-shadow: 0 18px 40px rgba(56, 189, 248, 0.20);
    }

    .chip {display:inline-block; padding:6px 10px; border-radius:999px; margin:4px 6px 0 0; font-size:0.85rem;}
    .chip-positive {background: rgba(34,197,94,0.18); border:1px solid rgba(34,197,94,0.35); color:#86EFAC;}
    .chip-calm {background: rgba(59,130,246,0.18); border:1px solid rgba(59,130,246,0.35); color:#93C5FD;}
    .chip-other {background: rgba(148,163,184,0.18); border:1px solid rgba(148,163,184,0.35); color:#E5E7EB;}

    .red-kw { color: #F87171; font-weight: 900; }

    [data-testid="stSidebar"] img { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Punch lines ----------------
PUNCHLINES = [
    "A clearer message gives a clearer insight.",
    "Write a real sentence — get a more reliable result.",
    "More detail = better prediction.",
    "Your words help the model understand your moment.",
    "Small words, better understanding.",
]
random.seed(datetime.now().strftime("%Y-%m-%d"))
punchline = random.choice(PUNCHLINES)

# ---------------- Load models (cached) ----------------
@st.cache_resource
def load_artifacts():
    tfidf_word = joblib.load(MODELS_DIR / "tfidf_word.joblib")
    tfidf_char = joblib.load(MODELS_DIR / "tfidf_char.joblib")
    model = joblib.load(MODELS_DIR / "mental_health_svm_hybrid_calibrated.joblib")
    return tfidf_word, tfidf_char, model

tfidf_word, tfidf_char, model = load_artifacts()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## About")
    st.write(
        "This app predicts a mental-health related category from text using a machine learning model. "
        "It is **educational** and **not a medical diagnosis**."
    )

    st.markdown("## Developer")
    st.write("**MD. MOSTAFIZUR RAHMAN**")
    st.write("📧 mdmrmanik1000@gmail.com")

    img_path = ASSETS_DIR / "developer.jpg"
    if img_path.exists():
        st.image(str(img_path), caption="Developer", use_container_width=True)
    else:
        st.info("Add your photo at: assets/developer.jpg")

    st.markdown("---")
    st.markdown("## Quick tips")
    st.write("- Write 1–3 full sentences\n- Include context (work, sleep, mood, thoughts)")

    st.markdown("## Resources")
    st.write(
        "- If you feel unsafe, contact local emergency services.\n"
        "- Talk to someone you trust or a mental health professional."
    )

# ---------------- Title + Punchline ----------------
st.title("Mental Health Detection")
st.markdown(f'<div class="quote-card">{punchline}</div>', unsafe_allow_html=True)

# -------- Dictionaries --------
MOOD_TAGS = {
    "Positive / Happy": {
        "happy","happiness","joy","joyful","cheerful","glad","grateful","thankful",
        "optimistic","hopeful","proud","great","awesome","fantastic","amazing","smile","smiling","laugh","laughing"
    },
    "Excited / Energetic": {
        "excited","exciting","thrilled","pumped","energetic","motivated","eager","enthusiastic","can’t wait","cant wait"
    },
    "Calm / Relaxed": {
        "calm","relaxed","peaceful","content","fine","okay","stable","comfortable","at ease"
    },
    "Angry / Irritable": {
        "angry","furious","mad","annoyed","irritated","irritable","frustrated","rage","snapped","snapping"
    },
    "Caring / Supportive": {
        "care","caring","support","supportive","help","helping","kind","kindness","love","loved","compassion","empathy"
    },
    "Funny / Playful": {
        "funny","hilarious","joking","joke","playful","silly","lol","haha","lmao"
    }
}

SUICIDE_TRIGGERS = {
    "suicide", "suicidal", "kill myself", "end my life", "take my life",
    "self harm", "self-harm", "hurt myself", "die", "dying", "want to die",
    "overdose", "cut myself"
}

STRESS_KEYWORDS = {
    "stress", "stressed", "stressful", "overwhelmed", "overwhelming",
    "pressure", "pressured", "deadline", "deadlines",
    "workload", "overwork", "burnout", "burned out",
    "exam", "exams", "assignment", "assignments",
    "responsibilities", "too much to do",
    "can't cope", "cant cope", "under pressure",
    "sleep deprived", "no sleep", "insomnia"
}

def simple_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text.lower())

def has_suicide_trigger(text: str) -> bool:
    t = text.lower()
    return any(trg in t for trg in SUICIDE_TRIGGERS)

def has_stress_cue(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in STRESS_KEYWORDS)

def detect_mood_tags(text: str) -> list[str]:
    t = text.lower()
    tokens = set(simple_tokens(text))
    matched = []
    for tag, words in MOOD_TAGS.items():
        if any((" " in w and w in t) for w in words) or len(tokens & words) > 0:
            matched.append(tag)
    return matched

def vectorize_hybrid(text: str):
    xw = tfidf_word.transform([text])
    xc = tfidf_char.transform([text])
    return hstack([xw, xc])

def rerank_stress_safe(text: str, probs: np.ndarray, classes: np.ndarray,
                       max_gap: float = 0.12, dep_block: float = 0.75) -> str:
    order = probs.argsort()[::-1]
    top1 = str(classes[order[0]])
    top1_p = float(probs[order[0]])

    if has_suicide_trigger(text):
        return top1

    class_list = list(map(str, classes))
    stress_p = float(probs[class_list.index("Stress")])
    dep_p = float(probs[class_list.index("Depression")])

    if dep_p >= dep_block:
        return top1

    if has_stress_cue(text) and (top1_p - stress_p) <= max_gap:
        return "Stress"
    return top1

def is_unclear_text(text: str) -> tuple[bool, str]:
    t = text.strip()
    if len(t) < 8:
        return True, "Text too short."
    words = simple_tokens(t)
    if len(words) < 3:
        return True, "Not enough meaningful words. Please write 1–2 full sentences."
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    if avg_len > 12:
        return True, "Looks like random text. Please write a clear sentence."
    if re.search(r"(.)\1{5,}", t.lower()):
        return True, "Too many repeated characters. Please write a clear sentence."
    return False, ""

def clear_text():
    st.session_state["text"] = ""

def make_donut(ax, sizes, labels):
    """
    Donut chart with:
    - no overlapping labels on wedges
    - legend on the right
    - center text = top1 label + %
    """
    wedges, _ = ax.pie(
        sizes,
        labels=None,
        startangle=90,
        wedgeprops={"width": 0.38, "linewidth": 0.8, "edgecolor": "white"}
    )
    ax.axis("equal")

    legend_labels = [f"{lab} ({val*100:.1f}%)" for lab, val in zip(labels, sizes)]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9
    )

    top_i = int(np.argmax(sizes))
    ax.text(
        0, 0,
        f"{labels[top_i]}\n{sizes[top_i]*100:.1f}%",
        ha="center", va="center",
        fontsize=12, fontweight="bold"
    )

# ---------------- Main card ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("**Write a clear message (1–3 sentences).** Example: “I feel stressed due to exams and deadlines.”")

text = st.text_area("Enter your text:", height=160, key="text")

c1, c2 = st.columns([1, 1])
with c1:
    predict_clicked = st.button("Predict")
with c2:
    st.button("Clear", on_click=clear_text)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction ----------------
if predict_clicked:
    if not text.strip():
        st.error("Please enter some text.")
    else:
        unclear, reason = is_unclear_text(text)
        if unclear:
            st.warning(
                f"Input text is not clear: {reason}\n\n"
                "Please write a meaningful sentence so the model can understand it."
            )
        else:
            vec = vectorize_hybrid(text)
            probs = model.predict_proba(vec)[0]
            classes = model.classes_
            order = np.argsort(probs)[::-1]

            pred_idx = int(order[0])
            pred = str(classes[pred_idx])
            top1_prob = float(probs[pred_idx])

            mood_tags = detect_mood_tags(text)
            matched_stress = [k for k in sorted(STRESS_KEYWORDS) if k in text.lower()]
            matched_stress = matched_stress[:6]

            # Safety override for suicidal
            if pred == "Suicidal" and not has_suicide_trigger(text):
                for i in order[1:]:
                    if str(classes[i]) != "Suicidal":
                        pred = str(classes[i])
                        pred_idx = int(i)
                        top1_prob = float(probs[pred_idx])
                        break
                st.warning(
                    'Safety note: <span class="red-kw">Suicidal</span> label is shown only when self-harm related words are present.',
                    unsafe_allow_html=True
                )

            # Safe Stress rerank
            if not has_suicide_trigger(text):
                pred_after_stress = rerank_stress_safe(text, probs, classes)
                if pred_after_stress != pred:
                    pred = pred_after_stress
                    if matched_stress:
                        st.info("Detected stress indicators: " + ", ".join(matched_stress))

            # Contradiction override
            POSITIVE_LIKE = {"Positive / Happy", "Calm / Relaxed", "Caring / Supportive", "Excited / Energetic"}
            has_positive_like = any(t in POSITIVE_LIKE for t in mood_tags)
            if has_positive_like and pred in {"Depression", "Suicidal"} and top1_prob < 0.60:
                st.info("Your text looks positive/calm. Showing Normal as a safer default.")
                pred = "Normal"

            st.subheader("Result")
            st.success(f"Predicted Category: {pred}")

            # --- Bar chart (top-5) ---
            pairs = sorted([(str(c), float(p)) for c, p in zip(classes, probs)], key=lambda x: x[1], reverse=True)
            top5 = pairs[:5]
            top5_dict = {c: p for c, p in top5}

            st.write("Prediction probabilities (top 5):")
            st.bar_chart(top5_dict)

            # --- Donut chart (top-5 + Others) ---
            other_sum = float(sum(p for _, p in pairs[5:]))
            donut_labels = [c for c, _ in top5] + (["Others"] if other_sum > 0 else [])
            donut_sizes = [p for _, p in top5] + ([other_sum] if other_sum > 0 else [])

            fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=130)
            make_donut(ax, donut_sizes, donut_labels)

            st.write("Probability distribution (Donut):")
            st.pyplot(fig, use_container_width=False)

            # --- Top-3 list with red suicidal ---
            st.write("Top predictions (raw model output):")
            for c, p in pairs[:3]:
                if c == "Suicidal":
                    st.markdown(f"- **:red[{c}]**: {p*100:.2f}%")
                else:
                    st.write(f"- {c}: {p*100:.2f}%")

            if top1_prob < 0.45:
                st.info("Result is uncertain. Please add more details (1–2 more sentences) and try again.")

            # --- Mood tags as colored chips ---
            if mood_tags:
                st.write("Extra segments (non-clinical mood tags):")
                chip_html = ""
                for tag in mood_tags:
                    if tag in {"Positive / Happy", "Excited / Energetic"}:
                        chip_html += f'<span class="chip chip-positive">{tag}</span>'
                    elif tag in {"Calm / Relaxed", "Caring / Supportive"}:
                        chip_html += f'<span class="chip chip-calm">{tag}</span>'
                    else:
                        chip_html += f'<span class="chip chip-other">{tag}</span>'
                st.markdown(chip_html, unsafe_allow_html=True)

            if pred == "Suicidal" and has_suicide_trigger(text):
                st.error(
                    "If you are feeling unsafe or thinking about self-harm, please seek help immediately from a trusted person or a professional. "
                    "If this is an emergency, contact your local emergency number."
                )