
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EyeScan AI | Eye Disease Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# PROFESSIONAL UI STYLING
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F5F7FB;
        color: #172B4D;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        color: #123C69;
        letter-spacing: -0.8px;
    }

    .brand span {
        color: #159A91;
    }

    .top-tag {
        display: inline-block;
        padding: 6px 12px;
        background: #E2F5F1;
        color: #087F72;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .hero {
        background: linear-gradient(120deg, #123C69 0%, #176B78 100%);
        padding: 38px 38px;
        border-radius: 20px;
        margin-top: 22px;
        margin-bottom: 26px;
        color: white;
    }

    .hero h1 {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: 35px;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 12px;
    }

    .hero p {
        color: #E1EDF5;
        font-size: 15px;
        line-height: 1.8;
        max-width: 680px;
        margin-bottom: 0;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 21px;
        font-weight: 800;
        color: #172B4D;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .section-caption {
        font-size: 13px;
        color: #64748B;
        margin-bottom: 20px;
    }

    .panel {
        background: #FFFFFF;
        padding: 24px;
        border: 1px solid #E3EAF2;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(23, 43, 77, 0.035);
    }

    .panel-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .result-card {
        background: #FFFFFF;
        padding: 26px;
        border: 1px solid #DCE8EC;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(23, 43, 77, 0.04);
    }

    .result-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        color: #64748B;
        margin-bottom: 12px;
    }

    .result-name {
        font-family: 'Manrope', sans-serif;
        font-size: 28px;
        font-weight: 800;
        color: #087F72;
        line-height: 1.3;
        margin-bottom: 18px;
        overflow-wrap: anywhere;
    }

    .score-number {
        font-family: 'Manrope', sans-serif;
        font-size: 32px;
        font-weight: 800;
        color: #123C69;
    }

    .score-caption {
        font-size: 12px;
        color: #64748B;
        margin-top: 2px;
    }

    .info-box {
        background: #EEF5FB;
        border-left: 4px solid #3977A8;
        border-radius: 8px;
        padding: 14px 16px;
        color: #294766;
        font-size: 13px;
        line-height: 1.7;
        margin-top: 18px;
    }

    .disclaimer {
        background: #FFF8E8;
        border: 1px solid #F3DCA4;
        border-radius: 10px;
        padding: 15px 18px;
        color: #795A18;
        font-size: 12px;
        line-height: 1.7;
        margin-top: 22px;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 12px;
        padding-top: 26px;
        border-top: 1px solid #E2E8F0;
        margin-top: 36px;
    }

    div.stButton > button {
        width: 100%;
        background: #087F72;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 12px 20px;
        min-height: 48px;
        font-weight: 700;
        font-size: 15px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #06685E;
        color: white;
        border: none;
    }

    div[data-testid="stFileUploader"] {
        background: #F8FAFC;
        border: 1px dashed #B8C8D8;
        border-radius: 12px;
        padding: 12px;
    }

    div[data-testid="stProgressBar"] > div > div {
        background-color: #159A91;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E3EAF2;
        padding: 18px;
        border-radius: 12px;
    }

    hr {
        border-color: #E2E8F0;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 25px;
        }

        .hero h1 {
            font-size: 27px;
        }

        .result-name {
            font-size: 23px;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "best_eye_model_v2.keras"

CLASS_NAMES = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal"
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="brand">EyeScan <span>AI</span></div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div style="text-align:right; padding-top:8px;">'
        '<span class="top-tag">AI RESEARCH PROJECT</span></div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>Eye Disease Classification<br>Using Deep Learning</h1>
    <p>
        An AI-powered image classification system that analyses uploaded
        eye fundus images and predicts one of four categories using a
        trained MobileNetV2 deep learning model.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# IMAGE UPLOAD SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Image Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'Upload a fundus image to generate a model prediction.'
    '</div>',
    unsafe_allow_html=True
)

upload_col, preview_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-label">01 / Upload Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an eye image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG."
    )

    st.markdown("""
        <div class="info-box">
            <strong>Image requirements</strong><br>
            Upload a clear eye fundus photograph. The model resizes
            the image to 224 × 224 pixels before prediction.
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


with preview_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-label">02 / Image Preview</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded fundus image",
            use_container_width=True
        )
    else:
        st.markdown("""
            <div style="
                background:#F8FAFC;
                border:1px dashed #CBD5E1;
                border-radius:12px;
                min-height:250px;
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
                text-align:center;
                padding:25px;
            ">
                <div style="font-size:17px; font-weight:700; color:#334155;">
                    No image uploaded
                </div>
                <div style="font-size:13px; color:#94A3B8; margin-top:8px;">
                    Your selected image will appear here.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

st.write("")

if uploaded_file is not None:

    if st.button("Analyze Image", type="primary"):

        with st.spinner("Analysing image..."):

            try:
                model = load_model()

                resized_image = image.resize((224, 224))

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Preprocessing is already included in the trained model.
                # Do not apply MobileNetV2 preprocessing a second time.

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

                st.session_state["prediction_result"] = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "scores": predictions.tolist(),
                    "filename": uploaded_file.name
                }

            except Exception as error:
                st.error(f"Unable to analyse this image: {error}")


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

result = st.session_state.get("prediction_result")

if result is not None:

    # Only display the latest result for the currently uploaded file.
    if uploaded_file is not None and result["filename"] == uploaded_file.name:

        st.write("")
        st.divider()

        st.markdown(
            '<div class="section-title">Analysis Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-caption">'
            'Prediction generated by the trained classification model.'
            '</div>',
            unsafe_allow_html=True
        )

        result_col, details_col = st.columns([1, 1], gap="large")

        with result_col:
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Predicted Category</div>
                    <div class="result-name">{result["class"]}</div>
                    <div class="result-label">Model Score</div>
                    <div class="score-number">{result["confidence"]:.2f}%</div>
                    <div class="score-caption">
                        Confidence score reported by the model
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with details_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown(
                '<div class="panel-label">Class-wise Scores</div>',
                unsafe_allow_html=True
            )

            for name, score in zip(
                CLASS_NAMES,
                result["scores"]
            ):
                percentage = float(score) * 100

                st.markdown(
                    f'<div style="font-size:13px; font-weight:600; '
                    f'color:#334155; margin-top:13px;">'
                    f'{name} <span style="float:right;">'
                    f'{percentage:.2f}%</span></div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    min(max(float(score), 0.0), 1.0)
                )

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
            <div class="disclaimer">
                <strong>Important notice</strong><br>
                This system is an academic deep learning prototype and
                is not a clinically validated diagnostic tool. Its
                predictions may be incorrect, particularly for images
                that differ from its training data. Consult a qualified
                eye-care professional for medical assessment.
            </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    EyeScan AI &nbsp; | &nbsp; MobileNetV2 Transfer Learning
    <br><br>
    Academic Project Demonstration
</div>
""", unsafe_allow_html=True)