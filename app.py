
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "/content/drive/MyDrive/Deep learning/"
    "Dataset Vehicle/Models/"
    "best_vehicle_mobilenetv2_finetuned.keras"
)

IMG_SIZE = 160

CLASS_NAMES = [
    "Bike",
    "Bus",
    "CNG",
    "Easy-Bike",
    "Hatchback",
    "MPV",
    "Pickup",
    "SEDAN",
    "SUV",
    "Truck"
]

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Vehicle Classification AI",
    page_icon="Vehicle",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM DARK UI
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.hero {
    text-align: center;
    padding: 20px 10px 30px 10px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.70;
    margin-top: 8px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 15px;
}

.result-title {
    font-size: 18px;
    font-weight: 600;
}

.big-result {
    font-size: 42px;
    font-weight: 800;
    margin-top: 5px;
}

.confidence-text {
    font-size: 20px;
    margin-top: 5px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    opacity: 0.55;
    padding-top: 30px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
Vehicle Classification AI
</div>

<div class="hero-subtitle">
10-Class Vehicle Recognition using MobileNetV2
</div>

</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Model Information")

    st.metric(
        "Test Accuracy",
        "79.53%"
    )

    st.metric(
        "Best Validation",
        "78.50%"
    )

    st.write("")

    st.write("**Architecture**")
    st.write("MobileNetV2")

    st.write("**Transfer Learning**")
    st.write("ImageNet")

    st.write("**Fine-Tuning**")
    st.write("Enabled")

    st.write("**Training**")
    st.write("30 Epochs")

    st.write("**Input Size**")
    st.write("160 × 160")

    st.write("**Classes**")
    st.write("10")

    st.divider()

    st.markdown("### Vehicle Classes")

    for i, class_name in enumerate(CLASS_NAMES, 1):
        st.write(f"{i}. {class_name}")

# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Upload Vehicle Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image containing a vehicle."
)

# ============================================================
# NO IMAGE STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a vehicle image above to start classification."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Vehicle Classes",
            "10"
        )

    with col2:
        st.metric(
            "Test Accuracy",
            "79.53%"
        )

    with col3:
        st.metric(
            "Training Epochs",
            "30"
        )

# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    resized_image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner("Analyzing vehicle image..."):

        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    top3_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    # ========================================================
    # RESULT LAYOUT
    # ========================================================

    st.divider()

    image_col, result_col = st.columns(
        [1.1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_col:

        st.markdown(
            '<div class="section-title">Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with result_col:

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):

            st.markdown(
                '<div class="result-title">'
                'Predicted Vehicle'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="big-result">'
                f'{predicted_class}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="confidence-text">'
                f'Confidence: {confidence:.2%}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(
                confidence
            )

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Top 3 Predictions'
            '</div>',
            unsafe_allow_html=True
        )

        for rank, index in enumerate(
            top3_indices,
            start=1
        ):

            name = CLASS_NAMES[index]
            probability = float(
                probabilities[index]
            )

            st.write(
                f"**{rank}. {name}**"
            )

            st.progress(
                probability
            )

            st.caption(
                f"{probability:.2%}"
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">

Vehicle Classification AI  
MobileNetV2 Transfer Learning + Fine-Tuning  
10 Vehicle Classes | 160×160 Input | 30 Epoch Training

</div>
""", unsafe_allow_html=True)
