
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image, ImageOps
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import hashlib


# 1. APPLICATION CONFIGURATION

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pth"
UPLOAD_DIR = BASE_DIR / "uploaded_images"

UPLOAD_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# 2. CNN ARCHITECTURE

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 3. LOAD TRAINED MODEL
@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = CNN()

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

    # Support both a raw state_dict and a checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    return model


# 4. IMAGE PREPROCESSING
transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# 5. IMAGE SAVING FUNCTION
def save_uploaded_image(image_bytes, extension):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    unique_id = uuid4().hex[:8]

    filename = f"{timestamp}_{unique_id}.{extension}"

    save_path = UPLOAD_DIR / filename

    with open(save_path, "wb") as file:
        file.write(image_bytes)

    return save_path


# 6. IMAGE PREDICTION FUNCTION
def predict_image(image, model):

    # Convert image into model-compatible format
    image = image.convert("RGB")

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move input to CPU/GPU
    image_tensor = image_tensor.to(DEVICE)

    # Disable gradient calculation
    with torch.no_grad():

        outputs = model(image_tensor)

        # Convert logits into probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get highest probability
        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    predicted_label = CLASS_NAMES[
        predicted_class.item()
    ]

    confidence_score = confidence.item() * 100

    return predicted_label, confidence_score, probabilities


# 7. MAIN WEB INTERFACE
st.title("🧠 CIFAR-10 Image Classification")

st.write(
    "Upload an image and let the trained PyTorch CNN "
    "identify its category."
)

st.caption(
    f"Running on: {DEVICE}"
)

st.divider()


# Load model
try:

    model = load_model()

    st.success("Trained CNN model loaded successfully!")

except Exception as error:

    st.error(f"Unable to load model: {error}")

    st.info(
        "Place your trained model.pth file "
        "in the same directory as app.py."
    )

    st.stop()



# 8. IMAGE UPLOADER
uploaded_file = st.file_uploader(

    "Choose an image",

    type=["jpg", "jpeg", "png"],

    help="Upload a JPG, JPEG, or PNG image."
)


if uploaded_file is None:

    # Reset upload tracking when the uploader is cleared
    st.session_state.pop("last_saved_upload", None)

    st.info("Please upload an image to get started.")

else:

    image_bytes = uploaded_file.getvalue()

    # Create a fingerprint to avoid saving the same
    # upload repeatedly on Streamlit reruns.
    file_hash = hashlib.sha256(image_bytes).hexdigest()

    try:

        # Open and validate image
        image = Image.open(
            uploaded_file
        )

        image = ImageOps.exif_transpose(image)

        image.load()

        image = image.convert("RGB")

    except Exception:

        st.error("Invalid image. Please upload a valid JPG or PNG.")

        st.stop()


    # 9. SAVE UPLOADED IMAGE
    if st.session_state.get("last_saved_upload") != file_hash:

        extension = Path(uploaded_file.name).suffix.lower().lstrip(".")

        saved_path = save_uploaded_image(
            image_bytes,
            extension
        )

        st.session_state["last_saved_upload"] = file_hash

        st.success(
            f"Image saved successfully: {saved_path.name}"
        )


    # 10. DISPLAY IMAGE AND PREDICTION
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption=uploaded_file.name,
            width=350
        )

        st.write(
            f"Original image size: {image.size[0]} × {image.size[1]}"
        )

        st.write(
            "Model input size: 32 × 32 × 3"
        )


    with col2:

        st.subheader("Classification Result")

        with st.spinner("Analyzing image..."):

            predicted_label, confidence, probabilities = predict_image(
                image,
                model
            )

        st.success("Prediction completed!")

        st.markdown("### Predicted Class")

        st.markdown(
            f"## {predicted_label.upper()}"
        )

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(
            confidence / 100
        )


    # 11. TOP 3 PREDICTIONS
    st.divider()

    st.subheader("Top 3 Predictions")

    top_probabilities, top_indices = torch.topk(
        probabilities[0],
        k=3
    )

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        label = CLASS_NAMES[index.item()]

        score = probability.item() * 100

        st.write(
            f"**{label.capitalize()}** — {score:.2f}%"
        )

        st.progress(score / 100)


# 12. SIDEBAR
st.sidebar.title("About the Project")

st.sidebar.write(
    "This application uses a Convolutional Neural Network "
    "trained on the CIFAR-10 dataset."
)

st.sidebar.write("**Model:** Custom CNN")

st.sidebar.write("**Framework:** PyTorch")

st.sidebar.write("**Dataset:** CIFAR-10")

st.sidebar.write("**Classes:** 10")

st.sidebar.write("**Input Size:** 32 × 32 RGB")

st.sidebar.divider()

st.sidebar.write("**Supported Images:** JPG, JPEG, PNG")

st.sidebar.write(
    f"**Saved Images:** {len(list(UPLOAD_DIR.iterdir()))}"
)
