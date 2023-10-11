# Python packages
from pathlib import Path
import PIL
import streamlit as st


# Local Modules
import model


# Setting page layout
st.set_page_config(
    page_title="Object Detection and Segmentation using OpenCV's YOLO-v8 model",
    page_icon="https://www.emoji.co.uk/files/twitter-emojis/objects-twitter/11072-right-pointing-magnifying-glass.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Sidebar
st.sidebar.header("⚙️ Model parameters ⚙️",divider="gray")

#Adding blank spaces
st.sidebar.markdown("")
st.sidebar.markdown("")

# Model Options
st.sidebar.markdown("<h4 style='margin: 0; padding: 0;'>Select the Task</h4>", unsafe_allow_html=True)
model_type = st.sidebar.radio("42", ['Detection', 'Segmentation'],label_visibility="hidden")

#Adding empty mardown to add an empty line
st.sidebar.markdown("")
st.sidebar.markdown("")

st.sidebar.markdown("<h4 style='margin: 0; padding: 0;'>Select Model Confidence (Treshold)</h4>", unsafe_allow_html=True)
confidence = float(st.sidebar.slider('', 20, 100, 40)) / 100

# Selecting Detection Or Segmentation - Up to user - Here model L puis v8n and v8m are also available
if model_type == 'Detection':
    model_path = "weights/yolov8l.pt"
elif model_type == 'Segmentation':
    model_path = "weights/yolov8l-seg.pt"
# Load YOLO Pre-trained ML Model
try:
    st.cache_resource()
    model = model.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
source_img = None

#Adding empty mardown to add an empty line
st.sidebar.markdown("")
st.sidebar.markdown("")
st.title("Demo app : Upload your image")

# Setting subheader
st.subheader("Object Detection and Segmentation using OpenCV's YOLO-v8 model")

# If image is selected
st.sidebar.markdown("<h4 style='margin: 0; padding: 0;'>Choose an image</h4>", unsafe_allow_html=True)
source_img = st.sidebar.file_uploader('42', type=("jpg", "jpeg", "png", 'bmp', 'webp'),label_visibility="hidden")
col1, col2 = st.columns(2)
with col1:
    try:
        if source_img is None:
            default_image_path = "images/default-image.jpeg"
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Default Image", use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            caption_uploaded_image = "Uploaded Image"
            st.image(source_img, caption=caption_uploaded_image, use_column_width=True)
    except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
with col2:
    if source_img is None:
        default_detected_image_path = "images/default-processed-image.jpg"
        default_detected_image = PIL.Image.open(default_detected_image_path)
        caption_detected_image = "Default Image Processed"
        st.image(default_detected_image_path, caption=caption_detected_image, use_column_width=True)
    else:
        if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Processed Image', use_column_width=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# Setting second header/title
st.title("A brief introduction to Object Detection and Image segmentation")
col1, col2 = st.columns(2)

with col1:
    st.header("Object Detection")
    st.write("It's a process of identifying and locating specific objects within an image or video. It typically involves outlining objects with bounding boxes and specifying the type of object. For instance, in an image containing cars, pedestrians, and trees, object detection would pinpoint where these objects are and classify them as cars, pedestrians, trees, etc. Applications of Object Detection : Autonomous vehicles, Identifying and tracking...")
    st.header("Image Segmentation")
    st.write("Image segmentation is the task of dividing an image into distinct regions or segments, where each region corresponds to an object or a part of an object. This can be applied to various fields : Medical imaging, Robotics... There are two main types of segmentation: semantic segmentation and instance segmentation.")
    st.write("Semantic Segmentation: It involves labeling each pixel in the image with an object class, indicating which class each pixel belongs to. For example, in an image of a street, each pixel might be labeled as road, sidewalk, car, pedestrian, etc.")
    st.write("Instance Segmentation: It goes beyond semantic segmentation by not only distinguishing object classes but also individual instances of objects within the same class. For example, in an image with multiple cars, each car's instance would be identified individually.")

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.image("images/object-detection.png",width=400, use_column_width=False, caption="Object detection")
    st.write("")
    st.write("")
    st.image("images/semantic-segmentation.png",width=400, use_column_width=False, caption="Semantic segmentation")
    st.write("")
    st.image("images/instance-segmentation.png", width=400, use_column_width=False, caption="Instance segmentation")
