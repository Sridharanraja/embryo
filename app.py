# import streamlit as st
# import cv2
# import tempfile
# import numpy as np
# from ultralytics import YOLO

# LABEL_MEANINGS = {
#     "1-0_Unfertilized": ("Unfertilized Oocyte", "Not an embryo", "NON-VIABLE"),

#     "2-1": ("2–8 Cell Stage", "Excellent", "GOOD EMBRYO"),
#     "2-2": ("2–8 Cell Stage", "Good", "AVERAGE EMBRYO"),
#     "2-3": ("2–8 Cell Stage", "Poor", "BAD EMBRYO"),
#     "2-4": ("2–8 Cell Stage", "Very Poor", "BAD EMBRYO"),

#     "3-1": ("Early Morula", "Excellent", "GOOD EMBRYO"),
#     "3-2": ("Early Morula", "Good", "AVERAGE EMBRYO"),
#     "3-3": ("Early Morula", "Poor", "BAD EMBRYO"),
#     "3-4": ("Early Morula", "Very Poor", "BAD EMBRYO"),

#     "4-1": ("Morula", "Excellent", "GOOD EMBRYO"),
#     "4-2": ("Morula", "Good", "AVERAGE EMBRYO"),
#     "4-3": ("Morula", "Poor", "BAD EMBRYO"),
#     "4-4": ("Morula", "Very Poor", "BAD EMBRYO"),

#     "5-1": ("Early Blastocyst", "Excellent", "GOOD EMBRYO"),
#     "5-2": ("Early Blastocyst", "Good", "AVERAGE EMBRYO"),
#     "5-3": ("Early Blastocyst", "Poor", "BAD EMBRYO"),
#     "5-4": ("Early Blastocyst", "Very Poor", "BAD EMBRYO"),

#     "6-1": ("Blastocyst", "Excellent", "GOOD EMBRYO"),
#     "6-2": ("Blastocyst", "Good", "AVERAGE EMBRYO"),
#     "6-3": ("Blastocyst", "Poor", "BAD EMBRYO"),
#     "6-4": ("Blastocyst", "Very Poor", "BAD EMBRYO"),

#     "7-1": ("Expanded Blastocyst", "Excellent", "GOOD EMBRYO"),
#     "7-2": ("Expanded Blastocyst", "Good", "AVERAGE EMBRYO"),
#     "7-3": ("Expanded Blastocyst", "Poor", "BAD EMBRYO"),
#     "7-4": ("Expanded Blastocyst", "Very Poor", "BAD EMBRYO"),

#     "8-1": ("Hatched Blastocyst", "Excellent", "GOOD EMBRYO"),
#     "8-2": ("Hatched Blastocyst", "Good", "AVERAGE EMBRYO"),
#     "8-3": ("Hatched Blastocyst", "Poor", "BAD EMBRYO"),
#     "8-4": ("Hatched Blastocyst", "Very Poor", "BAD EMBRYO"),

#     "9-1": ("Degenerated", "Mild", "BAD EMBRYO"),
#     "9-2": ("Degenerated", "Strong", "BAD EMBRYO"),
#     "9-3": ("Degenerated", "Severe", "BAD EMBRYO"),
#     "9-4": ("Degenerated", "Complete", "BAD EMBRYO"),

#     "Arrested": ("Arrested Embryo", "Stopped developing", "NON-VIABLE"),
#     "Empty_zona": ("Empty Zona", "No embryo present", "NON-VIABLE"),
#     "Unfertilized": ("Unfertilized Oocyte", "No fertilization", "NON-VIABLE"),
#     "Fragmented": ("Fragmented Embryo", "High fragmentation", "BAD EMBRYO"),
#     "Dead": ("Dead Embryo", "Non-viable", "NON-VIABLE")
# }

# def parse_label(label: str):
#     return LABEL_MEANINGS.get(label, ("Unknown", "Unknown", "Unknown"))


# def run_inference(model, img_path):
#     results = model(img_path, conf=0.50, verbose=False)
#     output = []

#     for r in results:
#         boxes = r.boxes
#         masks = r.masks

#         for i, box in enumerate(boxes):
#             cls_id = int(box.cls[0])
#             label = r.names[cls_id]
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

#             stage_name, quality_text, category = parse_label(label)

#             mask = None
#             if masks is not None:
#                 mask = masks.data[i].cpu().numpy()  # binary mask

#             output.append({
#                 "label": label,
#                 "stage_name": stage_name,
#                 "quality_text": quality_text,
#                 "category": category,
#                 "bbox": [x1, y1, x2, y2],
#                 "mask": mask
#             })
#     return output


# def draw_boxes(image_path, detections):
#     img = cv2.imread(image_path)

#     for det in detections:
#         x1, y1, x2, y2 = det["bbox"]
#         label = det["label"]
#         mask = det["mask"]

#         if mask is not None:
#             mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
#             overlay = np.zeros_like(img)
#             overlay[:, :, 1] = (mask * 255).astype("uint8")
#             img = cv2.addWeighted(img, 1, overlay, 0.4, 0)

#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# def crop_embryo(image_path, bbox):
#     img = cv2.imread(image_path)
#     x1, y1, x2, y2 = bbox
#     crop = img[y1:y2, x1:x2]
#     return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# st.set_page_config(page_title="Embryo Grading System", layout="wide")
# st.title("🧬 Cow Embryo Grading System")

# model = YOLO("./weight/best.pt")
# uploaded_image_path = None

# uploaded_file = st.file_uploader("Upload embryo microscope image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
#     temp_file.write(uploaded_file.read())
#     uploaded_image_path = temp_file.name

#     img = cv2.imread(uploaded_image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     st.image(img, caption="Uploaded Image", use_container_width=True)


# if uploaded_image_path:
#     detections = run_inference(model, uploaded_image_path)

#     if not detections:
#         st.error("No embryos detected!")
#         st.stop()

#     output_img = draw_boxes(uploaded_image_path, detections)
#     st.image(output_img, caption="Segmented Embryos", use_container_width=True)

#     st.subheader("🧫 Embryo-by-Embryo Results")

#     for i, det in enumerate(detections, start=1):
#         st.markdown(f"## 🔹 Embryo {i}")

#         # ------ Show Cropped Image ------
#         crop_img = crop_embryo(uploaded_image_path, det["bbox"])
#         st.image(crop_img, width=250, caption=f"Embryo {i} – Cropped")

#         # ------ Report ------
#         st.write("**Label:**", det["label"])
#         st.write("**Stage:**", det["stage_name"])
#         st.write("**Quality:**", det["quality_text"])
#         st.write("**Category:**", det["category"])

#         st.write("**Morphology Notes:**")
#         if det["category"] == "GOOD EMBRYO":
#             st.success("Healthy embryo with excellent structural quality.")
#         elif det["category"] == "AVERAGE EMBRYO":
#             st.warning("Moderately viable embryo with minor irregularities.")
#         elif det["category"] == "BAD EMBRYO":
#             st.error("Low-quality embryo with poor developmental features.")
#         else:
#             st.info("Not a viable embryo.")

#         st.divider()




import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------
# 1. LABEL MAPPING (Your Full Custom Set)
# -----------------------------------------------------------
LABEL_MEANINGS = {
    "1-0_Unfertilized": ("Unfertilized Oocyte", "Not an embryo", "NON-VIABLE"),

    "2-1": ("2–8 Cell Stage", "Excellent", "GOOD EMBRYO"),
    "2-2": ("2–8 Cell Stage", "Good", "AVERAGE EMBRYO"),
    "2-3": ("2–8 Cell Stage", "Poor", "BAD EMBRYO"),
    "2-4": ("2–8 Cell Stage", "Very Poor", "BAD EMBRYO"),

    "3-1": ("Early Morula", "Excellent", "GOOD EMBRYO"),
    "3-2": ("Early Morula", "Good", "AVERAGE EMBRYO"),
    "3-3": ("Early Morula", "Poor", "BAD EMBRYO"),
    "3-4": ("Early Morula", "Very Poor", "BAD EMBRYO"),

    "4-1": ("Morula", "Excellent", "GOOD EMBRYO"),
    "4-2": ("Morula", "Good", "AVERAGE EMBRYO"),
    "4-3": ("Morula", "Poor", "BAD EMBRYO"),
    "4-4": ("Morula", "Very Poor", "BAD EMBRYO"),

    "5-1": ("Early Blastocyst", "Excellent", "GOOD EMBRYO"),
    "5-2": ("Early Blastocyst", "Good", "AVERAGE EMBRYO"),
    "5-3": ("Early Blastocyst", "Poor", "BAD EMBRYO"),
    "5-4": ("Early Blastocyst", "Very Poor", "BAD EMBRYO"),

    "6-1": ("Blastocyst", "Excellent", "GOOD EMBRYO"),
    "6-2": ("Blastocyst", "Good", "AVERAGE EMBRYO"),
    "6-3": ("Blastocyst", "Poor", "BAD EMBRYO"),
    "6-4": ("Blastocyst", "Very Poor", "BAD EMBRYO"),

    "7-1": ("Expanded Blastocyst", "Excellent", "GOOD EMBRYO"),
    "7-2": ("Expanded Blastocyst", "Good", "AVERAGE EMBRYO"),
    "7-3": ("Expanded Blastocyst", "Poor", "BAD EMBRYO"),
    "7-4": ("Expanded Blastocyst", "Very Poor", "BAD EMBRYO"),

    "8-1": ("Hatched Blastocyst", "Excellent", "GOOD EMBRYO"),
    "8-2": ("Hatched Blastocyst", "Good", "AVERAGE EMBRYO"),
    "8-3": ("Hatched Blastocyst", "Poor", "BAD EMBRYO"),
    "8-4": ("Hatched Blastocyst", "Very Poor", "BAD EMBRYO"),

    "9-1": ("Degenerated", "Mild", "BAD EMBRYO"),
    "9-2": ("Degenerated", "Strong", "BAD EMBRYO"),
    "9-3": ("Degenerated", "Severe", "BAD EMBRYO"),
    "9-4": ("Degenerated", "Complete", "BAD EMBRYO"),

    "Arrested": ("Arrested Embryo", "Stopped developing", "NON-VIABLE"),
    "Empty_zona": ("Empty Zona", "No embryo present", "NON-VIABLE"),
    "Unfertilized": ("Unfertilized Oocyte", "No fertilization", "NON-VIABLE"),
    "Fragmented": ("Fragmented Embryo", "High fragmentation", "BAD EMBRYO"),
    "Dead": ("Dead Embryo", "Non-viable", "NON-VIABLE")
}

def parse_label(label: str):
    return LABEL_MEANINGS.get(label, ("Unknown", "Unknown", "Unknown"))


# -----------------------------------------------------------
# 2. YOLO DETECTION INFERENCE (NO SEGMENTATION)
# -----------------------------------------------------------
def run_inference(model, img_path):
    results = model(img_path, conf=0.50, verbose=False)
    output = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            stage_name, quality_text, category = parse_label(label)

            output.append({
                "label": label,
                "stage_name": stage_name,
                "quality_text": quality_text,
                "category": category,
                "bbox": [x1, y1, x2, y2]
            })

    return output


# -----------------------------------------------------------
# 3. DRAW DETECTION BOXES ONLY (NO MASKS)
# -----------------------------------------------------------
def draw_boxes(image_path, detections):
    img = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------
# 4. CROP FUNCTION
# -----------------------------------------------------------
def crop_embryo(image_path, bbox):
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------
# 5. STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="Embryo Grading System", layout="wide")
st.title("🧬 Cow Embryo Grading System")

model = YOLO("./weight/best_500.pt")   # 👉 detection model
uploaded_image_path = None

uploaded_file = st.file_uploader("Upload embryo microscope image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    uploaded_image_path = temp_file.name

    img = cv2.imread(uploaded_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption="Uploaded Image", use_container_width=True)


# -----------------------------------------------------------
# 6. RUN MODEL + SHOW RESULTS
# -----------------------------------------------------------
if uploaded_image_path:
    detections = run_inference(model, uploaded_image_path)

    if not detections:
        st.error("No embryos detected!")
        st.stop()

    output_img = draw_boxes(uploaded_image_path, detections)
    st.image(output_img, caption="Detected Embryos", use_container_width=True)

    st.subheader("🧫 Embryo-by-Embryo Results")

    for i, det in enumerate(detections, start=1):
        st.markdown(f"## 🔹 Embryo {i}")

        # ------ Cropped Image ------
        crop_img = crop_embryo(uploaded_image_path, det["bbox"])
        st.image(crop_img, width=250, caption=f"Embryo {i} – Cropped")

        # ------ Report ------
        st.write("**Label:**", det["label"])
        st.write("**Stage:**", det["stage_name"])
        st.write("**Quality:**", det["quality_text"])
        st.write("**Category:**", det["category"])

        if det["category"] == "GOOD EMBRYO":
            st.success("Healthy embryo with excellent structural quality.")
        elif det["category"] == "AVERAGE EMBRYO":
            st.warning("Moderately viable embryo with minor irregularities.")
        elif det["category"] == "BAD EMBRYO":
            st.error("Low-quality embryo with poor developmental features.")
        else:
            st.info("Not a viable embryo.")

        st.divider()
