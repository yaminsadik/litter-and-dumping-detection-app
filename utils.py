"""
utils.py - Inference Utility Functions
========================================

This module provides all the helper functions used by app.py to run YOLOv8
object detection inference on different input sources (image, video, webcam).

Functions
---------
load_model(model_path)
    Load and cache a YOLOv8 model from disk.
_display_detected_frames(conf, model, st_count, st_frame, image)
    Internal helper: run inference on a single frame and render results.
infer_uploaded_image(conf, model)
    Handle image upload, run inference, and display annotated results,
    detection statistics, bar chart, and download buttons.
infer_uploaded_video(conf, model)
    Handle video upload, run frame-by-frame inference, stream results with a
    progress bar.
infer_uploaded_webcam(conf, model)
    Capture frames from the default webcam and stream live detections.

Notes
-----
- YOLOv8 inference is performed via the ``ultralytics`` library.
- OpenCV (cv2) is used for video capture and frame decoding.
- Streamlit widgets are used for uploading files and rendering output.
"""

from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import io
import pandas as pd
import config


def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Run YOLOv8 inference on a single video frame and render the annotated
    result inside the Streamlit app.

    This is an internal helper called on each frame by both
    ``infer_uploaded_video`` and ``infer_uploaded_webcam``.

    Parameters
    ----------
    conf : float
        Confidence threshold in the range [0.0, 1.0].  Detections below this
        score are discarded.
    model : ultralytics.YOLO
        A loaded YOLOv8 model instance (returned by ``load_model``).
    st_count : streamlit.delta_generator.DeltaGenerator
        Streamlit placeholder used to display object count text above the
        video frame.
    st_frame : streamlit.delta_generator.DeltaGenerator
        Streamlit placeholder used to display the annotated video frame.
    image : numpy.ndarray
        A single BGR video frame as a NumPy array (H × W × 3).

    Returns
    -------
    None
    """
    # Run YOLOv8 inference; results contain bounding boxes, classes, scores
    res = model.predict(image, conf=conf)
    
    # Build object count summary strings from global counters.
    # OBJECT_COUNTER1 tracks "in" events; OBJECT_COUNTER tracks "out" events.
    inText = 'Vehicle In'
    outText = 'Vehicle Out'
    if config.OBJECT_COUNTER1 != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER1.items()):
            inText += ' - ' + str(key) + ": " +str(value)
    if config.OBJECT_COUNTER != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER.items()):
            outText += ' - ' + str(key) + ": " +str(value)
    
    # Render the count text and annotated frame inside the Streamlit placeholders
    st_count.write(inText + '\n\n' + outText)
    res_plotted = res[0].plot()   # Returns the frame with bounding boxes drawn
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    
@st.cache_resource
def load_model(model_path):
    """
    Load a YOLOv8 model from disk and cache it for the Streamlit session.

    Using ``@st.cache_resource`` ensures the model is only read from disk
    once per session, avoiding redundant I/O on every Streamlit rerun.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Absolute or relative path to a YOLOv8 ``.pt`` weights file.

    Returns
    -------
    ultralytics.YOLO
        An initialised YOLOv8 model ready for inference.

    Raises
    ------
    FileNotFoundError
        If the weights file does not exist at the given path.
    """
    model = YOLO(model_path)
    return model

def infer_uploaded_image(conf, model):
    """
    Handle image upload and run YOLOv8 detection on the uploaded file.

    Renders a two-column layout: the original image on the left and the
    annotated result (with bounding boxes) on the right.  Detection is
    triggered by clicking the "Execution" button.

    Below the annotated image, an expandable panel lists the bounding-box
    coordinates (x-centre, y-centre, width, height) for every detection.

    Parameters
    ----------
    conf : float
        Confidence threshold in [0.0, 1.0] controlling which predictions are
        shown.  Set by the sidebar slider in app.py.
    model : ultralytics.YOLO
        A loaded YOLOv8 model instance (returned by ``load_model``).

    Returns
    -------
    None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # Display the original (un-annotated) uploaded image
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )
    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                # Convert from BGR (OpenCV) to RGB (PIL/Streamlit) for display
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)

                    # ── Download annotated image ───────────────────────────
                    img_buf = io.BytesIO()
                    Image.fromarray(res_plotted).save(img_buf, format="PNG")
                    st.download_button(
                        label="⬇ Download Annotated Image",
                        data=img_buf.getvalue(),
                        file_name="ecowatch_result.png",
                        mime="image/png",
                    )

                # ── Detection statistics ───────────────────────────────
                with st.expander("📊 Detection Results", expanded=True):
                    try:
                        if boxes is not None and len(boxes) > 0:
                            names = model.names  # {class_id: class_name}
                            rows = []
                            for box in boxes:
                                cls_id   = int(box.cls[0])
                                conf_val = float(box.conf[0])
                                xywh     = box.xywh[0].tolist()
                                rows.append({
                                    "Class":      names[cls_id],
                                    "Confidence": round(conf_val, 3),
                                    "X Centre":   round(xywh[0], 1),
                                    "Y Centre":   round(xywh[1], 1),
                                    "Width":      round(xywh[2], 1),
                                    "Height":     round(xywh[3], 1),
                                })
                            df = pd.DataFrame(rows)

                            st.metric("Total Detections", len(df))
                            st.dataframe(df, use_container_width=True)

                            # Bar chart of detections per class
                            st.subheader("Detections by Class")
                            class_counts = (
                                df["Class"]
                                .value_counts()
                                .rename_axis("Class")
                                .reset_index(name="Count")
                            )
                            st.bar_chart(class_counts.set_index("Class")["Count"])

                            # CSV export
                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇ Download Detections CSV",
                                data=csv_bytes,
                                file_name="ecowatch_detections.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info(
                                "No objects detected at the current confidence "
                                "threshold.  Try lowering the slider."
                            )
                    except Exception as ex:
                        st.write("Error rendering detection results.")
                        st.write(ex)




def infer_uploaded_video(conf, model):
    """
    Handle video upload and run frame-by-frame YOLOv8 detection.

    The uploaded video is written to a temporary file so that OpenCV can open
    it with ``VideoCapture``.  Each frame is processed sequentially and
    displayed live in the Streamlit app via placeholder elements.

    Object counters in ``config`` are reset to ``None`` at the start of each
    inference run so stale counts from a previous video are cleared.

    Parameters
    ----------
    conf : float
        Confidence threshold in [0.0, 1.0].
    model : ultralytics.YOLO
        A loaded YOLOv8 model instance.

    Returns
    -------
    None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    # Reset global object counters for this inference session
                    config.OBJECT_COUNTER1 = None
                    config.OBJECT_COUNTER = None
                    # Write the uploaded bytes to a temp file for OpenCV
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)

                    # Determine total frame count for progress reporting
                    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_num    = 0

                    st_progress = st.progress(0, text="Starting…")
                    st_count = st.empty()   # Placeholder for object count text
                    st_frame = st.empty()   # Placeholder for the annotated frame
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            frame_num += 1
                            # Update progress bar (guard against divide-by-zero)
                            progress_pct = (
                                min(frame_num / total_frames, 1.0)
                                if total_frames > 0 else 0
                            )
                            st_progress.progress(
                                progress_pct,
                                text=f"Frame {frame_num} / {total_frames}",
                            )
                            _display_detected_frames(conf,
                                                     model,
                                                     st_count,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()  # Release capture when video ends
                            break
                    st_progress.progress(1.0, text="Done!")
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Capture frames from the default webcam and run live YOLOv8 detection.

    Opens the system's primary camera (device index 0) via OpenCV and streams
    annotated frames into the Streamlit app.  Inference continues until either
    the "Stop running" button is pressed or the camera becomes unavailable.

    Parameters
    ----------
    conf : float
        Confidence threshold in [0.0, 1.0].
    model : ultralytics.YOLO
        A loaded YOLOv8 model instance.

    Returns
    -------
    None

    Notes
    -----
    - Requires a connected camera accessible at device index 0.
    - In cloud-hosted environments (e.g., Streamlit Cloud) the webcam source
      will not be available; use Image or Video mode instead.
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # Open default camera (index 0)
        st_count = st.empty()   # Placeholder for object count text
        st_frame = st.empty()   # Placeholder for the annotated frame
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_count,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()  # Release camera if frame read fails
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")