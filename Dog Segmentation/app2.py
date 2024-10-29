import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO
import cv2

# Carga el modelo YOLO preentrenado
model = YOLO("best.pt")

def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

def predict_video(video_path, conf_threshold, iou_threshold):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,
            show_conf=True,
            imgsz=640,
        )

        for r in results:
            frame = r.plot()

        # Convierte el frame a formato PIL para mostrar en la interfaz
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        yield pil_img

    cap.release()

def predict_webcam(conf_threshold, iou_threshold):
    cap = cv2.VideoCapture(0)  # Accede a la webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,
            show_conf=True,
            imgsz=640,
        )

        for r in results:
            frame = r.plot()

        # Convierte el frame a formato PIL para mostrar en la interfaz
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        yield pil_img

    cap.release()

iface_img = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image", sources=['upload']),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Dog Segmentation - Image",
    description="Upload images for inference.",
    allow_flagging="never"
)

iface_video = gr.Interface(
    fn=predict_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Dog Segmentation - Video",
    description="Upload videos for inference.",
    allow_flagging="never"
)

iface_webcam = gr.Interface(
    fn=predict_webcam,
    inputs=[
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Dog Segmentation - Webcam",
    description="Use your webcam for real-time inference.",
    allow_flagging="never",
    live=True
)

if __name__ == "__main__":
    # Lanzar todas las interfaces en una única interfaz combinada
    gr.TabbedInterface([iface_img, iface_video, iface_webcam], ["Image Upload", "Video Upload", "Webcam"]).launch()
