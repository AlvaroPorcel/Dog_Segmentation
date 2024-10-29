import numpy as np
from collections import Counter
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gradio as gr
from PIL import Image

# Cargar el modelo YOLO preentrenado
model = YOLO("best.pt")


def categorize_tail_speed(speed):
    """Categoriza la velocidad de la cola."""
    if speed > 130:
        return "Ansioso"
    elif speed > 25:
        return "Calmado"
    elif speed > 5:
        return "Muy relajado"
    else:
        return "Muy relajado"


def calculate_behavior(body_posture, tail_speed):
    """Evalúa el estado emocional del perro basado en la postura corporal y velocidad de la cola."""
    if tail_speed > 130 and body_posture == "Tenso":
        return "Ansioso"
    elif tail_speed > 130:
        return "Muy activo"
    elif 25 < tail_speed <= 130 and body_posture == "Relajado":
        return "Calmado"
    elif tail_speed <= 25 and body_posture == "Relajado":
        return "Muy relajado"
    else:
        return "Calmado"


def analyze_body_posture(bounding_boxes):
    """Analiza la postura corporal del perro."""
    positions = {'Head': None, 'Tail': None, 'Paws': []}

    for label, box in bounding_boxes:
        x1, y1, x2, y2 = box
        if label == 'Head':
            positions['Head'] = ((x1 + x2) / 2, (y1 + y2) / 2)
        elif label == 'Tail':
            positions['Tail'] = ((x1 + x2) / 2, (y1 + y2) / 2)
        elif label == 'Paws':
            positions['Paws'].append(((x1 + x2) / 2, (y1 + y2) / 2))

    if positions['Head'] and positions['Tail']:
        head_y = positions['Head'][1]
        tail_y = positions['Tail'][1]

        if head_y > tail_y:
            return "Tenso"
        else:
            return "Relajado"

    return "Desconocido"


def process_video(video_path, conf_threshold=0.25, iou_threshold=0.45):
    cap = cv2.VideoCapture(video_path)
    prev_tail_centroid = None
    tail_speed = 0
    frame_count = 0
    behaviors = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
        )

        frame_count += 1
        if frame_count % 5 == 0:  # Procesar cada 5 fotogramas
            for r in results:
                bounding_boxes = [(r.names[int(box.cls)], box.xyxy[0].tolist()) for box in r.boxes if
                                  r.names[int(box.cls)] in ['Tail', 'Head', 'Paws']]

                if bounding_boxes:
                    body_posture = analyze_body_posture(bounding_boxes)
                    tail_box = next((box for label, box in bounding_boxes if label == 'Tail'), None)
                    if tail_box:
                        x1, y1, x2, y2 = tail_box
                        tail_centroid = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)

                        if prev_tail_centroid is not None:
                            distance = np.linalg.norm(np.array(tail_centroid) - np.array(prev_tail_centroid))
                            tail_speed = distance

                        prev_tail_centroid = tail_centroid

                    behavior = calculate_behavior(body_posture, tail_speed)
                    behaviors.append(behavior)

    cap.release()
    return behaviors


def generate_behavior_graph(behaviors):
    """Genera una gráfica de comportamientos."""
    behavior_counts = Counter(behaviors)
    labels = list(behavior_counts.keys())
    counts = list(behavior_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color=['green', 'blue', 'red', 'orange'])
    plt.xlabel("Comportamiento")
    plt.ylabel("Cantidad de detecciones")
    plt.title("Distribución de Comportamientos del Perro")

    # Agregar etiquetas encima de las barras
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')

    # Guardar la gráfica como imagen
    plt.savefig("behavior_distribution.png")
    plt.close()
    return Image.open("behavior_distribution.png")


def predict_video(video_path, conf_threshold, iou_threshold):
    """Procesa el video, analiza los comportamientos, y genera la gráfica final."""
    behaviors = process_video(video_path, conf_threshold, iou_threshold)

    # Generar una imagen de la gráfica de comportamientos
    behavior_graph = generate_behavior_graph(behaviors)

    # Texto resumen del análisis
    summary_text = f"Resultados del video:\n\n" \
                   f"Calmado: {behaviors.count('Calmado')}\n" \
                   f"Muy relajado: {behaviors.count('Muy relajado')}\n" \
                   f"Ansioso: {behaviors.count('Ansioso')}"

    # Devolver la última imagen del video procesado, texto resumen, y gráfica de comportamiento
    return None, summary_text, behavior_graph


# Crear la interfaz Gradio para cargar videos y visualizar el análisis
iface_video = gr.Interface(
    fn=predict_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Frame Analizado"),
        gr.Text(label="Resumen de Comportamientos"),
        gr.Image(type="pil", label="Gráfica de Comportamientos"),
    ],
    title="Análisis de Comportamiento del Perro en Video",
    description="Analiza el comportamiento del perro en base al movimiento de la cola y la postura corporal.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface_video.launch()
