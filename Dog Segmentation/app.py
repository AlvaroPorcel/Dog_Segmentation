import numpy as np
from PIL import Image, ImageDraw, ImageOps
import gradio as gr
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Cargar el modelo YOLO preentrenado
model = YOLO("best.pt")


def categorize_tail_speed(speed):
    """Categoriza la velocidad de la cola."""
    if speed > 130:
        return "Altos (más de 130 píxeles/segundo): El perro puede estar ansioso o agitado."
    elif speed > 25:
        return "Moderados (25-130 píxeles/segundo): El perro está en calma o moderadamente activo."
    elif speed > 5:
        return "Bajos (5-25 píxeles/segundo): Indica tranquilidad o relajación."
    else:
        return "Muy bajo (menos de 5 píxeles/segundo): El perro está muy relajado."


def calculate_behavior(body_posture, tail_speed):
    """Evalúa el estado emocional del perro basado en la postura corporal y velocidad de la cola."""
    # Evaluar el estado general del perro según la velocidad de la cola y la postura
    if tail_speed > 130 and body_posture == "Tenso":
        return "El perro está ansioso."
    elif tail_speed > 130:
        return "El perro está muy activo o agitado."
    elif 25 < tail_speed <= 130 and body_posture == "Relajado":
        return "El perro está tranquilo y contento."
    elif tail_speed <= 25 and body_posture == "Relajado":
        return "El perro está muy relajado."
    else:
        return "Comportamiento indefinido."


def analyze_body_posture(image, bounding_boxes):
    """Analiza la postura corporal del perro."""
    # Evaluar la relación entre la cabeza, patas y cola
    positions = {'Head': None, 'Tail': None, 'Paws': []}

    # Obtener las posiciones del cuerpo
    for label, box in bounding_boxes:
        x1, y1, x2, y2 = box
        if label == 'Head':
            positions['Head'] = ((x1 + x2) / 2, (y1 + y2) / 2)
        elif label == 'Tail':
            positions['Tail'] = ((x1 + x2) / 2, (y1 + y2) / 2)
        elif label == 'Paws':
            positions['Paws'].append(((x1 + x2) / 2, (y1 + y2) / 2))

    # Analizar la postura según la posición de la cabeza y la cola
    if positions['Head'] and positions['Tail']:
        head_y = positions['Head'][1]
        tail_y = positions['Tail'][1]

        if head_y > tail_y:  # Si la cabeza está más baja que la cola
            return "Tenso"
        else:
            return "Relajado"

    return "Desconocido"


def predict_video(video_path, conf_threshold, iou_threshold):
    """Procesa un video para predecir la dirección y estado del perro cuadro a cuadro."""
    cap = cv2.VideoCapture(video_path)

    # Variable para almacenar la posición anterior de la cola
    prev_tail_centroid = None
    tail_speed = 0

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
            num_boxes = len(r.boxes)  # Contar el número de bounding boxes detectadas

            # Obtener las bounding boxes necesarias (cuerpo, orejas, cola, etc.)
            bounding_boxes = [(r.names[int(box.cls)], box.xyxy[0].tolist()) for box in r.boxes if
                              r.names[int(box.cls)] in ['Tail', 'Head', 'Paws']]

            if bounding_boxes:
                # Análisis de postura corporal
                body_posture = analyze_body_posture(frame, bounding_boxes)

                # Obtener la posición de la cola
                tail_box = next((box for label, box in bounding_boxes if label == 'Tail'), None)
                if tail_box:
                    x1, y1, x2, y2 = tail_box
                    tail_centroid = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)

                    if prev_tail_centroid is not None:
                        # Calcular la velocidad de la cola
                        distance = np.linalg.norm(np.array(tail_centroid) - np.array(prev_tail_centroid))
                        # Asumiendo un intervalo de tiempo de 1 segundo entre cuadros
                        tail_speed = distance  # Puedes ajustar el cálculo dependiendo de los fps del video

                    prev_tail_centroid = tail_centroid

            # Convertir el frame a formato PIL para mostrar en la interfaz
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)

            # Categorizar la velocidad de la cola
            speed_description = categorize_tail_speed(tail_speed)
            behavior = calculate_behavior(body_posture, tail_speed)
            final_message = f"Velocidad de la cola: {tail_speed:.2f} píxeles/segundo. {speed_description} El estado emocional es: {behavior}"

            yield pil_img, final_message

    cap.release()


# Interfaz Gradio para cargar videos
iface_video = gr.Interface(
    fn=predict_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Result"),
        gr.Text(label="Behavior"),
    ],
    title="Dog Tail and Body Posture Detection - Video",
    description="Analyze the dog's tail movement and body posture to infer emotional state.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface_video.launch()
