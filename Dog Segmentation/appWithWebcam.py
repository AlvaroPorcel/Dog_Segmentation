import numpy as np
from PIL import Image, ImageDraw, ImageOps
import gradio as gr
from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO preentrenado
model = YOLO("best.pt")

def analyze_pupil(image, box):
    """Analiza la cantidad de negro en el iris."""
    x1, y1, x2, y2 = box
    eye_image = image.crop((x1, y1, x2, y2))
    eye_image = ImageOps.grayscale(eye_image)
    eye_array = np.array(eye_image)

    # Umbral para identificar negro
    black_threshold = 50
    black_pixels = np.sum(eye_array < black_threshold)
    total_pixels = eye_array.size

    # Porcentaje de negro en el iris
    black_percentage = black_pixels / total_pixels

    return black_percentage

def categorize_tail_speed(speed):
    """Categoriza la velocidad de la cola."""
    if speed > 130:
        return "Altos: Indican que la cola se está moviendo rápidamente. Esto podría ser un signo de excitación o agitación del perro."
    elif speed > 25:
        return "Moderados: Pueden indicar un movimiento moderado de la cola, posiblemente asociado con calma o interés."
    elif speed > 5:
        return "Bajos: Sugieren que la cola se está moviendo lentamente, lo que podría ser una señal de tranquilidad o relajación."
    else:
        return "Muy bajo: La cola está casi inmóvil, indicando posible tranquilidad."

def calculate_direction_and_state(image, bounding_boxes):
    """Calcula la dirección del perro y su estado, además de la posición de las orejas y cola."""
    draw = ImageDraw.Draw(image)
    centers = {'Eyes': [], 'Snout': [], 'Ears': [], 'Tail': []}

    ear_boxes = []
    tail_box = None
    snout_box = None

    # Extraer los centroides de las bounding boxes
    for label, box in bounding_boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers[label].append((center_x, center_y))

        # Dibujar el centro de cada box
        draw.ellipse([(center_x - 3, center_y - 3), (center_x + 3, center_y + 3)], fill='blue', outline='blue')

        if label == 'Ears':
            ear_boxes.append((x1, y1, width, height))
        elif label == 'Tail':
            tail_box = (x1, y1, width, height)
        elif label == 'Snout':
            snout_box = (x1, y1, width, height)

    if len(centers['Eyes']) == 2 and len(centers['Snout']) == 1:
        eye_right, eye_left = centers['Eyes']
        nose = centers['Snout'][0]

        # Dibujar líneas para formar un triángulo conectando los puntos azules
        draw.line([eye_right, eye_left], fill='blue', width=2)
        draw.line([eye_left, nose], fill='blue', width=2)
        draw.line([nose, eye_right], fill='blue', width=2)

        # Calcular la dirección del perro en función de la posición del hocico
        eye_mid_x = (eye_right[0] + eye_left[0]) / 2
        distance_to_eye_right = abs(nose[0] - eye_right[0])
        distance_to_eye_left = abs(nose[0] - eye_left[0])
        eye_distance = abs(eye_right[0] - eye_left[0])
        threshold = 0.2 * eye_distance  # 20% de la distancia entre los ojos

        if abs(nose[0] - eye_mid_x) < threshold:
            direction = 'Straight'
        elif distance_to_eye_right < distance_to_eye_left:
            direction = 'Right'
        elif distance_to_eye_left < distance_to_eye_right:
            direction = 'Left'
        else:
            direction = 'Straight'

        # Analizar los ojos para determinar el estado (nervioso o calmado)
        black_percentage_right = analyze_pupil(image, (eye_right[0] - 20, eye_right[1] - 20, eye_right[0] + 20,
                                                       eye_right[1] + 20))  # Ajusta el tamaño de la región según sea necesario
        black_percentage_left = analyze_pupil(image, (eye_left[0] - 20, eye_left[1] - 20, eye_left[0] + 20, eye_left[
            1] + 20))  # Ajusta el tamaño de la región según sea necesario
        black_threshold = 0.3  # Ajusta este umbral según sea necesario

        if black_percentage_right > black_threshold or black_percentage_left > black_threshold:
            state = 'Nervous'
        else:
            state = 'Calm'

    else:
        direction, state = "Unknown", "Unknown"

    # Añadir el análisis de orejas y cola
    if ear_boxes and tail_box and snout_box:
        ear_centroids = []
        for x1, y1, w, h in ear_boxes:
            centroid = (x1 + w / 2, y1 + h / 2)
            ear_centroids.append(centroid)
            draw.rectangle([(x1, y1), (x1 + w, y1 + h)], outline="red", width=2)
            draw.line((centroid[0] - 5, centroid[1], centroid[0] + 5, centroid[1]), fill="yellow", width=2)
            draw.line((centroid[0], centroid[1] - 5, centroid[0], centroid[1] + 5), fill="yellow", width=2)
            draw.line((x1, y1 + h / 2, x1 + w, y1 + h / 2), fill="yellow", width=2)  # Línea horizontal en la caja
            draw.line((x1 + w / 2, y1, x1 + w / 2, y1 + h), fill="yellow", width=2)  # Línea vertical en la caja

        x1, y1, w, h = tail_box
        tail_centroid = (x1 + w / 2, y1 + h / 2)
        draw.rectangle([(x1, y1), (x1 + w, y1 + h)], outline="blue", width=2)
        draw.line((tail_centroid[0] - 5, tail_centroid[1], tail_centroid[0] + 5, tail_centroid[1]), fill="blue", width=2)
        draw.line((tail_centroid[0], tail_centroid[1] - 5, tail_centroid[0], tail_centroid[1] + 5), fill="blue", width=2)

        # Dibujar una línea horizontal que pase por el centroide de la cola a través de toda la imagen
        image_width, image_height = image.size
        draw.line([(0, tail_centroid[1]), (image_width, tail_centroid[1])], fill="blue", width=2)

        # Determinar si el `box` de las orejas está arriba o abajo de la línea horizontal de la cola
        ears_above_line = any([centroid[1] < tail_centroid[1] for centroid in ear_centroids])

        # Determinar si la cola está arriba o abajo
        tail_up = not ears_above_line

        # Determinar el estado del perro
        if ears_above_line and tail_up:
            behavior_state = "Juguetón"
        elif not ears_above_line and not tail_up:
            behavior_state = "Intranquilo"
        elif not ears_above_line and tail_up:
            behavior_state = "Tranquilo"
        else:
            behavior_state = "Enfadado"

        # Posiciones detalladas de orejas y cola
        ears_pos = "arriba" if ears_above_line else "abajo"
        tail_pos = "arriba" if tail_up else "abajo"

        # Mensaje final
        final_message = f"Las orejas están {ears_pos}. La cola está {tail_pos}. El perro está: {behavior_state}"

    else:
        final_message = "No se detectaron orejas o cola suficientes para determinar el comportamiento."

    return image, final_message

def predict_image(img, conf_threshold, iou_threshold):
    """Predice la dirección y estado del perro en la imagen usando YOLO."""
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
        num_boxes = len(r.boxes)  # Contar el número de bounding boxes detectadas

        # Obtener las bounding boxes necesarias (ojos, nariz, orejas y cola)
        bounding_boxes = [(r.names[int(box.cls)], box.xyxy[0].tolist()) for box in r.boxes if
                          r.names[int(box.cls)] in ['Eyes', 'Snout', 'Ears', 'Tail']]

        if bounding_boxes:
            im, final_message = calculate_direction_and_state(im, bounding_boxes)
        else:
            final_message = "No se detectaron orejas o cola suficientes para determinar el comportamiento."

    return im, f"Number of boxes detected: {num_boxes}", final_message

def process_frame(frame, conf_threshold, iou_threshold, prev_tail_centroid):
    """Procesa un solo frame y calcula la velocidad de la cola."""
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

        # Obtener las bounding boxes necesarias (orejas y cola)
        bounding_boxes = [(r.names[int(box.cls)], box.xyxy[0].tolist()) for box in r.boxes if
                          r.names[int(box.cls)] in ['Tail']]

        if bounding_boxes:
            x1, y1, w, h = bounding_boxes[0][1]
            tail_centroid = (x1 + w / 2, y1 + h / 2)

            if prev_tail_centroid is not None:
                # Calcular la velocidad de la cola
                distance = np.linalg.norm(np.array(tail_centroid) - np.array(prev_tail_centroid))
                # Asumiendo un intervalo de tiempo de 1 segundo entre cuadros
                tail_speed = distance  # Puedes ajustar el cálculo dependiendo de los fps del video
            else:
                tail_speed = 0

            return frame, tail_centroid, tail_speed

    return frame, prev_tail_centroid, 0

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

        # Procesar el frame en un thread separado
        frame, prev_tail_centroid, tail_speed = process_frame(frame, conf_threshold, iou_threshold, prev_tail_centroid)

        # Convertir el frame a formato PIL para mostrar en la interfaz
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        speed_description = categorize_tail_speed(tail_speed)
        final_message = f"Velocidad de la cola: {tail_speed:.2f} píxeles/segundo. {speed_description}"

        yield pil_img, final_message

    cap.release()

def predict_webcam(conf_threshold, iou_threshold):
    """Procesa la transmisión en vivo de la webcam para predecir la dirección y estado del perro."""
    cap = cv2.VideoCapture(0)  # Accede a la webcam

    # Variable para almacenar la posición anterior de la cola
    prev_tail_centroid = None
    tail_speed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar el frame en un thread separado
        frame, prev_tail_centroid, tail_speed = process_frame(frame, conf_threshold, iou_threshold, prev_tail_centroid)

        # Convertir el frame a formato PIL para mostrar en la interfaz
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        speed_description = categorize_tail_speed(tail_speed)
        final_message = f"Velocidad de la cola: {tail_speed:.2f} píxeles/segundo. {speed_description}"

        yield pil_img, final_message

    cap.release()

# Interfaz Gradio para cargar imágenes
iface_img = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image", sources=['upload']),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Result"),
        gr.Text(label="Bounding Box Count"),
        gr.Text(label="Behavior"),
    ],
    title="Dog Orientation and State Detection - Image",
    description="Upload images for inference.",
    allow_flagging="never"
)

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
        gr.Text(label="Bounding Box Count"),
    ],
    title="Dog Orientation and State Detection - Video",
    description="Upload videos for inference.",
    allow_flagging="never"
)

# Interfaz Gradio para transmisión en vivo desde la webcam
iface_webcam = gr.Interface(
    fn=predict_webcam,
    inputs=[
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Result"),
        gr.Text(label="Bounding Box Count"),
    ],
    title="Dog Orientation and State Detection - Webcam",
    description="Use your webcam for real-time inference.",
    allow_flagging="never",
    live=True
)

if __name__ == "__main__":
    # Lanzar todas las interfaces en una única interfaz combinada
    gr.TabbedInterface([iface_img, iface_video, iface_webcam], ["Image Upload", "Video Upload", "Webcam"]).launch()


