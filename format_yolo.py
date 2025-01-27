import os
import cv2

# Convertir coordenadas a formato YOLO
def convert_to_yolo_format(image_width, image_height, bbox):
    """
    Convierte las coordenadas absolutas de la bounding box al formato YOLO normalizado.
    Args:
        image_width (int): Ancho de la imagen.
        image_height (int): Alto de la imagen.
        bbox (list): Bounding box en formato [x_min, y_min, width, height].
    Returns:
        list: Coordenadas en formato YOLO [x_center, y_center, width, height].
    """
    x_min = max(0, min(bbox[0], image_width))
    y_min = max(0, min(bbox[1], image_height))
    x_max = max(0, min(bbox[0] + bbox[2], image_width))
    y_max = max(0, min(bbox[1] + bbox[3], image_height))
    
    width = x_max - x_min
    height = y_max - y_min

    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    width = width / image_width
    height = height / image_height

    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        raise ValueError(f"Coordenadas normalizadas fuera de rango: {[x_center, y_center, width, height]}")

    return x_center, y_center, width, height

# Procesar un archivo de anotaciones
def process_annotations(input_annotations, images_dir, labels_dir):
    """
    Procesa un archivo de anotaciones y genera etiquetas en formato YOLO.
    Args:
        input_annotations (str): Ruta al archivo de anotaciones.
        images_dir (str): Directorio de las imágenes.
        labels_dir (str): Directorio donde se guardarán las etiquetas.
    """
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    with open(input_annotations, "r") as f:
        lines = f.readlines()

    image_name = ""
    for i, line in enumerate(lines):
        line = line.strip()
        if ".jpg" in line:  
            image_name = line
            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                print(f"Imagen no encontrada: {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"No se pudo leer la imagen: {image_path}")
                continue
            image_height, image_width = image.shape[:2]

            label_path = os.path.join(labels_dir, image_name.replace(".jpg", ".txt"))
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            label_file = open(label_path, "w")
            unique_labels = set()
        elif line.isdigit():  
            continue
        elif line:  
            bbox = list(map(int, line.split()[:4]))  
            try:
                yolo_bbox = convert_to_yolo_format(image_width, image_height, bbox)
                if tuple(yolo_bbox) not in unique_labels:
                    unique_labels.add(tuple(yolo_bbox))
                    label_file.write(f"0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")
            except ValueError as e:
                print(f"Error procesando bbox en línea {i}: {e}")
    label_file.close()

# Directorios y archivos de anotaciones
datasets = [
    {
        "input_annotations": "data/wider_face/wider_face_split/wider_face_train_bbx_gt.txt",
        "images_dir": "data/wider_face/WIDER_train/images/",
        "labels_dir": "data/wider_face/WIDER_train/labels/",
    },
    {
        "input_annotations": "data/wider_face/wider_face_split/wider_face_val_bbx_gt.txt",
        "images_dir": "data/wider_face/WIDER_val/images/",
        "labels_dir": "data/wider_face/WIDER_val/labels/",
    },
]

# Procesar entrenamiento y validación
for dataset in datasets:
    process_annotations(dataset["input_annotations"], dataset["images_dir"], dataset["labels_dir"])
    print(f"Procesado: {dataset['input_annotations']}")