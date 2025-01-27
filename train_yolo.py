from ultralytics import YOLO
import torch

# Cargar el modelo YOLO base
model = YOLO("/models/yolo11n.pt")  # Modelo base

# Entrenar el modelo
model.train(data="/configs/wider_face.yaml",  # Configuración del dataset
            epochs=50,                   # Número de épocas
            batch=16,                    # Tamaño de batch
            device=device)               # Usar MPS o CPU según disponibilidad

# Exportar el modelo entrenado en formato ONNX (opcional)
model.export(format="onnx")