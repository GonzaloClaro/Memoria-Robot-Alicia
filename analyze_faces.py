def analyze_faces_and_save(self, run_id: str, frame: cv2.typing.MatLike, conf_threshold=0.7) -> None:
    """
    Detecta rostros en un frame, analiza género, edad y emociones usando AWS Rekognition,
    y guarda el resultado JSON en S3.
    """
    # Procesar el frame en formato JPEG
    blob = cv2.imencode('.jpg', frame)[1].tobytes()
    results_file_name = f"rekognition/{CAM_DETECTION_PATH}/{run_id}.json"

    # Llamar a la API de AWS Rekognition para analizar el frame
    faces = detect_faces(self.aws_client, blob, ['ALL'])
    self.count_api_calls += 1

    if len(faces) == 0:
        print("No se detectaron rostros")
        return

    # Procesar la respuesta para extraer datos relevantes
    self.faces_data = list(map(
        lambda x: {
            "BoundingBox": x['BoundingBox'],
            "AgeRange": x['AgeRange'],
            "Smile": {
                "Value": x['Smile']['Value'],
                "Confidence": x['Smile']['Confidence']
            },
            "Gender": {
                "Value": x['Gender']['Value'],
                "Confidence": x['Gender']['Confidence']
            },
            "Emotions": [
                {"Type": emotion['Type'], "Confidence": emotion['Confidence']}
                for emotion in x['Emotions']
            ]
        },
        faces
    ))

    # Convertir los resultados a JSON y subir a S3
    jsonData = json.dumps(faces)
    json_file_obj = BytesIO(jsonData.encode())
    upload_file_to_s3(json_file_obj, results_file_name)

    print(f"Datos de rostros guardados en {results_file_name}")
