import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import pytesseract

# Cargar el modelo
model_path = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Inicializar la cámara
cap = cv2.VideoCapture('/dev/video1')

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def detect_objects(interpreter, image, threshold):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()

    # Obtener resultados
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Caja delimitadora
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Clase
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confianza
    count = int(interpreter.get_tensor(output_details[3]['index'])[0])

    return boxes, classes, scores, count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar para la detección
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))

    # Detección de objetos
    boxes, classes, scores, count = detect_objects(interpreter, image_resized, 0.5)

    # Mostrar los resultados
    for i in range(count):
        if scores[i] > 0.5:
            # Dibujar la caja delimitadora
            ymin, xmin, ymax, xmax = boxes[i]
            (startX, startY, endX, endY) = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]),
                                            int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))

            # Extraer la región de la placa
            placa = frame[startY:endY, startX:endX]
            # Reconocer el texto de la placa con OCR
            text = pytesseract.image_to_string(placa, config='--psm 8')
            print(f"Placa detectada: {text}")

            # Dibujar la caja alrededor de la placa
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Mostrar el resultado
    cv2.imshow('Detección de placas', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
