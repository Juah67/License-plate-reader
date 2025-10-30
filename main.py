import cv2
import numpy as np
from funciones import *
import torch
from yolov5.utils.plots import Annotator, colors
import easyocr
import os
from datetime import datetime
import re
import warnings
#Cargar modelos
COCO_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
license_plate_detector = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

#Cargar video
print("1. Archivo de video")
print("2. Webcam USB")
print("3. Cámara IP")
a = int(input("Selecciona la fuente (1/2/3): "))

try:
    if a == 1:
        # Archivo de video
        video_path = r"video.mp4"
        cap = cv2.VideoCapture(video_path)
        source_name = os.path.basename(video_path)
        
    elif a == 2:
        # Webcam USB
        webcam_index = 0

        cap = cv2.VideoCapture(webcam_index)
        source_name = f"webcam_{webcam_index}"
        
    elif a == 3:
        # Cámara IP RTSP
        rtsp_url = r"rtsp://admin:admin@192.168.88.38:554/live"

        cap = cv2.VideoCapture(rtsp_url)
        source_name = "camara_ip"
    else:
        exit()   
except Exception as e:

    exit()

# Crear carpeta principal para todas las detecciones
main_output_dir = "detecciones_historico"
os.makedirs(main_output_dir, exist_ok=True)

# Crear carpeta específica para esta ejecución
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(main_output_dir, f"run_{timestamp}_{source_name}")


# Configurar VideoWriter para guardar el video procesado
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:  
    fps = 30  # FPS predeterminado
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_output_path = os.path.join(run_dir, "video_procesado.mp4")
video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))



vehicles = [2, 3]  # Car, motorcycle

FramesPerPlate = 3
FramesPerOCR = 5  # Leer OCR cada 5 frames
reader = easyocr.Reader(["es"], gpu=True)

#Leer frames
ret = True
frame_number = -1
ocr_counter = 0
ultimo_texto = ""
ultima_bbox_plate = None
detecciones_guardadas = []

while ret:
    frame_number += 1
    ocr_counter += 1
    ret, frame = cap.read()
    
  
    if ret:
        frame_original = frame.copy()  # Copiar frame original
        
        #Detectar vehículos
        detections = COCO_model(frame)
        annotator = Annotator(frame) #Donde se dibuja texto i bbox
        detections_v = []
        for *box, score, class_id in detections.xyxy[0]:
            x1, y1, x2, y2 = box
            if int(class_id) in vehicles:
                detections_v.append([x1.item(), y1.item(), x2.item(), y2.item(), score.item()])

        if len(detections_v):

            #Centrarse en el vehículo central
            alto, ancho, canales = frame.shape
            centered_car = centered_bbox(detections_v, ancho, alto)
            x1, y1, x2, y2, conf = centered_car
            ancho_bbox = x2 - x1
            alto_bbox = y2 - y1
            margen_x = int(ancho_bbox * 0.1)
            margen_y = int(alto_bbox * 0.1)
            
            x1 = max(0, int(x1 - margen_x))
            y1 = max(0, int(y1 - margen_y))
            x2 = min(frame.shape[1], int(x2 + margen_x))
            y2 = min(frame.shape[0], int(y2 + margen_y))
            x1_car, y1_car = x1, y1
            annotator.box_label((x1, y1, x2, y2), "CAR", color=(0,255, 0))
            
            #Lectura de matricula cada FramesPerPlate
            if frame_number%FramesPerPlate == 0:

                #Recortar frame a solo el coche
                frame_crop = frame_original[y1:y2, x1:x2]
                cv2.imshow("Coche", frame_crop)
                cv2.waitKey(1)
                #Detectar Matrícula
                license_plate = license_plate_detector(frame_crop)
                if len(license_plate.xyxy[0]) > 0:
                    detection = license_plate.xyxy[0][0]
                    x1_plate = int(detection[0].item())
                    y1_plate = int(detection[1].item())
                    x2_plate = int(detection[2].item())
                    y2_plate = int(detection[3].item())

                    # Guardar posición de matrícula
                    ultima_bbox_plate = (x1_plate + x1_car, y1_plate + y1_car, x2_plate + x1_car, y2_plate + y1_car)

                    frame_plate = frame_crop[y1_plate:y2_plate, x1_plate:x2_plate]
                    if frame_plate.size > 0: #Existe matrícula

                        #Filtros
                        frame_crop_grey = cv2.cvtColor(frame_plate, cv2.COLOR_BGR2GRAY)
                        
                        _, frame_crop_threshold = cv2.threshold(frame_crop_grey, 140, 255, cv2.THRESH_BINARY_INV)
                        cv2.imshow("Matrícula", frame_crop_threshold)
                        cv2.waitKey(1)

                        #Leer matrícula cada FramesPerOCR
                        if ocr_counter >= FramesPerOCR:
                            ocr_counter = 0  # Resetear contador
                            results = reader.readtext(frame_crop_threshold)
                            if results:
                                # Ordenar por posición X (coordenada horizontal)
                                results_ordenados = sorted(results, key=lambda x: x[0][0][0])
                                texto_final = []
                                for (bbox_ocr, text, conf_ocr) in results_ordenados:
                                    # Limpiar texto: solo letras y números
                                    texto_limpio = re.sub(r'[^A-Za-z0-9]', '', text.upper())
                                    if texto_limpio:  # Solo añadir si no está vacío
                                        texto_final.append(texto_limpio)
                                ultimo_texto = ''.join(texto_final)  # Sin espacios ni caracteres especiales

                            else:
                                ultimo_texto = "NaN"
                            
            # Mostrar el último texto leído FUERA del condicional de detección
            if ultimo_texto and ultima_bbox_plate:
                annotator.box_label(ultima_bbox_plate, ultimo_texto, color=(0,255, 0))
        
        # Guardar frame procesado en el video de salida
        video_writer.write(frame)
        
        cv2.imshow("a", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir
            break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

