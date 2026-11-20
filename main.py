import argparse
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import easyocr
import torch
from yolov5.utils.plots import Annotator

from funciones import (
    DetectorMatriculas,
    PromedioMatriculas,
    centered_bbox,
    guardar_foto_deteccion,
    leer_matricula_ocr,
    seleccionar_mejor_matricula,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Lector de matrículas con YOLOv5 + EasyOCR")
    parser.add_argument("source", nargs="?", help="Ruta al archivo de video, índice de webcam o URL RTSP")
    parser.add_argument("--no-display", action="store_true", help="Ejecutar sin abrir ventanas OpenCV")
    return parser.parse_args()


def abrir_fuente(source_arg=None):
    """Abre la fuente de video desde argumento CLI o menú interactivo."""
    if source_arg is not None:
        if source_arg.isdigit():
            idx = int(source_arg)
            return cv2.VideoCapture(idx), f"webcam_{idx}"
        cap = cv2.VideoCapture(source_arg)
        if source_arg.lower().startswith("rtsp://"):
            return cap, "camara_ip"
        return cap, os.path.basename(source_arg)

    print("1. Archivo de video")
    print("2. Webcam USB")
    print("3. Cámara IP")
    try:
        opcion = int(input("Selecciona la fuente (1/2/3): "))
    except (ValueError, EOFError):
        return None, None

    if opcion == 1:
        video_path = "video.mp4"
        return cv2.VideoCapture(video_path), os.path.basename(video_path)
    if opcion == 2:
        webcam_index = 0
        return cv2.VideoCapture(webcam_index), f"webcam_{webcam_index}"
    if opcion == 3:
        rtsp_url = r"rtsp://admin:admin@192.168.88.38:554/live"
        return cv2.VideoCapture(rtsp_url), "camara_ip"
    return None, None


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    license_plate_detector = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=os.path.join(base_dir, "best.pt"),
    )
    coco_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    cap, source_name = abrir_fuente(args.source)
    if cap is None or not cap.isOpened():
        print("Error: no se pudo abrir la fuente de video.")
        sys.exit(1)

    main_output_dir = os.path.join(base_dir, "detecciones_historico")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(main_output_dir, f"run_{timestamp}_{source_name}")
    fotos_dir = os.path.join(run_dir, "fotos")
    os.makedirs(run_dir, exist_ok=True)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_path = os.path.join(run_dir, "video_procesado.mp4")
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    vehicles = [2, 3, 5, 7]  # Coche, moto, autobús, camión/furgoneta
    frames_per_plate = 2
    frames_per_ocr = 2
    max_frames_sin_placa = 30

    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["es"], gpu=use_gpu)

    detector = DetectorMatriculas(
        buffer_size=12,
        max_frames_sin_deteccion=max_frames_sin_placa,
        max_salto_relativo=0.22,
    )
    promedio = PromedioMatriculas(buffer_size=10)

    print(f"¿Cap abierto? {cap.isOpened()}")
    print(f"FPS: {fps}, Width: {width}, Height: {height}")
    print(f"Run dir: {run_dir}")
    print(f"¿VideoWriter abierto? {video_writer.isOpened()}")

    frame_number = -1
    ocr_counter = frames_per_ocr
    matricula_estable = ""
    ultima_bbox_plate = None
    ultimo_frame_placa = -1
    matriculas_guardadas = set()
    mostrar_ventanas = not args.no_display

    try:
        while True:
            frame_number += 1
            ocr_counter += 1
            ret, frame = cap.read()
            if not ret:
                break

            frame_original = frame.copy()
            annotator = Annotator(frame)

            detections = coco_model(frame)
            detections_v = []
            for *box, score, class_id in detections.xyxy[0]:
                if int(class_id) in vehicles:
                    detections_v.append([
                        box[0].item(), box[1].item(),
                        box[2].item(), box[3].item(),
                        score.item(),
                    ])

            centered_car = None
            if detections_v:
                centered_car = centered_bbox(detections_v, width, height)
                vx1, vy1, vx2, vy2, _ = centered_car
                margen_x = int((vx2 - vx1) * 0.1)
                margen_y = int((vy2 - vy1) * 0.1)
                cx1 = max(0, int(vx1 - margen_x))
                cy1 = max(0, int(vy1 - margen_y))
                cx2 = min(width, int(vx2 + margen_x))
                cy2 = min(height, int(vy2 + margen_y))
                annotator.box_label((cx1, cy1, cx2, cy2), "CAR", color=(0, 255, 0))
            else:
                cx1, cy1, cx2, cy2 = 0, 0, width, height

            if frame_number % frames_per_plate == 0:
                mejor_placa = None
                offset_x, offset_y = cx1, cy1
                src_crop = frame_original[cy1:cy2, cx1:cx2]

                if centered_car is not None and src_crop.size > 0:
                    if mostrar_ventanas:
                        try:
                            cv2.imshow("Coche", src_crop)
                            cv2.waitKey(1)
                        except cv2.error:
                            mostrar_ventanas = False
                    lp_crop = license_plate_detector(src_crop)
                    mejor_placa = seleccionar_mejor_matricula(
                        lp_crop.xyxy[0], src_crop.shape[1], src_crop.shape[0]
                    )

                # Respaldo sobre fotograma completo si no hubo vehículo central o no se halló matrícula en él
                if mejor_placa is None:
                    lp_full = license_plate_detector(frame_original)
                    mejor_placa = seleccionar_mejor_matricula(lp_full.xyxy[0], width, height)
                    src_crop = frame_original
                    offset_x, offset_y = 0, 0

                if mejor_placa is not None:
                    x1_p, y1_p, x2_p, y2_p, _ = mejor_placa
                    ultima_bbox_plate = (
                        int(x1_p) + offset_x,
                        int(y1_p) + offset_y,
                        int(x2_p) + offset_x,
                        int(y2_p) + offset_y,
                    )
                    ultimo_frame_placa = frame_number

                    pw = x2_p - x1_p
                    ph = y2_p - y1_p
                    px1 = max(0, int(x1_p - pw * 0.04))
                    py1 = max(0, int(y1_p - ph * 0.08))
                    px2 = min(src_crop.shape[1], int(x2_p + pw * 0.04))
                    py2 = min(src_crop.shape[0], int(y2_p + ph * 0.08))
                    frame_plate = src_crop[py1:py2, px1:px2]

                    if frame_plate.size > 0 and ocr_counter >= frames_per_ocr:
                        ocr_counter = 0
                        texto_norm, score_ocr, img_ocr = leer_matricula_ocr(reader, frame_plate)

                        if mostrar_ventanas and img_ocr is not None:
                            try:
                                cv2.imshow("Matrícula", img_ocr)
                                cv2.waitKey(1)
                            except cv2.error:
                                mostrar_ventanas = False

                        if texto_norm:
                            resultado_pos = detector.procesar_deteccion(
                                texto_norm,
                                score=score_ocr,
                                bbox=ultima_bbox_plate,
                                frame_idx=frame_number,
                                frame_width=width,
                            )
                            if detector.vehiculo_cambiado:
                                promedio.reset()

                            estabilizada = promedio.procesar_promedio(resultado_pos)
                            if estabilizada:
                                matricula_estable = estabilizada
                                print(
                                    f"[Frame {frame_number}] Lectura: {texto_norm} "
                                    f"(score={score_ocr:.2f}) -> Estable: {matricula_estable}"
                                )
                                if (
                                    (
                                        detector.contar_consistentes(matricula_estable) >= 3
                                        or (
                                            detector.contar_consistentes(matricula_estable) == 2
                                            and len(detector.detecciones) == 2
                                            and detector.sumar_score(matricula_estable) >= 1.65
                                        )
                                    )
                                    and matricula_estable not in matriculas_guardadas
                                ):
                                    guardar_foto_deteccion(frame_original, matricula_estable, carpeta=fotos_dir)
                                    matriculas_guardadas.add(matricula_estable)
                        else:
                            if detector.reset_if_new_vehicle(
                                bbox=ultima_bbox_plate,
                                frame_idx=frame_number,
                                frame_width=width,
                            ):
                                promedio.reset()
                                matricula_estable = ""

            if ultimo_frame_placa >= 0 and (frame_number - ultimo_frame_placa) > max_frames_sin_placa:
                ultima_bbox_plate = None
                matricula_estable = ""

            if matricula_estable and ultima_bbox_plate:
                annotator.box_label(ultima_bbox_plate, matricula_estable, color=(0, 255, 0))

            frame_out = annotator.result()
            video_writer.write(frame_out)

            if mostrar_ventanas:
                try:
                    cv2.imshow("PlateReader", frame_out)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    mostrar_ventanas = False
    finally:
        cap.release()
        video_writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"Procesamiento finalizado. Matrículas confirmadas: {sorted(matriculas_guardadas)}")
    print(f"Video guardado en: {video_output_path}")


if __name__ == "__main__":
    main()
