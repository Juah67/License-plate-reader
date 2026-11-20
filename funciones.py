import os
import re
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np

# Tablas de desambiguación posicional para matrículas españolas (NNNN LLL)
DIGIT_MAP = {
    'O': '0', 'D': '0', 'Q': '0', 'U': '0',
    'I': '1', 'L': '1', 'T': '1', 'J': '1',
    'Z': '2', 'E': '3', 'A': '4', 'H': '4',
    'S': '5', 'G': '6', 'C': '6', 'B': '8',
}
LETTER_MAP = {
    '0': 'D', '1': 'L', '2': 'Z', '3': 'B',
    '4': 'H', '5': 'S', '6': 'G', '7': 'T',
    '8': 'B', '9': 'P', 'A': 'H', 'E': 'F',
    'I': 'L', 'O': 'D', 'Q': 'G', 'U': 'V',
}
VALID_CONSONANTS = set('BCDFGHJKLMNPQRSTVWXYZ')
BORDER_CHARS = set('E1I7LTJF|')
GAP_CHARS = set('1IL27JT|')

PROVINCIAL_RE = re.compile(r'^[A-Z]{1,2}\d{4}[A-Z]{1,2}$')
SPECIAL_RE = re.compile(r'^[CRHE]\d{4}[BCDFGHJKLMNPQRSTVWXYZ]{3}$')


def centered_bbox(detecciones, ancho_frame, alto_frame):
    """Selecciona la detección de vehículo más relevante por área y cercanía al centro."""
    if len(detecciones) == 0:
        return None

    centro_frame_x = ancho_frame / 2
    centro_frame_y = alto_frame / 2
    max_distancia = ((ancho_frame / 2) ** 2 + (alto_frame / 2) ** 2) ** 0.5
    area_frame = max(1.0, ancho_frame * alto_frame)

    mejor_score = -1.0
    mejor_deteccion = None

    for det in detecciones:
        x1, y1, x2, y2 = det[:4]
        centro_x = (x1 + x2) / 2
        centro_y = (y1 + y2) / 2

        distancia_centro = ((centro_x - centro_frame_x) ** 2 + (centro_y - centro_frame_y) ** 2) ** 0.5
        distancia_normalizada = distancia_centro / max(1.0, max_distancia)
        area_normalizada = ((x2 - x1) * (y2 - y1)) / area_frame

        score = (0.6 * area_normalizada) + (0.4 * (1 - distancia_normalizada))
        if score > mejor_score:
            mejor_score = score
            mejor_deteccion = det

    return mejor_deteccion


def guardar_foto_deteccion(frame, matricula, carpeta='detecciones/fotos'):
    """Guarda una captura del fotograma etiquetada con la matrícula detectada."""
    os.makedirs(carpeta, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre_archivo = os.path.join(carpeta, f"{matricula}_{timestamp}.jpg")
    cv2.imwrite(nombre_archivo, frame)
    return nombre_archivo


def recortar_eurobanda(plate_bgr):
    """Recorta la franja azul izquierda de la eurobanda para evitar falsos caracteres iniciales."""
    if plate_bgr is None or plate_bgr.size == 0:
        return plate_bgr
    h, w = plate_bgr.shape[:2]
    if w < 40 or h < 12:
        return plate_bgr

    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array([95, 70, 40]), np.array([135, 255, 255]))
    left_limit = int(w * 0.22)
    col_ratios = (mask_blue[:, :left_limit] > 0).mean(axis=0)
    blue_cols = np.where(col_ratios > 0.35)[0]
    if len(blue_cols) > 0:
        cut_x = min(int(w * 0.18), int(blue_cols[-1]) + 2)
        if cut_x < w - 10:
            return plate_bgr[:, cut_x:]
    return plate_bgr


def preprocesar_matricula(frame_plate):
    """Genera variantes preprocesadas (escala de grises reescalada, bilateral, CLAHE) para OCR."""
    if frame_plate is None or frame_plate.size == 0:
        return []

    crop = recortar_eurobanda(frame_plate)
    if crop.ndim == 3:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        grey = crop.copy()

    scale = max(1.0, 96.0 / max(1, grey.shape[0]))
    up_grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    up_bilat = cv2.bilateralFilter(up_grey, 5, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    up_clahe = clahe.apply(up_grey)

    return [
        (up_grey, 1.0),
        (up_bilat, 0.95),
        (up_clahe, 0.90),
    ]


def normalizar_matricula(raw_text, ocr_conf=1.0):
    """Valida y corrige posicionalmente la lectura OCR según formatos de matrícula."""
    s = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    n = len(s)
    if n < 5:
        return None, 0.0

    best_cand = None
    best_cost = 999.0

    # Formato principal español moderno: 4 dígitos + 3 consonantes (NNNN LLL)
    if n >= 7:
        for start in range(n - 6):
            for gap in (0, 1):
                end = start + 7 + gap
                if end > n:
                    continue
                prefix = s[:start]
                digits_part = s[start:start + 4]
                gap_part = s[start + 4:start + 4 + gap]
                letters_part = s[start + 4 + gap:end]
                suffix = s[end:]

                cost = 0.0
                for pc in prefix:
                    cost += 0.20 if pc in BORDER_CHARS else 0.70
                for gc in gap_part:
                    cost += 0.50 if gc in GAP_CHARS else 1.10
                for sc in suffix:
                    cost += 0.25 if sc in BORDER_CHARS else 0.50

                chars = []
                valid = True
                for c in digits_part:
                    if c.isdigit():
                        chars.append(c)
                    elif c in DIGIT_MAP:
                        chars.append(DIGIT_MAP[c])
                        cost += 0.55
                    else:
                        valid = False
                        break
                if not valid:
                    continue

                for c in letters_part:
                    if c in VALID_CONSONANTS:
                        chars.append(c)
                    elif c in LETTER_MAP:
                        chars.append(LETTER_MAP[c])
                        cost += 0.65
                    else:
                        valid = False
                        break

                if valid and cost < best_cost:
                    best_cost = cost
                    best_cand = ''.join(chars)

    if best_cand is not None and best_cost <= 2.2:
        pattern_score = max(0.15, 1.0 - best_cost * 0.25)
        return best_cand, float(ocr_conf) * pattern_score

    # Formatos secundarios españoles (provincial histórico y especiales)
    if (PROVINCIAL_RE.match(s) or SPECIAL_RE.match(s)) and ocr_conf >= 0.68:
        return s, float(ocr_conf) * 0.85

    # Respaldo para matrículas europeas o formatos no estándar, rechazando fragmentos truncados dígitos+letras
    is_truncated_es = re.match(r'^\d+[A-Z]{1,3}$', s) is not None
    if (
        not is_truncated_es
        and 5 <= n <= 8
        and any(c.isdigit() for c in s)
        and any(c.isalpha() for c in s)
        and ocr_conf >= 0.68
    ):
        return s, float(ocr_conf) * 0.45

    return None, 0.0


def leer_matricula_ocr(reader, frame_plate):
    """Ejecuta OCR multipaso sobre las variantes preprocesadas y devuelve (matricula, score, imagen)."""
    variantes = preprocesar_matricula(frame_plate)
    if not variantes:
        return None, 0.0, None

    best_plate = None
    best_score = 0.0
    best_img = variantes[0][0]

    for idx, (img_var, base_weight) in enumerate(variantes):
        results = reader.readtext(img_var, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if not results:
            continue

        results_ord = sorted(results, key=lambda r: (r[0][0][0] + r[0][1][0]) / 2.0)
        raw_txt = ''.join(r[1] for r in results_ord)
        avg_conf = sum(float(r[2]) for r in results_ord) / len(results_ord)
        cand, score = normalizar_matricula(raw_txt, avg_conf)
        weighted_score = score * base_weight

        if cand and weighted_score > best_score:
            best_score = weighted_score
            best_plate = cand
            best_img = img_var

        # Cortocircuito si la imagen limpia ya logra una lectura sólida
        if idx == 0 and best_score >= 0.72:
            break

    return best_plate, best_score, best_img


def seleccionar_mejor_matricula(detecciones_yolo, ancho_crop, alto_crop, conf_min=0.35):
    """Filtra detecciones YOLO por confianza y proporción geométrica, devolviendo [x1, y1, x2, y2, conf]."""
    if detecciones_yolo is None or len(detecciones_yolo) == 0:
        return None

    area_crop = max(1.0, float(ancho_crop * alto_crop))
    mejor_det = None
    mejor_score = -1.0

    for det in detecciones_yolo:
        x1, y1, x2, y2, conf = [float(v) for v in det[:5]]
        if conf < conf_min:
            continue
        pw = x2 - x1
        ph = max(1.0, y2 - y1)
        if (pw / ph) < 1.1:
            continue

        area_norm = min(1.0, ((pw * ph) / area_crop) * 20.0)
        score = (0.75 * conf) + (0.25 * area_norm)
        if score > mejor_score:
            mejor_score = score
            mejor_det = [x1, y1, x2, y2, conf]

    return mejor_det


class DetectorMatriculas:
    """Consenso temporal ponderado por posición de carácter con reinicio automático entre vehículos."""

    def __init__(self, buffer_size=12, max_frames_sin_deteccion=30, max_salto_relativo=0.22):
        self.detecciones = deque(maxlen=buffer_size)
        self.max_frames_sin_deteccion = max_frames_sin_deteccion
        self.max_salto_relativo = max_salto_relativo
        self.ultimo_bbox = None
        self.ultimo_frame = None
        self.vehiculo_cambiado = False
        self._divergent_streak = 0

    def reset(self):
        self.detecciones.clear()
        self.ultimo_bbox = None
        self.ultimo_frame = None
        self._divergent_streak = 0

    def reset_if_new_vehicle(self, bbox=None, frame_idx=None, frame_width=None):
        """Reinicia el historial si hubo una pausa prolongada o un salto espacial brusco."""
        if frame_idx is not None and self.ultimo_frame is not None:
            if (frame_idx - self.ultimo_frame) > self.max_frames_sin_deteccion:
                self.reset()
                return True

        if bbox is not None and self.ultimo_bbox is not None and frame_width:
            cx_new = (bbox[0] + bbox[2]) / 2.0
            cy_new = (bbox[1] + bbox[3]) / 2.0
            cx_old = (self.ultimo_bbox[0] + self.ultimo_bbox[2]) / 2.0
            cy_old = (self.ultimo_bbox[1] + self.ultimo_bbox[3]) / 2.0
            dist = ((cx_new - cx_old) ** 2 + (cy_new - cy_old) ** 2) ** 0.5
            if dist > (self.max_salto_relativo * frame_width):
                self.reset()
                return True

        return False

    def _consenso_actual(self):
        if not self.detecciones:
            return None

        len_scores = Counter()
        for mat, w in self.detecciones:
            len_scores[len(mat)] += w
        target_len = len_scores.most_common(1)[0][0]

        resultado_final = []
        for i in range(target_len):
            char_scores = Counter()
            for mat, w in self.detecciones:
                if len(mat) == target_len:
                    char_scores[mat[i]] += w
            caracter_ganador, _ = char_scores.most_common(1)[0]
            resultado_final.append(caracter_ganador)

        return "".join(resultado_final)

    def contar_consistentes(self, matricula):
        """Devuelve cuántas lecturas del buffer coinciden exactamente con la matrícula dada."""
        if not matricula:
            return 0
        return sum(1 for mat, _ in self.detecciones if mat == matricula)

    def sumar_score(self, matricula):
        """Devuelve la suma de puntuaciones de las lecturas coincidentes en el buffer."""
        if not matricula:
            return 0.0
        return sum(w for mat, w in self.detecciones if mat == matricula)

    def procesar_deteccion(self, nueva_matricula, score=1.0, bbox=None, frame_idx=None, frame_width=None):
        changed = self.reset_if_new_vehicle(bbox=bbox, frame_idx=frame_idx, frame_width=frame_width)

        if nueva_matricula and len(self.detecciones) >= 3 and score >= 0.72:
            cons_actual = self._consenso_actual()
            if cons_actual:
                diffs = (
                    len(cons_actual)
                    if len(cons_actual) != len(nueva_matricula)
                    else sum(1 for a, b in zip(nueva_matricula, cons_actual) if a != b)
                )
                if diffs >= 4:
                    self._divergent_streak += 1
                    if self._divergent_streak >= 2:
                        self.reset()
                        changed = True
                else:
                    self._divergent_streak = 0
        elif nueva_matricula:
            self._divergent_streak = 0

        self.vehiculo_cambiado = changed
        if bbox is not None:
            self.ultimo_bbox = bbox
        if frame_idx is not None:
            self.ultimo_frame = frame_idx

        if nueva_matricula and not (self._divergent_streak == 1 and len(self.detecciones) >= 3):
            self.detecciones.append((nueva_matricula, max(0.05, float(score))))

        return self._consenso_actual()


class PromedioMatriculas:
    """Estabiliza la matrícula final ponderando más las lecturas recientes."""

    def __init__(self, buffer_size=10):
        self.promedios = deque(maxlen=buffer_size)

    def reset(self):
        self.promedios.clear()

    def procesar_promedio(self, nueva_matricula):
        if nueva_matricula:
            self.promedios.append(nueva_matricula)
        if not self.promedios:
            return None

        scores = Counter()
        for idx, mat in enumerate(self.promedios):
            scores[mat] += 1.0 + (0.1 * idx)

        matricula_mas_comun, _ = scores.most_common(1)[0]
        return matricula_mas_comun
