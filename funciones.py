import easyocr

def centered_bbox(detecciones, ancho_frame, alto_frame):
    """
    Encuentra el coche más centrado y grande de las detecciones.
    
    Args:
        detecciones: Lista de detecciones de YOLOv5 (tensor o lista)
        ancho_frame: Ancho del frame/imagen
        alto_frame: Alto del frame/imagen
    
    Returns:
        La detección con mejor score (más centrada y grande), o None si no hay detecciones
    """
    if len(detecciones) == 0:
        return None
    
    centro_frame_x = ancho_frame / 2
    centro_frame_y = alto_frame / 2
    
    mejor_score = -1
    mejor_deteccion = None
    
    for det in detecciones:
        # Coordenadas del bounding box: x1, y1, x2, y2
        x1, y1, x2, y2 = det[:4]
        
        # Centro del bounding box
        centro_x = (x1 + x2) / 2
        centro_y = (y1 + y2) / 2
        
        # Distancia al centro del frame (normalizada)
        distancia_centro = ((centro_x - centro_frame_x)**2 + (centro_y - centro_frame_y)**2)**0.5
        max_distancia = ((ancho_frame/2)**2 + (alto_frame/2)**2)**0.5
        distancia_normalizada = distancia_centro / max_distancia
        
        # Área del bounding box (normalizada)
        area = (x2 - x1) * (y2 - y1)
        area_frame = ancho_frame * alto_frame
        area_normalizada = area / area_frame
        
        # Score combinado: prioriza área grande y distancia pequeña
        # Puedes ajustar los pesos (0.6 y 0.4) según tus necesidades
        score = (0.6 * area_normalizada) + (0.4 * (1 - distancia_normalizada))
        
        if score > mejor_score:
            mejor_score = score
            mejor_deteccion = det
    
    return mejor_deteccion
# Función para guardar foto de detección
def guardar_foto_deteccion(frame, matricula, carpeta='detecciones/fotos'):
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre_archivo = f"{carpeta}/{matricula}_{timestamp}.jpg"
    cv2.imwrite(nombre_archivo, frame)
    return nombre_archivo