import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import os
import sys
from fer import FER
import traceback

class PoseSatisfactionDetector:
    def __init__(self):
        print("Iniciando sistema de detección...")
        # Inicializar detectores de MediaPipe con mayor precisión
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,  # Aumentado para mayor precisión
            min_tracking_confidence=0.5,   # Aumentado para mejor seguimiento
            model_complexity=2,            # Máxima complejidad para mejor precisión
            enable_segmentation=False      # Deshabilitamos la segmentación
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.satisfaction_data = []
        self.last_person_coords = None  # Almacenar coordenadas de la última persona detectada
        self.emotion_detector = FER()  # Inicializar detector de emociones
        
        # Cargar modelo YOLO para detección de personas
        weights_path = os.path.join(os.path.dirname(__file__), 'yolov4-tiny.weights')
        config_path = os.path.join(os.path.dirname(__file__), 'yolov4-tiny.cfg')
        
        if not os.path.exists(weights_path):
            print("Descargando modelo YOLO...")
            self.download_yolo_weights(weights_path)
        
        print("Cargando modelo YOLO...")
        try:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Modelo YOLO cargado correctamente")
            
        except Exception as e:
            print(f"Error al cargar el modelo YOLO: {e}")
            raise
            
        # Cargar clases
        self.classes = []
        with open(os.path.join(os.path.dirname(__file__), 'coco.names'), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        print("Sistema iniciado correctamente")

    def download_yolo_weights(self, weights_path):
        """Descarga los pesos del modelo YOLO si no existen"""
        import urllib.request
        
        url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
        print(f"Descargando {url}...")
        urllib.request.urlretrieve(url, weights_path)
        print("Descarga completada")

    def calculate_angle(self, point1, point2, point3):
        """Calcula el ángulo entre tres puntos"""
        try:
            vector1 = np.array([point1.x - point2.x, point1.y - point2.y])
            vector2 = np.array([point3.x - point2.x, point3.y - point2.y])
            
            # Verificar que los vectores no sean nulos
            if np.all(vector1 == 0) or np.all(vector2 == 0):
                return 0
                
            cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return np.degrees(angle)
        except Exception as e:
            print(f"Error al calcular ángulo: {e}")
            return 0

    def analyze_body_language(self, landmarks):
        """Analiza el lenguaje corporal con métricas más precisas"""
        try:
            # Puntos clave
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            
            # Punto medio entre hombros y caderas
            shoulders_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2,
                                    (left_shoulder.y + right_shoulder.y) / 2])
            hips_mid = np.array([(left_hip.x + right_hip.x) / 2,
                                (left_hip.y + right_hip.y) / 2])
            
            # 1. Postura de la espalda (35%)
            spine_angle = self.calculate_angle(nose, 
                                            type('Point', (), {'x': shoulders_mid[0], 'y': shoulders_mid[1]}),
                                            type('Point', (), {'x': hips_mid[0], 'y': hips_mid[1]}))
            posture_straight = 80 <= spine_angle <= 100
            posture_score = 0.35 if posture_straight else (0.175 if 70 <= spine_angle <= 110 else 0)
            
            # 2. Nivel de hombros (20%)
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            shoulders_level = shoulder_diff < 0.05
            shoulder_score = 0.20 if shoulders_level else (0.10 if shoulder_diff < 0.1 else 0)
            
            # 3. Posición de los brazos (25%)
            left_arm_tension = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_tension = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            arms_relaxed = (left_arm_tension > 100 and right_arm_tension > 100)
            arms_score = 0.25 if arms_relaxed else (0.125 if left_arm_tension > 100 or right_arm_tension > 100 else 0)
            
            # 4. Centrado del cuerpo (20%)
            hip_center = (left_hip.x + right_hip.x) / 2
            shoulder_center = (left_shoulder.x + right_shoulder.x) / 2
            body_centered = abs(hip_center - shoulder_center) < 0.1
            center_score = 0.20 if body_centered else (0.10 if abs(hip_center - shoulder_center) < 0.15 else 0)
            
            # Calcular score total
            total_score = posture_score + shoulder_score + arms_score + center_score
            
            # Detalles del análisis
            details = {
                'postura': {
                    'score': posture_score,
                    'angulo': spine_angle,
                    'estado': 'Excelente' if posture_straight else 'Regular' if 70 <= spine_angle <= 110 else 'Mala'
                },
                'hombros': {
                    'score': shoulder_score,
                    'diferencia': shoulder_diff,
                    'estado': 'Nivelados' if shoulders_level else 'Desnivelados'
                },
                'brazos': {
                    'score': arms_score,
                    'tension_izq': left_arm_tension,
                    'tension_der': right_arm_tension,
                    'estado': 'Relajados' if arms_relaxed else 'Tensos'
                },
                'centrado': {
                    'score': center_score,
                    'desviacion': abs(hip_center - shoulder_center),
                    'estado': 'Centrado' if body_centered else 'Desviado'
                }
            }
            
            return {
                'score': total_score,
                'details': details,
                'spine_angle': spine_angle
            }
            
        except Exception as e:
            print(f"Error en analyze_body_language: {e}")
            raise

    def analyze_emotions(self, frame):
        """Analiza las emociones en el frame dado"""
        try:
            emotions = self.emotion_detector.detect_emotions(frame)
            if emotions:
                # Obtener la emoción más prominente
                dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                print(f"Emoción detectada: {dominant_emotion}")  # Mensaje de diagnóstico
                return dominant_emotion
            else:
                print("No se detectaron emociones")  # Mensaje de diagnóstico
        except Exception as e:
            print(f"Error al analizar emociones: {e}")
        return None

    def detect_people(self, frame):
        """Detecta personas usando YOLO con mayor precisión"""
        height, width = frame.shape[:2]
        
        # Preparar imagen para YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        try:
            outs = self.net.forward(self.output_layers)
            boxes = []
            confidences = []
            
            # Procesar cada detección
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Solo detectar personas con alta confianza
                    if class_id == 0 and confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
            
            # Aplicar non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Seleccionar solo la detección más confiable
            if len(indices) > 0:
                indices = indices.flatten()
                sorted_indices = sorted([(i, confidences[i]) for i in indices], 
                                     key=lambda x: x[1], reverse=True)
                
                # Tomar solo la primera detección (la más confiable)
                i = sorted_indices[0][0]
                box = boxes[i]
                x, y, w, h = box
                
                # Agregar margen para mejor detección de pose
                margin_x = int(w * 0.2)
                margin_y = int(h * 0.2)
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                w = min(width - x, w + 2*margin_x)
                h = min(height - y, h + 2*margin_y)
                
                self.last_person_coords = (x, y, w, h)  # Actualizar coordenadas de la última persona detectada
                return [(x, y, w, h)]
            
            # Si no se detecta a la persona, intentar buscar cerca de la última posición
            if self.last_person_coords:
                x, y, w, h = self.last_person_coords
                search_margin = 50  # Margen de búsqueda
                x = max(0, x - search_margin)
                y = max(0, y - search_margin)
                w = min(width - x, w + 2 * search_margin)
                h = min(height - y, h + 2 * search_margin)
                return [(x, y, w, h)]
            
            return []
            
        except Exception as e:
            print(f"Error en la detección YOLO: {e}")
            return []

    def process_frame(self, frame):
        if frame is None:
            print("Frame nulo recibido")
            return frame
            
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        # Detectar personas usando YOLO
        people_regions = self.detect_people(frame)
        
        # Lista para almacenar análisis
        all_analyses = []
        
        # Procesar cada región donde se detectó una persona
        for i, (x, y, w, h) in enumerate(people_regions):
            try:
                # Extraer región de la persona con un margen adicional
                margin_x = int(w * 0.2)
                margin_y = int(h * 0.2)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(frame.shape[1], x + w + margin_x)
                y2 = min(frame.shape[0], y + h + margin_y)
                
                person_frame = frame[y1:y2, x1:x2].copy()
                
                # Verificar tamaño mínimo
                if person_frame.shape[0] < 64 or person_frame.shape[1] < 64:
                    print(f"Región {i+1} demasiado pequeña: {person_frame.shape}")
                    continue
                
                # Redimensionar si es necesario para mantener proporciones consistentes
                target_height = 480
                aspect_ratio = person_frame.shape[1] / person_frame.shape[0]
                target_width = int(target_height * aspect_ratio)
                person_frame = cv2.resize(person_frame, (target_width, target_height))
                
                # Convertir a RGB para MediaPipe
                person_frame_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
                
                # Procesar con MediaPipe
                results = self.pose.process(person_frame_rgb)
                
                if results.pose_landmarks:
                    # Dibujar landmarks en la región
                    self.mp_drawing.draw_landmarks(
                        person_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Analizar lenguaje corporal
                    body_analysis = self.analyze_body_language(results.pose_landmarks.landmark)
                    all_analyses.append(body_analysis)
                    
                    # Detección de emociones
                    dominant_emotion = self.analyze_emotions(person_frame)
                    if dominant_emotion:
                        cv2.putText(person_frame, f"Emoción: {dominant_emotion}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Redimensionar de vuelta y colocar en el frame original
                    person_frame = cv2.resize(person_frame, (x2-x1, y2-y1))
                    frame[y1:y2, x1:x2] = person_frame
                    
                    # Dibujar rectángulo y etiqueta
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Persona {i+1}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            except Exception as e:
                print(f"Error al procesar persona {i+1}: {e}")
                continue
        
        # Guardar datos si hay análisis
        if all_analyses:
            self.satisfaction_data.append({
                'timestamp': datetime.now().isoformat(),
                'num_people': len(all_analyses),
                'analyses': all_analyses
            })
        
        # Mostrar información para cada persona
        for i, analysis in enumerate(all_analyses):
            try:
                self.draw_satisfaction_info(frame, analysis, person_index=i, total_people=len(all_analyses))
            except Exception as e:
                print(f"Error al dibujar información: {e}")
        
        # Agregar contador de personas
        cv2.putText(frame, f"Personas detectadas: {len(people_regions)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def draw_satisfaction_info(self, frame, analysis, person_index=0, total_people=1):
        height, width = frame.shape[:2]
        score = analysis['score']
        details = analysis['details']
        
        # Ajustar posición según el número de persona
        section_height = height // total_people
        vertical_offset = person_index * section_height
        
        # Color basado en el score (rojo a verde)
        color = (0, int(255 * score), int(255 * (1 - score)))
        
        # Barra de satisfacción principal
        bar_width = 300
        bar_height = 30
        bar_x = width - bar_width - 20
        bar_y = vertical_offset + 20
        
        # Dibujar barra principal
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (int(bar_x + bar_width * score), bar_y + bar_height), color, -1)
        
        # Texto principal
        satisfaction_text = f"Persona {person_index + 1} - Satisfacción: {score:.1%}"
        cv2.putText(frame, satisfaction_text, (bar_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Métricas detalladas
        metrics_y = bar_y + bar_height + 25
        metrics_x = bar_x
        
        # Mostrar detalles de cada métrica
        for metric_name, metric_data in details.items():
            if 'score' in metric_data and 'estado' in metric_data:
                # Color basado en el score de la métrica
                metric_color = (0, int(255 * (metric_data['score'] / 0.3)), int(255 * (1 - metric_data['score'] / 0.3)))
                
                # Texto de la métrica
                metric_text = f"{metric_name.title()}: {metric_data['estado']} ({metric_data['score']*100:.0f}%)"
                cv2.putText(frame, metric_text, (metrics_x, metrics_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, metric_color, 2)
                metrics_y += 25
        
        # Mostrar ángulo de la columna
        if 'spine_angle' in analysis:
            spine_text = f"Ángulo columna: {analysis['spine_angle']:.1f}°"
            cv2.putText(frame, spine_text, (metrics_x, metrics_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def save_data(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        
        filename = f"data/satisfaction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.satisfaction_data, f, indent=4)
        return filename

    def run(self, source):
        print(f"\nIniciando análisis de video: {source}")
        
        try:
            # Verificar si el archivo existe
            if isinstance(source, str) and not os.path.exists(source):
                print(f"ERROR: El archivo {source} no existe")
                return None
            
            print("Verificando archivo de video...")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"ERROR: No se pudo abrir el video: {source}")
                return None
                
            print("Video abierto correctamente")
            
            # Configurar ventana
            window_name = "Detector de Satisfacción"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            print("Ventana creada")
            
            # Configurar FPS
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_delay = int(1000/fps) if fps > 0 else 30
            print(f"FPS del video: {fps}")
            
            frame_count = 0
            while True:
                print(f"\rLeyendo frame {frame_count + 1}", end="")
                ret, frame = cap.read()
                
                if not ret:
                    print("\nNo se pudo leer el siguiente frame")
                    break
                    
                frame_count += 1
                if frame_count == 1:
                    print("\nPrimer frame leído correctamente")
                    print(f"Dimensiones del frame: {frame.shape}")
                
                try:
                    # Procesar frame
                    processed_frame = self.process_frame(frame.copy())
                    
                    # Mostrar frame
                    cv2.imshow(window_name, processed_frame)
                    print(f"\rProcesando frame {frame_count}", end="")
                    
                    # Esperar por tecla
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nSaliendo del programa (tecla q)")
                        break
                    elif key == ord('p'):
                        print("\nPausa - presiona cualquier tecla para continuar")
                        cv2.waitKey(0)
                        
                except Exception as e:
                    print(f"\nError procesando frame {frame_count}: {e}")
                    continue
            
            print("\nLiberando recursos...")
            cap.release()
            cv2.destroyAllWindows()
            print("Recursos liberados")
            
            # Guardar datos
            if self.satisfaction_data:
                data_file = self.save_data()
                print(f"\nDatos guardados en: {data_file}")
                print(f"Total frames procesados: {frame_count}")
                print(f"Frames con detecciones: {len(self.satisfaction_data)}")
            
            return True
            
        except Exception as e:
            print(f"\nError general en run: {e}")
            return False

def list_cameras():
    """Lista todas las cámaras disponibles"""
    print("\nBuscando cámaras disponibles...")
    available_cameras = []
    for i in range(5):  # Probar los primeros 5 índices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Cámara {i} - OK")
            cap.release()
        else:
            print(f"Cámara {i} - No disponible")
    return available_cameras

if __name__ == "__main__":
    print("=== Detector de Satisfacción por Postura Corporal ===")
    
    try:
        # Verificar argumentos de línea de comandos
        if len(sys.argv) > 1:
            # Si se proporciona un argumento, asumimos que es un archivo de video
            video_path = sys.argv[1]
            print(f"\nVerificando archivo de video: {video_path}")
            
            # Verificar si el archivo existe
            if not os.path.exists(video_path):
                print(f"ERROR: El archivo {video_path} no existe")
                sys.exit(1)
                
            print(f"Archivo encontrado: {video_path}")
            print("Iniciando detector...")
            
            try:
                detector = PoseSatisfactionDetector()
                print("Detector iniciado correctamente")
                print("Intentando procesar video...")
                data_file = detector.run(video_path)
            except Exception as e:
                print(f"Error al iniciar el detector: {e}")
                sys.exit(1)
        else:
            print("\nBuscando cámaras disponibles...")
            available_cameras = list_cameras()
            
            if not available_cameras:
                print("\nERROR: No se encontraron cámaras disponibles")
                sys.exit(1)
            
            camera_to_use = available_cameras[0]
            print(f"\nUsando cámara {camera_to_use}")
            
            detector = PoseSatisfactionDetector()
            data_file = detector.run(camera_to_use)
        
        if data_file:
            print("\nPrograma finalizado correctamente")
        else:
            print("\nError al ejecutar el programa")
            
    except Exception as e:
        print(f"\nError general del programa: {e}")
        sys.exit(1) 