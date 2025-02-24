import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import os
import sys

class PoseSatisfactionDetector:
    def __init__(self):
        print("Iniciando sistema de detección...")
        # Inicializar detectores de MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.3,  # Reducido para detectar mejor
            min_tracking_confidence=0.3,   # Reducido para mejor seguimiento
            model_complexity=1,
            enable_segmentation=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.satisfaction_data = []
        
        # Cargar modelo YOLO para detección de personas
        weights_path = os.path.join(os.path.dirname(__file__), 'yolov4-tiny.weights')
        config_path = os.path.join(os.path.dirname(__file__), 'yolov4-tiny.cfg')
        
        if not os.path.exists(weights_path):
            print("Descargando modelo YOLO...")
            self.download_yolo_weights(weights_path)
        
        print("Cargando modelo YOLO...")
        try:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Intentar usar GPU si está disponible
            try:
                print("Intentando usar GPU (CUDA)...")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("GPU activada correctamente")
            except:
                print("GPU no disponible, usando CPU")
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
        """Analiza el lenguaje corporal en detalle"""
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
            
            # 1. Postura de la espalda
            spine_angle = self.calculate_angle(nose, 
                                            type('Point', (), {'x': shoulders_mid[0], 'y': shoulders_mid[1]}),
                                            type('Point', (), {'x': hips_mid[0], 'y': hips_mid[1]}))
            posture_straight = 80 <= spine_angle <= 100
            
            # 2. Nivel de hombros
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            shoulders_level = shoulder_diff < 0.05
            
            # 3. Posición de los brazos
            # Detectar brazos cruzados
            arms_crossed = (
                abs(left_wrist.x - right_shoulder.x) < 0.2 and
                abs(right_wrist.x - left_shoulder.x) < 0.2
            )
            
            # 4. Apertura corporal
            left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            arms_open = (left_arm_angle > 30 and right_arm_angle > 30)
            
            # 5. Centrado del cuerpo
            hip_center = (left_hip.x + right_hip.x) / 2
            shoulder_center = (left_shoulder.x + right_shoulder.x) / 2
            body_centered = abs(hip_center - shoulder_center) < 0.1
            
            # Calcular score de postura
            posture_score = 0
            if posture_straight: posture_score += 0.3  # Postura recta
            if shoulders_level: posture_score += 0.2   # Hombros nivelados
            if not arms_crossed: posture_score += 0.2  # Brazos no cruzados
            if arms_open: posture_score += 0.2         # Brazos abiertos
            if body_centered: posture_score += 0.1     # Cuerpo centrado
            
            return {
                'score': posture_score,
                'posture_straight': posture_straight,
                'shoulders_level': shoulders_level,
                'arms_crossed': arms_crossed,
                'arms_open': arms_open,
                'body_centered': body_centered,
                'spine_angle': spine_angle
            }
        except Exception as e:
            print(f"Error en analyze_body_language: {e}")
            raise

    def detect_people(self, frame):
        """Detecta personas usando MediaPipe Pose directamente"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Mejorar el contraste y brillo de la imagen
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        frame_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Detectar poses
        results = self.pose.process(frame_rgb)
        people_regions = []
        
        if results.pose_landmarks:
            height, width = frame.shape[:2]
            
            # Obtener coordenadas de los landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Encontrar los límites del cuerpo
            x_coordinates = [landmark.x for landmark in landmarks]
            y_coordinates = [landmark.y for landmark in landmarks]
            
            # Calcular el bounding box
            x_min = max(0, int(min(x_coordinates) * width))
            y_min = max(0, int(min(y_coordinates) * height))
            x_max = min(width, int(max(x_coordinates) * width))
            y_max = min(height, int(max(y_coordinates) * height))
            
            # Agregar margen
            margin = 20
            x = max(0, x_min - margin)
            y = max(0, y_min - margin)
            w = min(width - x, (x_max - x_min) + 2*margin)
            h = min(height - y, (y_max - y_min) + 2*margin)
            
            people_regions.append((x, y, w, h))
            print(f"Detectadas {len(people_regions)} personas")
            
        return people_regions

    def resize_frame_aspect_ratio(self, frame, max_width=1280, max_height=720):
        """Redimensiona el frame manteniendo la proporción de aspecto"""
        height, width = frame.shape[:2]
        
        # Si el frame es más pequeño que los máximos, lo dejamos como está
        if width <= max_width and height <= max_height:
            return frame
            
        # Calcular ratio de aspecto
        aspect_ratio = width / height
        
        # Calcular nuevas dimensiones manteniendo proporción
        if width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            
        # Si después de ajustar el ancho, la altura sigue siendo muy grande
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            
        return cv2.resize(frame, (new_width, new_height))

    def process_frame(self, frame):
        # Redimensionar el frame si es necesario
        frame = self.resize_frame_aspect_ratio(frame)
        
        # Asegurarse de que el frame sea continuo en memoria
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Procesar el frame completo con MediaPipe
        results = self.pose.process(frame_rgb)
        
        # Lista para almacenar análisis de todas las personas detectadas
        all_analyses = []
        
        if results.pose_landmarks:
            try:
                # Analizar lenguaje corporal
                body_analysis = self.analyze_body_language(results.pose_landmarks.landmark)
                all_analyses.append(body_analysis)
                
                # Dibujar landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Obtener bounding box para la persona
                height, width = frame.shape[:2]
                landmarks = results.pose_landmarks.landmark
                x_coordinates = [landmark.x for landmark in landmarks]
                y_coordinates = [landmark.y for landmark in landmarks]
                
                x_min = max(0, int(min(x_coordinates) * width))
                y_min = max(0, int(min(y_coordinates) * height))
                x_max = min(width, int(max(x_coordinates) * width))
                y_max = min(height, int(max(y_coordinates) * height))
                
                # Dibujar rectángulo alrededor de la persona
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, "Persona 1", (x_min, y_min-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error al procesar persona: {e}")
        
        # Guardar datos de todas las personas
        if all_analyses:
            self.satisfaction_data.append({
                'timestamp': datetime.now().isoformat(),
                'num_people': len(all_analyses),
                'analyses': all_analyses
            })
        
        # Mostrar información para cada persona
        for i, analysis in enumerate(all_analyses):
            self.draw_satisfaction_info(frame, analysis, person_index=i, total_people=len(all_analyses))
        
        # Agregar contador de personas con fondo semitransparente
        personas_text = f"Personas detectadas: {len(all_analyses)}"
        # Obtener el tamaño del texto
        (text_width, text_height), _ = cv2.getTextSize(personas_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        
        # Dibujar un rectángulo semitransparente como fondo
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (text_width + 20, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Dibujar el texto
        cv2.putText(frame, personas_text, 
                   (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
        return frame

    def draw_satisfaction_info(self, frame, analysis, person_index=0, total_people=1):
        height, width = frame.shape[:2]
        score = analysis['score']
        
        # Ajustar posición según el número de persona
        vertical_offset = person_index * (height // total_people)
        
        # Color basado en el score (rojo a verde)
        score_color = (0, int(255 * score), int(255 * (1 - score)))
        texto_color = (255, 255, 255)  # Color blanco para el texto principal
        
        # Barra de satisfacción - Ajustada para dejar más espacio
        bar_width = 500  # Aumentado para texto más largo
        bar_height = 40
        bar_x = width - bar_width - 100  # Más margen a la derecha
        bar_y = vertical_offset + 80    # Más espacio desde arriba
        
        # Dibujar título con borde negro para mejor visibilidad
        satisfaction_text = f"Persona {person_index + 1} - Satisfaccion: {score:.2%}"
        
        # Dibujar el texto con borde negro
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        # Borde negro
        cv2.putText(frame, satisfaction_text, (bar_x, bar_y - 20), 
                   font, font_scale, (0, 0, 0), thickness + 2)
        # Texto blanco
        cv2.putText(frame, satisfaction_text, (bar_x, bar_y - 20), 
                   font, font_scale, texto_color, thickness)
        
        # Dibujar barra de progreso con borde
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (0, 0, 0), thickness + 2)  # Borde negro
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), thickness)  # Borde blanco
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (int(bar_x + bar_width * score), bar_y + bar_height), 
                     score_color, -1)
        
        # Métricas detalladas
        metrics_y = bar_y + bar_height + 50
        metrics_names = {
            'posture_straight': 'Postura recta',
            'shoulders_level': 'Hombros nivelados',
            'arms_crossed': 'Brazos cruzados',
            'arms_open': 'Postura abierta',
            'body_centered': 'Cuerpo centrado'
        }
        
        for metric, name in metrics_names.items():
            if metric in analysis:
                value = analysis[metric]
                if metric == 'arms_crossed':  # Invertir para arms_crossed
                    value = not value
                status = "OK" if value else "NO"
                status_color = (0, 255, 0) if value else (0, 0, 255)
                
                # Dibujar el nombre de la métrica con borde
                # Borde negro
                cv2.putText(frame, f"{name}:", (bar_x, metrics_y), 
                           font, 1.0, (0, 0, 0), thickness + 2)
                # Texto blanco
                cv2.putText(frame, f"{name}:", (bar_x, metrics_y), 
                           font, 1.0, texto_color, thickness)
                
                # Dibujar el status con borde
                status_x = bar_x + 350  # Aumentado para evitar solapamiento
                # Borde negro
                cv2.putText(frame, status, (status_x, metrics_y), 
                           font, 1.0, (0, 0, 0), thickness + 2)
                # Texto en color
                cv2.putText(frame, status, (status_x, metrics_y), 
                           font, 1.0, status_color, thickness)
                
                metrics_y += 45  # Aumentado el espacio entre líneas
        
        # Mostrar ángulo de la columna
        if 'spine_angle' in analysis:
            spine_text = f"Angulo columna:"
            angle_value = f"{analysis['spine_angle']:.1f} grados"
            
            # Borde negro para el texto
            cv2.putText(frame, spine_text, (bar_x, metrics_y), 
                       font, 1.0, (0, 0, 0), thickness + 2)
            # Texto blanco
            cv2.putText(frame, spine_text, (bar_x, metrics_y), 
                       font, 1.0, texto_color, thickness)
            
            # Borde negro para el valor
            cv2.putText(frame, angle_value, (status_x, metrics_y), 
                       font, 1.0, (0, 0, 0), thickness + 2)
            # Texto blanco
            cv2.putText(frame, angle_value, (status_x, metrics_y), 
                       font, 1.0, texto_color, thickness)

    def save_data(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Función auxiliar para convertir tipos de NumPy a tipos nativos de Python
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            else:
                return obj
        
        # Convertir todos los datos antes de guardar
        converted_data = convert_numpy_types(self.satisfaction_data)
        
        filename = f"data/satisfaction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(converted_data, f, indent=4)
        return filename

    def run(self, source):
        """
        Ejecuta el detector en una fuente de video (cámara o archivo)
        source: puede ser un número de cámara (int) o ruta a un archivo de video (str)
        """
        print(f"\nIntentando abrir la fuente de video: {source}")
        
        # Si source es un string, asumimos que es un archivo de video
        if isinstance(source, str):
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"ERROR: No se pudo abrir el archivo de video: {source}")
                return None
            
            # Configurar una velocidad de reproducción más rápida para videos
            frame_delay = 1  # Mínimo delay posible
        else:
            # Intentar diferentes backends de cámara para webcam
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            cap = None
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(source + backend)
                    if cap.isOpened():
                        print(f"Cámara abierta correctamente usando backend {backend}")
                        break
                except:
                    continue
            
            if not cap or not cap.isOpened():
                print(f"ERROR: No se pudo abrir la cámara {source}")
                return None
            
            frame_delay = 1
        
        # Configurar resolución más baja para mejor rendimiento
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Resolución original: {width}x{height} @ {fps}fps")
        if width > 1280 or height > 720:
            print("El video será redimensionado automáticamente para mejor visualización")
            print("Resolución máxima: 1280x720 (manteniendo proporción de aspecto)")
        
        if isinstance(source, str):
            print(f"Total de frames: {total_frames}")
        print("\nPresiona 'q' para salir, 's' para saltar frame, 'p' para pausar/continuar")
        
        frame_count = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str):
                        print("\nFin del video")
                        break
                    else:
                        print("Error al leer frame de la cámara")
                        break
                
                frame_count += 1
                if frame_count == 1:
                    print("Primer frame capturado correctamente")
                
                if isinstance(source, str):
                    print(f"\rProcesando frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end="")
            
            try:
                # Procesar frame
                processed_frame = self.process_frame(frame.copy())
                
                # Mostrar resultado
                cv2.imshow("Detector de Satisfaccion", processed_frame)
                
                # Control de reproducción
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q'):
                    print("\nCerrando programa...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("\nVideo pausado" if paused else "\nVideo continuando")
                elif key == ord('s'):
                    print("\nSaltando frame")
                    paused = False
                
            except Exception as e:
                print(f"\nError al procesar frame {frame_count}: {e}")
                continue
        
        # Guardar datos y limpiar
        data_file = self.save_data()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDatos guardados en: {data_file}")
        
        # Mostrar resumen del análisis
        if self.satisfaction_data:
            print("\nResumen del análisis:")
            print(f"Total de frames analizados: {frame_count}")
            print(f"Frames con detecciones: {len(self.satisfaction_data)}")
            if len(self.satisfaction_data) > 0:
                avg_people = sum(data['num_people'] for data in self.satisfaction_data) / len(self.satisfaction_data)
                print(f"Promedio de personas detectadas por frame: {avg_people:.2f}")
        
        return data_file

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
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        # Si se proporciona un argumento, asumimos que es un archivo de video
        video_path = sys.argv[1]
        print(f"\nAnalizando video: {video_path}")
        detector = PoseSatisfactionDetector()
        data_file = detector.run(video_path)
    else:
        # Si no hay argumentos, usar la cámara
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