# 📊 Detector de Satisfacción por Postura Corporal

Este proyecto implementa un sistema de detección de satisfacción del cliente basado en el análisis de postura corporal en tiempo real utilizando MediaPipe y OpenCV.

## 🚀 Características

- Detección de postura en tiempo real
- Análisis de satisfacción basado en métricas corporales
- Visualización en tiempo real con barra de satisfacción
- Exportación de datos a JSON para análisis posterior
- Interfaz visual intuitiva

## 📋 Requisitos Previos

- Python 3.8 o superior
- Webcam o video pregrabado

## ⚙️ Instalación

1. Clonar el repositorio o descargar los archivos

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## 🎮 Uso

1. Ejecutar el detector:
```bash
python pose_satisfaction_detector.py
```

2. Para salir del programa, presionar 'q'

## 📊 Métricas Analizadas

- Postura recta
- Nivel de hombros
- Centrado del cuerpo
- Score general de satisfacción

## 💾 Datos Exportados

Los datos se guardan automáticamente en la carpeta `data` en formato JSON con la siguiente estructura:
```json
{
    "timestamp": "2024-02-XX...",
    "satisfaction_score": 0.8,
    "metrics": {
        "posture_straight": true,
        "shoulders_level": true,
        "body_centered": false
    }
}
```

## 🔧 Personalización

Puedes ajustar los umbrales de detección en la clase `PoseSatisfactionDetector` según tus necesidades específicas. 