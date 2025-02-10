# Motor de Recomendación y Publicidad en Videos mediante Segmentación Semántica y Aprendizaje por Refuerzo
 
<p align="center">
  <img src="https://cdn-images-1.medium.com/v2/resize:fit:2600/1*u9L_UJbV0Qfg1PZQkHna2g.png" alt="Deep Learning">
</p>

## Descripción del Proyecto

Este proyecto de fin de máster se centra en la creación de un sistema de recomendación de vídeos basado en el contenido de los mismos, utilizando técnicas avanzadas de Deep Learning. El objetivo es identificar y recomendar vídeos similares basándose en los objetos presentes en los fotogramas de los vídeos, para posteriormente implementar un recomendador publicitario basado en tecnicas de aprendizaje por refuerzo.

## Objetivos

- **Descarga y preprocesamiento** de vídeos de diferentes categorías.
- **Extracción de fotogramas** y preprocesamiento de imágenes.
- **Detección de objetos** en los fotogramas utilizando modelos preentrenados.
- **Clasificación de vídeos** basándose en los objetos detectados.
- **Recomendación de vídeos similares** utilizando técnicas de aprendizaje supervisado y no supervisado.
- **Implementación de un recomendador publicitario** basado en **Reinforcement Learning**.

## Estructura del Proyecto

La estructura del proyecto es la siguiente:

```
Proyecto
├── config
│   └── settings.conf
├── data
│   ├── detection_results_semantic/
│   ├── models/
│   ├── processed/
│   └── vectorized.csv
├── logger
│   └── logger_setup.py
├── logs
│   └── ...
├── notebooks
│   ├── 00_video_scraping.ipynb
│   ├── 01_video_preprocessing.ipynb
│   ├── 02_transfer_learning.ipynb
│   ├── 03_feature_extraction_classification.ipynb
│   ├── 04_Semantic_segmentation.ipynb
│   ├── 05_vectorization.ipynb
│   ├── 06_video_classifier.ipynb
│   ├── 07_video_recommender.ipynb
│   └── 08_rl_ad_recommender.ipynb
├── scripts
│   ├── 01_video_preprocessing.py
│   ├── 02_transfer_learning.py
│   ├── 03_feature_extraction_classification.py
│   ├── 04_Semantic_segmentation.py
│   ├── 05_vectorization.py
│   ├── 06_video_classifier.py
│   ├── 07_video_recommender.py
│   └── 08_rl_ad_recommender.py
├── LICENSE
├── main.py
├── README.md
├── SPECIFICATIONS.md
└── requirements.txt
```


### Descripción de los directorios y archivos principales

- **`00_video_scraping.ipynb`**: Notebook para la descarga de vídeos de diferentes categorías.
- **`02_transfer_learning.ipynb`**: Notebook para el ajuste fino de un modelo preentrenado utilizando Transfer Learning.
- **`03_feature_extraction_classification.ipynb`**: Notebook para la extracción de características y clasificación de fotogramas.
- **`04_semantic_segmentation.ipynb`**: Notebook para la detección de objetos utilizando segmentación semántica.
- **`05_vectorization.ipynb`**: Notebook para la creación de representaciones vectoriales de los vídeos.
- **`06_video_classifier.ipynb`**: Notebook para la clasificación de vídeos basándose en los objetos detectados.
- **`07_video_recommender.ipynb`**: Notebook para la recomendación de vídeos similares.
- **`08_rl_ad_recommender.ipynb`**: Notebook para la implementación de un recomendador publicitario basado en Reinforcement Learning.
- **`logger_setup.py`**: Script para la configuración del logger.
- **`settings.conf`**: Archivo de configuración que contiene rutas, hiperparámetros y ajustes para los modelos.
- **`data/`**: Carpeta que contiene los datos en sus diferentes estados (crudo, preprocesado, procesado).
- **`models/`**: Contiene los distintos modelos, sus pesos, métricas y resultados.
- **`logs/`**: Directorio donde se almacenan los registros de ejecución.

## Dataset

El dataset utilizado incluye vídeos de diferentes categorías descargados de YouTube. Los datos contienen información detallada de cada vídeo, como su categoría y los objetos presentes en los fotogramas.

## Explicación de los scripts y cómo ejecutarlos

### Requisitos previos

- Python 3.7 o superior.
- Bibliotecas listadas en `requirements.txt` (asegúrese de instalar todas las dependencias).

## Autor

**Pedro Ruiz**

- **LinkedIn**: [linkedin.com/in/pdro-ruiz](https://linkedin.com/in/pdro-ruiz/)

## Licencia

Este proyecto está bajo la Licencia MIT. Consulte el archivo [LICENSE](LICENSE) para obtener más información.

## Agradecimientos

A todos los profesores por su paciencia y dedicación, y los compañeros de la maestría por su apoyo.
