# Objetivo de la Práctica 

**NOTA**: Esta práctica es el proyecto de Fin de Máster, es completamente opcional. Su realización exitosa permite, eso sí, la obtención de un título extra y acceso prioritario a la bolsa de empleo de Atrium.

# Objetivo de la Práctica

El objetivo de esta práctica consiste en resolver una problemática de negocio mediante las técnicas que hemos aprendido en el módulo de Deep Learning. Asimismo, para emular el tipo de proyectos a los que un científico de datos puede enfrentarse, se dejará bastante libertad creativa al alumno, para que elija y busque sus propias fuentes de datos, algoritmos pre-hechos y pre-entrenados, que investigue el estado del arte, para de manera creativa y original resolver el problema de negocio que se propone combinando todos estos elementos inteligentemente.

En cuanto a los directorios, archivos, nomenclaturas, etc., el alumno deberá utilizar la lógica y las buenas prácticas de programación y estructuración del código aprendidas hasta el momento para crear los módulos, ficheros, directorios, clases… más coherentes en base a los problemas que se pide solucionar.

# Problema de Negocio: Recomendador de Vídeos

Este será un recomendador de vídeos basado en contenido, es decir, basado en los objetos que aparecen en los fotogramas de los propios vídeos. La premisa en la que basaremos este sistema es que dos vídeos serán más similares entre ellos cuantos más objetos similares haya entre ambos. Por ejemplo, si en ambos vídeos aparecen coches todo el rato, serán más parecidos que si en uno aparecen coches frecuentemente y en el otro, perros y gatos casi todo el tiempo. Para resolverlo se deben seguir los siguientes pasos:

1. **Descargar vídeos**: Se elegirán al menos 3 categorías diferentes de vídeos (ej: vídeos musicales, vídeos de mascotas y vídeos de coches) y se descargarán al menos 100 ejemplos por categoría. Se puede utilizar cualquier fuente que se desee, como por ejemplo YouTube.

2. **Preprocesador de vídeos**: Se deberá crear un script de Python que convierta cada vídeo en un conjunto de fotogramas con nombres consecutivos (ej: 1.jpg, 2.jpg, 3.jpg…) y guarde cada conjunto de fotogramas en un directorio distinto (se recomienda crear un directorio por categoría y dentro un directorio por vídeo). Las imágenes guardadas deberán tener todas la misma resolución (ej: 224x224) y se aconseja que se use el mismo framerate en todos (por ejemplo, extraer 3 frames por segundo). Para no hacer el sistema muy pesado, se recomienda mantener la resolución y el framerate bajos.

3. **Detector de objetos basado en clasificador pre-entrenado**: Se deberá conseguir una red neuronal convolucional de clasificación pre-entrenada que sea capaz de distinguir entre un buen número de objetos diferentes en imágenes (ej: Las 1000 clases diferentes de ImageNet). Se creará un modelo a partir de esta red, cuya inferencia reciba un directorio con los fotogramas de un vídeo en orden y la salida sea un DataFrame de pandas con 3 columnas: el número de fotograma, la clase más probable detectada y el grado de confianza que da la red para dicha clasificación.

    **Bonus para subir nota**: Basándonos en un modelo pre-entrenado, usaremos nuestro propio dataset de imágenes más reducido con menos categorías que sea relevante al tipo de vídeos que queremos clasificar. Y cambiaremos la parte final del modelo para que tenga menos clases y lo reentrenaremos con nuestro propio dataset, dejando la parte convolucional “congelada” con los pesos originales obtenidos en el conjunto de datos original (ej: ImageNet). De esta manera, los outputs de la red serán más valiosos para nuestra tarea.

4. **Clasificador de vídeos**: Se utilizarán las inferencias del modelo anterior como input para entrenar el clasificador de tipo de vídeo. Hay que tener en cuenta que los inputs son variables en tamaño ya que dependerán de la longitud del vídeo y su framerate, por lo que habrá que realizar un post procesado de estos datos para que puedan ser utilizados como input a un clasificador al uso. Se sugiere, por ejemplo, crear una variable de entrada por cada tipo de objeto posible y asignarle como valor la "importancia" de dicho objeto en el vídeo. Esta "importancia" se podría calcular en base al número de fotogramas en los que aparece, así como teniendo en cuenta la "confiabilidad".

    **Bonus para subir nota**: Detector de objetos basado en segmentación semántica. Las imágenes de los vídeos pueden contener más de un objeto. Se sugiere que se use un modelo pre-entrenado de segmentación semántica (como MaskRCNN entrenado con MSCOCO) para crear un detector similar al primero. Este nuevo detector debe inferir un CSV parecido al anterior, pero añadiendo una línea por cada objeto detectado en cada fotograma. Además, sería recomendable poner una nueva columna con el % de la imagen que ocupa dicho objeto (no es lo mismo un objeto que aparezca en primer plano que uno alejado). También, al igual que en el primer detector, se podrá hacer un transfer learning para un conjunto de clases que nos interesen especialmente si se cree necesario.

5. **Recomendador de vídeos similares**: Basándonos en una de las inferencias de los detectores de objetos, se debe crear un sistema que las postprocese de tal manera que cada vídeo sea un punto en un espacio vectorial cuyas dimensiones estén dadas por cada objeto analizado. El post procesamiento creado para el clasificador, si se hizo correctamente, puede ser suficiente. Mediante técnicas como KNN y distancia coseno, se debe crear un sistema que te diga los 10 vídeos más parecidos dado un vídeo de entrada.

    **Bonus para subir nota**: Recomendador publicitario basado en Reinforcement Learning. Se creará un sistema que recomiende un anuncio publicitario cada vez que vea un vídeo de la plataforma. Al carecer de usuarios reales, se deberá realizar una simulación probabilística en la que un usuario ficticio vea vídeos de la plataforma y tenga mayor probabilidad de ver un anuncio cuanto más corto sea este y si es de la misma categoría que el vídeo que quiere visualizar. Habrá en total 2 tipos de anuncios por categoría, uno corto y uno largo; el corto tendrá más probabilidades de ser visualizado por el usuario que el largo, pero el rendimiento económico derivado de ello será menor, por lo que su recompensa asociada si se visualiza será menor también. Se deberá entrenar el modelo de Aprendizaje por Refuerzo basándonos en dicha simulación.

    **Ejemplo ilustrativo con un caso concreto**:

    Tenemos 3 categorías y 2 anuncios por cada una de ellas:

    - **Gatos:**
        - Anuncio Largo -> 30 segundos -> Probabilidad de visualización 0.3 -> Recompensa +3
        - Anuncio Corto -> 10 segundos -> Probabilidad de visualización 0.7 -> Recompensa +1
    - **Carreras de coches:**
        - Anuncio Largo -> 30 segundos -> Probabilidad de visualización 0.4 -> Recompensa +2
        - Anuncio Corto -> 10 segundos -> Probabilidad de visualización 0.8 -> Recompensa +1
    - **Conciertos:**
        - Anuncio Largo -> 30 segundos -> Probabilidad de visualización 0.5 -> Recompensa +2
        - Anuncio Corto -> 10 segundos -> Probabilidad de visualización 0.9 -> Recompensa +1

    Si no se visualiza el anuncio, la recompensa sería de 0.

    El usuario comienza viendo un vídeo aleatorio del set de datos y luego en la simulación continúa viendo los vídeos que el recomendador le dice. El sistema entrena para maximizar el rendimiento económico de los anuncios.