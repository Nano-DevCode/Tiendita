# 📄 Licencia y Términos de Propiedad Intelectual

**Copyright (c) 2026 Mayka — Todos los Derechos Reservados.**

El código fuente, arquitectura de software, algoritmos de interfaz visual (UI/UX), máquina de estados, sistema de cobro y documentación contenidos en este repositorio son obra intelectual original y propiedad exclusiva de su autor.

---

### 1. Términos de Uso y Permisos (Propiedad Intelectual)

Se concede permiso de manera gratuita a cualquier persona que obtenga una copia de este software y los archivos de documentación asociados con los siguientes propósitos:

1. **Revisión y Evaluación Profesional:** Lectura, análisis de código y evaluación técnica para fines de contratación, reclutamiento o demostración de competencias de portafolio.
2. **Fines Educativos y Académicos:** Estudio de la arquitectura de visión por computadora, inferencia con DirectML y procesamiento de tensores en tiempo real sin fines de lucro.
3. **Restricción Comercial:** Queda estrictamente prohibida la comercialización, sublicenciamiento, distribución cerrada con fines de lucro o apropiación de este código fuente sin el consentimiento explícito y por escrito del autor original.

---

### 2. Compatibilidad y Atribución de Licencias de Terceros

Este proyecto fue desarrollado integrando bibliotecas y tecnologías de código abierto de clase mundial. Cada una de ellas conserva sus respectivas licencias y derechos de autor originales:

| Biblioteca / Framework | Titular / Proyecto | Licencia de Terceros | Compatibilidad con Código Propietario |
| :--- | :--- | :--- | :--- |
| **OpenCV** (`cv2`) | OpenCV Team | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) | **Permisiva:** Permite uso, modificación e integración en software propietario. |
| **ONNX Runtime** | Microsoft Corporation | [MIT License](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | **Permisiva:** Permite uso comercial e integración en software cerrado sin restricciones. |
| **Supervision** (ByteTrack) | Roboflow Inc. | [MIT License](https://github.com/roboflow/supervision/blob/main/LICENSE) | **Permisiva:** Totalmente libre para integración de seguimiento visual. |
| **NumPy** | NumPy Developers | [BSD 3-Clause License](https://numpy.org/doc/stable/license.html) | **Permisiva:** Permite uso comercial y redistribución con preservación del copyright. |
| **Pillow** (`PIL`) | Alex Clark and Contributors | [HPND License](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | **Permisiva:** Similar a MIT/BSD para renderizado de fuentes y gráficos. |
| **Python Standard Library** | Python Software Foundation | [PSF License](https://docs.python.org/3/license.html) | **Permisiva:** Bibliotecas base de ejecución del sistema operativo. |

---

### 3. Consideraciones Especiales sobre YOLOv8 (Ultralytics)

* **Herramienta de Entrenamiento:** El framework Ultralytics YOLOv8 está licenciado bajo [GNU AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) para proyectos de código abierto y bajo licencias comerciales (Enterprise) para implementaciones propietarias cerradas.
* **Aislamiento Arquitectónico de Runtime:** En este proyecto, la aplicación en tiempo real (`main.py`, `FinalGPU.py`, `FinalCPU.py` y el paquete `src/`) **no importa ni ejecuta la librería `ultralytics`**. La inferencia se realiza de forma desacoplada y nativa utilizando el estándar abierto **ONNX** ejecutado por el motor permisivo **Microsoft ONNX Runtime (`onnxruntime-directml`)**.
* **Uso Académico / Portafolio:** Si este proyecto se utiliza o distribuye con fines de investigación, docencia o código abierto público, cumple con los lineamientos de la comunidad de visión artificial. Para implementaciones comerciales masivas a terceros basadas en modelos derivados de YOLOv8, se recomienda adquirir una licencia comercial de Ultralytics o alternativamente exportar hacia arquitecturas de licencia permisiva como RT-DETR (Apache 2.0).

---

*Para solicitudes de licencia comercial personalizada, dudas o autorizaciones, contactar directamente al autor a través de su perfil de GitHub.*
