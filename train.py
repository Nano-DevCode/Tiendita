"""
=============================================================================
PIPELINE DE ENTRENAMIENTO Y ACTUALIZACIÓN AUTOMÁTICA DE MODELO YOLOv8
=============================================================================
Entrena el modelo sobre el dataset de productos (ProyectoFinalIA.v5i.yolov8).
Al finalizar con éxito:
1. Extrae los mejores pesos (best.pt).
2. Limpia todos los metadatos residuales (rutas, fechas, licencias).
3. Exporta a formato ONNX optimizado (opset 12) para aceleración DirectML (GPU).
4. Reemplaza automáticamente 'MiModelo_YOLO_BEST.pt' y 'MiModelo_YOLO_BEST.onnx'
   para que la aplicación 'main.py' lo utilice de inmediato con máxima precisión.
=============================================================================
"""

import os
import sys
import shutil
import argparse
import torch
import onnx
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, 'ProyectoFinalIA.v5i.yolov8', 'data.yaml')
MODELO_PT_DESTINO = os.path.join(BASE_DIR, 'MiModelo_YOLO_BEST.pt')
MODELO_ONNX_DESTINO = os.path.join(BASE_DIR, 'MiModelo_YOLO_BEST.onnx')


def limpiar_metadatos_pt(pt_path: str):
    """Elimina metadatos de entrenamiento, rutas absolutas y marcas de tiempo del archivo .pt."""
    print("🧹 Limpiando metadatos del modelo PyTorch (.pt)...")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    claves_eliminar = ['date', 'license', 'docs', 'git', 'train_args', 'train_metrics', 'train_results']
    for k in claves_eliminar:
        if k in ckpt:
            del ckpt[k]

    if hasattr(ckpt.get('model'), 'args'):
        ckpt['model'].args = None

    torch.save(ckpt, pt_path)
    print("✅ Archivo .pt limpio y optimizado.")


def limpiar_metadatos_onnx(onnx_path: str):
    """Elimina metadatos de exportación y rutas del archivo ONNX."""
    print("🧹 Limpiando metadatos del modelo ONNX (.onnx)...")
    model = onnx.load(onnx_path)

    # Conservar únicamente la topología y nombres de clases
    keep_keys = {'names', 'imgsz', 'stride', 'task'}
    new_props = [p for p in model.metadata_props if p.key in keep_keys]
    del model.metadata_props[:]
    model.metadata_props.extend(new_props)

    model.doc_string = ''
    model.producer_name = 'ONNX'
    model.producer_version = '1.0'

    onnx.save(model, onnx_path)
    print("✅ Archivo .onnx limpio y optimizado para inferencia DirectML.")


def entrenar(modelo_base: str = 'MiModelo_YOLO_BEST.pt', epochs: int = 30,
             batch: int = 16, imgsz: int = 640, device: str = 'cpu'):
    print("=" * 75)
    print("🎯 INICIANDO ENTRENAMIENTO DE MODELO DE VISIÓN ARTIFICIAL")
    print(f"📦 Dataset: {DATASET_YAML}")
    print(f"🧠 Modelo base: {modelo_base}")
    print(f"⚙️ Épocas: {epochs} | Batch: {batch} | Resolución: {imgsz} | Dispositivo: {device}")
    print("=" * 75)

    if not os.path.exists(DATASET_YAML):
        print(f"❌ Error: No se encontró el archivo de datos: {DATASET_YAML}")
        sys.exit(1)

    # Si no existe el modelo base local, usar yolov8s.pt como fallback
    if not os.path.exists(modelo_base):
        print(f"⚠️ Aviso: '{modelo_base}' no encontrado. Descargando 'yolov8s.pt'...")
        modelo_base = 'yolov8s.pt'

    # 1. Cargar modelo base para Transfer Learning (aprendizaje por transferencia)
    model = YOLO(modelo_base)

    # 2. Iniciar entrenamiento con hiperparámetros optimizados para convergencia rápida
    resultados = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,                 # Evita problemas de multiprocesamiento en Windows
        patience=10,               # Early stopping si no hay mejora en 10 épocas
        optimizer='auto',
        lr0=0.001,                 # Tasa de aprendizaje adaptativa
        lrf=0.01,                  # Reducción suave con función coseno
        mosaic=0.8,                # Data augmentation con mosaicos
        mixup=0.1,
        project='runs/train',
        name='tiendita_model',
        exist_ok=True,
        save=True,
        verbose=True
    )

    # 3. Localizar los mejores pesos obtenidos
    runs_dir = os.path.join(BASE_DIR, 'runs', 'train', 'tiendita_model', 'weights')
    best_pt = os.path.join(runs_dir, 'best.pt')

    if not os.path.exists(best_pt):
        print("❌ Error: No se encontró 'best.pt' al finalizar el entrenamiento.")
        sys.exit(1)

    print(f"\n🎉 ¡Entrenamiento completado con éxito! Mejor modelo en: {best_pt}")

    # 4. Copiar y limpiar el archivo .pt
    print("📦 Actualizando modelo principal 'MiModelo_YOLO_BEST.pt'...")
    shutil.copy2(best_pt, MODELO_PT_DESTINO)
    limpiar_metadatos_pt(MODELO_PT_DESTINO)

    # 5. Exportar a formato ONNX optimizado
    print("🔄 Exportando modelo a ONNX para aceleración por GPU DirectML...")
    nuevo_modelo = YOLO(MODELO_PT_DESTINO)
    onnx_generado = nuevo_modelo.export(
        format='onnx',
        imgsz=imgsz,
        opset=12,
        simplify=True,
        dynamic=False
    )

    if os.path.exists(onnx_generado):
        if onnx_generado != MODELO_ONNX_DESTINO:
            shutil.copy2(onnx_generado, MODELO_ONNX_DESTINO)
        limpiar_metadatos_onnx(MODELO_ONNX_DESTINO)
        print(f"✅ Nuevo modelo ONNX listo en: {MODELO_ONNX_DESTINO}")

    print("\n" + "=" * 75)
    print("🚀 ¡ACTUALIZACIÓN COMPLETADA CON ÉXITO!")
    print("Tu aplicación 'main.py' ya está lista para ejecutarse con el nuevo modelo entrenado.")
    print("Ejecuta: python main.py")
    print("=" * 75)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenador automático de modelo Tiendita IA")
    parser.add_argument('--epochs', type=int, default=25, help="Número de épocas de entrenamiento (default: 25)")
    parser.add_argument('--batch', type=int, default=16, help="Tamaño de lote (batch size, default: 16)")
    parser.add_argument('--model', type=str, default='MiModelo_YOLO_BEST.pt', help="Modelo base para transfer learning")
    parser.add_argument('--device', type=str, default='cpu', help="Dispositivo ('cpu' o '0' para GPU CUDA)")
    args = parser.parse_args()

    entrenar(
        modelo_base=args.model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device
    )
