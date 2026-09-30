"""
=============================================================================
TIENDITA INTELIGENTE IA - LANZADOR MODO CPU (PORTFOLIO EDITION)
=============================================================================
Fuerza la ejecución en CPU utilizando ONNX Runtime con optimización AVX2.
Ideal para computadoras portátiles o estaciones sin GPU dedicada compatible.
=============================================================================
"""

from main import ejecutar_aplicacion

if __name__ == '__main__':
    print("🖥️ Iniciando Tiendita IA forzando modo CPU...")
    ejecutar_aplicacion(modo_inicial='ticket', forzar_cpu=True)