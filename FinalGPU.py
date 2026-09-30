"""
=============================================================================
TIENDITA INTELIGENTE IA - LANZADOR MODO GPU DIRECTML (PORTFOLIO EDITION)
=============================================================================
Ejecuta la aplicación aprovechando aceleración nativa DirectML sobre GPU
(AMD Radeon RX 6600M / 760M, Nvidia RTX/GTX o Intel Iris).
=============================================================================
"""

from main import ejecutar_aplicacion

if __name__ == '__main__':
    print("🚀 Iniciando Tiendita IA en modo GPU DirectML...")
    ejecutar_aplicacion(modo_inicial='ticket', forzar_cpu=False)