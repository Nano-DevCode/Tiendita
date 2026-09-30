"""
Módulo de Interfaz de Usuario (UI).
Incluye diseño Glassmorphism Dark, componentes gráficos y composición responsiva.
"""

from .theme import Theme
from .renderer import UIRenderer
from .layout import ResponsiveLayout

__all__ = ['Theme', 'UIRenderer', 'ResponsiveLayout']
