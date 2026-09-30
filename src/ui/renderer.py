"""
Renderizador Visual de Alta Definición (Anti-Aliased High-DPI Renderer).
Utiliza Pillow y el motor TrueType de Segoe UI para eliminar por completo la
pixelación y generar una interfaz estilo Fintech / Terminal Comercial moderna.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .theme import Theme, FontManager
from ..config import CATALOGO_PRECIOS, CLASES_NOMBRES


def _limpiar_para_cv2(texto: str) -> str:
    """
    Elimina emojis, caracteres multibyte o símbolos no-ASCII que OpenCV
    convierte en signos de interrogación '????'.
    Garantiza que el texto renderizado en pantalla sea 100% nítido y legible.
    """
    if not texto:
        return ""
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N', '¡': '', '¿': '', '•': '-',
        '✓': 'OK', '✔': 'OK'
    }
    for k, v in reemplazos.items():
        texto = texto.replace(k, v)
    # Conservar únicamente caracteres ASCII estándar imprimibles (32 al 126)
    return "".join(c for c in texto if 32 <= ord(c) <= 126).strip()


class UIRenderer:
    """Generador gráfico de alta resolución con renderizado sub-píxel anti-aliasing."""

    @staticmethod
    def render_sidebar_hd(width: int, height: int, estado, scale: float,
                          clases_presentes: set, video_offset: int = 0) -> np.ndarray:
        """
        Construye la barra lateral completa con Pillow utilizando fuentes TrueType,
        bordes redondeados y degradados. Retorna un arreglo BGR para OpenCV.
        Registra hitboxes clicables en estado.hitboxes_sidebar.
        """
        im = Image.new('RGB', (width, height), color=Theme.RGB_BG_DARK)
        draw = ImageDraw.Draw(im)

        estado.hitboxes_sidebar = []

        # Cargar fuentes del sistema escaladas
        f_title = FontManager.get_font(int(19 * scale), bold=True)
        f_subtitle = FontManager.get_font(int(11 * scale), bold=False)
        f_section = FontManager.get_font(int(11 * scale), bold=True)
        f_body = FontManager.get_font(int(13 * scale), bold=False)
        f_body_bold = FontManager.get_font(int(13 * scale), bold=True)
        f_badge = FontManager.get_font(int(11 * scale), bold=True)
        f_total_lbl = FontManager.get_font(int(12 * scale), bold=True)
        f_total_val = FontManager.get_font(int(24 * scale), bold=True)
        f_footer = FontManager.get_font(int(10 * scale), bold=False)
        f_footer_bold = FontManager.get_font(int(10 * scale), bold=True)

        pad = int(14 * scale)
        cw = width - (pad * 2)

        # -------------------------------------------------------------
        # 1. ENCABEZADO PRINCIPAL (HEADER)
        # -------------------------------------------------------------
        hdr_y = pad
        hdr_h = int(60 * scale)
        draw.rounded_rectangle(
            [pad, hdr_y, pad + cw, hdr_y + hdr_h],
            radius=int(10 * scale),
            fill=Theme.RGB_CARD_BG,
            outline=Theme.RGB_CARD_BORDER,
            width=1
        )

        # Encabezado dinámico según el modo activo
        if estado.modo_actual == 'inventario':
            header_accent = Theme.RGB_CYAN
            header_title = "AUDITORIA DE INVENTARIO IA"
            header_sub = "MONITOR DE STOCK Y ANAQUEL EN TIEMPO REAL"
        elif estado.modo_actual == 'completo':
            header_accent = Theme.RGB_GOLD
            header_title = "SISTEMA INTEGRAL POS & METRICAS"
            header_sub = "CAJERO COMERCIAL + TELEMETRIA EN VIVO"
        else:
            header_accent = Theme.RGB_GREEN
            header_title = "TIENDITA INTELIGENTE IA"
            header_sub = "TERMINAL PUNTO DE VENTA AUTONOMO (POS)"

        draw.rounded_rectangle(
            [pad + int(10 * scale), hdr_y + int(8 * scale), pad + cw - int(10 * scale), hdr_y + int(11 * scale)],
            radius=int(2 * scale),
            fill=header_accent
        )

        draw.text((pad + int(14 * scale), hdr_y + int(17 * scale)), header_title,
                  font=f_title, fill=Theme.RGB_TEXT_TITLE)
        draw.text((pad + int(14 * scale), hdr_y + int(40 * scale)), header_sub,
                  font=f_subtitle, fill=header_accent)

        # -------------------------------------------------------------
        # 2. VISOR PICTURE-IN-PICTURE (ULTIMO ARTICULO INSPECCIONADO)
        # -------------------------------------------------------------
        pip_y = hdr_y + hdr_h + int(10 * scale)
        pip_h = int(128 * scale)
        draw.rounded_rectangle(
            [pad, pip_y, pad + cw, pip_y + pip_h],
            radius=int(10 * scale),
            fill=Theme.RGB_CARD_BG,
            outline=Theme.RGB_CARD_BORDER,
            width=1
        )

        if estado.modo_actual == 'inventario':
            # PANEL DE AUDITORÍA Y COBERTURA DE STOCK
            num_presentes = sum(1 for p in CLASES_NOMBRES if p in clases_presentes)
            total_clases = len(CLASES_NOMBRES)
            pct_surtido = (num_presentes / total_clases) * 100.0

            draw.text((pad + int(14 * scale), pip_y + int(10 * scale)), "COBERTURA DE PRODUCTOS EN ANAQUEL",
                      font=f_section, fill=Theme.RGB_CYAN)

            # Barra gráfica visual de porcentaje de surtido
            bar_w = cw - int(28 * scale)
            bar_h = int(14 * scale)
            bar_x = pad + int(14 * scale)
            bar_y = pip_y + int(32 * scale)

            draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                                  radius=int(5 * scale), fill=(20, 26, 36), outline=Theme.RGB_CARD_BORDER, width=1)
            fill_w = int((bar_w * (pct_surtido / 100.0)))
            if fill_w > 0:
                bar_color = Theme.RGB_GREEN if pct_surtido >= 60 else Theme.RGB_CYAN
                draw.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h],
                                      radius=int(5 * scale), fill=bar_color)

            draw.text((bar_x, pip_y + int(52 * scale)),
                      f"SURTIDO EN CAMARA: {num_presentes} de {total_clases} productos ({pct_surtido:.0f}%)",
                      font=f_body_bold, fill=Theme.RGB_TEXT_TITLE)

            faltantes = [p for p in CLASES_NOMBRES if p not in clases_presentes]
            if not faltantes:
                txt_status = "[OK] TODOS LOS PRODUCTOS PRESENTES EN ANAQUEL"
                col_status = Theme.RGB_GREEN
            else:
                txt_status = f"[!] Faltantes a resurtir: {', '.join(faltantes[:3])}" + ("..." if len(faltantes) > 3 else "")
                col_status = Theme.RGB_CORAL

            draw.text((bar_x, pip_y + int(74 * scale)), txt_status,
                      font=f_subtitle, fill=col_status)

            badge_st_w = int(220 * scale)
            badge_st_y = pip_y + int(96 * scale)
            draw.rounded_rectangle([bar_x, badge_st_y, bar_x + badge_st_w, badge_st_y + int(20 * scale)],
                                  radius=int(4 * scale),
                                  fill=(16, 42, 34) if num_presentes >= 3 else (45, 20, 24),
                                  outline=Theme.RGB_GREEN if num_presentes >= 3 else Theme.RGB_CORAL, width=1)
            txt_badge = "ESTADO: ANAQUEL OPTIMO" if num_presentes >= 3 else "ESTADO: ALERTA DE RESURTIDO"
            draw.text((bar_x + int(8 * scale), badge_st_y + int(3 * scale)), txt_badge,
                      font=f_badge, fill=Theme.RGB_MINT_LIGHT if num_presentes >= 3 else Theme.RGB_CORAL)

        elif estado.modo_actual == 'completo':
            # MODO COMPLETO: Split View (PiP compacto a la izquierda + Telemetría a la derecha)
            draw.text((pad + int(14 * scale), pip_y + int(10 * scale)), "TELEMETRIA DE INFERENCIA DIRECTML & ESCANEO",
                      font=f_section, fill=Theme.RGB_GOLD)

            marco_x = pad + int(14 * scale)
            marco_y = pip_y + int(28 * scale)
            marco_w = int(95 * scale)
            marco_h = int(88 * scale)

            if estado.ultimo_crop is not None:
                crop_rgb = cv2.cvtColor(estado.ultimo_crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(cv2.resize(crop_rgb, (marco_w, marco_h), interpolation=cv2.INTER_LINEAR))
                im.paste(crop_pil, (marco_x, marco_y))
                draw.rounded_rectangle([marco_x, marco_y, marco_x + marco_w, marco_y + marco_h],
                                      radius=int(6 * scale), outline=Theme.RGB_GOLD, width=1)
            else:
                draw.rounded_rectangle([marco_x, marco_y, marco_x + marco_w, marco_y + marco_h],
                                      radius=int(6 * scale), fill=(18, 22, 30), outline=Theme.RGB_CARD_BORDER, width=1)
                draw.text((marco_x + int(14 * scale), marco_y + int(36 * scale)), "ESCANER",
                          font=f_section, fill=Theme.RGB_TEXT_DIM)

            info_x = marco_x + marco_w + int(14 * scale)
            draw.text((info_x, pip_y + int(28 * scale)), "ACELERACION DIRECTML ACTIVA",
                      font=f_body_bold, fill=Theme.RGB_GOLD)
            draw.text((info_x, pip_y + int(48 * scale)), f"Ultimo Articulo: {estado.ultimo_nombre or 'Ninguno'}",
                      font=f_body, fill=Theme.RGB_TEXT_TITLE)
            draw.text((info_x, pip_y + int(68 * scale)), f"Precio: ${estado.ultimo_precio:.2f} | Conf: {estado.ultima_conf:.0%}",
                      font=f_subtitle, fill=Theme.RGB_TEXT_MUTED)
            draw.text((info_x, pip_y + int(88 * scale)), "ByteTrack: 30 fps buffer | NMS: 0.45",
                      font=f_subtitle, fill=Theme.RGB_CYAN)

        else:
            # MODO TICKET: Visor PiP del último artículo escaneado
            draw.text((pad + int(14 * scale), pip_y + int(10 * scale)), "ULTIMO ARTICULO INSPECCIONADO",
                      font=f_section, fill=Theme.RGB_TEXT_MUTED)

            marco_x = pad + int(14 * scale)
            marco_y = pip_y + int(28 * scale)
            marco_w = int(105 * scale)
            marco_h = int(88 * scale)

            if estado.ultimo_crop is not None:
                crop_rgb = cv2.cvtColor(estado.ultimo_crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(cv2.resize(crop_rgb, (marco_w, marco_h), interpolation=cv2.INTER_LINEAR))
                im.paste(crop_pil, (marco_x, marco_y))
                draw.rounded_rectangle(
                    [marco_x, marco_y, marco_x + marco_w, marco_y + marco_h],
                    radius=int(6 * scale),
                    outline=Theme.RGB_GREEN,
                    width=1
                )

                info_x = marco_x + marco_w + int(14 * scale)
                draw.text((info_x, pip_y + int(28 * scale)), estado.ultimo_nombre.upper(),
                          font=FontManager.get_font(int(17 * scale), bold=True), fill=Theme.RGB_TEXT_TITLE)
                draw.text((info_x, pip_y + int(52 * scale)), f"Precio Unit.:  ${estado.ultimo_precio:.2f} MXN",
                          font=f_body_bold, fill=Theme.RGB_GREEN)
                draw.text((info_x, pip_y + int(72 * scale)), f"Certeza: {estado.ultima_conf:.0%}   |   Hora: {estado.ultimo_tiempo}",
                          font=f_subtitle, fill=Theme.RGB_TEXT_MUTED)

                badge_w = int(115 * scale)
                badge_h = int(18 * scale)
                badge_y = pip_y + int(94 * scale)
                draw.rounded_rectangle(
                    [info_x, badge_y, info_x + badge_w, badge_y + badge_h],
                    radius=int(5 * scale),
                    fill=(12, 45, 34),
                    outline=Theme.RGB_GREEN,
                    width=1
                )
                draw.text((info_x + int(10 * scale), badge_y + int(2 * scale)), "OK - ESCANEADO",
                          font=f_badge, fill=Theme.RGB_MINT_LIGHT)
            else:
                draw.rounded_rectangle(
                    [marco_x, marco_y, marco_x + marco_w, marco_y + marco_h],
                    radius=int(6 * scale),
                    fill=(15, 20, 28),
                    outline=Theme.RGB_CARD_BORDER,
                    width=1
                )
                draw.text((marco_x + int(20 * scale), marco_y + int(36 * scale)), "EN ESPERA",
                          font=f_section, fill=Theme.RGB_TEXT_DIM)

                info_x = marco_x + marco_w + int(14 * scale)
                draw.text((info_x, pip_y + int(38 * scale)), "Esperando producto...",
                          font=f_body_bold, fill=Theme.RGB_TEXT_MUTED)
                draw.text((info_x, pip_y + int(60 * scale)), "Coloca un articulo frente",
                          font=f_subtitle, fill=Theme.RGB_TEXT_DIM)
                draw.text((info_x, pip_y + int(78 * scale)), "al lente de la camara web.",
                          font=f_subtitle, fill=Theme.RGB_TEXT_DIM)

        # -------------------------------------------------------------
        # 3. TABLA DE PRODUCTOS / CATÁLOGO
        # -------------------------------------------------------------
        tbl_y = pip_y + pip_h + int(10 * scale)
        row_h = int(32 * scale)
        tbl_h = int(28 * scale) + (len(CLASES_NOMBRES) * row_h) + int(8 * scale)
        draw.rounded_rectangle(
            [pad, tbl_y, pad + cw, tbl_y + tbl_h],
            radius=int(10 * scale),
            fill=Theme.RGB_CARD_BG,
            outline=Theme.RGB_CARD_BORDER,
            width=1
        )

        hdr_row_y = tbl_y + int(8 * scale)
        hdr_prod_txt = "AUDITORIA DE STOCK" if estado.modo_actual == 'inventario' else "PRODUCTO"
        col_hdr = Theme.RGB_CYAN if estado.modo_actual == 'inventario' else Theme.RGB_TEXT_MUTED
        draw.text((pad + int(16 * scale), hdr_row_y), hdr_prod_txt, font=f_section, fill=col_hdr)
        draw.text((width - int(205 * scale), hdr_row_y), "P. UNIT", font=f_section, fill=Theme.RGB_TEXT_MUTED)
        draw.text((width - int(145 * scale), hdr_row_y), "STOCK" if estado.modo_actual == 'inventario' else "AJUSTE", font=f_section, fill=col_hdr if estado.modo_actual == 'inventario' else Theme.RGB_TEXT_MUTED)
        draw.text((width - int(72 * scale), hdr_row_y), "SUBTOTAL", font=f_section, fill=Theme.RGB_TEXT_MUTED)

        draw.line([pad + int(10 * scale), hdr_row_y + int(18 * scale),
                   width - pad - int(10 * scale), hdr_row_y + int(18 * scale)],
                  fill=Theme.RGB_CARD_BORDER, width=1)

        curr_y = hdr_row_y + int(24 * scale)
        for i, prod in enumerate(CLASES_NOMBRES, start=1):
            precio_u = CATALOGO_PRECIOS.get(prod, 0.0)
            cant = estado.inventario.get(prod, 0)
            sub_prod = cant * precio_u
            en_vista = prod in clases_presentes

            if en_vista:
                draw.rounded_rectangle(
                    [pad + int(6 * scale), curr_y - int(4 * scale),
                     width - pad - int(6 * scale), curr_y + row_h - int(8 * scale)],
                    radius=int(6 * scale),
                    fill=Theme.RGB_CARD_ACTIVE
                )
                draw.ellipse([pad + int(12 * scale), curr_y + int(5 * scale),
                              pad + int(18 * scale), curr_y + int(11 * scale)],
                             fill=Theme.RGB_CYAN)
            else:
                draw.ellipse([pad + int(12 * scale), curr_y + int(5 * scale),
                              pad + int(17 * scale), curr_y + int(10 * scale)],
                             fill=(45, 55, 70))

            col_nombre = Theme.RGB_TEXT_TITLE if cant > 0 or en_vista else Theme.RGB_TEXT_DIM
            draw.text((pad + int(24 * scale), curr_y), f"[{i}] {prod}", font=f_body, fill=col_nombre)
            draw.text((width - int(205 * scale), curr_y), f"${precio_u:.2f}",
                      font=f_body, fill=Theme.RGB_TEXT_MUTED)

            if estado.modo_actual == 'inventario':
                # En modo inventario: Controles táctiles de stock en bodega [-] y [+]
                stk_disp = estado.obtener_stock_disponible(prod)
                btn_stk_w = int(18 * scale)
                btn_stk_h = int(18 * scale)

                # Botón [-] para decrementar stock de almacén
                btn_m_x = width - int(158 * scale)
                btn_m_y = curr_y - int(1 * scale)
                if stk_disp > 0:
                    draw.rounded_rectangle(
                        [btn_m_x, btn_m_y, btn_m_x + btn_stk_w, btn_m_y + btn_stk_h],
                        radius=int(4 * scale), fill=(45, 20, 24), outline=Theme.RGB_CORAL, width=1
                    )
                    draw.text((btn_m_x + int(5 * scale), btn_m_y + int(1 * scale)), "-", font=f_badge, fill=Theme.RGB_CORAL)
                    estado.hitboxes_sidebar.append({
                        'type': 'stock_minus',
                        'prod': prod,
                        'box': (video_offset + btn_m_x, btn_m_y,
                                video_offset + btn_m_x + btn_stk_w, btn_m_y + btn_stk_h)
                    })

                # Píldora de existencias disponibles
                pill_stk_x = width - int(136 * scale)
                pill_stk_y = curr_y - int(1 * scale)
                pill_stk_w = int(26 * scale)
                pill_stk_h = int(18 * scale)
                if stk_disp > 0:
                    draw.rounded_rectangle(
                        [pill_stk_x, pill_stk_y, pill_stk_x + pill_stk_w, pill_stk_y + pill_stk_h],
                        radius=int(4 * scale), fill=(12, 45, 30), outline=Theme.RGB_GREEN, width=1
                    )
                    txt_stk = f"{stk_disp:2d}"
                    draw.text((pill_stk_x + int(4 * scale), curr_y), txt_stk, font=f_badge, fill=Theme.RGB_MINT_LIGHT)
                else:
                    draw.rounded_rectangle(
                        [pill_stk_x, pill_stk_y, pill_stk_x + pill_stk_w, pill_stk_y + pill_stk_h],
                        radius=int(4 * scale), fill=(45, 20, 24), outline=Theme.RGB_CORAL, width=1
                    )
                    draw.text((pill_stk_x + int(6 * scale), curr_y), "0", font=f_badge, fill=Theme.RGB_CORAL)

                # Botón [+] para incrementar stock de almacén
                btn_p_x = width - int(106 * scale)
                btn_p_y = curr_y - int(1 * scale)
                draw.rounded_rectangle(
                    [btn_p_x, btn_p_y, btn_p_x + btn_stk_w, btn_p_y + btn_stk_h],
                    radius=int(4 * scale), fill=(18, 42, 30), outline=Theme.RGB_GREEN, width=1
                )
                draw.text((btn_p_x + int(4 * scale), btn_p_y + int(1 * scale)), "+", font=f_badge, fill=Theme.RGB_GREEN)
                estado.hitboxes_sidebar.append({
                    'type': 'stock_plus',
                    'prod': prod,
                    'box': (video_offset + btn_p_x, btn_p_y,
                            video_offset + btn_p_x + btn_stk_w, btn_p_y + btn_stk_h)
                })

                val_stock = stk_disp * precio_u
                col_sub = Theme.RGB_CYAN if stk_disp > 0 else Theme.RGB_CORAL
                draw.text((width - int(72 * scale), curr_y), f"${val_stock:.2f}", font=f_body_bold, fill=col_sub)

            else:
                # En modo ticket / completo: Controles interactivos de carrito [-] y [+]
                btn_minus_w = int(18 * scale)
                btn_minus_h = int(18 * scale)
                btn_minus_x = width - int(158 * scale)
                btn_minus_y = curr_y - int(1 * scale)

                if cant > 0:
                    draw.rounded_rectangle(
                        [btn_minus_x, btn_minus_y, btn_minus_x + btn_minus_w, btn_minus_y + btn_minus_h],
                        radius=int(4 * scale),
                        fill=(45, 20, 24),
                        outline=Theme.RGB_CORAL,
                        width=1
                    )
                    draw.text((btn_minus_x + int(5 * scale), btn_minus_y + int(1 * scale)), "-",
                              font=f_badge, fill=Theme.RGB_CORAL)
                    estado.hitboxes_sidebar.append({
                        'type': 'minus',
                        'prod': prod,
                        'box': (video_offset + btn_minus_x, btn_minus_y,
                                video_offset + btn_minus_x + btn_minus_w, btn_minus_y + btn_minus_h)
                    })

                # Píldora de cantidad en carrito
                pill_x = width - int(136 * scale)
                pill_y = curr_y - int(1 * scale)
                pill_w = int(26 * scale)
                pill_h = int(18 * scale)

                if cant > 0:
                    draw.rounded_rectangle(
                        [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                        radius=int(4 * scale),
                        fill=(16, 45, 60),
                        outline=Theme.RGB_CYAN,
                        width=1
                    )
                    draw.text((pill_x + int(4 * scale), curr_y), f"x{cant}", font=f_badge, fill=Theme.RGB_CYAN)
                else:
                    draw.text((pill_x + int(9 * scale), curr_y), "0", font=f_body, fill=Theme.RGB_TEXT_DIM)

                # Botón interactivo de suma [+] (clicable para agregar manual)
                btn_plus_w = int(18 * scale)
                btn_plus_h = int(18 * scale)
                btn_plus_x = width - int(106 * scale)
                btn_plus_y = curr_y - int(1 * scale)

                draw.rounded_rectangle(
                    [btn_plus_x, btn_plus_y, btn_plus_x + btn_plus_w, btn_plus_y + btn_plus_h],
                    radius=int(4 * scale),
                    fill=(18, 42, 30),
                    outline=Theme.RGB_GREEN,
                    width=1
                )
                draw.text((btn_plus_x + int(4 * scale), btn_plus_y + int(1 * scale)), "+",
                          font=f_badge, fill=Theme.RGB_GREEN)
                estado.hitboxes_sidebar.append({
                    'type': 'plus',
                    'prod': prod,
                    'box': (video_offset + btn_plus_x, btn_plus_y,
                            video_offset + btn_plus_x + btn_plus_w, btn_plus_y + btn_plus_h)
                })

                col_sub = Theme.RGB_GREEN if cant > 0 else Theme.RGB_TEXT_DIM
                draw.text((width - int(72 * scale), curr_y), f"${sub_prod:.2f}", font=f_body_bold, fill=col_sub)

            curr_y += row_h

        # -------------------------------------------------------------
        # 4. BARRA DE CONTROLES INFERIOR (FOOTER)
        # -------------------------------------------------------------
        ftr_h = int(58 * scale)
        ftr_y = height - ftr_h - pad
        draw.rounded_rectangle(
            [pad, ftr_y, pad + cw, ftr_y + ftr_h],
            radius=int(10 * scale),
            fill=Theme.RGB_CARD_BG,
            outline=Theme.RGB_CARD_BORDER,
            width=1
        )

        if estado.modo_actual == 'inventario':
            draw.text((pad + int(14 * scale), ftr_y + int(10 * scale)),
                      "[L] Resurtir Stock (+10)   |   [M] Cambiar Modo   |   [Q] Salir",
                      font=f_footer_bold, fill=Theme.RGB_CYAN)
            draw.text((pad + int(14 * scale), ftr_y + int(32 * scale)),
                      "[P] Pausar Camara   |   [+] / [-] Calibrar Umbral   |   [H] Ayuda",
                      font=f_footer_bold, fill=Theme.RGB_TEXT_MUTED)
        elif estado.modo_actual == 'completo':
            draw.text((pad + int(14 * scale), ftr_y + int(10 * scale)),
                      "[M] Siguiente Modo   |   [C] Cobrar   |   [B] Borrar   |   [Q] Salir",
                      font=f_footer_bold, fill=Theme.RGB_GOLD)
            draw.text((pad + int(14 * scale), ftr_y + int(32 * scale)),
                      "[F] Pantalla Completa   |   [P] Pausar   |   [H] Ayuda de Sistema",
                      font=f_footer_bold, fill=Theme.RGB_TEXT_MUTED)
        else:
            draw.text((pad + int(14 * scale), ftr_y + int(10 * scale)),
                      "[B] Borrado Manual   |   [Z] Deshacer   |   [1-6] Quitar   |   [Q] Salir",
                      font=f_footer_bold, fill=Theme.RGB_TEXT_TITLE)
            draw.text((pad + int(14 * scale), ftr_y + int(32 * scale)),
                      "[C] Cobrar   |   [S] Ticket   |   [D] -10%   |   [R] Reset   |   [M] Modo",
                      font=f_footer_bold, fill=Theme.RGB_GREEN)

        estado.hitboxes_sidebar.append({
            'type': 'btn_modal_borrado',
            'prod': '',
            'box': (video_offset + pad, ftr_y, video_offset + pad + int(160 * scale), ftr_y + int(26 * scale))
        })

        # -------------------------------------------------------------
        # 5. RESUMEN FINANCIERO Y METRICAS DE STOCK
        # -------------------------------------------------------------
        res_y = tbl_y + tbl_h + int(10 * scale)
        res_h = ftr_y - res_y - int(10 * scale)

        if res_h > int(80 * scale):
            draw.rounded_rectangle(
                [pad, res_y, pad + cw, res_y + res_h],
                radius=int(10 * scale),
                fill=Theme.RGB_CARD_BG,
                outline=Theme.RGB_CARD_BORDER,
                width=1
            )

            if estado.modo_actual == 'inventario':
                total_piezas_bodega = sum(estado.stock_tienda.values())
                valor_total_bodega = sum(estado.stock_tienda.get(p, 0) * CATALOGO_PRECIOS.get(p, 0.0) for p in CLASES_NOMBRES)
                num_agotados = sum(1 for p in CLASES_NOMBRES if estado.obtener_stock_disponible(p) <= 0)

                draw.text((pad + int(16 * scale), res_y + int(10 * scale)),
                          f"Stock en Bodega: {total_piezas_bodega} pzas  |  Agotados: {num_agotados}",
                          font=f_body, fill=Theme.RGB_CYAN)
                # Botón táctil para resurtir +10 a todos los productos
                btn_res_w = int(148 * scale)
                btn_res_h = int(24 * scale)
                btn_res_x = width - pad - btn_res_w - int(4 * scale)
                btn_res_y = res_y + int(6 * scale)

                draw.rounded_rectangle(
                    [btn_res_x, btn_res_y, btn_res_x + btn_res_w, btn_res_y + btn_res_h],
                    radius=int(6 * scale),
                    fill=(16, 48, 32),
                    outline=Theme.RGB_GREEN,
                    width=1
                )
                draw.text((btn_res_x + int(10 * scale), btn_res_y + int(4 * scale)),
                          "+10 A TODO (Clic/L)",
                          font=f_badge, fill=Theme.RGB_MINT_LIGHT)

                estado.hitboxes_sidebar.append({
                    'type': 'stock_resurtir_todo',
                    'prod': '',
                    'box': (video_offset + btn_res_x, btn_res_y,
                            video_offset + btn_res_x + btn_res_w, btn_res_y + btn_res_h)
                })

                total_y = res_y + int(48 * scale)
                total_h = max(int(46 * scale), res_h - int(56 * scale))
                draw.rounded_rectangle(
                    [pad + int(10 * scale), total_y, pad + cw - int(10 * scale), total_y + total_h],
                    radius=int(8 * scale),
                    fill=(14, 28, 40),
                    outline=Theme.RGB_CYAN,
                    width=2
                )
                draw.text((pad + int(20 * scale), total_y + int(6 * scale)),
                          "VALOR TOTAL DE INVENTARIO EN BODEGA:",
                          font=f_total_lbl, fill=Theme.RGB_CYAN)
                draw.text((pad + int(20 * scale), total_y + int(22 * scale)),
                          f"${valor_total_bodega:8.2f} MXN",
                          font=f_total_val, fill=Theme.RGB_TEXT_TITLE)

            elif estado.modo_actual == 'completo':
                draw.text((pad + int(16 * scale), res_y + int(10 * scale)),
                          f"POS + Telemetria  |  Articulos: {estado.total_items} pzas",
                          font=f_body, fill=Theme.RGB_GOLD)
                draw.text((width - int(185 * scale), res_y + int(10 * scale)),
                          f"Subtotal: ${estado.subtotal:7.2f}",
                          font=f_body_bold, fill=Theme.RGB_TEXT_TITLE)

                total_y = res_y + int(48 * scale)
                total_h = max(int(46 * scale), res_h - int(56 * scale))
                draw.rounded_rectangle(
                    [pad + int(10 * scale), total_y, pad + cw - int(10 * scale), total_y + total_h],
                    radius=int(8 * scale),
                    fill=(30, 26, 16),
                    outline=Theme.RGB_GOLD,
                    width=2
                )
                draw.text((pad + int(20 * scale), total_y + int(6 * scale)),
                          "TOTAL COMBINADO (MXN):",
                          font=f_total_lbl, fill=Theme.RGB_GOLD)
                draw.text((pad + int(20 * scale), total_y + int(22 * scale)),
                          f"${estado.total_precio:8.2f} MXN",
                          font=f_total_val, fill=Theme.RGB_TEXT_TITLE)

            else:
                draw.text((pad + int(16 * scale), res_y + int(10 * scale)),
                          f"Artículos Totales:  {estado.total_items} piezas",
                          font=f_body, fill=Theme.RGB_TEXT_MUTED)
                draw.text((width - int(185 * scale), res_y + int(10 * scale)),
                          f"Subtotal:  ${estado.subtotal:7.2f} MXN",
                          font=f_body_bold, fill=Theme.RGB_TEXT_TITLE)

                if estado.descuento_pct > 0:
                    monto_desc = estado.subtotal * (estado.descuento_pct / 100.0)
                    draw.text((pad + int(16 * scale), res_y + int(28 * scale)),
                              f"Descuento Activo (10%):  -${monto_desc:.2f} MXN",
                              font=f_body, fill=Theme.RGB_GOLD)

                total_y = res_y + int(48 * scale)
                total_h = max(int(46 * scale), res_h - int(56 * scale))
                draw.rounded_rectangle(
                    [pad + int(10 * scale), total_y, pad + cw - int(10 * scale), total_y + total_h],
                    radius=int(8 * scale),
                    fill=Theme.RGB_TOTAL_BG_TOP,
                    outline=Theme.RGB_GREEN,
                    width=2
                )
                draw.text((pad + int(20 * scale), total_y + int(6 * scale)),
                          "TOTAL A PAGAR (MXN):",
                          font=f_total_lbl, fill=Theme.RGB_MINT_LIGHT)
                draw.text((pad + int(20 * scale), total_y + int(22 * scale)),
                          f"${estado.total_precio:8.2f} MXN",
                          font=f_total_val, fill=Theme.RGB_TEXT_TITLE)

        sidebar_rgb = np.array(im)
        return cv2.cvtColor(sidebar_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def render_hud_superior(frame: np.ndarray, fps: float, device_name: str,
                            modo_nombre: str, toast_mensaje: str, toast_alfa: float,
                            descuento_activo: bool, pausado: bool, scale: float = 1.0):
        """Dibuja la barra de estado superior con anti-aliasing sobre el fotograma."""
        h, w = frame.shape[:2]
        hud_h = int(38 * scale)
        hud_w = int(410 * scale)

        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 12), (15 + hud_w, 12 + hud_h), Theme.BGR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (15, 12), (15 + hud_w, 12 + hud_h), Theme.BGR_CARD_BORDER, 1)

        dot_col = Theme.BGR_CORAL if pausado else Theme.BGR_GREEN
        cv2.circle(frame, (15 + int(16 * scale), 12 + int(19 * scale)), max(3, int(5 * scale)), dot_col, -1)

        st_txt = "PAUSADO" if pausado else "EN VIVO"
        cv2.putText(frame, st_txt, (15 + int(28 * scale), 12 + int(24 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44 * scale, dot_col, 1, cv2.LINE_AA)

        color_fps = Theme.BGR_GREEN if fps >= 22 else (Theme.BGR_GOLD if fps >= 15 else Theme.BGR_CORAL)
        cv2.putText(frame, f"FPS: {fps:4.1f}", (15 + int(115 * scale), 12 + int(24 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44 * scale, color_fps, 1, cv2.LINE_AA)

        cv2.putText(frame, f"| {device_name[:18]}", (15 + int(210 * scale), 12 + int(24 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42 * scale, Theme.BGR_TEXT_MUTED, 1, cv2.LINE_AA)

        if modo_nombre == 'inventario':
            badge_str = "MODO: AUDITORIA DE STOCK"
            badge_col = Theme.BGR_CYAN
        elif modo_nombre == 'completo':
            badge_str = "MODO: COMPLETO (POS + TELEMETRIA)"
            badge_col = Theme.BGR_GOLD
        else:
            badge_str = "MODO: PUNTO DE VENTA (TICKET)"
            badge_col = Theme.BGR_GREEN

        if descuento_activo:
            badge_str += " | [10% OFF]"

        (bw, bh), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_DUPLEX, 0.44 * scale, 1)
        bx = w - bw - int(32 * scale)
        cv2.rectangle(frame, (bx - 10, 12), (w - 15, 12 + hud_h), Theme.BGR_CARD_BG, -1)
        cv2.rectangle(frame, (bx - 10, 12), (w - 15, 12 + hud_h), badge_col, 1)
        cv2.putText(frame, badge_str, (bx, 12 + int(24 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44 * scale, badge_col, 1, cv2.LINE_AA)

        if toast_mensaje and toast_alfa > 0.05:
            txt_limpio = _limpiar_para_cv2(toast_mensaje)
            (tw, th), _ = cv2.getTextSize(txt_limpio, cv2.FONT_HERSHEY_DUPLEX, 0.50 * scale, 1)
            toast_w = tw + int(42 * scale)
            toast_h = int(38 * scale)
            toast_x = (w - toast_w) // 2
            toast_y = h - int(60 * scale)

            t_over = frame.copy()
            cv2.rectangle(t_over, (toast_x, toast_y), (toast_x + toast_w, toast_y + toast_h), (16, 42, 30), -1)
            cv2.addWeighted(t_over, min(1.0, toast_alfa * 0.90), frame, 1.0 - min(1.0, toast_alfa * 0.90), 0, frame)
            cv2.rectangle(frame, (toast_x, toast_y), (toast_x + toast_w, toast_y + toast_h), Theme.BGR_GREEN, 1)
            cv2.putText(frame, txt_limpio, (toast_x + int(20 * scale), toast_y + int(25 * scale)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.50 * scale, Theme.BGR_TEXT_TITLE, 1, cv2.LINE_AA)

    @staticmethod
    def render_modal_ayuda(frame_completo: np.ndarray, scale: float = 1.0):
        """Renderiza el modal de arquitectura y ayuda centrado."""
        h, w = frame_completo.shape[:2]
        mw, mh = int(min(w * 0.88, 800 * scale)), int(min(h * 0.90, 540 * scale))
        mx, my = (w - mw) // 2, (h - mh) // 2

        overlay = frame_completo.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (5, 8, 12), -1)
        cv2.addWeighted(overlay, 0.85, frame_completo, 0.15, 0, frame_completo)

        cv2.rectangle(frame_completo, (mx, my), (mx + mw, my + mh), (22, 28, 38), -1)
        cv2.rectangle(frame_completo, (mx, my), (mx + mw, my + mh), Theme.BGR_CYAN, 2)

        cv2.putText(frame_completo, "PROYECTO DE PORTAFOLIO: TIENDITA INTELIGENTE IA",
                    (mx + int(25 * scale), my + int(38 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62 * scale, Theme.BGR_TEXT_TITLE, 2, cv2.LINE_AA)
        cv2.line(frame_completo, (mx + int(25 * scale), my + int(50 * scale)),
                 (mx + mw - int(25 * scale), my + int(50 * scale)), Theme.BGR_CARD_BORDER, 1)

        texto_lineas = [
            ("ARQUITECTURA DEL SISTEMA (Clean Architecture):", Theme.BGR_CYAN),
            (" - Inferencia: ONNX Runtime con aceleracion DirectML nativa (GPU AMD/Intel/Nvidia).", Theme.BGR_TEXT_TITLE),
            (" - Tracking: ByteTrack multi-objeto asignando IDs unicos para evitar cobros dobles.", Theme.BGR_TEXT_TITLE),
            (" - Post-procesamiento: Pipeline 100% vectorizado con NumPy (0 bucles Python).", Theme.BGR_TEXT_TITLE),
            (" - Interfaz: High-DPI TrueType (Segoe UI) con renderizado anti-aliasing sin pixelacion.", Theme.BGR_TEXT_TITLE),
            ("", Theme.BGR_TEXT_TITLE),
            ("ATAJOS DE TECLADO:", Theme.BGR_GOLD),
            (" - [Z] / [Backspace] / [B]: Deshacer o borrar el ultimo producto escaneado del ticket.", Theme.BGR_CYAN),
            (" - [1] al [6]: Restar 1 unidad del producto correspondiente (ej. [2] para quitar Atun).", Theme.BGR_CYAN),
            (" - [C]: Simular cobro en caja (timbrado fiscal y reseteo para nuevo cliente).", Theme.BGR_TEXT_TITLE),
            (" - [S]: Exportar ticket digital detallado a la carpeta /tickets en formato TXT.", Theme.BGR_TEXT_TITLE),
            (" - [D]: Activar / Desactivar promocion especial de 10% de descuento.", Theme.BGR_TEXT_TITLE),
            (" - [M]: Cambiar modo de visualizacion (Ticket -> Inventario -> Completo).", Theme.BGR_TEXT_TITLE),
            (" - [L]: Resurtir stock (+10 piezas a todos los productos en stock.json).", Theme.BGR_CYAN),
            (" - [R] / [T]: Reiniciar carrito, inventario y registros de seguimiento a cero.", Theme.BGR_TEXT_TITLE),
            (" - [P]: Pausar / Reanudar la deteccion y captura en tiempo real.", Theme.BGR_TEXT_TITLE),
            (" - [F]: Alternar modo de pantalla completa (Fullscreen).", Theme.BGR_TEXT_TITLE),
            (" - [X] o [Q] o [ESC]: Cerrar la aplicacion de forma segura.", Theme.BGR_TEXT_TITLE),
        ]

        ty = my + int(76 * scale)
        step_y = int(22 * scale)
        for linea, col in texto_lineas:
            if linea:
                cv2.putText(frame_completo, _limpiar_para_cv2(linea), (mx + int(30 * scale), ty),
                            cv2.FONT_HERSHEY_DUPLEX, 0.38 * scale, col, 1, cv2.LINE_AA)
            ty += step_y

        cv2.putText(frame_completo, "Presiona [H] o [ESPACIO] para volver al sistema en vivo",
                    (mx + int(140 * scale), my + mh - int(16 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42 * scale, Theme.BGR_GREEN, 1, cv2.LINE_AA)

    @staticmethod
    def render_panel_borrado(frame_completo: np.ndarray, estado, scale: float = 1.0):
        """
        Renderiza el modal centrado para borrado y cancelación manual de artículos.
        Permite al usuario/cajero elegir cualquier producto escaneado mediante clic
        o presionando la tecla numérica [1-6], sin alterar otros artículos.
        """
        h, w = frame_completo.shape[:2]
        mw, mh = int(min(w * 0.88, 760 * scale)), int(min(h * 0.90, 520 * scale))
        mx, my = (w - mw) // 2, (h - mh) // 2

        # 1. Fondo oscurecido con desenfoque suave
        overlay = frame_completo.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (4, 6, 10), -1)
        cv2.addWeighted(overlay, 0.84, frame_completo, 0.16, 0, frame_completo)

        # 2. Marco principal de la tarjeta modal
        cv2.rectangle(frame_completo, (mx, my), (mx + mw, my + mh), (20, 25, 34), -1)
        cv2.rectangle(frame_completo, (mx, my), (mx + mw, my + mh), Theme.BGR_CORAL, 2)

        # 3. Encabezado del modal
        cv2.putText(frame_completo, "PANEL DE CANCELACION Y BORRADO MANUAL",
                    (mx + int(24 * scale), my + int(36 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60 * scale, Theme.BGR_TEXT_TITLE, 2, cv2.LINE_AA)
        cv2.putText(frame_completo, "Retira productos especificos del ticket sin alterar el resto de la compra",
                    (mx + int(24 * scale), my + int(56 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38 * scale, Theme.BGR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.line(frame_completo, (mx + int(20 * scale), my + int(68 * scale)),
                 (mx + mw - int(20 * scale), my + int(68 * scale)), Theme.BGR_CARD_BORDER, 1)

        estado.hitboxes_modal = []
        start_y = my + int(82 * scale)
        row_h = int(52 * scale)

        for i, prod in enumerate(CLASES_NOMBRES, start=1):
            cant = estado.inventario.get(prod, 0)
            precio = CATALOGO_PRECIOS.get(prod, 0.0)
            ry = start_y + (i - 1) * row_h
            rx = mx + int(20 * scale)
            rw = mw - int(40 * scale)
            rh = row_h - int(8 * scale)

            # Fondo de la fila
            bg_col = (28, 34, 46) if cant > 0 else (16, 20, 28)
            border_col = Theme.BGR_CYAN if cant > 0 else (35, 42, 55)
            cv2.rectangle(frame_completo, (rx, ry), (rx + rw, ry + rh), bg_col, -1)
            cv2.rectangle(frame_completo, (rx, ry), (rx + rw, ry + rh), border_col, 1)

            # Indicador de número / atajo
            cv2.putText(frame_completo, f"[{i}]", (rx + int(14 * scale), ry + int(28 * scale)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52 * scale, Theme.BGR_GOLD, 2, cv2.LINE_AA)

            # Nombre y precio unitario
            name_col = Theme.BGR_TEXT_TITLE if cant > 0 else Theme.BGR_TEXT_MUTED
            cv2.putText(frame_completo, f"{prod.upper()} (${precio:.2f} c/u)",
                        (rx + int(55 * scale), ry + int(28 * scale)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48 * scale, name_col, 1, cv2.LINE_AA)

            # Cantidad actual en carrito
            if cant > 0:
                txt_cant = f"En ticket: {cant} pza{'s' if cant > 1 else ''}  (${cant * precio:.2f})"
                cv2.putText(frame_completo, txt_cant, (rx + int(260 * scale), ry + int(28 * scale)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.42 * scale, Theme.BGR_CYAN, 1, cv2.LINE_AA)

                # Botón [ -1 QUITAR ]
                btn_w1 = int(120 * scale)
                btn_x1 = rx + rw - btn_w1 - int(130 * scale)
                btn_y = ry + int(6 * scale)
                btn_h = rh - int(12 * scale)
                cv2.rectangle(frame_completo, (btn_x1, btn_y), (btn_x1 + btn_w1, btn_y + btn_h), (45, 20, 30), -1)
                cv2.rectangle(frame_completo, (btn_x1, btn_y), (btn_x1 + btn_w1, btn_y + btn_h), Theme.BGR_CORAL, 1)
                cv2.putText(frame_completo, "-1 QUITAR", (btn_x1 + int(12 * scale), btn_y + int(20 * scale)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.40 * scale, Theme.BGR_CORAL, 1, cv2.LINE_AA)

                estado.hitboxes_modal.append({
                    'action': 'quitar_1',
                    'prod': prod,
                    'box': (btn_x1, btn_y, btn_x1 + btn_w1, btn_y + btn_h)
                })

                # Botón [ VACIAR TODO ]
                btn_w2 = int(115 * scale)
                btn_x2 = rx + rw - btn_w2 - int(10 * scale)
                cv2.rectangle(frame_completo, (btn_x2, btn_y), (btn_x2 + btn_w2, btn_y + btn_h), (25, 22, 28), -1)
                cv2.rectangle(frame_completo, (btn_x2, btn_y), (btn_x2 + btn_w2, btn_y + btn_h), (70, 70, 80), 1)
                cv2.putText(frame_completo, "VACIAR TODO", (btn_x2 + int(10 * scale), btn_y + int(20 * scale)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.36 * scale, Theme.BGR_TEXT_MUTED, 1, cv2.LINE_AA)

                estado.hitboxes_modal.append({
                    'action': 'vaciar',
                    'prod': prod,
                    'box': (btn_x2, btn_y, btn_x2 + btn_w2, btn_y + btn_h)
                })
            else:
                cv2.putText(frame_completo, "(0 piezas en ticket)", (rx + int(260 * scale), ry + int(28 * scale)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.40 * scale, (60, 70, 85), 1, cv2.LINE_AA)

        # 4. Pie del modal con botón de cerrar
        close_btn_y = my + mh - int(45 * scale)
        close_btn_w = int(240 * scale)
        close_btn_x = (w - close_btn_w) // 2
        close_btn_h = int(32 * scale)

        cv2.rectangle(frame_completo, (close_btn_x, close_btn_y),
                      (close_btn_x + close_btn_w, close_btn_y + close_btn_h), (30, 38, 50), -1)
        cv2.rectangle(frame_completo, (close_btn_x, close_btn_y),
                      (close_btn_x + close_btn_w, close_btn_y + close_btn_h), Theme.BGR_CYAN, 1)
        cv2.putText(frame_completo, "[ESC / B] VOLVER AL ESCANEO",
                    (close_btn_x + int(16 * scale), close_btn_y + int(21 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38 * scale, Theme.BGR_TEXT_TITLE, 1, cv2.LINE_AA)

        estado.hitboxes_modal.append({
            'action': 'cerrar',
            'prod': '',
            'box': (close_btn_x, close_btn_y, close_btn_x + close_btn_w, close_btn_y + close_btn_h)
        })

    @staticmethod
    def render_modal_cobro_exitoso(frame_completo: np.ndarray, estado, scale: float = 1.0):
        """
        Renderiza el modal centrado de confirmación de venta completada y emisión de ticket.
        Muestra el folio, el monto cobrado, la cantidad de artículos y la confirmación
        del ticket guardado en /tickets.
        """
        h, w = frame_completo.shape[:2]
        mw, mh = int(min(w * 0.82, 700 * scale)), int(min(h * 0.75, 420 * scale))
        mx, my = (w - mw) // 2, (h - mh) // 2

        # 1. Overlay oscuro de fondo
        overlay = frame_completo.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (3, 8, 5), -1)
        cv2.addWeighted(overlay, 0.85, frame_completo, 0.15, 0, frame_completo)

        # 2. Marco principal con estilo de éxito (Verde Esmeralda)
        cv2.rectangle(frame_completo, (mx, my), (mx + mw, my + mh), (18, 28, 24), -1)
        cv2.rectangle(frame_completo, (mx, my), (mx + mw, my + mh), Theme.BGR_GREEN, 2)

        # 3. Encabezado de éxito
        header_h = int(55 * scale)
        cv2.rectangle(frame_completo, (mx, my), (mx + mw, my + header_h), (12, 38, 24), -1)
        cv2.rectangle(frame_completo, (mx, my), (mx + mw, my + header_h), Theme.BGR_GREEN, 1)

        cv2.putText(frame_completo, "[ COBRO EXITOSO - TICKET EMITIDO ]",
                    (mx + int(30 * scale), my + int(36 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65 * scale, Theme.BGR_GREEN, 2, cv2.LINE_AA)

        # 4. Datos del cobro
        datos = estado.datos_ultimo_cobro or {}
        folio = datos.get('folio', 1001)
        total = datos.get('total', 0.0)
        subtotal = datos.get('subtotal', 0.0)
        descuento = datos.get('descuento_pct', 0.0)
        articulos = datos.get('articulos', 0)
        archivo = datos.get('archivo', '')
        hora = datos.get('hora', '')

        # Tarjeta destacada con el Monto Total
        card_w = mw - int(60 * scale)
        card_x = mx + int(30 * scale)
        card_y = my + int(72 * scale)
        card_h = int(95 * scale)
        cv2.rectangle(frame_completo, (card_x, card_y), (card_x + card_w, card_y + card_h), (24, 38, 30), -1)
        cv2.rectangle(frame_completo, (card_x, card_y), (card_x + card_w, card_y + card_h), Theme.BGR_GREEN, 1)

        cv2.putText(frame_completo, f"TOTAL COBRADO: ${total:.2f} MXN",
                    (card_x + int(24 * scale), card_y + int(42 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.80 * scale, Theme.BGR_TEXT_TITLE, 2, cv2.LINE_AA)

        detalle_str = f"Subtotal: ${subtotal:.2f}  |  IVA (16%): ${(subtotal * 0.16):.2f}"
        if descuento > 0:
            detalle_str += f"  |  Desc: {descuento:.0f}%"
        cv2.putText(frame_completo, detalle_str,
                    (card_x + int(25 * scale), card_y + int(74 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44 * scale, Theme.BGR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Detalles del recibo
        ty = my + int(200 * scale)
        step = int(30 * scale)

        cv2.putText(frame_completo, f"FOLIO FISCAL: #{folio:04d}   |   HORA: {hora}   |   PIEZAS: {articulos}",
                    (mx + int(35 * scale), ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48 * scale, Theme.BGR_CYAN, 1, cv2.LINE_AA)
        ty += step

        nom_archivo = os.path.basename(archivo) if archivo else "tickets/ticket.txt"
        cv2.putText(frame_completo, f"COMPROBANTE: Guardado en /tickets/{nom_archivo}",
                    (mx + int(35 * scale), ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44 * scale, Theme.BGR_TEXT_TITLE, 1, cv2.LINE_AA)
        ty += step

        cv2.putText(frame_completo, "[!] Retire los articulos para iniciar la siguiente venta",
                    (mx + int(35 * scale), ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46 * scale, Theme.BGR_GOLD, 1, cv2.LINE_AA)

        # 5. Botón / atajo para continuar
        btn_w = int(320 * scale)
        btn_x = (w - btn_w) // 2
        btn_y = my + mh - int(52 * scale)
        btn_h = int(36 * scale)

        cv2.rectangle(frame_completo, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (20, 50, 32), -1)
        cv2.rectangle(frame_completo, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), Theme.BGR_GREEN, 1)
        cv2.putText(frame_completo, "PULSA [C], [ESPACIO] O CLIC",
                    (btn_x + int(28 * scale), btn_y + int(24 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45 * scale, Theme.BGR_TEXT_TITLE, 1, cv2.LINE_AA)

