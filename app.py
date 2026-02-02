import sys
import pandas as pd
import numpy as np
from scipy.interpolate import RBFInterpolator
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QPushButton, QCheckBox, 
                               QGroupBox, QFileDialog, QSplitter, QDoubleSpinBox, QFormLayout, QMessageBox, QProgressDialog, QComboBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QVector3D, QFont

import pyqtgraph.opengl as gl
import pyqtgraph as pg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PySide6.QtWidgets import QStackedWidget, QDialog, QSizePolicy

# ==========================================
# 0. Localization
# ==========================================
TRANSLATIONS = {
    'zh': {
        'WINDOW_TITLE': "时空数据可视化与剖面分析工具 v1.0",
        'CONTROL_PANEL': "控制面板",
        'BTN_LOAD': "加载数据 (CSV)",
        'BTN_EXPORT': "导出剖面图 (SVG/PNG)",
        'CHK_PROJECTION': "地球投影模式",
        'CHK_NORMALIZE': "数据归一化 (0-1)",
        'LBL_VAR_SELECT': "选择变量:",
        'BTN_PLAY': "播放",
        'BTN_PAUSE': "暂停",
        'LBL_TIME': "时间",
        'LBL_INFO_WAIT': "请加载数据...",
        'LBL_INFO_READY': "就绪: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "无变量被选中",
        'PLOT_TITLE': "切面剖面图",
        'PLOT_X': "归一化距离 (0-1)",
        'PLOT_Y_NORM': "归一化数值 (0-1)",
        'PLOT_Y_RAW': "原始数值",
        'GRP_SLICE': "切面设置",
        'LBL_START_LAT': "起点 纬度:",
        'LBL_START_LON': "起点 经度:",
        'LBL_END_LAT': "终点 纬度:",
        'LBL_END_LON': "终点 经度:",
        'MSG_SPARSE_TITLE': "稀疏数据检测",
        'MSG_SPARSE_BODY': "检测到非规则网格数据，是否进行插值处理？\n(这可能需要几秒钟)",
        'DIALOG_INTERP': "正在插值...",
        'BTN_CANCEL': "取消"
    },
    'en': {
        'WINDOW_TITLE': "Spatio-Temporal Data Visualization & Profile Analysis Tool v1.0",
        'CONTROL_PANEL': "Control Panel",
        'BTN_LOAD': "Load Data (CSV)",
        'BTN_EXPORT': "Export Profile (SVG/PNG)",
        'CHK_PROJECTION': "Earth Projection Mode",
        'CHK_NORMALIZE': "Normalize Data (0-1)",
        'LBL_VAR_SELECT': "Select Variables:",
        'BTN_PLAY': "Play",
        'BTN_PAUSE': "Pause",
        'LBL_TIME': "Time",
        'LBL_INFO_WAIT': "Please load data...",
        'LBL_INFO_READY': "Ready: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "No variables selected",
        'PLOT_TITLE': "Slice Profile Plot",
        'PLOT_X': "Normalized Distance (0-1)",
        'PLOT_Y_NORM': "Normalized Value (0-1)",
        'PLOT_Y_RAW': "Raw Value",
        'GRP_SLICE': "Slice Settings",
        'LBL_START_LAT': "Start Lat:",
        'LBL_START_LON': "Start Lon:",
        'LBL_END_LAT': "End Lat:",
        'LBL_END_LON': "End Lon:",
        'MSG_SPARSE_TITLE': "Sparse Data Detected",
        'MSG_SPARSE_BODY': "Irregular grid detected. Perform interpolation?\n(This may take a few seconds)",
        'DIALOG_INTERP': "Interpolating...",
        'BTN_CANCEL': "Cancel"
    },
    'ja': {
        'WINDOW_TITLE': "時空間データ可視化・断面分析ツール v1.0",
        'CONTROL_PANEL': "コントロールパネル",
        'BTN_LOAD': "データ読み込み (CSV)",
        'CHK_PROJECTION': "地球投影モード",
        'CHK_NORMALIZE': "データ正規化 (0-1)",
        'LBL_VAR_SELECT': "変数を選択:",
        'BTN_PLAY': "再生",
        'BTN_PAUSE': "一時停止",
        'LBL_TIME': "時間",
        'LBL_INFO_WAIT': "データを読み込んでください...",
        'LBL_INFO_READY': "準備完了: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "変数が選択されていません",
        'PLOT_TITLE': "断面プロファイル図",
        'PLOT_X': "正規化距離 (0-1)",
        'PLOT_Y_NORM': "正規化値 (0-1)",
        'PLOT_Y_RAW': "生の値",
        'GRP_SLICE': "断面設定",
        'LBL_START_LAT': "開始 緯度:",
        'LBL_START_LON': "開始 経度:",
        'LBL_END_LAT': "終了 緯度:",
        'LBL_END_LON': "終了 経度:",
        'MSG_SPARSE_TITLE': "スパースデータ検出",
        'MSG_SPARSE_BODY': "不規則なグリッドが検出されました。補間処理を行いますか？\n（数秒かかる場合があります）",
        'DIALOG_INTERP': "補間中...",
        'BTN_CANCEL': "キャンセル"
    },
    'fr': {
        'WINDOW_TITLE': "Outil de Visualisation et d'Analyse de Profil Spatio-Temporel v1.0",
        'CONTROL_PANEL': "Panneau de configuration",
        'BTN_LOAD': "Charger données (CSV)",
        'CHK_PROJECTION': "Mode Projection Terrestre",
        'CHK_NORMALIZE': "Normaliser (0-1)",
        'LBL_VAR_SELECT': "Sélectionner variables:",
        'BTN_PLAY': "Lire",
        'BTN_PAUSE': "Pause",
        'LBL_TIME': "Temps",
        'LBL_INFO_WAIT': "Veuillez charger les données...",
        'LBL_INFO_READY': "Prêt: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "Aucune variable sélectionnée",
        'PLOT_TITLE': "Profil de Coupe",
        'PLOT_X': "Distance Normalisée (0-1)",
        'PLOT_Y_NORM': "Valeur Normalisée (0-1)",
        'PLOT_Y_RAW': "Valeur Brute",
        'GRP_SLICE': "Paramètres de Coupe",
        'LBL_START_LAT': "Lat Début:",
        'LBL_START_LON': "Lon Début:",
        'LBL_END_LAT': "Lat Fin:",
        'LBL_END_LON': "Lon Fin:",
        'MSG_SPARSE_TITLE': "Données Éparses Détectées",
        'MSG_SPARSE_BODY': "Grille irrégulière détectée. Effectuer l'interpolation ?\n(Cela peut prendre quelques secondes)",
        'DIALOG_INTERP': "Interpolation en cours...",
        'BTN_CANCEL': "Annuler"
    },
    'ru': {
        'WINDOW_TITLE': "Инструмент пространственно-временной визуализации и анализа профилей v1.0",
        'CONTROL_PANEL': "Панель управления",
        'BTN_LOAD': "Загрузить данные (CSV)",
        'CHK_PROJECTION': "Проекция Земли",
        'CHK_NORMALIZE': "Нормализация (0-1)",
        'LBL_VAR_SELECT': "Выбрать переменные:",
        'BTN_PLAY': "Воспр.",
        'BTN_PAUSE': "Пауза",
        'LBL_TIME': "Время",
        'LBL_INFO_WAIT': "Загрузите данные...",
        'LBL_INFO_READY': "Готово: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "Переменные не выбраны",
        'PLOT_TITLE': "График профиля среза",
        'PLOT_X': "Норм. расстояние (0-1)",
        'PLOT_Y_NORM': "Норм. значение (0-1)",
        'PLOT_Y_RAW': "Исходное значение",
        'GRP_SLICE': "Настройки среза",
        'LBL_START_LAT': "Нач. Широта:",
        'LBL_START_LON': "Нач. Долгота:",
        'LBL_END_LAT': "Кон. Широта:",
        'LBL_END_LON': "Кон. Долгота:",
        'MSG_SPARSE_TITLE': "Обнаружены разреженные данные",
        'MSG_SPARSE_BODY': "Обнаружена нерегулярная сетка. Выполнить интерполяцию?\n(Это может занять несколько секунд)",
        'DIALOG_INTERP': "Интерполяция...",
        'BTN_CANCEL': "Отмена"
    },
    'de': {
        'WINDOW_TITLE': "Raum-Zeit-Datenvisualisierung & Profilanalyse-Tool v1.0",
        'CONTROL_PANEL': "Bedienfeld",
        'BTN_LOAD': "Daten laden (CSV)",
        'CHK_PROJECTION': "Erdprojektionsmodus",
        'CHK_NORMALIZE': "Daten normalisieren (0-1)",
        'LBL_VAR_SELECT': "Variablen auswählen:",
        'BTN_PLAY': "Abspielen",
        'BTN_PAUSE': "Pause",
        'LBL_TIME': "Zeit",
        'LBL_INFO_WAIT': "Bitte Daten laden...",
        'LBL_INFO_READY': "Bereit: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "Keine Variablen ausgewählt",
        'PLOT_TITLE': "Schnittprofil-Diagramm",
        'PLOT_X': "Norm. Abstand (0-1)",
        'PLOT_Y_NORM': "Norm. Wert (0-1)",
        'PLOT_Y_RAW': "Rohwert",
        'GRP_SLICE': "Schnitteinstellungen",
        'LBL_START_LAT': "Start Lat:",
        'LBL_START_LON': "Start Lon:",
        'LBL_END_LAT': "Ende Lat:",
        'LBL_END_LON': "Ende Lon:",
        'MSG_SPARSE_TITLE': "Dünne Daten erkannt",
        'MSG_SPARSE_BODY': "Unregelmäßiges Raster erkannt. Interpolation durchführen?\n(Dies kann einige Sekunden dauern)",
        'DIALOG_INTERP': "Interpoliere...",
        'BTN_CANCEL': "Abbrechen"
    },
    'it': {
        'WINDOW_TITLE': "Strumento di Visualizzazione e Analisi Spazio-Temporale v1.0",
        'CONTROL_PANEL': "Pannello di Controllo",
        'BTN_LOAD': "Carica Dati (CSV)",
        'CHK_PROJECTION': "Modalità Proiezione Terra",
        'CHK_NORMALIZE': "Normalizza Dati (0-1)",
        'LBL_VAR_SELECT': "Seleziona Variabili:",
        'BTN_PLAY': "Riproduci",
        'BTN_PAUSE': "Pausa",
        'LBL_TIME': "Tempo",
        'LBL_INFO_WAIT': "Carica i dati...",
        'LBL_INFO_READY': "Pronto: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "Nessuna variabile selezionata",
        'PLOT_TITLE': "Profilo della Sezione",
        'PLOT_X': "Distanza Norm. (0-1)",
        'PLOT_Y_NORM': "Valore Norm. (0-1)",
        'PLOT_Y_RAW': "Valore Grezzo",
        'GRP_SLICE': "Impostazioni Sezione",
        'LBL_START_LAT': "Lat Inizio:",
        'LBL_START_LON': "Lon Inizio:",
        'LBL_END_LAT': "Lat Fine:",
        'LBL_END_LON': "Lon Fine:",
        'MSG_SPARSE_TITLE': "Dati Sparsi Rilevati",
        'MSG_SPARSE_BODY': "Griglia irregolare rilevata. Eseguire interpolazione?\n(Potrebbe richiedere alcuni secondi)",
        'DIALOG_INTERP': "Interpolazione...",
        'BTN_CANCEL': "Annulla"
    },
    'es': {
        'WINDOW_TITLE': "Herramienta de Visualización y Análisis Espacio-Temporal v1.0",
        'CONTROL_PANEL': "Panel de Control",
        'BTN_LOAD': "Cargar Datos (CSV)",
        'CHK_PROJECTION': "Modo Proyección Terrestre",
        'CHK_NORMALIZE': "Normalizar Datos (0-1)",
        'LBL_VAR_SELECT': "Seleccionar Variables:",
        'BTN_PLAY': "Reproducir",
        'BTN_PAUSE': "Pausa",
        'LBL_TIME': "Tiempo",
        'LBL_INFO_WAIT': "Por favor cargue datos...",
        'LBL_INFO_READY': "Listo: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "Sin variables seleccionadas",
        'PLOT_TITLE': "Perfil de Corte",
        'PLOT_X': "Distancia Norm. (0-1)",
        'PLOT_Y_NORM': "Valor Norm. (0-1)",
        'PLOT_Y_RAW': "Valor Bruto",
        'GRP_SLICE': "Ajustes de Corte",
        'LBL_START_LAT': "Lat Inicio:",
        'LBL_START_LON': "Lon Inicio:",
        'LBL_END_LAT': "Lat Fin:",
        'LBL_END_LON': "Lon Fin:",
        'MSG_SPARSE_TITLE': "Datos Dispersos Detectados",
        'MSG_SPARSE_BODY': "Rejilla irregular detectada. ¿Realizar interpolación?\n(Puede tardar unos segundos)",
        'DIALOG_INTERP': "Interpolando...",
        'BTN_CANCEL': "Cancelar"
    },
    'pt': {
        'WINDOW_TITLE': "Ferramenta de Visualização e Análise Espaço-Temporal v1.0",
        'CONTROL_PANEL': "Painel de Controle",
        'BTN_LOAD': "Carregar Dados (CSV)",
        'CHK_PROJECTION': "Modo Projeção Terra",
        'CHK_NORMALIZE': "Normalizar Dados (0-1)",
        'LBL_VAR_SELECT': "Selecionar Variáveis:",
        'BTN_PLAY': "Reproduzir",
        'BTN_PAUSE': "Pausa",
        'LBL_TIME': "Tempo",
        'LBL_INFO_WAIT': "Por favor carregue dados...",
        'LBL_INFO_READY': "Pronto: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "Nenhuma variável selecionada",
        'PLOT_TITLE': "Perfil de Corte",
        'PLOT_X': "Distância Norm. (0-1)",
        'PLOT_Y_NORM': "Valor Norm. (0-1)",
        'PLOT_Y_RAW': "Valor Bruto",
        'GRP_SLICE': "Configurações de Corte",
        'LBL_START_LAT': "Lat Início:",
        'LBL_START_LON': "Lon Início:",
        'LBL_END_LAT': "Lat Fim:",
        'LBL_END_LON': "Lon Fim:",
        'MSG_SPARSE_TITLE': "Dados Esparsos Detectados",
        'MSG_SPARSE_BODY': "Grade irregular detectada. Realizar interpolação?\n(Isso pode levar alguns segundos)",
        'DIALOG_INTERP': "Interpolando...",
        'BTN_CANCEL': "Cancelar"
    },
    'ko': {
        'WINDOW_TITLE': "시공간 데이터 시각화 및 프로파일 분석 도구 v1.0",
        'CONTROL_PANEL': "제어 패널",
        'BTN_LOAD': "데이터 로드 (CSV)",
        'CHK_PROJECTION': "지구 투영 모드",
        'CHK_NORMALIZE': "데이터 정규화 (0-1)",
        'LBL_VAR_SELECT': "변수 선택:",
        'BTN_PLAY': "재생",
        'BTN_PAUSE': "일시 중지",
        'LBL_TIME': "시간",
        'LBL_INFO_WAIT': "데이터를 로드하십시오...",
        'LBL_INFO_READY': "준비: {t}T x {lat}Lat x {lon}Lon",
        'LBL_INFO_NO_VAR': "선택된 변수 없음",
        'PLOT_TITLE': "단면 프로파일 플롯",
        'PLOT_X': "정규화 거리 (0-1)",
        'PLOT_Y_NORM': "정규화 값 (0-1)",
        'PLOT_Y_RAW': "원시 값",
        'GRP_SLICE': "단면 설정",
        'LBL_START_LAT': "시작 위도:",
        'LBL_START_LON': "시작 경도:",
        'LBL_END_LAT': "종료 위도:",
        'LBL_END_LON': "종료 경도:",
        'MSG_SPARSE_TITLE': "희소 데이터 감지됨",
        'MSG_SPARSE_BODY': "불규칙 그리드가 감지되었습니다. 보간을 수행하시겠습니까?\n(몇 초 정도 걸릴 수 있습니다)",
        'DIALOG_INTERP': "보간 중...",
        'BTN_CANCEL': "취소"
    }
}

# ==========================================
# 1. Data Processing Module (Interpolation)
# ==========================================
class DataProcessor:
    @staticmethod
    def process_sparse_data(df, progress_callback=None):
        """
        Interpolate sparse dataframe to dense grid.
        df: DataFrame with 'Latitude', 'Longitude', 'Age' (or 'Time') and variables.
        """
        # 1. Identify Columns
        coords_cols = ['Latitude', 'Longitude', 'Time', 'Age']
        # Map Age to Time if needed
        if 'Age' in df.columns and 'Time' not in df.columns:
            df['Time'] = df['Age']
            
        var_cols = [c for c in df.columns if c not in coords_cols and c != 'SiteID']
        
        # 2. Define Target Grid
        grid_lat_res = 50
        grid_lon_res = 100
        grid_time_res = 20
        
        lats = np.linspace(-90, 90, grid_lat_res)
        lons = np.linspace(-180, 180, grid_lon_res)
        times = np.linspace(df['Time'].min(), df['Time'].max(), grid_time_res)
        
        # Create meshgrid (3D)
        grid_lat_mesh, grid_lon_mesh, grid_time_mesh = np.meshgrid(lats, lons, times, indexing='ij')
        
        # Flatten target grid
        target_points = np.column_stack([
            grid_lat_mesh.ravel(),
            grid_lon_mesh.ravel(),
            grid_time_mesh.ravel()
        ])
        
        # 3. Normalization Helpers
        def normalize(v, v_min, v_max):
            return (v - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(v)
        
        min_lat, max_lat = -90, 90
        min_lon, max_lon = -180, 180
        min_time, max_time = df['Time'].min(), df['Time'].max()
        
        # 4. Interpolation
        dense_data_dict = {
            'Latitude': grid_lat_mesh.ravel(),
            'Longitude': grid_lon_mesh.ravel(),
            'Time': grid_time_mesh.ravel()
        }
        
        # Normalize Target once
        tgt_lat_n = normalize(target_points[:,0], min_lat, max_lat)
        tgt_lon_n = normalize(target_points[:,1], min_lon, max_lon)
        tgt_time_n = normalize(target_points[:,2], min_time, max_time)
        tgt_points_n = np.column_stack([tgt_lat_n, tgt_lon_n, tgt_time_n])
        
        total_vars = len(var_cols)
        for i, var in enumerate(var_cols):
            if progress_callback:
                progress_callback(int((i / total_vars) * 100), f"Interpolating {var}...")
                
            # Filter NaN
            sub_df = df.dropna(subset=[var, 'Latitude', 'Longitude', 'Time'])
            
            if len(sub_df) < 10:
                dense_data_dict[var] = np.nan
                continue
                
            src_lat = sub_df['Latitude'].values
            src_lon = sub_df['Longitude'].values
            src_time = sub_df['Time'].values
            src_vals = sub_df[var].values
            
            # Normalize Source
            src_lat_n = normalize(src_lat, min_lat, max_lat)
            src_lon_n = normalize(src_lon, min_lon, max_lon)
            src_time_n = normalize(src_time, min_time, max_time)
            
            src_points_n = np.column_stack([src_lat_n, src_lon_n, src_time_n])
            
            try:
                # RBF Interpolation
                rbf = RBFInterpolator(src_points_n, src_vals, kernel='thin_plate_spline', smoothing=0.1)
                interpolated_vals = rbf(tgt_points_n)
                dense_data_dict[var] = interpolated_vals
            except Exception as e:
                print(f"Interpolation error for {var}: {e}")
                dense_data_dict[var] = np.nan
                
        dense_df = pd.DataFrame(dense_data_dict)
        dense_df.sort_values(by=['Time', 'Latitude', 'Longitude'], inplace=True)
        return dense_df

# ==========================================
# 2. Interactive 3D View Widget
# ==========================================
class InteractiveGLViewWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dragging_idx = -1  # -1: none, 0: Start, 1: End
        self.parent_viz = None
        self.z_plane = 12

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        # Adjust zoom factor for better responsiveness
        zoom_factor = 0.9 if delta > 0 else 1.1
        
        new_dist = self.opts['distance'] * zoom_factor
        
        # Limit distance to prevent getting stuck or going too far
        new_dist = max(1.0, min(new_dist, 500.0))
        
        self.opts['distance'] = new_dist
        self.update()

    def get_ray(self, pos):
        """Maps 2D screen position to a 3D world-space ray."""
        x = pos.x()
        y = pos.y()
        w = self.width()
        h = self.height()
        
        # NDC
        x_ndc = (2.0 * x / w) - 1.0
        y_ndc = 1.0 - (2.0 * y / h) 
        
        view_matrix = self.viewMatrix()
        proj_matrix = self.projectionMatrix()
        
        mvp = proj_matrix * view_matrix
        inv_mvp, success = mvp.inverted()
        
        if not success: return None, None
            
        p_near = inv_mvp * QVector3D(x_ndc, y_ndc, -1.0)
        p_far = inv_mvp * QVector3D(x_ndc, y_ndc, 1.0)
        
        direction = (p_far - p_near).normalized()
        return p_near, direction

    def intersect_z_plane(self, start, direction):
        if start is None or direction is None: return None
            
        # Check projection mode
        if self.parent_viz and self.parent_viz.chk_projection.isChecked():
            # Spherical Intersection |P| = R
            R = 20.5
            a = 1
            b = 2 * QVector3D.dotProduct(start, direction)
            c = QVector3D.dotProduct(start, start) - R*R
            
            discriminant = b*b - 4*a*c
            if discriminant < 0: return None
            
            t1 = (-b - np.sqrt(discriminant)) / (2*a)
            t2 = (-b + np.sqrt(discriminant)) / (2*a)
            
            t = None
            if t1 > 0: t = t1
            elif t2 > 0: t = t2
            
            if t is None: return None
            hit = start + direction * t
            return hit
        else:
            # Planar Intersection Z = z_plane
            if abs(direction.z()) < 1e-6: return None
            t = (self.z_plane - start.z()) / direction.z()
            hit = start + direction * t
            return hit

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            start, direction = self.get_ray(ev.position())
            hit = self.intersect_z_plane(start, direction)
            
            if hit and self.parent_viz:
                lat1 = self.parent_viz.spin_lat1.value()
                lon1 = self.parent_viz.spin_lon1.value()
                lat2 = self.parent_viz.spin_lat2.value()
                lon2 = self.parent_viz.spin_lon2.value()
                
                p1_x, p1_y, p1_z = self.parent_viz.latlon_to_local(lat1, lon1)
                p2_x, p2_y, p2_z = self.parent_viz.latlon_to_local(lat2, lon2)
                
                # Check proximity
                if self.parent_viz.chk_projection.isChecked():
                    p1 = np.array([p1_x, p1_y, p1_z])
                    p1 = p1 / np.linalg.norm(p1) * 20.5
                    p2 = np.array([p2_x, p2_y, p2_z])
                    p2 = p2 / np.linalg.norm(p2) * 20.5
                    hit_arr = np.array([hit.x(), hit.y(), hit.z()])
                    dist1 = np.linalg.norm(hit_arr - p1)
                    dist2 = np.linalg.norm(hit_arr - p2)
                else:
                    dist1 = np.hypot(hit.x() - p1_x, hit.y() - p1_y)
                    dist2 = np.hypot(hit.x() - p2_x, hit.y() - p2_y)
                
                threshold = 2.0
                if dist1 < threshold:
                    self.dragging_idx = 0
                    return
                elif dist2 < threshold:
                    self.dragging_idx = 1
                    return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.dragging_idx != -1 and self.parent_viz:
            start, direction = self.get_ray(ev.position())
            hit = self.intersect_z_plane(start, direction)
            if hit:
                lat, lon = self.parent_viz.local_to_latlon(hit.x(), hit.y(), hit.z())
                if self.dragging_idx == 0:
                    self.parent_viz.spin_lat1.setValue(lat)
                    self.parent_viz.spin_lon1.setValue(lon)
                else:
                    self.parent_viz.spin_lat2.setValue(lat)
                    self.parent_viz.spin_lon2.setValue(lon)
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.dragging_idx != -1:
            self.dragging_idx = -1
            return
        super().mouseReleaseEvent(ev)

# ==========================================
# 3. Main Application Window
# ==========================================
class SpatioTemporalViz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400, 900)
        
        # Localization
        self.current_lang = 'zh' # Default
        self.tr = TRANSLATIONS[self.current_lang]
        
        # Data storage
        self.df = None
        self.grid_data = {}
        self.unique_times = []
        self.unique_lats = []
        self.unique_lons = []
        self.current_time_idx = 0
        self.active_variables = set() 
        self.is_playing = False
        self.var_order = []
        self.var_colors = {}
        
        # Rendering objects
        self.surfaces = {}
        self.surface_labels = {}
        self.axis_labels = []
        
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_step)
        
    def init_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)
        
        # --- LEFT PANEL: 3D View ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.view = InteractiveGLViewWidget()
        self.view.parent_viz = self
        self.view.setCameraPosition(distance=80, elevation=30, azimuth=45)
        self.view.opts['distance'] = 80
        left_layout.addWidget(self.view, stretch=1)
        
        # Grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=20, y=20, z=0)
        self.grid.setSpacing(x=1, y=1, z=1)
        self.view.addItem(self.grid)
        
        # Slice Indicators
        self.slice_line = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,0]]), color=(1,1,1,1), width=2, antialias=True)
        self.view.addItem(self.slice_line)
        self.slice_points = gl.GLScatterPlotItem(pos=np.array([[0,0,0], [0,0,0]]), color=(1,1,1,1), size=10, pxMode=True)
        self.view.addItem(self.slice_points)
        
        # Earth Sphere
        md = gl.MeshData.sphere(rows=40, cols=80)
        self.earth_sphere = gl.GLMeshItem(
            meshdata=md, smooth=True, color=(0.1, 0.1, 0.3, 0.3), 
            shader='shaded', glOptions='translucent', drawEdges=True, edgeColor=(0.3, 0.3, 0.5, 0.2)
        )
        self.earth_sphere.scale(19.9, 19.9, 19.9)
        self.earth_sphere.setVisible(False)
        self.view.addItem(self.earth_sphere)
        
        # Controls Group
        self.controls_group = QGroupBox()
        controls_layout = QHBoxLayout(self.controls_group)
        left_layout.addWidget(self.controls_group)
        
        # File Loading
        f_layout = QVBoxLayout()
        
        # Language Selector
        self.combo_lang = QComboBox()
        self.combo_lang.addItems([
            "中文 (Chinese)", "English", "日本語 (Japanese)", "Français (French)", 
            "Русский (Russian)", "Deutsch (German)", "Italiano (Italian)", 
            "Español (Spanish)", "Português (Portuguese)", "한국어 (Korean)"
        ])
        # Map index to code
        self.lang_codes = ['zh', 'en', 'ja', 'fr', 'ru', 'de', 'it', 'es', 'pt', 'ko']
        self.combo_lang.currentIndexChanged.connect(self.on_language_changed)
        f_layout.addWidget(self.combo_lang)
        
        self.btn_load = QPushButton()
        self.btn_load.clicked.connect(self.open_file_dialog)
        f_layout.addWidget(self.btn_load)
        
        self.chk_projection = QCheckBox()
        self.chk_projection.stateChanged.connect(self.on_projection_changed)
        f_layout.addWidget(self.chk_projection)
        
        # Normalization Mode
        self.chk_normalize = QCheckBox()
        self.chk_normalize.stateChanged.connect(self.on_normalization_changed)
        f_layout.addWidget(self.chk_normalize)
        
        f_layout.addStretch()
        controls_layout.addLayout(f_layout)
        
        # Variable Selection
        self.v_layout = QVBoxLayout()
        self.lbl_var_select = QLabel()
        self.v_layout.addWidget(self.lbl_var_select)
        self.check_boxes = {}
        controls_layout.addLayout(self.v_layout)
        
        # Play Controls
        p_layout = QVBoxLayout()
        self.btn_play = QPushButton()
        self.btn_play.clicked.connect(self.toggle_play)
        p_layout.addWidget(self.btn_play)
        controls_layout.addLayout(p_layout)
        
        # Time Slider
        t_layout = QVBoxLayout()
        self.lbl_time = QLabel()
        t_layout.addWidget(self.lbl_time)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider_change)
        t_layout.addWidget(self.slider)
        controls_layout.addLayout(t_layout, stretch=1)
        
        self.lbl_info = QLabel()
        controls_layout.addWidget(self.lbl_info)
        
        # --- RIGHT PANEL: Slice Analysis ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.plot_widget = pg.PlotWidget()
        # Academic Style: White background, Black axes
        self.plot_widget.setBackground('w')
        self.plot_widget.getAxis('bottom').setPen('k')
        self.plot_widget.getAxis('left').setPen('k')
        self.plot_widget.getAxis('bottom').setTextPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')
        
        self.plot_widget.addLegend()
        right_layout.addWidget(self.plot_widget, stretch=1)
        
        self.slice_group = QGroupBox()
        slice_layout = QFormLayout(self.slice_group)
        
        self.spin_lat1 = QDoubleSpinBox()
        self.spin_lon1 = QDoubleSpinBox()
        self.spin_lat2 = QDoubleSpinBox()
        self.spin_lon2 = QDoubleSpinBox()
        
        for spin in [self.spin_lat1, self.spin_lon1, self.spin_lat2, self.spin_lon2]:
            spin.setDecimals(2)
            spin.setRange(-999, 999)
            spin.valueChanged.connect(self.update_slice_view)
        
        self.lbl_start_lat = QLabel()
        self.lbl_start_lon = QLabel()
        self.lbl_end_lat = QLabel()
        self.lbl_end_lon = QLabel()
        
        slice_layout.addRow(self.lbl_start_lat, self.spin_lat1)
        slice_layout.addRow(self.lbl_start_lon, self.spin_lon1)
        slice_layout.addRow(self.lbl_end_lat, self.spin_lat2)
        slice_layout.addRow(self.lbl_end_lon, self.spin_lon2)
        
        self.btn_export = QPushButton()
        self.btn_export.clicked.connect(self.export_slice_plot)
        slice_layout.addRow(self.btn_export)
        
        # New Button for Multi-Site Analysis
        self.btn_multi_analysis = QPushButton("时空演化分析 (Spatio-Temporal)")
        self.btn_multi_analysis.clicked.connect(self.open_multi_analysis)
        slice_layout.addRow(self.btn_multi_analysis)
        
        right_layout.addWidget(self.slice_group)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([900, 400])
        
        self.update_ui_text()

    def on_language_changed(self, index):
        code = self.lang_codes[index]
        self.current_lang = code
        self.tr = TRANSLATIONS[code]
        self.update_ui_text()

    def update_ui_text(self):
        tr = self.tr
        self.setWindowTitle(tr['WINDOW_TITLE'])
        self.controls_group.setTitle(tr['CONTROL_PANEL'])
        self.btn_load.setText(tr['BTN_LOAD'])
        self.chk_projection.setText(tr['CHK_PROJECTION'])
        self.chk_normalize.setText(tr['CHK_NORMALIZE'])
        self.lbl_var_select.setText(tr['LBL_VAR_SELECT'])
        self.btn_play.setText(tr['BTN_PAUSE'] if self.is_playing else tr['BTN_PLAY'])
        
        # Time Label update
        current_time_val = 0
        if self.unique_times is not None and len(self.unique_times) > self.current_time_idx:
            current_time_val = self.unique_times[self.current_time_idx]
        self.lbl_time.setText(f"{tr['LBL_TIME']}: {current_time_val:.2f}")
        
        # Info Label
        if self.df is None:
            self.lbl_info.setText(tr['LBL_INFO_WAIT'])
        else:
             # Refresh info text logic if needed, or just keep current status
             pass 

        self.plot_widget.setTitle(tr['PLOT_TITLE'])
        self.plot_widget.setLabel('bottom', tr['PLOT_X'])
        is_norm = self.chk_normalize.isChecked()
        self.plot_widget.setLabel('left', tr['PLOT_Y_NORM'] if is_norm else tr['PLOT_Y_RAW'])
        
        self.slice_group.setTitle(tr['GRP_SLICE'])
        self.lbl_start_lat.setText(tr['LBL_START_LAT'])
        self.lbl_start_lon.setText(tr['LBL_START_LON'])
        self.lbl_end_lat.setText(tr['LBL_END_LAT'])
        self.lbl_end_lon.setText(tr['LBL_END_LON'])
        self.btn_export.setText(tr.get('BTN_EXPORT', "Export"))
        
        # Update Plot Labels if data loaded
        if self.df is not None:
             self.update_plot()
             self.update_slice_view()

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, self.tr['BTN_LOAD'], "", "CSV Files (*.csv)")
        if filename:
            self.load_data(filename)

    def normalize_data(self, df):
        # Rename standard columns
        mapping = {
            'lat': 'Latitude', 'latitude': 'Latitude', 'y': 'Latitude',
            'lon': 'Longitude', 'longitude': 'Longitude', 'long': 'Longitude', 'x': 'Longitude',
            'time': 'Time', 'date': 'Time', 't': 'Time', 'age': 'Time', 'Age': 'Time'
        }
        
        df = df.rename(columns=lambda x: mapping.get(x.lower(), mapping.get(x, x)))
        return df

    def load_data(self, filepath):
        try:
            print(f"Loading {filepath}...")
            df_raw = pd.read_csv(filepath)
            df_norm = self.normalize_data(df_raw)
            
            required = ['Latitude', 'Longitude', 'Time']
            if not all(col in df_norm.columns for col in required):
                 # Try heuristics
                 raise ValueError("Could not find Latitude, Longitude, or Time columns.")

            # Check if sparse / needs interpolation
            # Heuristic: irregular unique counts or explicit SiteID
            unique_lats = df_norm['Latitude'].unique()
            unique_lons = df_norm['Longitude'].unique()
            unique_times = df_norm['Time'].unique()
            
            expected_rows = len(unique_lats) * len(unique_lons) * len(unique_times)
            is_sparse = len(df_norm) != expected_rows
            
            if 'SiteID' in df_norm.columns or is_sparse:
                reply = QMessageBox.question(self, self.tr['MSG_SPARSE_TITLE'], 
                                           self.tr['MSG_SPARSE_BODY'],
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    # Show progress
                    progress = QProgressDialog(self.tr['DIALOG_INTERP'], self.tr['BTN_CANCEL'], 0, 100, self)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.show()
                    
                    def cb(val, msg):
                        progress.setValue(val)
                        progress.setLabelText(msg)
                        QApplication.processEvents()
                        
                    self.df = DataProcessor.process_sparse_data(df_norm, progress_callback=cb)
                    progress.setValue(100)
                else:
                    self.df = df_norm
            else:
                self.df = df_norm
            
            # --- Common Load Logic ---
            self.setup_visualization()
            
        except Exception as e:
            self.lbl_info.setText(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def setup_visualization(self):
        # Reset
        self.grid_data = {}
        self.active_variables = set()
        self.var_colors = {}
        
        while self.v_layout.count() > 1:
            item = self.v_layout.takeAt(1)
            if item.widget(): item.widget().deleteLater()
            
        for item in self.view.items[:]:
            if isinstance(item, (gl.GLMeshItem, gl.GLTextItem)) and item != self.earth_sphere:
                self.view.removeItem(item)
        self.surfaces = {}
        self.surface_labels = {}
        self.axis_labels = []

        # Dimensions
        self.unique_times = np.sort(self.df['Time'].unique())
        self.unique_lats = np.sort(self.df['Latitude'].unique())
        self.unique_lons = np.sort(self.df['Longitude'].unique())
        
        n_t, n_lat, n_lon = len(self.unique_times), len(self.unique_lats), len(self.unique_lons)
        
        self.slider.setMaximum(n_t - 1)
        self.slider.setValue(0)
        
        min_lat, max_lat = self.unique_lats[0], self.unique_lats[-1]
        min_lon, max_lon = self.unique_lons[0], self.unique_lons[-1]
        
        for spin in [self.spin_lat1, self.spin_lat2]:
            spin.setRange(min_lat, max_lat)
            spin.setSingleStep((max_lat - min_lat) / 20)
        for spin in [self.spin_lon1, self.spin_lon2]:
            spin.setRange(min_lon, max_lon)
            spin.setSingleStep((max_lon - min_lon) / 20)
            
        self.spin_lat1.setValue(min_lat)
        self.spin_lon1.setValue(min_lon)
        self.spin_lat2.setValue(max_lat)
        self.spin_lon2.setValue(max_lon)
        
        # Process Variables
        reserved = ['Time', 'Latitude', 'Longitude', 'SiteID']
        self.var_order = [c for c in self.df.columns if c not in reserved]
        
        palette = [(255, 50, 50), (50, 255, 50), (50, 100, 255), (255, 255, 0), 
                   (0, 255, 255), (255, 0, 255), (255, 128, 0)]
        
        for i, var in enumerate(self.var_order):
            # Safe reshape
            try:
                data_array = self.df[var].values.reshape(n_t, n_lat, n_lon)
            except:
                print(f"Skipping {var} due to shape mismatch")
                continue
                
            self.grid_data[var] = data_array
            self.var_colors[var] = palette[i % len(palette)]
            
            cb = QCheckBox(var)
            if i == 0: 
                cb.setChecked(True)
                self.active_variables.add(var)
            cb.stateChanged.connect(self.on_var_check_changed)
            self.check_boxes[var] = cb
            self.v_layout.addWidget(cb)
            
        self.lbl_info.setText(self.tr['LBL_INFO_READY'].format(t=n_t, lat=n_lat, lon=n_lon))
        self.init_surfaces()
        self.update_axis_labels()
        self.update_plot()
        self.update_slice_view()

    def generate_grid_faces(self, rows, cols):
        r_idx = np.arange(rows - 1)
        c_idx = np.arange(cols - 1)
        r_grid, c_grid = np.meshgrid(r_idx, c_idx, indexing='ij')
        
        p00 = r_grid * cols + c_grid
        p01 = r_grid * cols + (c_grid + 1)
        p10 = (r_grid + 1) * cols + c_grid
        p11 = (r_grid + 1) * cols + (c_grid + 1)
        
        t1 = np.column_stack((p00.flatten(), p01.flatten(), p11.flatten()))
        t2 = np.column_stack((p00.flatten(), p11.flatten(), p10.flatten()))
        return np.vstack((t1, t2)).astype(np.uint32)

    def init_surfaces(self):
        for var in self.var_order:
            if var not in self.grid_data: continue
            surface = gl.GLMeshItem(meshdata=None, smooth=True, shader='shaded', glOptions='translucent')
            surface.setVisible(False)
            self.view.addItem(surface)
            self.surfaces[var] = surface
            
            label = gl.GLTextItem(text=var, color=(1, 1, 1, 1))
            label.setVisible(False)
            self.view.addItem(label)
            self.surface_labels[var] = label

    def update_axis_labels(self):
        for item in self.axis_labels:
            try: self.view.removeItem(item)
            except: pass
        self.axis_labels = []
        
        if len(self.unique_lons) == 0: return
        n_labels = 5
        is_spherical = self.chk_projection.isChecked()
        min_lat, max_lat = self.unique_lats[0], self.unique_lats[-1]
        min_lon, max_lon = self.unique_lons[0], self.unique_lons[-1]
        
        # Lon
        lon_indices = np.linspace(0, len(self.unique_lons)-1, n_labels, dtype=int)
        for idx in lon_indices:
            lon_val = self.unique_lons[idx]
            if is_spherical:
                R = 24.0
                phi = np.radians(90 - max(-90, min_lat - 10))
                theta = np.radians(lon_val)
                x = R * np.sin(phi) * np.cos(theta)
                y = R * np.sin(phi) * np.sin(theta)
                z = R * np.cos(phi)
                label = gl.GLTextItem(pos=(x,y,z), text=f"{lon_val:.0f}", font=QFont('Arial', 8))
            else:
                x = -10 + (idx / (len(self.unique_lons)-1)) * 20
                label = gl.GLTextItem(pos=(x, -12, 0), text=f"{lon_val:.0f}", font=QFont('Arial', 8))
            self.view.addItem(label)
            self.axis_labels.append(label)
            
        # Lat
        lat_indices = np.linspace(0, len(self.unique_lats)-1, n_labels, dtype=int)
        for idx in lat_indices:
            lat_val = self.unique_lats[idx]
            if is_spherical:
                R = 24.0
                phi = np.radians(90 - lat_val)
                theta = np.radians(min_lon - 10)
                x = R * np.sin(phi) * np.cos(theta)
                y = R * np.sin(phi) * np.sin(theta)
                z = R * np.cos(phi)
                label = gl.GLTextItem(pos=(x,y,z), text=f"{lat_val:.0f}", font=QFont('Arial', 8))
            else:
                y = -10 + (idx / (len(self.unique_lats)-1)) * 20
                label = gl.GLTextItem(pos=(-13, y, 0), text=f"{lat_val:.0f}", font=QFont('Arial', 8))
            self.view.addItem(label)
            self.axis_labels.append(label)

    def generate_gradient_colors(self, data, base_color_rgb):
        d_min = np.nanmin(data)
        d_max = np.nanmax(data)
        if d_max == d_min: norm = np.zeros_like(data)
        else: norm = (data - d_min) / (d_max - d_min)
        norm = np.nan_to_num(norm, nan=0.0)
        norm_flat = norm.flatten()
        colors = np.zeros((len(norm_flat), 4), dtype=float)
        r, g, b = [c/255.0 for c in base_color_rgb]
        colors[:, 0] = r; colors[:, 1] = g; colors[:, 2] = b
        colors[:, 3] = 0.05 + 0.8 * (norm_flat ** 1.5)
        return colors

    def update_plot(self):
        if not self.grid_data: return
        info_texts = []
        is_spherical = self.chk_projection.isChecked()
        is_norm = self.chk_normalize.isChecked()
        n_lat = len(self.unique_lats)
        n_lon = len(self.unique_lons)
        
        if not hasattr(self, '_cached_faces') or self._cached_faces_shape != (n_lat, n_lon):
            self._cached_faces = self.generate_grid_faces(n_lat, n_lon)
            self._cached_faces_shape = (n_lat, n_lon)
        faces = self._cached_faces
        
        lat_grid, lon_grid = np.meshgrid(self.unique_lats, self.unique_lons, indexing='ij')
        
        if is_spherical:
            R_base = 20
            phi = np.radians(90 - lat_grid)
            theta = np.radians(lon_grid)
            bx = np.sin(phi) * np.cos(theta)
            by = np.sin(phi) * np.sin(theta)
            bz = np.cos(phi)
        else:
            min_lat, max_lat = self.unique_lats[0], self.unique_lats[-1]
            min_lon, max_lon = self.unique_lons[0], self.unique_lons[-1]
            norm_lat = (lat_grid - min_lat) / (max_lat - min_lat)
            norm_lon = (lon_grid - min_lon) / (max_lon - min_lon)
            px = -10 + norm_lon * 20
            py = -10 + norm_lat * 20

        for var in self.var_order:
            if var not in self.surfaces: continue
            surface = self.surfaces[var]
            label = self.surface_labels[var]
            
            if var not in self.active_variables:
                surface.setVisible(False)
                label.setVisible(False)
                continue
                
            surface.setVisible(True)
            label.setVisible(True)
            
            z_data = self.grid_data[var][self.current_time_idx]
            mask = ~np.isnan(z_data)
            if not mask.any():
                surface.setVisible(False)
                continue
                
            z_min = np.nanmin(self.grid_data[var])
            z_max = np.nanmax(self.grid_data[var])
            z_norm = (z_data - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z_data)
            z_norm_filled = np.nan_to_num(z_norm, nan=0.0)
            
            colors = self.generate_gradient_colors(z_data, self.var_colors[var])
            colors[~mask.flatten(), 3] = 0.0
            
            if is_spherical:
                R = R_base + z_norm_filled * 2.0
                x = R * bx
                y = R * by
                z = R * bz
                verts = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
                
                max_idx = np.nanargmax(z_norm)
                idx_y, idx_x = np.unravel_index(max_idx, z_norm.shape)
                lp = np.array([x[idx_y, idx_x], y[idx_y, idx_x], z[idx_y, idx_x]])
                label.setData(pos=lp * 1.1)
            else:
                z_val = z_norm_filled * 10.0
                verts = np.column_stack((px.flatten(), py.flatten(), z_val.flatten()))
                
                max_idx = np.nanargmax(z_val)
                idx_y, idx_x = np.unravel_index(max_idx, z_val.shape)
                label.setData(pos=(px[idx_y, idx_x], py[idx_y, idx_x], z_val[idx_y, idx_x] + 1))
                
            md = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
            surface.setMeshData(meshdata=md)
            surface.resetTransform()
            
            info_texts.append(f"{var}: {np.nanmin(z_data):.1f}~{np.nanmax(z_data):.1f}" if not is_norm else f"{var}: 0.0~1.0")
            
        self.lbl_info.setText(" | ".join(info_texts) if info_texts else self.tr['LBL_INFO_NO_VAR'])

    def latlon_to_local(self, lat, lon):
        if len(self.unique_lats) == 0: return 0,0,0
        if self.chk_projection.isChecked():
            R = 20.0
            phi = np.radians(90 - lat)
            theta = np.radians(lon)
            return R * np.sin(phi) * np.cos(theta), R * np.sin(phi) * np.sin(theta), R * np.cos(phi)
        else:
            min_lat, max_lat = self.unique_lats[0], self.unique_lats[-1]
            min_lon, max_lon = self.unique_lons[0], self.unique_lons[-1]
            n_lat = (lat - min_lat) / (max_lat - min_lat)
            n_lon = (lon - min_lon) / (max_lon - min_lon)
            return -10 + n_lon*20, -10 + n_lat*20, 12

    def local_to_latlon(self, x, y, z=0):
        if len(self.unique_lats) == 0: return 0,0
        if self.chk_projection.isChecked():
            R = np.sqrt(x*x + y*y + z*z)
            if R == 0: return 0,0
            phi = np.arccos(z/R)
            theta = np.arctan2(y, x)
            return 90 - np.degrees(phi), np.degrees(theta)
        else:
            min_lat, max_lat = self.unique_lats[0], self.unique_lats[-1]
            min_lon, max_lon = self.unique_lons[0], self.unique_lons[-1]
            n_lat = (y + 10) / 20.0
            n_lon = (x + 10) / 20.0
            return min_lat + n_lat*(max_lat-min_lat), min_lon + n_lon*(max_lon-min_lon)

    def get_slice_data(self, n_samples=100):
        if not self.grid_data: return None, None
        
        lat1, lon1 = self.spin_lat1.value(), self.spin_lon1.value()
        lat2, lon2 = self.spin_lat2.value(), self.spin_lon2.value()
        
        sample_lat = np.linspace(lat1, lat2, n_samples)
        sample_lon = np.linspace(lon1, lon2, n_samples)
        
        min_lat, max_lat = self.unique_lats[0], self.unique_lats[-1]
        min_lon, max_lon = self.unique_lons[0], self.unique_lons[-1]
        
        results = {}
        
        for var in self.var_order:
            # For update_slice_view we check active, but for export maybe we want all?
            # Let's stick to active for now to match UI behavior.
            if var not in self.active_variables: continue
            
            grid = self.grid_data[var][self.current_time_idx]
            vals = []
            
            for i in range(n_samples):
                # Map to grid indices
                r = (sample_lat[i] - min_lat) / (max_lat - min_lat) * (grid.shape[0] - 1)
                c = (sample_lon[i] - min_lon) / (max_lon - min_lon) * (grid.shape[1] - 1)
                
                if not (0 <= r <= grid.shape[0]-1 and 0 <= c <= grid.shape[1]-1):
                    vals.append(np.nan)
                    continue
                
                r0 = int(np.floor(r))
                c0 = int(np.floor(c))
                r1 = min(r0 + 1, grid.shape[0] - 1)
                c1 = min(c0 + 1, grid.shape[1] - 1)
                
                dr = r - r0
                dc = c - c0
                
                v00 = grid[r0, c0]
                v01 = grid[r0, c1]
                v10 = grid[r1, c0]
                v11 = grid[r1, c1]
                
                if np.isnan([v00, v01, v10, v11]).any():
                    vals.append(grid[int(round(r)), int(round(c))])
                else:
                    val = (1-dr)*(1-dc)*v00 + (1-dr)*dc*v01 + dr*(1-dc)*v10 + dr*dc*v11
                    vals.append(val)
            
            results[var] = np.array(vals)
            
        dist_axis = np.linspace(0, 1, n_samples)
        return dist_axis, results

    def update_slice_view(self):
        if not self.grid_data: return
        lat1, lon1 = self.spin_lat1.value(), self.spin_lon1.value()
        lat2, lon2 = self.spin_lat2.value(), self.spin_lon2.value()
        
        # 1. Update 3D Indicators
        p1 = np.array(self.latlon_to_local(lat1, lon1))
        p2 = np.array(self.latlon_to_local(lat2, lon2))
        
        is_spherical = self.chk_projection.isChecked()
        if is_spherical:
            p1 = p1 / (np.linalg.norm(p1) + 1e-9) * 20.5
            p2 = p2 / (np.linalg.norm(p2) + 1e-9) * 20.5
            
            n_seg = 50
            lats = np.linspace(lat1, lat2, n_seg)
            lons = np.linspace(lon1, lon2, n_seg)
            path = []
            for i in range(n_seg):
                pt = np.array(self.latlon_to_local(lats[i], lons[i]))
                if np.linalg.norm(pt) > 0:
                    path.append(pt / np.linalg.norm(pt) * 20.5)
                else:
                    path.append(pt)
            self.slice_line.setData(pos=np.array(path))
        else:
            self.slice_line.setData(pos=np.array([p1, p2]))
            
        self.slice_points.setData(pos=np.array([p1, p2]))
        
        # 2. Update 2D Plot
        self.plot_widget.clear()
        is_norm = self.chk_normalize.isChecked()
        tr = self.tr
        self.plot_widget.setLabel('left', tr['PLOT_Y_NORM'] if is_norm else tr['PLOT_Y_RAW'])
        
        dist_axis, results = self.get_slice_data(n_samples=100)
        if results is None: return
        
        for var, vals in results.items():
            # Normalization logic
            if is_norm:
                z_min = np.nanmin(self.grid_data[var])
                z_max = np.nanmax(self.grid_data[var])
                if z_max > z_min:
                    vals = (vals - z_min) / (z_max - z_min)
                else:
                    vals = np.zeros_like(vals)
            
            base_color = self.var_colors.get(var, (200, 200, 200))
            pen = pg.mkPen(color=base_color, width=2)
            self.plot_widget.plot(dist_axis, vals, pen=pen, name=var, connect='finite')

    def export_slice_plot(self):
        if not self.grid_data or not self.active_variables:
            QMessageBox.warning(self, "Export", "No data to export.")
            return

        # High resolution data
        dist_axis, results = self.get_slice_data(n_samples=200)
        
        # Create Matplotlib Figure
        plt.style.use('default')
        # Try to match academic style
        font_options = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']}
        plt.rcParams.update(font_options)
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        is_norm = self.chk_normalize.isChecked()
        
        # Plot
        for var, vals in results.items():
            if is_norm:
                z_min = np.nanmin(self.grid_data[var])
                z_max = np.nanmax(self.grid_data[var])
                if z_max > z_min:
                    vals = (vals - z_min) / (z_max - z_min)
                else:
                    vals = np.zeros_like(vals)
            
            # Convert color from tuple (r,g,b) to #Hex
            c_rgb = self.var_colors.get(var, (0,0,0))
            c_hex = '#{:02x}{:02x}{:02x}'.format(*c_rgb)
            
            ax.plot(dist_axis, vals, label=var, color=c_hex, linewidth=2)
            
        # Styling
        ax.set_xlabel(self.tr['PLOT_X'], fontsize=12, fontweight='bold')
        ax.set_ylabel(self.tr['PLOT_Y_NORM'] if is_norm else self.tr['PLOT_Y_RAW'], fontsize=12, fontweight='bold')
        
        # Title
        title = f"Slice Profile: Lat({self.spin_lat1.value():.1f}->{self.spin_lat2.value():.1f}), Lon({self.spin_lon1.value():.1f}->{self.spin_lon2.value():.1f})"
        ax.set_title(title, fontsize=14)
        
        ax.legend(frameon=False, loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in')
        
        # File Dialog
        filename, filter_type = QFileDialog.getSaveFileName(self, self.tr['BTN_EXPORT'], "", "PNG Image (*.png);;SVG Image (*.svg);;PDF Document (*.pdf)")
        
        if filename:
            try:
                fig.savefig(filename, bbox_inches='tight', dpi=300)
                QMessageBox.information(self, "Success", f"Saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
        
        plt.close(fig)

    def open_multi_analysis(self):
        if not self.grid_data:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
            
        self.multi_win = MultiSiteVizWindow(self)
        self.multi_win.show()

    def on_var_check_changed(self):
        sender = self.sender()
        if sender.isChecked(): self.active_variables.add(sender.text())
        else: self.active_variables.discard(sender.text())
        self.update_plot()
        self.update_slice_view()

    def on_normalization_changed(self):
        self.update_plot()
        self.update_slice_view()

    def on_projection_changed(self):
        is_spherical = self.chk_projection.isChecked()
        self.earth_sphere.setVisible(is_spherical)
        self.grid.setVisible(not is_spherical)
        self.update_axis_labels()
        self.update_plot()
        self.update_slice_view()

    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("播放")
            self.is_playing = False
        else:
            self.timer.start(100)
            self.btn_play.setText("暂停")
            self.is_playing = True

    def next_step(self):
        if len(self.unique_times) > 0:
            self.slider.setValue((self.current_time_idx + 1) % len(self.unique_times))

    def on_slider_change(self, value):
        self.current_time_idx = value
        if len(self.unique_times) > value:
            self.lbl_time.setText(f"时间: {self.unique_times[value]:.2f}")
            self.update_plot()
            self.update_slice_view()

# ==========================================
# 4. Spatio-Temporal Evolution Window
# ==========================================
class MultiSiteVizWindow(QMainWindow):
    def __init__(self, parent_viz):
        super().__init__()
        self.parent_viz = parent_viz
        self.setWindowTitle("时空演化立体分析 (Spatio-Temporal Evolution Analysis)")
        self.resize(1000, 800)
        
        self.init_ui()
        self.update_data()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # --- Control Panel ---
        controls = QGroupBox("Analysis Controls")
        c_layout = QHBoxLayout(controls)
        
        # View Mode
        c_layout.addWidget(QLabel("View Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["3D Stereo (Interactive)", "2D Ridge Plot (Stacked)", "2D Heatmap"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        c_layout.addWidget(self.combo_mode)
        
        # Variable
        c_layout.addWidget(QLabel("Variable:"))
        self.combo_var = QComboBox()
        # Populate variables
        self.combo_var.addItems(self.parent_viz.var_order)
        self.combo_var.currentIndexChanged.connect(self.update_view)
        c_layout.addWidget(self.combo_var)
        
        # Samples
        c_layout.addWidget(QLabel("Samples:"))
        self.spin_samples = QDoubleSpinBox() # Hack: use double spin box for integer to get clean UI? No, QSpinBox
        # Wait, QSpinBox is better
        from PySide6.QtWidgets import QSpinBox
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(10, 200)
        self.spin_samples.setValue(50)
        self.spin_samples.setSingleStep(10)
        self.spin_samples.valueChanged.connect(self.update_data)
        c_layout.addWidget(self.spin_samples)
        
        # Export
        self.btn_export = QPushButton("Export (SVG/PNG/PDF)")
        self.btn_export.clicked.connect(self.export_view)
        c_layout.addWidget(self.btn_export)
        
        c_layout.addStretch()
        layout.addWidget(controls)
        
        # --- View Stack ---
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, stretch=1)
        
        # 1. 3D View (PyQtGraph)
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=40, elevation=30, azimuth=-90)
        self.stack.addWidget(self.gl_view)
        
        # 2. 2D View (Matplotlib)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.stack.addWidget(self.canvas)
        
    def get_spacetime_data(self):
        """
        Extracts matrix: Rows=Time, Cols=Location (Distance along slice)
        Returns: times, dists, matrix (T x D)
        """
        var = self.combo_var.currentText()
        if not var or var not in self.parent_viz.grid_data:
            return None, None, None
            
        n_samples = self.spin_samples.value()
        times = self.parent_viz.unique_times
        n_times = len(times)
        
        # Get slice coords
        lat1, lon1 = self.parent_viz.spin_lat1.value(), self.parent_viz.spin_lon1.value()
        lat2, lon2 = self.parent_viz.spin_lat2.value(), self.parent_viz.spin_lon2.value()
        
        sample_lat = np.linspace(lat1, lat2, n_samples)
        sample_lon = np.linspace(lon1, lon2, n_samples)
        dists = np.linspace(0, 1, n_samples)
        
        # Grid info
        min_lat, max_lat = self.parent_viz.unique_lats[0], self.parent_viz.unique_lats[-1]
        min_lon, max_lon = self.parent_viz.unique_lons[0], self.parent_viz.unique_lons[-1]
        
        full_grid = self.parent_viz.grid_data[var] # Shape (T, Lat, Lon)
        
        # Pre-calculate grid indices for the slice (constant over time)
        # grid shape is (n_t, n_lat, n_lon)
        # We need indices for lat/lon dimensions (1 and 2)
        
        n_lat_grid = full_grid.shape[1]
        n_lon_grid = full_grid.shape[2]
        
        r_float = (sample_lat - min_lat) / (max_lat - min_lat) * (n_lat_grid - 1)
        c_float = (sample_lon - min_lon) / (max_lon - min_lon) * (n_lon_grid - 1)
        
        # Clip
        r_float = np.clip(r_float, 0, n_lat_grid - 1)
        c_float = np.clip(c_float, 0, n_lon_grid - 1)
        
        # Vectorized Bilinear Interpolation for all time steps?
        # Maybe too heavy to do fully vectorized with meshgrid logic?
        # Let's do a semi-vectorized approach: Interpolate spatial slice first, then apply to all T
        
        # Indices
        r0 = np.floor(r_float).astype(int)
        c0 = np.floor(c_float).astype(int)
        r1 = np.minimum(r0 + 1, n_lat_grid - 1)
        c1 = np.minimum(c0 + 1, n_lon_grid - 1)
        
        dr = r_float - r0
        dc = c_float - c0
        
        # full_grid is (T, R, C)
        # We want to extract (T, Sample)
        
        # Advanced indexing
        # v00 shape will be (T, Samples)
        v00 = full_grid[:, r0, c0]
        v01 = full_grid[:, r0, c1]
        v10 = full_grid[:, r1, c0]
        v11 = full_grid[:, r1, c1]
        
        # Weights (broadcast over T)
        w00 = (1-dr)*(1-dc)
        w01 = (1-dr)*dc
        w10 = dr*(1-dc)
        w11 = dr*dc
        
        # Result (T, Samples)
        matrix = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
        
        return times, dists, matrix

    def update_data(self):
        self.update_view()

    def on_mode_changed(self):
        mode = self.combo_mode.currentIndex()
        if mode == 0:
            self.stack.setCurrentIndex(0)
        else:
            self.stack.setCurrentIndex(1)
        self.update_view()

    def update_view(self):
        times, dists, matrix = self.get_spacetime_data()
        if matrix is None: return
        
        mode = self.combo_mode.currentIndex()
        
        if mode == 0:
            self.plot_3d(times, dists, matrix)
        elif mode == 1:
            self.plot_ridge(times, dists, matrix)
        else:
            self.plot_heatmap(times, dists, matrix)

    def plot_3d(self, times, dists, matrix):
        self.gl_view.clear()
        
        # Add Grid
        g = gl.GLGridItem()
        g.setSize(x=max(times)-min(times), y=1, z=0)
        # Center grid?
        # Let's map coordinates:
        # X: Time (0 to T_range)
        # Y: Distance (0 to 1) -> mapped to e.g. -10 to 10
        # Z: Value
        
        # Normalize Data for display
        t_min, t_max = times[0], times[-1]
        
        # We want X to be centered around 0
        t_center = (t_max + t_min) / 2
        t_scale = 20.0 / (t_max - t_min) if t_max > t_min else 1
        
        y_scale = 20.0 # Distance 0-1 maps to -10 to 10
        
        z_min = np.nanmin(matrix)
        z_max = np.nanmax(matrix)
        z_range = z_max - z_min if z_max > z_min else 1
        z_scale = 10.0 / z_range
        
        # Axes
        axis = gl.GLAxisItem()
        axis.setSize(x=10, y=10, z=10)
        self.gl_view.addItem(axis)
        
        # Plot lines
        # matrix is (T, D)
        # We want lines along Time for each Distance step?
        # "Vertical to this plane (Time-Value) is the selected locations"
        # So we draw one line per location index.
        
        n_locs = len(dists)
        # Subsample lines if too many? 50 is fine.
        
        for i in range(n_locs):
            # Location i
            y_val = (dists[i] - 0.5) * y_scale # Map 0..1 to -10..10
            
            vals = matrix[:, i]
            
            # X: Time
            x_vals = (times - t_center) * t_scale
            
            # Z: Value
            z_vals = (vals - z_min) * z_scale
            
            # Points
            pts = np.column_stack([x_vals, np.full_like(x_vals, y_val), z_vals])
            
            # Color
            # Gradient based on distance? Or Value?
            # Let's use distance color (Rainbow)
            color = pg.intColor(i, n_locs, alpha=200)
            color = pg.glColor(color)
            
            plt_item = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
            self.gl_view.addItem(plt_item)
            
        # Labels? Simple text at corners
        
    def plot_ridge(self, times, dists, matrix):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d') # Use 3D projection for ridge/waterfall effectively?
        # Actually user asked for "Planar visualization" for the alternative.
        # But "Stacked lines" is often 2D with offset.
        
        # Let's do a 2D Waterfall (Offset lines)
        ax = self.fig.add_subplot(111)
        
        n_locs = len(dists)
        # Select subset of locations to avoid clutter?
        # If 50 samples, 50 lines is okay.
        
        offset_step = (np.nanmax(matrix) - np.nanmin(matrix)) * 0.2
        if offset_step == 0: offset_step = 1
        
        # Iterate backwards to draw front-to-back
        for i in range(n_locs):
            vals = matrix[:, i]
            # Add offset
            offset = i * offset_step * 0.1 # Small vertical offset
            
            ax.plot(times, vals + offset, color=plt.cm.jet(i/n_locs), alpha=0.8)
            
        ax.set_xlabel("Time")
        ax.set_ylabel("Value (+ Offset by Location)")
        ax.set_title(f"Stacked Time Series ({self.combo_var.currentText()})")
        self.canvas.draw()
        
    def plot_heatmap(self, times, dists, matrix):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Imshow
        # matrix is (T, D). We usually want Time on X.
        # So shape is correct for Transpose?
        # imshow expects (Rows, Cols). 
        # If we want X=Time, Y=Location.
        # Rows should be Location, Cols should be Time.
        # So Transpose matrix.
        
        im = ax.imshow(matrix.T, aspect='auto', 
                       extent=[times[0], times[-1], 0, 1],
                       origin='lower', cmap='viridis')
        
        self.fig.colorbar(im, ax=ax, label="Value")
        ax.set_xlabel("Time")
        ax.set_ylabel("Distance along Slice (0-1)")
        ax.set_title(f"Spatio-Temporal Heatmap ({self.combo_var.currentText()})")
        self.canvas.draw()

    def export_view(self):
        mode = self.combo_mode.currentIndex()
        
        filename, _ = QFileDialog.getSaveFileName(self, "Export Analysis", "", "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if not filename: return
        
        if mode == 0:
            # Export 3D View
            # Option A: Screenshot (PNG only)
            # Option B: Re-create in Matplotlib 3D (Vector support)
            # User wants SVG/PNG.
            
            if filename.endswith('.png'):
                # GL View grabFrameBuffer
                img = self.gl_view.grabFramebuffer()
                img.save(filename)
            else:
                # Vector export for 3D is tricky. 
                # Re-draw using Matplotlib 3D
                self.export_matplotlib_3d(filename)
                
        else:
            # 2D View
            self.fig.savefig(filename, bbox_inches='tight', dpi=300)
            
        QMessageBox.information(self, "Export", f"Saved to {filename}")

    def export_matplotlib_3d(self, filename):
        times, dists, matrix = self.get_spacetime_data()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        n_locs = len(dists)
        
        # Draw lines
        for i in range(n_locs):
            # X: Time, Y: Distance (fixed per line), Z: Value
            y_val = dists[i] # 0..1
            vals = matrix[:, i]
            
            # Matplotlib 3D: plot(x, y, z)
            # We want X=Time, Y=Location, Z=Value
            # Note: mplot3d Y axis is depth
            
            ax.plot(times, [y_val]*len(times), vals, 
                    color=plt.cm.jet(i/n_locs), alpha=0.8)
            
        ax.set_xlabel('Time')
        ax.set_ylabel('Location (Distance)')
        ax.set_zlabel('Value')
        ax.view_init(elev=30, azim=-60)
        
        # Pane colors transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpatioTemporalViz()
    window.show()
    sys.exit(app.exec())
