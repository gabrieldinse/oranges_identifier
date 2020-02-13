"""
This program is used to extract features of oranges passing by a conveyor
belt. The features are:
    - diameter;
    - RGB;
    - CCI (Citrus Color Index);
    - color (a string with human readable value of color like: 'green',
    'yellow', 'orange', etc).

The idenfitier works using techniques of image processing and computer vision,
based on the Opencv library to extract the image features, and using Qt (PyQt)
to create the GUI (Graphical User Interface). The conveyor is controlled by
an Arduino using QSerialPort from PyQt, and the data is stored in an CSV file.
"""

# Standard modules
from math import floor, hypot
import time
import sys
import os
import json

# Third party modules
from PyQt5.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene,
                             QMainWindow, QApplication)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from numpy import sqrt, pi
import cv2
import numpy as np
import imutils

# Local modules
from main_window import Ui_MainWindow
from conveyor_belt_controller import ConveyorController
from oranges_data_storager import OrangesDataWriter


def circular_kernel(size):
    """ Cria um uma janela circular para aplicacao de convolucao. """

    kernel = np.ones((size, size), dtype=np.uint8)
    center = floor(size / 2)
    for i in range(size):
        for j in range(size):
            if hypot(i - center, j - center) > center:
                kernel[i, j] = 0
    return kernel


class MainWindow(QMainWindow):
    """ Classe que representa a janela principal do programa. """

    storage_updated = pyqtSignal()

    def __init__(self):
        # Configuracoes principais da janela/GUI
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Sistema de visao funcionando ou nao
        self.identifier_running = False

        #  Configuracoes de pastas e arquivos
        self.gui_config_filename = 'gui_config.json'

        # Camera
        self.camera = cv2.VideoCapture(0)
        self.camera_resolution = (640.0, 480.0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
        self.capture_timer_delay = 3.0

        # Variaveis do processo
        self.conveyor = ConveyorController(debug=True)
        self.data_writer = OrangesDataWriter(
            os.path.dirname(os.path.abspath(__file__)), illuminant='D75')
        self.diameter_prop = 1/2.275

        # Visualizacao dos frames no framework do Qt
        self.main_scene = QGraphicsScene()
        self.ui.main_graphics_view.setScene(self.main_scene)
        self.main_pixmap = QGraphicsPixmapItem()
        self.main_scene.addItem(self.main_pixmap)
        self.segmentation_scene = QGraphicsScene()
        self.ui.segmentation_graphics_view.setScene(self.segmentation_scene)
        self.segmentation_pixmap = QGraphicsPixmapItem()
        self.segmentation_scene.addItem(self.segmentation_pixmap)

        # Coneccao dos signals e slots
        self.ui.min_h_slider.valueChanged.connect(
            self.min_h_slider_value_changed)
        self.ui.max_h_slider.valueChanged.connect(
            self.max_h_slider_value_changed)
        self.ui.min_s_slider.valueChanged.connect(
            self.min_s_slider_value_changed)
        self.ui.max_s_slider.valueChanged.connect(
            self.max_s_slider_value_changed)
        self.ui.min_v_slider.valueChanged.connect(
            self.min_v_slider_value_changed)
        self.ui.max_v_slider.valueChanged.connect(
            self.max_v_slider_value_changed)

        self.ui.gaussian_kernel_spin_box.valueChanged.connect(
            self.gaussian_kernel_spin_box_value_changed)
        self.ui.opening_kernel_spin_box.valueChanged.connect(
            self.opening_kernel_spin_box_value_changed)

        self.ui.min_frame_width_spin_box.valueChanged.connect(
            self.min_frame_width_spin_box_value_changed)
        self.ui.max_frame_width_spin_box.valueChanged.connect(
            self.max_frame_width_spin_box_value_changed)
        self.ui.min_frame_height_spin_box.valueChanged.connect(
            self.min_frame_height_spin_box_value_changed)
        self.ui.max_frame_height_spin_box.valueChanged.connect(
            self.max_frame_height_spin_box_value_changed)

        self.ui.start_conveyor_push_button.clicked.connect(
            self.start_conveyor_push_button_clicked)
        self.ui.stop_conveyor_push_button.clicked.connect(
            self.stop_conveyor_push_button_clicked)
        self.ui.start_identifier_push_button.clicked.connect(
            self.start_identifier_push_button_clicked)
        self.ui.stop_identifier_push_button.clicked.connect(
            self.stop_identifier_push_button_clicked)

        self.ui.capture_line_position_slider.valueChanged.connect(
            self.capture_line_position_slider_value_changed)
        self.ui.capture_box_left_width_spin_box.valueChanged.connect(
            self.capture_box_left_width_spin_box_value_changed)
        self.ui.capture_box_right_width_spin_box.valueChanged.connect(
            self.capture_box_right_width_spin_box_value_changed)

        self.storage_updated.connect(self.update_ui_oranges_info)

        # Configuracoes da gui (importante realizar as coneccoes antes)
        self.load_config()

        # Timer para capturar frames e processa-los
        self.frames_processor_timer = QTimer()
        self.frames_processor_timer.timeout.connect(
            self.process_and_verify_frame)
        self.frames_processor_timer.start(0)

    def load_config(self):
        """ Carrega as confiurações da gui no formato JSON. """

        with open(self.gui_config_filename) as config_file:
            config = json.load(config_file)

            # Configuracao dos sliders hsv
            hsv_sliders = config['color_range']
            self.min_h = hsv_sliders['min_h']
            self.max_h = hsv_sliders['max_h']
            self.min_s = hsv_sliders['min_s']
            self.max_s = hsv_sliders['max_s']
            self.min_v = hsv_sliders['min_v']
            self.max_v = hsv_sliders['max_v']
            self.ui.min_h_slider.setValue(self.min_h)
            self.ui.max_h_slider.setValue(self.max_h)
            self.ui.min_s_slider.setValue(self.min_s)
            self.ui.max_s_slider.setValue(self.max_s)
            self.ui.min_v_slider.setValue(self.min_v)
            self.ui.max_v_slider.setValue(self.max_v)

            # Configuracao do tamanho do Kernel
            kernel_size = config['kernel_size']
            self.gaussian_kernel_size = kernel_size['gaussian']
            self.opening_kernel_size = kernel_size['opening']
            self.opening_kernel = circular_kernel(self.opening_kernel_size)
            self.ui.opening_kernel_spin_box.setValue(self.opening_kernel_size)
            self.ui.gaussian_kernel_spin_box.setValue(self.gaussian_kernel_size)

            # Configuracao da area de interesse do frame
            frame_dimension_range = config['frame_dimension_range']
            self.min_frame_width = frame_dimension_range['min_width']
            self.max_frame_width = frame_dimension_range['max_width']
            self.min_frame_height = frame_dimension_range['min_height']
            self.max_frame_height = frame_dimension_range['max_height']
            self.ui.min_frame_width_spin_box.setValue(self.min_frame_width)
            self.ui.max_frame_width_spin_box.setValue(self.max_frame_width)
            self.ui.min_frame_height_spin_box.setValue(self.min_frame_height)
            self.ui.max_frame_height_spin_box.setValue(self.max_frame_height)
            
            # Configuracao da linha de captura
            capture_line = config['capture_line']
            self.capture_line_position = capture_line['position']
            self.capture_box_left_width = capture_line['left_width']
            self.capture_box_left_position = capture_line['position'] \
                - capture_line['left_width']
            self.capture_box_right_width = capture_line['right_width']
            self.capture_box_right_position = capture_line['position'] \
                - capture_line['right_width']
            self.ui.capture_box_left_width_spin_box.setValue(
                self.capture_box_left_width)
            self.ui.capture_box_right_width_spin_box.setValue(
                self.capture_box_right_width)
            self.ui.capture_line_position_slider.setValue(
                self.capture_line_position)

    def save_config(self):
        """ Salva as configuracoes da gui no formato JSON. """

        with open(self.gui_config_filename, 'w') as config_file:
            # Carrega as informacoes para o arquivo json na forma de um dict()
            config = {
                'color_range':
                {
                    'min_h': self.min_h,
                    'max_h': self.max_h,
                    'min_s': self.min_s,
                    'max_s': self.max_s,
                    'min_v': self.min_v,
                    'max_v': self.max_v
                },
                'kernel_size':
                {
                    'gaussian': self.gaussian_kernel_size,
                    'opening': self.opening_kernel_size
                },
                'frame_dimension_range':
                {
                    'min_width': self.min_frame_width,
                    'max_width': self.max_frame_width,
                    'min_height': self.min_frame_height,
                    'max_height': self.max_frame_height
                },
                'capture_line':
                {
                    'position': self.capture_line_position,
                    'left_width': self.capture_box_left_width,
                    'right_width': self.capture_box_right_width
                }
            }
            json.dump(config, config_file)

    def get_diameters_and_centroids(self):
        """ Extrai da imagem os diametros e centroides da cada laranja. """

        contours = cv2.findContours(self.segment_mask.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        self.diameters = []
        self.centroids = []
        self.contours = []
        for contour in contours:
            # Obtem os momentos do contorno:
            # m00 = area em pixels
            # m10 = momento de ordem 1 em x
            # m01 = momento de ordem 1 em y
            # m10/m00 = posicao x do centroide
            # m01/m00 = posicao y do centroide
            mom = cv2.moments(contour)

            # Filtra os contornos obtidos por metrica circular e area
            if (4*pi*mom['m00']/cv2.arcLength(contour, True)**2 > 0.8 and
                    mom['m00'] > 300):
                # Se o objeto eh aproximadamente circular, o diametro pode ser
                # estimado pela formula abaixo
                diameter = 2*sqrt(mom['m00']/pi)
                # Eh de interesse apenas a posicao do centroide em x
                centroid = (int(mom['m10'] / mom['m00']),
                            int(mom['m01'] / mom['m00']))
                self.diameters.append(diameter)
                self.centroids.append(centroid)
                self.contours.append(contour)

    def draw_contours_and_centroids(self):
        """ Desenha os centroides e o contorno obtido para cada laranja. """

        for contour, center in zip(self.contours, self.centroids):
            cv2.drawContours(self.processed_frame, [contour],
                             0, (0, 255, 0), 3)
            cv2.circle(self.processed_frame, center, 10, (255, 0, 0), -1)
            if self.identifier_running:
                cv2.drawContours(self.frame, [contour],
                                 0, (0, 255, 0), 3)
                cv2.circle(self.frame, center, 10, (255, 0, 0), -1)

    def draw_capture_lines(self):
        """ Desenha as linhas da caixa da captura no frame de segmentacao. """

        # Frame de segmentacao
        # Linha vertical central (linha de captura)
        cv2.line(self.processed_frame,
                 (self.capture_line_position, self.min_frame_height),
                 (self.capture_line_position, self.max_frame_height),
                 (0, 255, 0), 2)

        # Linha da esquerda
        cv2.line(self.processed_frame,
                 (self.capture_box_left_position, self.min_frame_height),
                 (self.capture_box_left_position, self.max_frame_height),
                 (0, 0, 255), 2)

        # Linha da direita
        cv2.line(self.processed_frame,
                 (self.capture_box_right_position, self.min_frame_height),
                 (self.capture_box_right_position, self.max_frame_height),
                 (0, 0, 255), 2)

        # Linha superior
        cv2.line(self.processed_frame,
                 (self.capture_box_left_position, self.max_frame_height),
                 (self.capture_box_right_position, self.max_frame_height),
                 (0, 0, 255), 2)

        # Linhas inferior
        cv2.line(self.processed_frame,
                 (self.capture_box_left_position, self.min_frame_height),
                 (self.capture_box_right_position, self.min_frame_height),
                 (0, 0, 255), 2)

    def segment_frame(self):
        """
        Segmenta o frame baseado nos parametros HSV e de convolucao
        configurados da gui.
        """

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # Corta o frame de acordo com os parametros de maximo e minimo de
        # largura e altura
        cropped_frame = np.zeros(self.frame.shape, dtype=np.uint8)
        cropped_frame[self.min_frame_height:self.max_frame_height,
                      self.min_frame_width:self.max_frame_width] = \
            self.frame[self.min_frame_height:self.max_frame_height,
                       self.min_frame_width:self.max_frame_width]
        self.frame = cropped_frame

        # Filtro gaussiano para suavizar ruidos na imagem
        blur = cv2.GaussianBlur(cropped_frame, (self.gaussian_kernel_size,
                                self.gaussian_kernel_size), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

        # Segmentacao de acordo com o invervalo de cores HSV
        in_range_mask = cv2.inRange(
            hsv, (self.min_h, self.min_s, self.min_v),
                 (self.max_h, self.max_s, self.max_v))
        self.segment_mask = cv2.morphologyEx(in_range_mask, cv2.MORPH_OPEN,
                                             self.opening_kernel)
        self.processed_frame = cv2.bitwise_and(cropped_frame, cropped_frame,
                                               mask=self.segment_mask)

    # Transforma o frame do formato do opencv (numpy.ndarray) para QImage e
    # mostra na gui
    def show_frames(self):
        """
        Converte os frames no formato Opencv para serem mostrados na
        interface do Qt
        """

        self.draw_contours_and_centroids()
        self.draw_capture_lines()

        height, width, _ = self.frame.shape
        bytes_per_line = 3 * width

        # Frame segmentado
        gui_frame = QImage(self.processed_frame.data, width, height,
                           bytes_per_line, QImage.Format_RGB888)
        gui_frame = gui_frame.scaled(355, 355, Qt.KeepAspectRatio)
        self.segmentation_pixmap.setPixmap(QPixmap.fromImage(gui_frame))

        # Frame original com realidade aumentada
        gui_frame = QImage(self.frame.data, width, height,
                           bytes_per_line, QImage.Format_RGB888)
        gui_frame = gui_frame.scaled(470, 470, Qt.KeepAspectRatio)
        self.main_pixmap.setPixmap(QPixmap.fromImage(gui_frame))

    def verify_frame(self):
        """
        Verifica a elegibilidade do frame e extrai as caracteristicas da
        laranja.
        """

        if self.identifier_running:
            if (time.time() - self.capture_timer) >= self.capture_timer_delay:
                for diameter, centroid in zip(self.diameters, self.centroids):
                    if (self.capture_line_position <= centroid[0]
                            <= self.capture_box_right_position):
                        self.create_capture_mask()

                        # cv2.mean retorna np.ndarray([R, G, B, alpha]), onde
                        # alpha eh a transparencia, nao utilizada neste caso
                        rgb_mean = np.array(cv2.mean(
                            self.processed_frame, mask=self.capture_mask)[0:3],
                                            dtype=np.uint8)
                        self.data_writer.add(
                            diameter * self.diameter_prop, rgb_mean)
                        self.storage_updated.emit()
                        self.capture_timer = time.time()
                        return

    def create_capture_mask(self):
        """ Cria uma mascara de captura baseada nos parametros de captura. """

        self.capture_mask = np.zeros(
            (int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))),
            dtype=np.uint8)

        self.capture_mask[
            self.min_frame_height:
            self.max_frame_height + 1,
            self.capture_box_left_position:
            self.capture_box_right_position + 1
        ] = np.ones((
            self.max_frame_height - self.min_frame_height + 1,
            self.capture_box_left_width + 1 +
            self.capture_box_right_width),
            dtype=np.uint8)

        self.capture_mask = cv2.bitwise_and(
            self.capture_mask, self.capture_mask,
            mask=self.segment_mask)

    # Reinplementacao do metodo closeEvent
    def closeEvent(self, event):
        """ Antes de encerrar o programa, salva os arquivos. """

        self.save_config()
        self.data_writer.stop()
        self.conveyor.stop()
        self.camera.release()
        event.accept()

    # Slots
    def process_and_verify_frame(self):
        """
        Slot principal. Processa o frame e extrai as caracteristicas
        da laranja.
        """

        self.grabbed, self.frame = self.camera.read()
        if self.grabbed:
            self.segment_frame()
            self.get_diameters_and_centroids()
            self.verify_frame()
            self.show_frames()

    def min_h_slider_value_changed(self):
        self.min_h = self.ui.min_h_slider.value()
        self.ui.min_h_label.setText(str(self.min_h))

    def max_h_slider_value_changed(self):
        self.max_h = self.ui.max_h_slider.value()
        self.ui.max_h_label.setText(str(self.max_h))

    def min_s_slider_value_changed(self):
        self.min_s = self.ui.min_s_slider.value()
        self.ui.min_s_label.setText(str(self.min_s))

    def max_s_slider_value_changed(self):
        self.max_s = self.ui.max_s_slider.value()
        self.ui.max_s_label.setText(str(self.max_s))

    def min_v_slider_value_changed(self):
        self.min_v = self.ui.min_v_slider.value()
        self.ui.min_v_label.setText(str(self.min_v))

    def max_v_slider_value_changed(self):
        self.max_v = self.ui.max_v_slider.value()
        self.ui.max_v_label.setText(str(self.max_v))

    def opening_kernel_spin_box_value_changed(self):
        if self.ui.opening_kernel_spin_box.value() % 2 == 0:
            self.ui.opening_kernel_spin_box.setValue(
                self.ui.opening_kernel_spin_box.value() + 1)
        self.opening_kernel_size = self.ui.opening_kernel_spin_box.value()
        self.opening_kernel = circular_kernel(self.opening_kernel_size)

    def gaussian_kernel_spin_box_value_changed(self):
        if self.ui.gaussian_kernel_spin_box.value() % 2 == 0:
            self.ui.gaussian_kernel_spin_box.setValue(
                self.ui.gaussian_kernel_spin_box.value() + 1)
        self.gaussian_kernel_size = self.ui.gaussian_kernel_spin_box.value()

    def min_frame_width_spin_box_value_changed(self):
        if (self.ui.min_frame_width_spin_box.value() >=
                self.ui.max_frame_width_spin_box.value()):
            self.ui.min_frame_width_spin_box.setValue(
                self.ui.min_frame_width_spin_box.minimum())
        self.min_frame_width = self.ui.min_frame_width_spin_box.value()

    def max_frame_width_spin_box_value_changed(self):
        if (self.ui.max_frame_width_spin_box.value() <=
                self.ui.min_frame_width_spin_box.value()):
            self.ui.max_frame_width_spin_box.setValue(
                self.ui.max_frame_width_spin_box.maximum())
        self.max_frame_width = self.ui.max_frame_width_spin_box.value()

    def min_frame_height_spin_box_value_changed(self):
        if (self.ui.min_frame_height_spin_box.value() >=
                self.ui.max_frame_height_spin_box.value()):
            self.ui.min_frame_height_spin_box.setValue(
                self.ui.min_frame_height_spin_box.minimum())
        self.min_frame_height = self.ui.min_frame_height_spin_box.value()

    def max_frame_height_spin_box_value_changed(self):
        if (self.ui.max_frame_height_spin_box.value() <=
                self.ui.min_frame_height_spin_box.value()):
            self.ui.max_frame_height_spin_box.setValue(
                self.ui.max_frame_height_spin_box.maximum())
        self.max_frame_height = self.ui.max_frame_height_spin_box.value()

    def start_conveyor_push_button_clicked(self):
        self.ui.conveyor_state_label.setText('ON')
        self.ui.conveyor_state_label.setStyleSheet(
            'border-color: rgb(147, 103, 53);'
            'border-width: 2px;'
            'border-style: solid;'
            'color: rgb(0, 255, 0);')
        self.conveyor.start()

    def stop_conveyor_push_button_clicked(self):
        self.ui.conveyor_state_label.setText('OFF')
        self.ui.conveyor_state_label.setStyleSheet(
            'border-color: rgb(147, 103, 53);'
            'border-width: 2px;'
            'border-style: solid;'
            'color: rgb(255, 0, 0);')
        self.conveyor.stop()

    def start_identifier_push_button_clicked(self):
        self.ui.identifier_state_label.setText('ON')
        self.ui.identifier_state_label.setStyleSheet(
            'border-color: rgb(147, 103, 53);'
            'border-width: 2px;'
            'border-style: solid;'
            'color: rgb(0, 255, 0);')
        self.capture_timer = time.time() - self.capture_timer_delay
        self.identifier_running = True

    def stop_identifier_push_button_clicked(self):
        self.ui.identifier_state_label.setText('OFF')
        self.ui.identifier_state_label.setStyleSheet(
            'border-color: rgb(147, 103, 53);'
            'border-width: 2px;'
            'border-style: solid;'
            'color: rgb(255, 0, 0);')
        self.data_writer.stop()
        self.identifier_running = False

    def capture_line_position_slider_value_changed(self):
        self.capture_line_position = \
            self.ui.capture_line_position_slider.value()
        if self.capture_line_position - self.capture_box_left_width < 0:
            self.ui.capture_box_left_width_spin_box.setValue(
                self.capture_line_position)
        else:
            self.capture_box_left_position = self.capture_line_position \
                                             - self.capture_box_left_width

        if (self.capture_line_position + self.capture_box_right_width >
                self.camera_resolution[0] - 1):
            self.capture_box_right_width_spin_box.setValue(
                self.camera_resolution[0] - 1 - self.capture_line_position)
        else:
            self.capture_box_right_position = self.capture_line_position \
                                              + self.capture_box_right_width

    def capture_box_left_width_spin_box_value_changed(self):
        if (self.capture_line_position -
                self.ui.capture_box_left_width_spin_box.value() < 0):
            self.capture_box_left_width = self.capture_line_position
        else:
            self.capture_box_left_width = \
                self.ui.capture_box_left_width_spin_box.value()
        self.capture_box_left_position = self.capture_line_position \
            - self.capture_box_left_width

    def capture_box_right_width_spin_box_value_changed(self):
        if (self.capture_line_position +
                self.ui.capture_box_right_width_spin_box.value() >
                self.camera_resolution[0] - 1):
            self.capture_box_right_width = self.camera_resolution[0] - 1
        else:
            self.capture_box_right_width = \
                self.ui.capture_box_right_width_spin_box.value()
        self.capture_box_right_position = self.capture_line_position \
            + self.capture_box_right_width

    def update_ui_oranges_info(self):
        # Ultima laranja
        formatted_diameter_text = '{:.2f}mm'.format(
            self.data_writer.oranges[-1].diameter)
        self.ui.last_diameter_label.setText(formatted_diameter_text)
        self.ui.last_color_label.setText(str(
            self.data_writer.oranges[-1].color.value))
        color = self.data_writer.oranges[-1].rgb
        frame_color = QColor(int(color[0]), int(color[1]), int(color[2]))
        self.ui.last_color_frame.setStyleSheet(
            'border-color: rgb(147, 103, 53);'
            'border-width : 2px;'
            'border-style:solid;'
            'background-color: {};'.format(frame_color.name()))

        # Media de todas as laranjas
        formatted_diameter_text = '{:.2f}mm'.format(
            self.data_writer.average_diameter)
        self.ui.average_color_label.setText(str(
            self.data_writer.average_color.value))
        self.ui.average_diameter_label.setText(formatted_diameter_text)
        color = self.data_writer.average_rgb
        frame_color = QColor(int(color[0]), int(color[1]), int(color[2]))
        self.ui.average_color_frame.setStyleSheet(
            'border-color: rgb(147, 103, 53);'
            'border-width : 2px;'
            'border-style:solid;'
            'background-color: {};'.format(frame_color.name()))

        # Numero total de laranjas
        self.ui.number_of_oranges_label.setText(
            str(self.data_writer.quantity))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
