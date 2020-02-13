# Author: Gabriel Dinse
# File: oranges_data_storager
# Date: 06/06/2019
# Made with PyCharm

"""
Library for extracting oranges information, like CCI and color. It also
provides ways to save all oranges features into a CSV file.
"""

# Standard Library
from enum import Enum
from datetime import datetime
import csv
import os

# Third party modules
import numpy as np

# Local application imports


# Erros
class Error(Exception):
    pass


class InvalidIlluminant(Error):
    pass


class NegativeNumberOfOranges(Error):
    pass


class OrangeColor(Enum):
    """ Nomeclatura das cores. """

    GREEN = 'Green'
    YELLOWISH_GREEN = 'Yellowish Green'
    YELLOW = 'Yellow'
    LIGHT_ORANGE = 'Light Orange'
    ORANGE = 'Orange'


def verify_color(cci):
    """ Retorna o nome da cor baseado no CCI da laranja. """

    if cci < -6.0:
        return OrangeColor.GREEN
    elif -6.0 <= cci < -1.0:
        return OrangeColor.YELLOWISH_GREEN
    elif -1.0 <= cci < 2.7:
        return OrangeColor.YELLOW
    elif 2.7 <= cci < 6.0:
        return OrangeColor.LIGHT_ORANGE
    else:  # cci >= 6
        return OrangeColor.ORANGE


def rgb_to_xyz(rgb_color):
    """ Transforma o espaco de cores de RGB para XYZ. """

    r = (rgb_color[0] / 255)
    g = (rgb_color[1] / 255)
    b = (rgb_color[2] / 255)

    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r = r / 12.92

    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g = g / 12.92

    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b = b / 12.92

    r = r * 100
    g = g * 100
    b = b * 100
    x = (r * 0.4124) + (g * 0.3576) + (b * 0.1805)
    y = (r * 0.2126) + (g * 0.7152) + (b * 0.0722)
    z = (r * 0.0193) + (g * 0.1192) + (b * 0.9505)

    return x, y, z


def xyz_to_hunterlab(xyz_color, illuminant):
    """ Transforma o espaco de cores de XYZ para HunterLab. """

    if illuminant == 'D65':
        ka = 172.10
        kb = 66.70
        xr = 94.83
        yr = 100.0
        zr = 107.38
    elif illuminant == 'D60':
        ka = 172.45
        kb = 64.28
        xr = 95.21
        yr = 100.0
        zr = 99.60
    elif illuminant == 'D75':
        ka = 171.76
        kb = 70.76
        xr = 94.45
        yr = 100.0
        zr = 120.70
    else:
        raise InvalidIlluminant("Wrong Illuminant. Value must be: 'D60',"
                                " 'D65' or 'D75'.")

    l = 100.0 * np.sqrt(xyz_color[1] / yr)
    a = ka * (((xyz_color[0] / xr) - (xyz_color[1] / yr)) / np.sqrt(
        xyz_color[1] / yr))
    b = kb * (((xyz_color[1] / yr) - (xyz_color[2] / zr)) / np.sqrt(
        xyz_color[1] / yr))
    return l, a, b


def rgb_to_hunterlab(rgb_color, illuminant):
    """ Transforma o espaco de cores de RGB para HunterLab. """

    return xyz_to_hunterlab(rgb_to_xyz(rgb_color), illuminant)


# Citrus Color Index
def calculate_cci(hunterlab):
    """
    Calcula o CCI baseado nas componentes de cor L, a e b do espaco de
    cores HunterLab.
    """
    return 1000 * (hunterlab[1]) / (hunterlab[0] * hunterlab[2])


# campos usados para armazenas e ler as informacoes das laranjas
field_names = ['diameter', 'cci', 'color', 'rgb']


class OrangeInfo:
    """ Classe que representa as informacoes extraidas de uma laranja. """
    def __init__(self, diameter, rgb, illuminant='D65'):
        self.diameter = diameter
        self.rgb = rgb
        self.cci = calculate_cci(rgb_to_hunterlab(rgb, illuminant))
        self.color = verify_color(self.cci)


# Identificacao das laranjas
class OrangesDataWriter:
    """
    Classe responsavel por organizar e salvar as caracteristicas
    extraidas dos frames.
    """

    def __init__(self, folder, illuminant='D65'):
        self.started = False
        self.illuminant = illuminant

        # Arquivo para salvar os dados da colheita
        self.harvest_folder_name = folder
        self.harvest_subfolder_name = 'harvest'

        # Dados da colheita
        self.quantity = 0
        self.oranges = []

    def stop(self):
        """
        Cria um arquivo no formato CSV para guardar todas as
        caracteristicas das laranjas.
        """

        if self.started and self.quantity > 0:
            with open(self.harvest_filepath, 'w', newline='') as harvest_file:
                self.harvest_file_writer = csv.DictWriter(
                    harvest_file, fieldnames=field_names)
                self.harvest_file_writer.writeheader()

                # Valores individuais
                for i in range(self.quantity):
                    self.harvest_file_writer.writerow(
                        {
                            'diameter': self.oranges[i].diameter,
                            'cci': self.oranges[i].cci,
                            'color': self.oranges[i].color.value,
                            'rgb': self.oranges[i].rgb.tolist()
                        }
                    )

                # Media
                self.harvest_file_writer.writerow(
                    {
                        'diameter': self.average_diameter,
                        'cci': self.average_cci,
                        'color': self.average_color.value,
                        'rgb': self.average_rgb.tolist()
                    }
                )

            # Reinicia as variaveis
            self.quantity = 0
            self.oranges.clear()

        self.started = False

    def add(self, diameter, rgb):
        """ Adiciona uma laranja ao banco de dados. """

        if not self.started:
            self.start_datetime = datetime.now()
            harvest_filename = self.start_datetime.strftime(
                'totalharvest_{}_%H-%M-%S.%d-%m-%Y.csv'.format(self.illuminant))
            self.harvest_filepath = os.path.join(self.harvest_folder_name,
                                                 self.harvest_subfolder_name,
                                                 harvest_filename)
            self.started = True

        self.oranges.append(OrangeInfo(diameter, rgb, self.illuminant))
        self.quantity += 1

    @property
    def average_cci(self):
        if self.quantity:
            total = 0
            for orange_info in self.oranges:
                total += orange_info.cci
            return total / self.quantity
        else:
            raise NegativeNumberOfOranges("Number of oranges must be "
                                          "greater than zero.")

    @average_cci.setter
    def average_cci(self, value):
        raise AttributeError("Can't set this attribute.")

    @property
    def average_diameter(self):
        if self.quantity:
            total = 0
            for orange_info in self.oranges:
                total += orange_info.diameter
            return total / self.quantity
        else:
            raise NegativeNumberOfOranges("Number of oranges must be "
                                          "greater than zero.")

    @average_diameter.setter
    def average_diameter(self, value):
        raise AttributeError("Can't set this attribute.")

    @property
    def average_rgb(self):
        if self.quantity:
            total = 0
            for orange_info in self.oranges:
                total += np.array(orange_info.rgb, dtype=np.float)
            return np.array(total / self.quantity, dtype=np.uint8)
        else:
            raise NegativeNumberOfOranges("Number of oranges must be "
                                          "greater than zero.")

    @average_rgb.setter
    def average_rgb(self, value):
        raise AttributeError("Can't set this attribute.")

    @property
    def average_color(self):
        if self.quantity:
            return verify_color(self.average_cci)
        else:
            raise NegativeNumberOfOranges("Number of oranges must be "
                                          "greater than zero.")

    @average_color.setter
    def average_color(self, value):
        raise AttributeError("Can't set this attribute.")


class OrangesDataReader:
    def __init__(self, filepath):
        self.filepath = filepath
        path, filename = os.path.split(filepath)
        name, self.illuminant, datetime_and_ext = filename.split('_')

        # Can access attributes like: hour, min, day, month, etc.
        self.time = datetime.strptime(
            datetime_and_ext, '%H-%M-%S.%d-%m-%Y.csv')

        self.file = open(filepath, 'r')
        self.reader = csv.DictReader(file, fieldnames=field_names)

    def __iter__(self):
        return self.reader

    def __next__(self):
        try:
            return next(self.reader)
        except StopIteration:
            self.file.close()
            raise

