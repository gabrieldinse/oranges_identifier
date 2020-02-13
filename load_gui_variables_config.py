# Author: Gabriel Dinse
# File: load_gui_variables_config
# Date: 13/05/2019
# Made with PyCharm

# Standard Library
import json

# Third party modules

# Local application imports


def main():
    config = {
        'color_range':
        {
            'min_h': 0,
            'max_h': 255,
            'min_s': 0,
            'max_s': 255,
            'min_v': 0,
            'max_v': 255
        },
        'kernel_size':
        {
            'gaussian': 5,
            'opening': 5
        },
        'frame_dimension_range':
        {
            'min_width': 0,
            'max_width': 640,
            'min_height': 0,
            'max_height': 480
        },
        'capture_line':
        {
            'position': 300,
            'left_width': 75,
            'right_width': 75
        }
    }

    with open('gui_config.json', 'w') as config_file:
        json.dump(config, config_file)


if __name__ == '__main__':
    main()
