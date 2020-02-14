## Oranges identifier
This program is used to extract features of oranges passing by a conveyor belt. The features are:
    - diameter (mm);
    - RGB;
    - CCI (Citrus Color Index);
    - color (a string with human readable value of color, like: 'green',
    'yellow', 'orange', etc), based on CCI.

## Libraries
The idenfitier works using techniques of image processing and computer vision, based on the Opencv library to extract the image features, and using Qt (PyQt) to create the GUI (Graphical User Interface). The conveyor is controlled by an Arduino using QSerialPort from PyQt, and the data is stored in an CSV file.