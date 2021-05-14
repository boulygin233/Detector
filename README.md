Самая простая версия детектора. Важно, чтобы в папке с ним лежал его чекпоинт (файл "checkpoint.pth.tar").
Запускаем с помощью команды "python3 detect.py /path/to/picture/"
Печатает словарь формата 
{'boxes':[[]], #(n_boxes, 4) в формате левый нижний - правый верхний
'labels': [], #(n_boxes) - циферка - лэйбл (1==рука, 2==товар)
'scores': [], #(n_boxes) - уверенность в своём предсказании}
#todo изменить формат выплевывания в удобный

Checkpoint - https://drive.google.com/file/d/13w-f5c6gTMTVuMRl3-GjXky4OgZjujGz/view?usp=sharing
