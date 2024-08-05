import torch
import torchvision

# Функция для декодирования результатов детекции для одного изображения
def decode_result(datum, threshold=1.0, cell=8, iou_threshold=0.7):
    # Структура для хранения результатов: координаты боксов, оценки и метки классов
    bboxes = {'boxes': [], 'scores': [], 'labels': []}
    
    # Разделение данных на два слоя: первый слой для класса gun (label 0), второй слой - пустой (label 1)
    datum = {0: datum[:5, :, :], 1: datum[5:, :, :]}

    # Обработка каждого класса (здесь только один класс - gun)
    for label in [0, 1]:
        # Создание маски для отсеивания значений ниже порога
        mask = (datum[label][0, :, :] >= threshold)
        
        # Генерация координат ячеек по оси x и y для маски
        x_cell = torch.arange(mask.shape[1], device=datum[label].device)
        y_cell = torch.arange(mask.shape[0], device=datum[label].device)
        
        # Создание сетки координат ячеек
        y_cell, x_cell = torch.meshgrid(y_cell, x_cell)
        
        # Применение маски для получения значений координат ячеек
        x_cell = x_cell[mask]
        y_cell = y_cell[mask]
        
        # Извлечение смещений координат
        x_shift = datum[label][2, :, :][mask]
        y_shift = datum[label][1, :, :][mask]
        
        # Вычисление абсолютных координат центров боксов
        x = (x_cell + x_shift) * cell
        y = (y_cell + y_shift) * cell
        
        # Вычисление ширины и высоты боксов с применением экспоненциальной функции
        w = datum[label][4, :, :][mask].exp() * cell
        h = datum[label][3, :, :][mask].exp() * cell
        
        # Извлечение оценок (scores) для текущих боксов
        scores = datum[label][0, :, :][mask]
        
        # Формирование итоговых боксов и добавление их в структуру результатов
        for index in range(len(x)):
            bboxes['boxes'].append([
                x[index].item() - w[index].item() / 2,
                y[index].item() - h[index].item() / 2,
                x[index].item() + w[index].item() / 2,
                y[index].item() + h[index].item() / 2,
            ])
            bboxes['scores'].append(scores[index].item())
            bboxes['labels'].append(label)
    
    # Применение Non-Maximum Suppression (NMS) для устранения пересекающихся боксов
    if len(bboxes['boxes']) > 0:
        bboxes['boxes'] = torch.tensor(bboxes['boxes'], dtype=torch.float32)
        bboxes['scores'] = torch.tensor(bboxes['scores'], dtype=torch.float32)
        bboxes['labels'] = torch.tensor(bboxes['labels'], dtype=torch.int64)
        
        to_keep = torchvision.ops.nms(bboxes['boxes'], 
                                      bboxes['scores'], 
                                      iou_threshold=iou_threshold)
        bboxes['boxes'] = bboxes['boxes'][to_keep]
        bboxes['scores'] = bboxes['scores'][to_keep]
        bboxes['labels'] = bboxes['labels'][to_keep]
    else:
        bboxes['boxes'] = torch.empty((0, 4), dtype=torch.float32)
        bboxes['scores'] = torch.empty((0,), dtype=torch.float32)
        bboxes['labels'] = torch.empty((0,), dtype=torch.int64)

    return bboxes

# Функция для декодирования результатов детекции для батча изображений
def decode_batch(batch, threshold=1.0, iou_threshold=0.7):
    res = []
    # Обработка каждого изображения в батче
    for index in range(batch.shape[0]):
        res.append(decode_result(batch[index],
                                 threshold=threshold,
                                 iou_threshold=iou_threshold))
    return res
