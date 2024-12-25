# epoxy-supervisor

Dataset
-------
- http://test.bevz.space/robot/dataset/epoxy-level-1774.zip
- http://test.bevz.space/robot/dataset/epoxy-level-1140.zip
- http://test.bevz.space/robot/dataset/epoxy-level-2640.zip


Weights
-------
Веса дообученных моделей.

epoxy-supervisor.20241218.pt - Дообученная на датасете из 1774 фото (70/20/10 - тренировка/валидация/тестирование) модель YOLO11n-pose.
Обучение проводилось в GoogleColabe на Tesla T4, заняло 0.403 часа и прошло за 20 эпох.
```
Ultralytics 8.3.51 🚀 Python-3.10.12 torch-2.5.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
YOLO11n-pose summary (fused): 257 layers, 2,664,805 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% 12/12 [00:12<00:00,  1.05s/it]
                   all        353        353          1          1      0.995      0.993          1          1      0.995      0.995
Speed: 0.2ms preprocess, 2.8ms inference, 0.0ms loss, 3.4ms postprocess per image
```
Более полная информация по обучению (параметры запуска, графики качества, примеры валидационных картинкок и пр.) тут - 2024-12-18.train.zip

epoxy-supervisor.20241221.pt - Дообученная на датасете из 1774 фото (70/20/10 - тренировка/валидация/тестирование) модель YOLO11n-pose.
Следующие параметры аугментации отличались от дефолтных. hsv_h=0.7 hsv_s=0.7 hsv_v=0.7 degrees=10 shear=10
Обучение проводилось в GoogleColabe на Tesla T4, заняло 2.218 часа и прошло за 300 эпох.
```
Ultralytics 8.3.52 🚀 Python-3.10.12 torch-2.5.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
YOLO11n-pose summary (fused): 257 layers, 2,664,805 parameters, 0 gradients, 6.6 GFLOPs
                 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% 12/12 [00:05<00:00,  2.39it/s]
                   all        353        353          1          1      0.995      0.993          1          1      0.995      0.995
Speed: 0.3ms preprocess, 3.0ms inference, 0.0ms loss, 2.1ms postprocess per image
```
Более полная информация по обучению (параметры запуска, графики качества, примеры валидационных картинкок и пр.) тут - 2024-12-21.train.zip
