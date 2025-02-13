# DETR-PYTORCH
This is the simple implementation of the Detection Transformer model from scratch on Pytorch. Also, you can train your custom dataset on it.
## Model Architecture
![DETR_framework](https://github.com/user-attachments/assets/3935f869-6ba4-4769-995c-d9aadde99245)
This is primarily trained on VOC 2007 dataset and have data loader according to it.
## To train on custom dataset
Change the dataset_params and num_classes in [config.yaml](config.yaml) file and then update the classes in [Dataloader.py](Dataloader.py) line 63.
```
trainer.py --config path_to_config_file
```
Currently, the backbone is ResNet50 and to train it set train_backbone to True in  [config.yaml](config.yaml) file. If you plan to change the backbone, be mindful of the feature map dimensions. 
Currently, the feature map has 2048 channels after exiting the backbone layer, so adjust accordingly.

Training with auxiliary loss hasn't been implemented yet. However, if you modify the trainer.py file, please submit a pull request, it will great to hear from you.
## Results
![output_detr_1](https://github.com/user-attachments/assets/29cefca1-e6fd-445a-9f19-e3ab549abd18)
![output_detr_6](https://github.com/user-attachments/assets/5747bf6b-cf8c-41d3-9cb4-abf9cabc9a29)
![output_detr_9](https://github.com/user-attachments/assets/905af354-80fc-44bd-9b02-ffbea2ae9869)
![output_detr_7](https://github.com/user-attachments/assets/0435357e-47ac-4209-8e87-109987383b75)
## Citations
```
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```
