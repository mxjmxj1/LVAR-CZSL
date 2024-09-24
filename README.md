# AOGN-CZSL: an Attributes and Objects Guided Network for Compositional Zero-Shot Learning

This is the official PyTorch code for the paper:

**LVAR-CZSL: Learning Visual Attributes Representation for Compositional Zero-Shot Learning**

***IEEE Transactions on Circuits and Systems for Video Technology***

**Xingjiang Ma**, **Jing Yang***, **Jiacheng Lin**, **Zhenzhe Zheng**, **Shaobo Li**, **Bingqi Hu and Xianghong Tang**.

[**Paper **](https://ieeexplore.ieee.org/document/10638107)**|**[**Code**](https://github.com/mxjmxj1/LVAR-CZSL)

<p align="center">
  <img src="img/Model.png"alt="" align=center />
</p>


## Setup

The model code is implemented based on the PyTorch framework. The experimental environment includes:

- Ubuntu20.04

- Intel(R) Core(TM) i9-12900K CPU
- 128GB RAM
- NVIDIA GeForce RTX 3090Ti GPU

Create a conda environment `lvar` using:

```
conda env create -f environment.yml
conda activate lvar
```

## Preparation

The datasets C-GQA and UT-Zappos50K used in our work need to be prepared before training and testing the model. They can be downloaded through a script:

```
bash utils/download_data.sh
```

The Vaw-CZSL can be downloaded by clicking [here](https://drive.google.com/drive/folders/1LaJnfVv-xjsr87mhgMAtMZ5tfo3v7DLZ?usp=drive_link).

In our method, since we use the pre-trained Wave-MLP as the backbone network of the image feature extractor, we need to load the pre-training file before starting the training. pretrained Wave-MLP can be found [here](https://drive.google.com/drive/u/0/folders/1vGai0PHtyFWuIyEBPd0JN8BZIVxeGIPN).

## Training

**Closed World.** To train LVAR-CZSL model, the command is simply:

```
    python train.py --config CONFIG_FILE
```

where `CONFIG_FILE` is the path to the configuration file of the model 

For example, we train  LVAR-CZSL on the dataset C-GQA in a closed world scenario:

```
    python train.py --config configs/cgqa.yml
```

**Open World.** To train  LVAR-CZSL in the open world scenario, you only need to set the `open-world`  parameter in the configuration file `CONFIG_FILE` to `True`.

## Test

After the above training, a `logs` file will be generated, which contains logs, model parameters and checkpoints. We can test our model through this `logs` file.

 For example, testing  LVAR-CZSL on the C-GQA dataset:

```
python test.py --logpath LOG_DIR
```

where `LOG_DIR` is the directory containing the logs of a model.

## References

If you use this code in your research, please consider citing our paper:

```
@article{ma2024lvar,
  title={LVAR-CZSL: Learning Visual Attributes Representation for Compositional Zero-Shot Learning},
  author={Ma, Xingjiang and Yang, Jing and Lin, Jiacheng and Zheng, Zhenzhe and Li, Shaobo and Hu, Bingqi and Tang, Xianghong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```

**Note:** Our work is based on  [CZSL](https://github.com/ExplainableML/czsl).  If you find those parts useful, please consider citing:

```
@inproceedings{naeem2021learning,
  title={Learning Graph Embeddings for Compositional Zero-shot Learning},
  author={Naeem, MF and Xian, Y and Tombari, F and Akata, Zeynep},
  booktitle={34th IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021},
  organization={IEEE}
}
```

Thanks for open source!

**If you have any questions you can contact us : gs.xjma22@gzu.edu.cn or jyang23@gzu.edu.cn**