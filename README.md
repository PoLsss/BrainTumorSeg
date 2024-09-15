This repository is used to store the codes of our paper [Enhancing brain tumor segmentation in MRI images: A hybrid approach using UNet, Attention Mechanisms, and Transformers](https://doi.org/10.1016/j.eij.2024.100528)


## Dataset
BraTS has always been focusing on the evaluation of state-of-the-art methods for the segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) scans.

We used the BraTS2019 dataset ([kaggle](https://www.kaggle.com/datasets/debobratachakraborty/brats2019-dataset), [CBICA](https://www.med.upenn.edu/cbica/brats2019/data.html)) and BraTS2020 dataset ([kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data?resource=download), [CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html))to conduct the study.


## Model
<img src="https://github.com/user-attachments/assets/eadab797-2017-4099-93f3-685ff3244961" width="500" alt="Description ò model">
We use a 3D U-Net model backbone combined with a Contextual Transformer (CoT) and Double Attention (DA)

## Training process:
1. Start at the main.ipynb file
2. Change data paths, LOGGER, output model
3. Change models to suit their intended use if necessary

## References
We refer to:
1. CoT model: [Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf).
2. DA model: [A^2-Nets: Double Attention Networks](https://proceedings.neurips.cc/paper_files/paper/2018/file/e165421110ba03099a1c0393373c5b43-Paper.pdf).

## Citation
If our work is useful for you, please cite as:
```
@article{nguyen2024enhancing,
  title={Enhancing brain tumor segmentation in MRI images: A hybrid approach using UNet, attention mechanisms, and transformers},
  author={Nguyen-Tat, Thien B and Nguyen, Thien-Qua T and Nguyen, Hieu-Nghia and Ngo, Vuong M},
  journal={Egyptian Informatics Journal},
  volume={27},
  pages={100528},
  year={2024},
  publisher={Elsevier}
}
```

For any questions, please contact: Mr. Thien Qua T.Nguyen at 20521783@gm.uit.edu.vn 
