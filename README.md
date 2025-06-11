# IR-DETR

Automatic target recognition is critical in infrared imaging guidance. Detecting small targets is challenging due to poorly defined silhouettes, low SNR and complex backgrounds. Traditional methods and deep learning approaches often struggle with SNR, target size, and insufficient feature extraction. To address these issues, we propose the IR-DETR network family based on DETR. Specifically, we design a MEB for efficient feature extraction of infrared small targets using a simple and efficient network structure. We then fuse local and global spatial features through LWAIFI. Our proposed TRC3 expands the model’s receptive field, which improves the detection accuracy of infrared targets while also reducing the model’s overall size. Additionally, we introduce the Adaptive Max-Sigmoid activation function to address the shortcomings of previous activation functions in small target detection. Finally, by incorporating NWD loss function, we further improve the detection performance of IR-DETR for small infrared targets. Compared to SOTA models on ATR dataset, NUDT-SIRST dataset and IRSTD-1k dataset, the IR-DETR network family achieved the best performance. 

<div align="center">
  <img src=https://github.com/Eason215xB/IR-DETR/blob/main/1.jpg>
</div>
 
## Installation
*[conda]* - Clone the repository and then create and activate a `IR-DETR` conda environment using the provided environment definition:

```shell
conda create -n irdetr python=3.8
conda activate irdetr
```

*[pip]* - Clone the repository and then use the provided `requirements.txt` to install the dependencies:

```shell
pip install -r requirements.txt
```
## Data preparation

*[ATR](1)*

*[NUDT-SIRST](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt)* (Extraction Code: nudt)

*[IRSTD-1K](https://drive.google.com/file/d/1JoGDGF96v4CncKZprDnoIor0k1opaLZa/view)*



## Usage

Train:

```shell
python train.py --data.yaml --IR-DETR-B0.yaml
```

Val:

```shell
python val.py --data.yaml --best.pt
```
## Reference

- If you found our work useful in your research, please consider citing our works at:

```tex

@article{yue2025ir,
  title={IR-DETR: An efficient detection transformer with multi-layer feature fusion for infrared small targets},
  author={Yue, Xinbo and Liu, Liwei and Du, Yue},
  journal={Infrared Physics \& Technology},
  pages={105926},
  year={2025},
  publisher={Elsevier}
}


