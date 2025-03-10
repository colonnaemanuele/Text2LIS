# Text2LIS: Towards Italian Sign Language generation for digital humans
If you need to create a new sign langugage dataset from webscraping check this [link](https://github.com/CarpiDiem98/downloader).

This repository contains the official code for the paper [Text-to-LIS]([url](https://www.researchgate.net/publication/387398556_Towards_Italian_Sign_Language_Generation_for_digital_humans)): Towards Italian Sign Language generation for digital humans

![Text2LISModel](assets/Text2LISModel.svg)

### Virtual Environment Creation (Conda)
```bash
conda create -n text2lis python=3.10
conda activate text2lis
```
### Virtual Environment Creation (pip)
```bash
python -m venv .venv
source .venv/bin/activate  # Su sistemi Unix o MacOS
.venv\Scripts\activate   # Su sistemi Windows
```

### Install dependences
```bash
pip install -r requirements.txt
```

### Inference
```bash
python __main__.py inference 
```

### Training 
```bash
python __main__.py train
```
