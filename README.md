# MoveSim
Codes for paper in KDD'20 (AI for COVID-19): Learning to Simulate Human Mobility

## Datasets

* **GeoLife**: This GPS trajectory dataset was collected by the MSRA GeoLife project with 182 users in a period of over five years.  
  * Link: https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F

## Requirements

* **Python 3.6**
* **PyTorch >= 1.0**
* Numpy
* Scipy

## Usage

Pretrain and train new model:

`python main.py --pretrain --data=geolife`

Evaluation with generated data:

`python evaluation.py`
