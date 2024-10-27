# MXNet-Vs.-TensorFlow-A-Comparative-Analysis-for-Intel-Image-Classification
Project Work for CIS552 - Advanced Mathematical Statistics, Spring 2024: Comparative analysis of MXNet and TensorFlow frameworks for Intel Image Classification using various CNN architectures.

This project provides a comparative analysis of MXNet and TensorFlow frameworks, including architectures such as LeNet, AlexNet, and VGG16 applied to a dataset sourced from Kaggle.

## Project Structure

```bash
├── main.py # Contains implementations of LeNet, AlexNet, and custom CNN models
├── VGG_image_prediction.ipynb # Jupyter notebook for VGG16 image classification using TensorFlow
├── VGG_TensorFlow # Folder containing VGG16 implementation in TensorFlow
├── intel_data # Folder with Intel image dataset
└── MTH522_Final_Project_Report.pdf # Project report detailing methodology, results, and analysis
```

## Requirements

To set up and run this project, install the following packages:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- MXNet
- Keras
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using:

```bash
pip install tensorflow mxnet keras numpy matplotlib scikit-learn
```

## Data Preprocessing

To make the dataset manageable, it is preprocessed as follows:

- **Original Dataset**: [Kaggle Intel Image Classification Dataset](https://www.kaggle.com/puneet6060/intel-image-classification)
- **Steps**:
  - **Reduce dataset size** to ensure it is manageable for training on standard hardware.
  - **Organize images** into folders for training and testing, distributed across six classes: `buildings`, `forests`, `glaciers`, `mountains`, `seas`, and `streets`.
  - **Store preprocessed data** in the `intel_Data` directory.


## Running the Models

### 1. `main.py`
This script contains CNN models for Intel image classification, including LeNet, AlexNet, and custom CNN models, excluding VGG16.

- **Usage**:
  ```bash
  python main.py
  ```

### 2. `VGG_image_prediction.ipynb`: VGG16 Model
This Jupyter notebook trains a VGG16 model using TensorFlow. The notebook compares VGG16's performance with other CNN models.

- **To Run:** Open the `VGG_image_prediction.ipynb` notebook and run all cells to train the VGG16 model on the preprocessed dataset.
  
## Analysis
Results, including model accuracy, training times, and statistical tests comparing MXNet and TensorFlow, are provided in the MTH522_Final_Project_Report.pdf.

## Future Work
Suggested future directions include:

- Training for additional epochs to ensure a fair comparison across models.
- Tune hyperparameters to optimize model performance.

## References

- **Intel Image Data**  
  Retrieved from [Kaggle Intel Image Classification Dataset](https://www.kaggle.com/puneet6060/intel-image-classification)

- **Pretrained VGG16**  
  Retrieved from [TensorFlow VGG16 Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16)

- **AlexNet Model Architecture**  
  Retrieved from [AlexNet and Image Classification](https://www.example.com/alexnet-architecture)

- **AlexNet**  
  Retrieved from [GitHub - Intel Image Classification](https://www.github.com/intel-image-classification)

- **LeNet**  
  Retrieved from [Image Classification Series - LeNet CNN Architecture](https://www.example.com/lenet-architecture)

