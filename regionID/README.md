# Region ID Classification
The task involves training a convolutional neural network to predict one of 15 region classes based on input images.

## Model Overview

- **Architecture**:  
  A ResNet-50 model pretrained on ImageNet is used as the backbone.

- **Modification**:  
  The final fully connected (FC) layer is replaced with a new `Linear` layer to output 15 classes, corresponding to Region IDs.

- **Fine-tuning**:  
  All layers are unfrozen and fine-tuned (`requires_grad = True`) to fully adapt the model to the dataset.

## Training Configuration

- **Loss Function**:  
  `CrossEntropyLoss` with label smoothing (factor = 0.1) is used to improve generalization and training stability.

- **Optimizer**:  
  Adam optimizer with a learning rate of 1e-4.

- **Scheduler**:  
  A `StepLR` scheduler is applied with:
  - Step size: 10 epochs  
  - Gamma (decay rate): 0.1

- **Batch Size**:  
  64 for training, validation, and test sets.

- **Epochs**:  
  Maximum of 20 epochs with early stopping (patience = 4 epochs) based on validation accuracy.

## Data Preprocessing

- **Training Transformations**:
  - Resize to 256×256  
  - Random horizontal flip  
  - Random rotation (±10 degrees)  
  - Color jitter (brightness, contrast, saturation)  
  - Convert to tensor

- **Validation/Test Transformations**:
  - Resize to 256×256  
  - Convert to tensor

## Output

- Region ID predictions for both validation and test sets are generated using the best model.
- The final combined predictions are saved in a CSV file:  
  `solution.csv`
