# HW0: Alohomora

## Phase 1: Shake My boundary

### Steps to run the code:

To run the PBLite boundary detection, use the following command:

```bash
python Wrapper.py
```
Wrappery.py reads all the input images from "BSDS500" folder and all the ouptuts are stored in the "Outputs" folder. Both the folder should be at the same hierarchy as Wrapper.py

## Phase 2: Deep Dive on Deep Learning

### Steps to run the code:

Train the model

```bash
python Train.py --NumEpochs 40 --MiniBatchSize 256 --ModelType Baseline --CustomLogs PATH_TO_CUSTOMLOGS
```
There are 5 model types: 'Baseline', 'BatchNorm', 'ResNet', 'ResNeXt', 'DenseNet'. This argument selects the model type for training.

--CustomLogs provides path to store model performance plots and best checkpoints dynamically during training.

Test the model

```bash
python Test.py --ModelPath PATH_TO_CHECKPOINT --SelectTestSet False --ModelType Baseline
```
--SelectTestSet selects the dataset on which the test operation is to be performed i.e. either TrainSet or TestSet. Default is TestSet.
