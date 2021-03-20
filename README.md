# 50.039-Deep-Learning
For Module 50.039

Students:

- Ong Xiang Qian - 1002646
- Glenn Chia Jin Wee - 1003118

# 1. Table of contents <a name="TOC"></a>

1. [Table of contents](#TOC)
2. [Directory structure](#DS)
3. [Final files and instructions on running them](#INSTRUCTIONS)
4. [Data analysis](#DA)
5. [Proposed model and design](#MODEL)
6. [Model Summary](#MODEL2)
7. [Evaluation of results](#EVAL)
8. [Challenges of predicting covid and non-covid](#CHALLENGE)
9. [Critically evaluating desired model](#OVERALL)
10. [How doctors evaluate covid](#DOCTORS)

# 2. Directory structure <a name="DS"></a>

```
- notebooks
  |_ colab
    |_ boilerplate
      |_ custom_dataset_dataloader_cascade.ipynb
      |_ custom_dataset_dataloader.ipynb
    |_ experiments
    |_ final
  |_ references
    |_ custom_dataset_dataloader_demo.ipynb
- instructions # contains the small project instructions 
  |_ Small_Project_instructions.pdf
README.md # contains the overview of the project and explanations for the different requirements
```

# 3. Final files and instructions on running them <a name="INSTRUCTIONS"></a>

## 3.1 Training the final model

## 3.2 Loading and testing the trained model

# 4. Data analysis <a name="DA"></a>

## 4.1 Custom Datasets and Dataloader

### 4.1.1 Binary cascade problem

### 4.1.2 Three-class problem

## 4.2 Distribution of data among classes and analysis

## 4.3 Data processing

### 4.3.1 Typical processing operations

### 4.3.2 Other potential pre-processing operations

## 4.4 Possible data augmentations

Reference link on how Doctors diagnose Covid-19: [The role of chest radiography in confirming covid-19 pneumonia](https://www.bmj.com/content/370/bmj.m2426#:~:text=Most%20people%20with%20covid%2D19,those%20with%20covid%2D19%20pneumonia.)

Relevant insight:

- "covid-19 pneumonia changes are mostly **bilateral on chest radiographs** (72.9%, 95% confidence interval 58.6 to 87.1) and have **ground glass opacity** in 68.5% of cases (95% CI 51.8 to 85.2)"
- The paper provides several images and analyzes them. From the images, it seems that the target areas are in the 4 corners of the lungs, perhaps data augmentation can perform crops to those areas

How it can be used in the model?

- Problem: The data classes are imbalanced for covid and non-covid and that the model has more problems differentiating the 2 compared to differentiating normal and infected.
  Solution: Since covid-19 is `bilateral` meaning that it affects both sides of the lungs at a high probability. Perhaps mirroring images could be a possible approach to increase the size of the dataset for covid images, giving the model a more balanced dataset and more covid images to recognize

# 5. Proposed model <a name="MODEL"></a>

## 5.1 3-class classifier vs 2 binary classifiers

**<u>Differences between the 2 architectures</u>**

**<u>Why we chose the 2 binary classifier approach</u>**

## 5.2 2 2 Binary classifiers architecture design

### 5.2.1 Referencing literature and traditional well-performing models

<u>**Naïve single convolution model**</u>

**<u>Re-implementing a scaled down version of resnet</u>**

<u>**Models from literature that tackled similar problems**</u>

### 5.2.2 Model parameters

Layers, channels, kernel size

### 5.2.3 Mini-batch size

**<u>Recommendation by literature</u>**

**<u>Experimenting with different batch-sizes</u>**

### 5.2.4 Loss function

<u>**Why cross-entropy loss?**</u>

<u>**Experimenting with weighted cross-entropy to account for imbalanced classes**</u>

## 5.3 Optimizer

### 5.3.1 Optimizer selection

<u>**Adam vs AdamW theory**</u>

<u>**Adam vs AdamW empirical**</u>

### 5.3.2 Regularization - Weight Decay

### 5.3.3 Learning rate

**<u>Recommendation by literature</u>**

**<u>Experimenting with different learning rates</u>**

### 5.3.4 Scheduled learning rate

## 5.4 Model parameters

**<u>Number of epochs</u>**

## 5.5 Implementing checkpoints

# 6. Model Summary <a name="MODEL2"></a>

Recapitulation

# 7. Evaluation <a name="EVAL"></a>

## 7.1 Learning curves

Loss vs epochs

Accuracy vs epochs

## 7.2 Key metrics and considerations

<u>**Confusion matrix**</u>

<u>**Recall and precision**</u>

<u>**F1 score**</u>

## 7.3 Accuracy and image diagrams

## 7.4 Investigating failures with feature maps

# 8. Challenges of predictions <a name="CHALLENGE"></a>

## 8.1 Differentiating covid and non-covid

# 9. Overall - what is the better model, accuracy vs low true negatives/false positives rates on certain classes <a name="OVERALL"></a>


# 10. How doctors diagnose infections based on x-rays. <a name="DOCTORS"></a>

Reference link: [How accurate is chest imaging for diagnosing COVID-19?](https://www.cochrane.org/CD013639/INFECTN_how-accurate-chest-imaging-diagnosing-covid-19)

- Defines X-Rays as "X-rays (radiography) use radiation to produce a 2-D image"
- This paper pooled results from several sources (if there were 4 or more results). For Chest X-rays, it found that the Chest X-ray "correctly diagnosed COVID-19 in 80.6% of the people who had COVID-19. However it incorrectly identified COVID-19 in 28.5% of the people who did not have COVID-19"

Critical insight

- It seems that classifications for X-rays have room for improvement, the numbers indicated above would provide a baseline for any model's performance

Citation:

> Islam N, Ebrahimzadeh S, Salameh J-P, Kazi S, Fabiano N, Treanor L, Absi M, Hallgrimson Z, Leeflang MMG, Hooft L, van der Pol CB, Prager R, Hare SS, Dennie C, Spijker R, Deeks JJ, Dinnes J, Jenniskens K, Korevaar DA, Cohen JF, Van den Bruel A, Takwoingi Y, van de Wijgert J, Damen JAAG, Wang J, McInnes MDF, Cochrane COVID-19 Diagnostic Test Accuracy Group. Thoracic imaging tests for the diagnosis of COVID-19. Cochrane Database of Systematic Reviews 2021, Issue 3. Art. No.: CD013639. DOI: 10.1002/14651858.CD013639.pub4.

Reference link 2: [The role of chest radiography in confirming covid-19 pneumonia](https://www.bmj.com/content/370/bmj.m2426#:~:text=Most%20people%20with%20covid%2D19,those%20with%20covid%2D19%20pneumonia.)

- "No single feature of covid-19 pneumonia on a chest radiograph is specific or diagnostic, but a combination of multifocal peripheral lung changes of ground glass opacity and/or consolidation, which are most commonly bilateral, may be present"
- "Diagnosis might be complicated as covid-19 pneumonia may or may not be visible on chest radiograph"
- "Most patients with covid-19 infection have a mild illness and do not develop pneumonia"
- Diagnose
  1. **"Like other pneumonias, covid-19 pneumonia causes the density of the lungs to increase**. This may be seen as **whiteness in the lungs** on radiography which, depending on the severity of the pneumonia, **obscures the lung markings** that are normally seen; however, this may be delayed in appearing or absent."
  2. "Review the radiograph systematically, looking for abnormalities of the **heart, mediastinum, lungs, diaphragm, and ribs**,[9](https://www.bmj.com/content/370/bmj.m2426#ref-9) and remembering that radiographic changes of covid-19 pneumonia can be **subtle or absent**."
  3. "covid-19 pneumonia changes are mostly **bilateral on chest radiographs** (72.9%, 95% confidence interval 58.6 to 87.1) and have **ground glass opacity** in 68.5% of cases (95% CI 51.8 to 85.2)"
  4. "common manifestations and patterns of **lung abnormality** seen on portable chest radiography in covid-19 patients"
- How it can be used in the model?
  1. The paper provides several images and analyzes them. From the images, it seems that the target areas are in the 4 corners of the lungs, perhaps data augmentation can perform crops to those areas
  2. In addition, covid-19 is "bilateral" meaning that it affects both sides of the lungs at a high probability. Perhaps mirroring images could be a possible approach to increase the size of the dataset.

Citation:

```bib
@article {Cleverleym2426,
	author = {Cleverley, Joanne and Piper, James and Jones, Melvyn M},
	title = {The role of chest radiography in confirming covid-19 pneumonia},
	volume = {370},
	elocation-id = {m2426},
	year = {2020},
	doi = {10.1136/bmj.m2426},
	publisher = {BMJ Publishing Group Ltd},
	URL = {https://www.bmj.com/content/370/bmj.m2426},
	eprint = {https://www.bmj.com/content/370/bmj.m2426.full.pdf},
	journal = {BMJ}
}
```


# 11. Common bug fixes

## Bug fix 1

**<u>Issue</u>**

Erroneous Code

```python
train_loss  = criterion(output, y_train_batch)
```

Error: `RuntimeError: 1D target tensor expected, multi-target not supported`

<u>**Background information**</u>

`criterion` is defined as `criterion = nn.CrossEntropyLoss()`

Tensors were initialized as `[1 0 0]` in the dataset class' `__getitem`. This was then passed to the data loader in `train_loader = DataLoader(ld_train, batch_size = bs_val, shuffle = True)` for the respective train, validation, test.

**<u>Explanation</u>**

The following link at [Pytorch discussion - RuntimeError: multi-target not supported](https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/3) explains it

This issue is caused by  because `CrossEntropyLoss` does not expect a one-hot encoded vector as the target, but class indices: The input is expected to contain scores for each class. Input has to be a 2D Tensor of size (minibatch, C). This criterion expects a class index (0 to C-1) as the target for each value of a 1D tensor of size minibatch

Example inputs

```python
loss = nn.CrossEntropyLoss()
# Input: (N, C). C is the number of classes
input = torch.randn(3, 5, requires_grad=True)
'''
tensor([[-0.9296, -0.6807,  0.1294,  0.0215,  1.1662],
        [-2.0247, -1.5964,  1.4188, -1.4765,  0.7276],
        [-0.3432, -2.3248,  0.2816, -0.4983, -0.7904]], requires_grad=True)
'''
target = torch.empty(3, dtype=torch.long).random_(5)
'''
tensor([2, 4, 1])
'''
output = loss(input, target)
'''
tensor(2.1256, grad_fn=<NllLossBackward>)
'''
```

**<u>Fix</u>**

**Possible fix 1:**

This fix was shown in the link. Change the loss to the following:

```python
loss = criterion(outputs, torch.max(y_train_batch, 1)[1])
```

What `torch.max` does. Refer to [PyTorch docs TORCH.MAX](https://pytorch.org/docs/stable/generated/torch.max.html)

```python
for k, v in train_loader:
    print(v)
    '''
    tensor([[0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.]])
    '''
    print(torch.max(v))
    '''
    tensor(1.)
    '''
    print(torch.max(v,1))
    '''
    values=tensor([1., 1., 1., 1.]),
    indices=tensor([1, 0, 1, 1]))
    '''
    print(torch.max(v,1)[1])
    '''
    tensor([1, 0, 1, 1]) 
    # this is the index of the max value per sample in the batch
    # example: first sample, the max is at index 1. Hence the first element in the tensor
    indicates that sample 1's max is at index 1
    '''
    # Forced stop
    break
```

