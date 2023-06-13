# Block 1

## Code 1

**Target:**

- Get the set-up right 
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training  & Test Loop

**Results:**

- Parameters: 6.3M
- Best Training Accuracy: 99.96
- Best Test Accuracy: 99.34

**Analysis:**

- Extremely Heavy Model for such a problem
- Model is over-fitting, changing our model in the next step

## Code 2

**Target:**

- Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible.
- No fancy stuff

**Results:**

- Parameters: 78.7k
- Best Training Accuracy: 99.54
- Best Test Accuracy: 99.04

**Analysis:**

- The model is still large, but working. 
- We see over-fitting

## Code 3

**Target:**

- Make the model lighter
**Results:**

- Parameters: 5.1k
- Best Training Accuracy: 98.70
- Best Test Accuracy: 98.66

**Analysis:**

- Good model!
- No over-fitting, model is capable if pushed further

## Code 4

**Target:**

- Add Batch-norm

**Results:**

- Parameters: 5.24k
- Best Training Accuracy: 99.42
- Best Test Accuracy: 98.98

**Analysis:**

- We have started to see over-fitting now. 

## Code 5

**Target:**

- Add regularisation, add dropout of 10 percent to every layer

**Results:**

- Parameters: 5.24k
- Best Training Accuracy: 98.50
- Best Test Accuracy: 99.18

**Analysis:**

- Model is not overfitting at all, now.

## Code 6

**Target:**

- Add gap and increase capacity by adding layer in end

**Results:**

- Parameters: 7.6k
- Best Training Accuracy: 98.50
- Best Test Accuracy: 99.18

**Analysis:**

- Model is not overfitting at all, but I don't think we can push it further to 99.4.