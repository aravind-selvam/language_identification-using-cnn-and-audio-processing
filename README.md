# Language Identification using CNN PyTorch

#### Language and Libraries

<p>
<a><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" alt="python"/></a>
<a><img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/></a>
<a><img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/></a>
<a><img src="https://img.shields.io/badge/PyAudio-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pyaudio"/></a>
<a><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pytorch"/></a>
<a><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)" alt="docker"/></a>
<a><img src="https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white" alt="gcp"/></a>
<a><img src="https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white" alt="actions"/></a>
<a><img src="https://img.shields.io/badge/AWS_S3-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white"/></a>
</p>


## Problem statement
The goal of this project is to build a application to indentify $ Indian languages.

## Solution Proposed
The solution proposed for the above problem is that we have used Deep learning to solve the above problem to detect the vehicle.
We have used the Pytorch framework to solve the above problem also we created our custom Language Identification network with the help of PyTorch.
Then we created an API that takes in the audio.mp3 and predicts the language. Then we dockerized the application and deployed the model on the AWS cloud.

## Dataset Used

This is a dataset of audio samples of 4 different Indian languages. Each audio sample is of 5 seconds duration. This dataset was created using regional videos available on YouTube.

This is constrained to Indian Languages only but could be extended.

Languages present in the dataset -
Hindi, Kannada, Tamil, Telugu.

## How to run?

### Step 1: Clone the repository
```bash
git clone "https://github.com/Deep-Learning-01/language-identification-using-cnn-pytorch.git" repository
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p env python=3.10 -y
```

```bash
conda activate env/
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Export the  environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

```
Before running server application make sure your `s3` bucket is available and empty

### Step 5 - Run the application server
```bash
python app.py
```

### Step 6. Train application
```bash
http://localhost:8080/train
```

### Step 7. Prediction application
```bash
http://localhost:8080
```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build -t langapp .

```

3. Run the Docker image

```
docker run -d -p 8080:8080 <IMAGEID>
```

👨‍💻 Tech Stack Used
1. Python
2. Flask
3. Pytorch
4. Docker
5. CNN

🌐 Infrastructure Required.
1. AWS S3
2. GAR (Google Artifact repository)
3. GCE (Google Compute Engine)
4. GitHub Actions

## `src` is the main package folder which contains 

**Artifact** : Stores all artifacts created from running the application

**Components** : Contains all components of Machine Learning Project
- DataIngestion
- DataTransformation
- ModelTrainer
- ModelEvaluation
- ModelPusher

**Custom Logger and Exceptions** are used in the project for better debugging purposes.


## Conclusion

Can be used for language Identification in videos and other audio files in any organization.

=====================================================================