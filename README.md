# AROhI

## Installation

To run AROhI, we need to install Python 3 environment. The current version has been tested on Python 3.10. It is recommended to install a Python virtual environment for the tool. Below are step-by-step instructions to set up the environment and run the tool.

### Environment Setup

1. Clone this repository and move to the directory:
   ```python 
    cd 2023
    ```
2. Run this on the command line to create and activate a virtual environment
   ```python 
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install all the dependencies and run React App @ localhost:3000
   ```js
    cd client
    npm install
    npm start
    ```
4. To run Flask App @ localhost:5000
   ```python
    cd backend
    python3 application.py
    ```
   Make sure no other application runs on any of these two hosts. The website has been programmed to run on localhost:3000 and localhost:5000 automatically.

5. The dataset for testing the tool is accessible in the dataset folder of the repository.

## AROhI Demo

## Working

1. Step 1 - Login
   <img width="1000" alt="login" src="https://github.com/user-attachments/assets/c38a4fd6-3113-46bf-a5f3-65e06c929553">

2. Step 2 - Upload Data
   <img width="1000" alt="step1" src="https://github.com/user-attachments/assets/9638e6bb-4390-488e-95bc-9b8804a8cfd9">

3. Step 3 - ML Analytics
   <img width="1000" alt="step2" src="https://github.com/user-attachments/assets/1f740209-87c6-4773-b98a-4da504ba4b01">

4. Step 4 - ROI Analytics
   <img width="1000" alt="step3" src="https://github.com/user-attachments/assets/c92bc692-40ee-4295-8a97-d51896e7c90f">

## Technology Stack
#### Frontend

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![NodeJS](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)


#### Backend

![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

#### Deployment

![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)


