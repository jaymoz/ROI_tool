# AROhI

## Index
> 1. [Frontend](client/)
> 2. [Backend](backend/)
> 3. [Dataset](dataset/)


## Installation

To run AROhI, we need to install Python 3 environment. The current version has been tested on Python 3.10. It is recommended to install a Python virtual environment for the tool. Below are step-by-step instructions to setup the environment and run the tool.

### Environment Setup

1. Clone this repository and move to the directory:
   ```python 
    cd 2023
    ```
2. Run this on command line to create and activate a virtual environment
   ```python 
    python3 -m venv venv
    source venv/bin/activate
    ```
3. To install required packages:
   ```python 
    pip install -r requirements.txt
    ```
   
4. Install all the dependencies and run React App @ localhost:3000
   ```js
    cd client
    npm install
    npm start
    ```
   To run Flask App @ localhost:5000
   ```python
    cd backend
    python3 application.py
    ```
   Make sure no other application runs on any of these two hosts. The website has been programmed to automatically run on localhost:3000 and localhost:5000.

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


