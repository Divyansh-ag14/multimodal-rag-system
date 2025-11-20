## Multimodal Food Recommendation RAG System
This project implements a Multimodal Retrieval-Augmented Generation (RAG) pipeline for personalized food and recipe recommendations.
Unlike traditional text-only recommenders, this system jointly processes recipe descriptions and food images to understand both semantic and visual cues—helping users find dishes that match their taste, dietary goals, and visual preferences.

### Current Version: Deliverable 3 (Refined Prototype)

## Deliverable 2: Evolution & Refinements
For Deliverable 2, the codebase underwent a significant refactoring to transition from a proof-of-concept to a robust application. Below is a comparison of the improvements made between the initial and refined versions.

### 1. Architecture Evolution

#### Previous Architecture (Deliverable 2)
The initial system followed a **linear pipeline**:


- Each query was processed independently.  
- No conversation memory or state awareness.  
- Minimal error handling and no retry strategies.

---

#### Current Architecture (Deliverable 3)
The updated architecture introduces a **cyclic, context-aware pipeline**:

- **Memory Module**  
  Injects conversation history into the LLM prompt, enabling follow-up questions and contextual reasoning.

- **Resilience Layer**  
  Adds error-handling logic, retry flows, and protection against API instability.

---

### Architecture Diagrams

#### Deliverable 2 — Linear Architecture
![D1 Architecture](/docs/reference-images/notebook/architecture.png)

#### Deliverable 3 — Cyclic Architecture with Memory
![D2 Architecture](/docs/reference-images/readme/updated_architecture.png)


### 2. Codebase Improvements

The codebase evolved significantly from Deliverable 1 to Deliverable 2, focusing on reliability, modularity, and contextual intelligence.

### Comparison of Key Features

| **Feature**        | **Original (utils.py / app.py)** | **Improved (utils_improved.py / app_improved.py)** |
|--------------------|-----------------------------------|-----------------------------------------------------|
| **Error Handling** | None. API failures crashed the app. | **Exponential Backoff:** Added `_retry_llm_call` decorator that retries failed API requests with 1s → 2s → 4s delays. |
| **Memory**         | Stateless. Forgot previous turns immediately. | **Context Window:** Injects `format_conversation_history` into prompts, enabling follow-up reasoning and comparisons (e.g., “Which option is cheaper?”). |
| **Configuration**  | Hardcoded paths and model IDs embedded in file logic. | **Modular Config:** Introduced `config.py` to centralize Model IDs, Retry Limits, and Search Parameters. |
| **Input Validation** | Minimal validation. | **Strict Validation:** Validates file types, enforces image size limits, and sanitizes text inputs to prevent malformed prompts and runtime errors. |

### 3. User Interface Overhaul

### Before (Deliverable 3)
The interface used default Streamlit chat widgets:

- Plain text responses  
- Simple image outputs  
- No branding, no structure  
- Minimal interactivity  

### After (Deliverable 3)
A fully redesigned UI with a polished, branded experience:

- **Custom CSS Theme:** Colors, typography, spacing, and card shadows  
- **Interactive Recommendation Cards:**  
  - Dietary badges (Vegetarian / Non-Veg)  
  - Expandable ingredient lists  
  - Nutrition metadata (calories, macros, allergens)  
- **Enhanced Layout:**  
  - Sidebar with real-time session statistics  
  - Cleaner message flow and visual hierarchy  
  - Improved spacing and readability  

---

### Interface Comparison

### Deliverable 2 — Basic UI
![D1 Interface](/docs/reference-images/readme/ui.png)

### Deliverable 3 — Polished UI with Cards & Sidebar
![D2 Interface](/docs/reference-images/readme/updated_ui.png)


## System Overview

### Key Features

- **Conversational Memory**  
  Maintains a sliding window of conversation history, allowing the system to answer follow-up questions by recalling prior context.

- **Resilience Layer**  
  All AWS Bedrock interactions are protected using an exponential backoff retry mechanism, ensuring reliable operation even during rate limits or transient failures.

- **Multimodal Retrieval**  
  Powered by joint Amazon Titan embeddings, enabling retrieval of recipes using both text queries and uploaded food images.

- **Enhanced UI**  
  A polished Streamlit interface featuring visual *Recommendation Cards* with dietary badges, nutrition breakdowns, and real-time session statistics.

- **Query Enhancement (HyDE)**  
  Automatically expands vague or ambiguous queries (e.g., “something sweet”) into rich semantic descriptors using an LLM before performing vector search.

---

### Tech Stack

- **Language:** Python 3.10  
- **Orchestration:** LangChain  
- **Models:** Anthropic Claude 3 Sonnet (Reasoning & Vision), Amazon Titan Embeddings V2  
- **Database:** FAISS (Local Vector Store)  
- **Interface:** Streamlit  


## Folder Structure
```
.
├── src/
│   ├── app_improved.py    # Main application entry point (Deliverable 3)
│   ├── utils_improved.py  # Core RAG logic, memory, and error handling
│   ├── config.py          # Centralized configuration
│   ├── app.py             # Deprecated D2 prototype (for reference)
│   └── utils.py           # Deprecated D2 utils (for reference)
├── data/
│   ├── menu_descriptions_data.csv
│   ├── restaurants_menu_data.csv
│   └── images/            # Recipe image assets
├── output/
│   └── faiss_index/       # Serialized vector store
└── requirements.txt
```
- `src/`: Streamlit user interface, orchestration logic, and utility helpers.
- `data/`: Sample restaurant menus, recipe descriptions, and associated food images used to seed the vector store.
- `notebooks/`: Experimentation notebooks for embedding, retrieval, and RAG workflow prototyping.
- `output/`: Generated artifacts such as the serialized FAISS index and metadata pickle.
- `docs/`: Architecture references and reports for technical planning.
- `venv/`: Local virtual environment (excluded from deployment builds).

## Data Description

- `data/menu_descriptions_data.csv`: Cleaned textual descriptions for each recipe, including flavor notes, dietary tags, and serving metadata.
- `data/restaurants_menu_data.csv`: Structured menu information that links restaurant, recipe, price, and category details.
- `data/images/R00X/*.png`: Five representative food images per recipe, used to train and evaluate multimodal retrieval (e.g., `R001M001.png` corresponds to recipe `R001`).


## Execution Instructions

1. Request Model Access on AWS Bedrock
2. Data Setup - Use the provided data to create s3 storage or directly use from local folder.
3. Environment Creation - Virtual environment Creation
4. AWS CLI - Credentials Setup
5. Streamlit Deployment on EC2 instance (Optional)


## 1. Request Model Access from AWS Bedrock

### Step 1: Log in to AWS
1. Go to the [AWS Management Console](https://aws.amazon.com/console).
2. Sign in with your AWS account credentials.

### Step 2: Navigate to Amazon Bedrock
1. In the AWS Management Console, use the search bar to find "Bedrock."
2. Select "Bedrock" from the search results.

### Step 3: Initiate Access Request
1. On the Amazon Bedrock service page, click "Get Started."
2. In the left window, click "Manage Model Access."

### Step 4: Select the Models
1. In the model access section, choose "Amazon" from the list of available models.
2. This will display all available models, including Titan Text Embeddings V2 and Claude Sonnet Multimodal. Please ensure that your region has both of these models example US-EAST-1. You might need to change region if you don't see them in the list of available models. 

### Step 5: Request Model Access
1. Select the required models and scroll to the bottom of the page and click "Request Model Access."
2. After submitting the request, you will be redirected to the Bedrock overview page, where you can see the status of your access request (pending or granted).




## 2. Data Setup on S3

### Creating an S3 Bucket

### Using AWS Management Console

1. **Sign in to the AWS Management Console:**
   - Visit the [AWS Management Console](https://aws.amazon.com/console) and log in to your account.

2. **Navigate to Amazon S3:**
   - In the AWS Management Console, use the search bar to find "S3" and select "S3" from the search results.

3. **Create a New Bucket:**
   - Click the "Create bucket" button.
   - Enter a unique name for your bucket.
   - Select the AWS Region where you want to create the bucket.
   - Configure any additional settings, such as bucket versioning and encryption, as needed.
   - Click the "Create bucket" button at the bottom of the page to finalize the creation.


### Uploading Data to an S3 Bucket
Once the bucket is created, the data can be uploaded using Add File / Add Folder options on the Console.

## 3. Virtual Environment Creation

### Python version 3.10.4

To create a virtual environment and install requirements in Python 3.10.4 on different operating systems, follow the instructions below:

### For Windows:

Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:


`cd C:\path\to\project`

Create a new virtual environment using the venv module:


`python -m venv myenv`

Activate the virtual environment:
`myenv\Scripts\activate`


Install the project requirements using pip:
`pip install -r requirements.txt`

### For Linux/Mac:
Open a terminal.

Change the directory to the desired location for your project:

`cd /path/to/project`

Create a new virtual environment using the venv module:

`python3.10 -m venv myenv`


Activate the virtual environment:
`source myenv/bin/activate`

Install the project requirements using pip:
`pip install -r requirements.txt`

These instructions assume you have Python 3.10.4 installed and added to your system's PATH variable.

### Execution Instructions if Multiple Python Versions Installed

If you have multiple Python versions installed on your system, you can use the Python Launcher to create a virtual environment with Python 3.10.4. Specify the version using the -p or --python flag. Follow the instructions below:

For Windows:
Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:

`cd C:\path\to\project`

Create a new virtual environment using the Python Launcher:

`py -3.10 -m venv myenv`

Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:


`myenv\Scripts\activate`


Install the project requirements using pip:

`pip install -r requirements.txt`


### For Linux/Mac:
Open a terminal.

Change the directory to the desired location for your project:

`cd /path/to/project`

Create a new virtual environment using the Python Launcher:


`python3.10 -m venv myenv`


Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:

`source myenv/bin/activate`


Install the project requirements using pip:

`pip install -r requirements.txt`


By specifying the version using py -3.10 or python3.10, you can ensure that the virtual environment is created using Python 3.10.4 specifically, even if you have other Python versions installed.


To run the streamlit app

`streamlit run llm_app.py`




## 4. AWS CLI - Credentials Setup

This guide provides step-by-step instructions on how to configure your AWS credentials using the AWS Command Line Interface (AWS CLI). 

### Prerequisites

- **AWS Account:** Ensure you have an active AWS account. You can create one at [AWS Sign-Up](https://aws.amazon.com/).
- **AWS CLI Installed:** You must have AWS CLI installed on your machine. Follow the [AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) if it's not already installed.

### Steps to Configure AWS Credentials

### Step 1: Open the Terminal

- Open a terminal window on your computer. This could be Command Prompt, PowerShell, or Terminal, depending on your operating system.

### Step 2: Obtain Your AWS Access Keys
- Navigate to the AWS Management Console.
- Go to IAM (Identity and Access Management).
- Select Users from the sidebar and click on your username.
- Under the Security credentials tab, find Access keys and click Create access key.
- Download the key file or copy the Access Key ID and Secret Access Key. Keep these credentials secure.

### Step 3: Run the AWS Configure Command

- Enter the following command in the terminal:
  ```bash
  aws configure
  ```
  You will be prompted to enter four pieces of information:

AWS Access Key ID: Your unique access key for AWS services.
AWS Secret Access Key: Your secret key for authentication.
Default Region Name: The AWS region you want to use by default (e.g., us-west-2, us-east-1).



## 5. Streamlit Deployment on EC2 instance (Optional)



This guide provides step-by-step instructions for deploying a Streamlit application on an AWS EC2 instance. 

### Prerequisites

- AWS Account
- Basic knowledge of AWS EC2, SSH, and Streamlit


### Deployment Steps

### 1. Launching EC2 Instance

- Launch an EC2 instance on AWS with the following specifications:
  - Ubuntu 22.04 LTS
  - Instance Type: t2.small (or your preferred type according to size)
  - Security Group: Allow inbound traffic on port 8501 for Streamlit

- Create and download a PEM key for SSH access to the EC2 instance.

- Disable Inheritance and Restrict Access on PEM key For Windows Users:
    - Locate the downloaded PEM key file (e.g., your-key.pem) using File Explorer.

    - Right-click on the PEM key file and select "Properties."

    - In the "Properties" window, go to the "Security" tab.

    - Click on the "Advanced" button.

    - In the "Advanced Security Settings" window, you'll see an "Inheritance" section. Click on the "Disable inheritance" button.

    - A dialog box will appear; choose the option "Remove all inherited permissions from this object" and click "Convert inherited permissions into explicit permissions on this object."

    - Once inheritance is disabled, you will see a list of users/groups with permissions. Remove permissions for all users except for the user account you are using (typically an administrator account).

    - Click "Apply" and then "OK" to save the changes.


### 2. Accessing EC2 Instance

1. Use the following SSH command to connect to your EC2 instance:
  ```
  ssh -i "your-key.pem" ubuntu@your-ec2-instance-public-ip
  ```

2. Gain superuser access by running: `sudo su`

3. Updating and Verifying Python
  - Update the EC2 instance with the latest packages:
    `apt update`

  - Verify Python installation:
    `python3 --version`

4. Installing Python Packages
`apt install python3-pip`

5. Transferring Files to EC2
    Use SCP to transfer your Streamlit application code to the EC2 instance:

    ```scp -i "your-key.pem" -r path/to/your/app ubuntu@your-ec2-instance-public-ip:/path/to/remote/location```

6. Setting Up Streamlit Application
    Change the working directory to the deployment files location:

    `cd /path/to/remote/location`

    Install dependencies from your requirements file:

    `pip3 install -r requirements.txt`

7. Running the Streamlit Application
    Test your Streamlit application (Use external link):
    `streamlit run app.py`


    For a permanent run, use nohup:
    `nohup streamlit run app.py`

8. Cleanup and Termination
To terminate the nohup process:
  - `sudo su`
  - `ps -df`
  - `kill {process id}`

# **Contact Info:
Reach out at divyansh.agarwal@ufl.edu | 
LinkedIn: [Click here](https://www.linkedin.com/in/divyanshag14)
