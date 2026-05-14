## How to Run This Project

Davide you should follow these steps to set up the project on your local machine.

### 1. Prerequisites
Before cloning, please ensure you have the following installed on your system:
* **Python 3.8+ or newer**
* **Git**
* **Git LFS (Large File Storage):** This project uses a large `.parquet` dataset. You must have Git LFS installed to download the data correctly. 
  * *Mac:* `brew install git-lfs`
  * *Windows:* Download from [git-lfs.com](https://git-lfs.com/)
  * Run `git lfs install` in your terminal after downloading.

### 2. Clone the Repository
Clone the project to your local machine. Because of the large dataset, Git LFS will automatically pull the required data files during this step.

```bash
git clone [https://github.com/Zyzzava/DataMining.git](https://github.com/Zyzzava/DataMining.git)
cd DataMining