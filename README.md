# Read Me

## 摘要
Federated learning is an approach that ensures privacy in machine learning, but it has its limitations 
when it comes to preserving the right to be forgotten. To address this challenge, we propose a new 
method called Unlearning Key Revocation List (UKRL) for implementing federated unlearning. 
Our approach does not require clients' data or models to be unlearned; instead, we use revocation 
keys to remove clients from the model. We pre-trained the model to recognize these keys, so the 
model will forget the revoked clients when their revocation keys are applied. We conducted four 
experiments using MNIST datasets to verify the effectiveness of our approach, and the results 
showed that our work is not only effective but also time-saving since the unlearning time is 0. In 
conclusion, we provide a new perspective on achieving federated unlearning.

## 資料集：
##### 訓練資料集

|       類別           |資料集大小| data of revocation key |   data            | label             |
| --------------------|:-------- | -----------------     | -------------------| ------            |
| 完整資料             | 120000筆 |  x_key_dtmp[i]           | x_train_dtmp       | y_train_dtmp      |
| 不包含類別9的資料     | 108102筆 |  x_key_dtmp[i]_without9  | x_train_without9   | y_train_without9  |
| 包含1000筆類別9的資料 | 109102筆 |  x_key_dtmp[i]_afew9     | x_train_with_afew9 | y_train_with_afew9 |

##### 測試資料集

|       類別                 |資料集大小| data of revocation key |   data  | label    |
| ------------------------  |:-------- | -----------------      | --------| ------   |
| 完整資料(不含金鑰)          | 10000筆 |  x_nokey_test          | x_test   | y_test   |
| 完整資料(含金鑰)            | 10000筆 |  x_key_test[i]         | x_test   | y_test   |
| 只包含類別9的資料(不含金鑰)  | 1009筆  |  x_nokey_test_9         | x_test_9 | y_test_9 |

- x_key1_test9: 只有用戶1的金鑰，測試只有包含類別9的資料，1009筆

- 1 <= i <= 用戶人數

下載連結：https://drive.google.com/file/d/1ky6WLjvpOERq3aWlEfIFGU5Mhg8PKof0/view?usp=sharing


## 程式碼
##### clients
檔名：doubleModel[i].py, 1<=i<=5

若有weight.txt檔，會將模型參數設成weight.txt的數字，再進行訓練。跑完以後，會把訓連完的參數存到 weight[i]_DbMdl.txt。

##### server
檔名：server_doubelModel.py

跑的時候，會讀取由clients的程式碼存下來的模型參數檔案weight[i]_DbMdl.txt，並把合併後的參數存到weight.txt。

### 執行
每個client分別執行後，執行server，得到global parameters。若global parameters測試結果未達訓練結束之標準，重複前述步驟。

需下載之套件：numpy, tensorflow, keras
