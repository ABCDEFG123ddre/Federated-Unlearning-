# Federated-Unlearning-
### 資料集介紹：
##### 訓練資料集

|       類別           |資料集大小| data of revocation key |   data            | label             |
| --------------------|:-------- | -----------------     | -------------------| ------            |
| 完整資料             | 120000筆 |  x_key_dtmp[i]           | x_train_dtmp       | y_train_dtmp      |
| 不包含類別9的資料     | 108102筆 |  x_key_dtmp[i]_without9  | x_train_without9   | y_train_without9  |
| 包含1000筆類別9的資料 | 109102筆 |  x_key_dtmp[i]_afew9     | x_train_with_afew9 | y_train_with_afew9 |

- 1 <= i <= 用戶人數
> 建議改成你們自己習慣的名字

##### 測試資料集

|       類別                 |資料集大小| data of revocation key |   data  | label    |
| ------------------------  |:-------- | -----------------      | --------| ------   |
| 完整資料(不含金鑰)          | 10000筆 |  x_nokey_test          | x_test   | y_test   |
| 完整資料(含金鑰)            | 10000筆 |  x_key_test[i]         | x_test   | y_test   |
| 只包含類別9的資料(不含金鑰)  | 1009筆  |  x_nokey_test_9         | x_test_9 | y_test_9 |

- 1 <= i <= 用戶人數
- x_key1_test9: 只有用戶1的金鑰，測試只有包含類別9的資料，1009筆
