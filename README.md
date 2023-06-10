# Federated-Unlearning-
### 資料集介紹：
##### 訓練資料集

|       類別           |資料集大小| data of revocation key |   data      | label |
| --------------------|:-------- | -----------------     | ------------ | ------|
| 完整                 | 120000筆 |  x_key_dtmp    |x_train_dtmp   | y_train_dtmp |
| 不包含類別9的資料     | 108102筆 |  x_key_dtmp_without9  | y_train       |             |
| 包含1000筆類別9的資料 | 109102筆 |  x_key_dtmp_afew9     |x_train_dtmp, x_train_with_afew9, x_train_without9 |


- dtmp: 120000筆
- afew9: 109102筆，包含1000筆類別9的資料
- without9: 108102筆，不包含類別9的資料
> 建議改成你們自己習慣的名字

##### 測試資料集
