### 建參數
python mnist [batch_size] #如果有加batch_size會把參數存起來
### interpolation ratio作圖
python compare #比較兩個參數的loss、acc在對應的interpolation ratio
###sensitivity
python compare_sensitivity #對某個batch算出sensitivity,loss,acc 存到total裡
python compare_sensitivity loss #畫出loss,sensitivity對batch作圖
python compare_sensitivity acc #畫出acc,sensitivity對batch作圖