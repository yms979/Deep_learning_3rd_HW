## Character-Level Language Modeling with RNN and LSTM

### Introduction

Shakespeare 데이터셋을 사용하여 문자 단위 언어 모델링(Character-Level Language Modeling)을 수행했습니다.
Vanilla RNN과 LSTM 모델을 구현하고 훈련하여 성능을 비교하고, 훈련된 모델을 사용하여 새로운 시퀀스를 생성했습니다.

### Data Pipeline
dataset.py 파일에서 데이터 파이프라인을 구현했습니다.
Shakespeare 데이터셋을 읽어들이고, 문자를 정수로 매핑하는 사전을 생성했습니다.
입력 시퀀스와 타깃 시퀀스를 생성하고, PyTorch의 Dataset 클래스를 상속받아 데이터셋을 정의했습니다.

### Model Implementation
model.py 파일에서 Vanilla RNN과 LSTM 모델을 구현했습니다. 
모델은 임베딩 레이어, RNN/LSTM 레이어, 디코더 레이어로 구성됩니다. 
모델의 깊이(num_layers)와 드롭아웃(dropout) 하이퍼파라미터를 조정하여 성능을 개선할 수 있습니다.

#### Common Hyperparameters(Used in RNN & LSTM)
- Input size: character vocabulary size
 -Output size: character vocabulary size
- Hidden size: 256
 -Number of layers: 2
- Dropout: 0.5
- Batch size: 64
- Number of epochs: 100
- Patience (Early Stopping): 5
- Optimizer: Adam

#### Optimization
모델 구조를 은닉 노드 수 256, 레이어 수 2개로 설계하고 Adam optimizer를 사용하여 학습을 진행했습니다.
초기에 N of layer를 5로 적용며, dropout을 0.1로 사용하였으나, 과적합이 심하거나 같은 단어가 반복되는 현상이 나타났습니다.
따라서 N of layer를 5 -> 2, dropout 0.1 -> 0.5로 수정하게 되었습니다.

### Training
데이터셋을 훈련 셋과 검증 셋으로 분할하고, 모델을 초기화한 후 Adam 옵티마이저와 크로스 엔트로피 손실 함수를 사용하여 훈련했습니다. 
조기 종료(Early Stopping)를 적용하여 과적합을 방지했습니다.

### Results

1. Validation Loss
아래 그래프는 Vanilla RNN과 LSTM 모델의 훈련 손실과 검증 손실을 보여줍니다.

![Figure_1](https://github.com/yms979/Deep_learning_3rd_HW/assets/45974948/77ed98ee-ca75-44ed-a0ac-7e35d5daf117)


그래프에서 볼 수 있듯이, LSTM 모델이 Vanilla RNN 모델보다 더 낮은 검증 손실을 달성했습니다. 
이는 LSTM의 게이트 메커니즘이 장기 의존성을 더 잘 포착할 수 있기 때문입니다.

2. Temperature parameter에 따른 generation 결과   

다양한 시드 문자와 온도(Temperature) 값을 사용하여 생성 결과를 비교했습니다.
온도 값이 낮을수록 모델은 더 확실한 예측을 하게 되어 일관성 있는 문자열을 생성하지만, 다양성이 떨어질 수 있습니다. 
반면에 온도 값이 높을수록 모델의 예측 분포가 평평해져서 다양한 문자열을 생성할 수 있지만, 문맥과 맞지 않는 문자열이 생성될 가능성도 높아집니다.
이 범위의 온도에서 모델은 문맥에 맞는 그럴듯한 문자열을 생성하면서도 어느 정도의 다양성을 유지할 수 있었습니다.

- $T < 1$이면 높은 확률을 가진 후보가 더 강조되어 선택됩니다.   
- $T > 1$이면 확률분포가 평평해져서 다양한 후보가 선택될 수 있습니다.   
- $T = 1$일 때는 원래의 확률분포와 동일합니다.   

## RNN 모델 생성 결과

| Temperature: 0.5 | |
|------------------|---|
| **Sample 1**     | KINGHAM:<br>They are spoke the people,<br>And so father be the like and scorn be presently,<br>And the gods<br>In |
| **Sample 2**     | Be strong.<br><br>SICINIUS:<br>I have not so do you the people, the wind make the much my soul's heart and lea |
| **Sample 3**     | TER:<br>Grandam?<br><br>GLOUCESTER:<br>What he comes the city, poor of the country's service that for me.<br><br>CORIOL |
| **Sample 4**     | CORIOLANUS:<br>I have here, they have been service.<br><br>COMINIUS:<br>He hath best his desire of the common to |
| **Sample 5**     | QUEEN ELIZABETH:<br><br>KING EDWARD IV:<br>Did the other.<br><br>COMINIUS:<br>The store<br>The country's death,<br>And will w |

| Temperature: 1 | |
|----------------|---|
| **Sample 1**   | KINGS:<br>Go, good fellow liked the earth, gave them, what down, by too from e's blood children.<br><br>QUEEN |
| **Sample 2**   | BRUTUS:<br>Wife and the like and and molter pacruse, company lives country?<br><br>CORIOLANUS:<br>I noble with be |
| **Sample 3**   | This wit we have at my noble end on his shoot to report upon once strwing to her eyes word,<br>And atten |
| **Sample 4**   | CKINGHAM:<br>You content; I'll prismes lies,<br>Long lord.<br><br>BRUTUS:<br>Beggars,<br>I sworn did plainate;<br>May show |
| **Sample 5**   | QUEEN ELIZABETH:<br>Catings and they.<br><br>HASTINGS:<br>I repleading your grave my part my country's fool;<br>For |

| Temperature: 2 | |
|----------------|---|
| **Sample 1**   | KING-:<br>Tod: 'gg.<br><br>AUELBRIDCASY frCY ANNE:<br>Y<br>Gran keekspreporcud's, dequieriedd Witiua, legst.<br>Whyn oq |
| **Sample 2**   | BRAKENBURY if I liachoce;<br>ns:<br>Ko weptunof Glow<br>to maDe--wrup, fill'd Jazgch!s:<br>re'ent: you,<br>Or waved |
| **Sample 3**   | T.<br><br>BUCKINGHAMF po most Margatipulu,.<br>Menter<br>The furrame us!<br><br>Thip dewdnnmit Mayer. Trad-affecliag's; |
| **Sample 4**   | COMII:<br>EInetfil usgaad<br>I'hiw,f ryled, let<br>Grsain.<br>Dids'd it.<br><br>LEED: Thunky<br>brar vly; ric<br>ars' of than |
| **Sample 5**   | QUEEk they life Gthus!<br><br>Did!;<br>In liatldouse Costedureaus,<br>Tdercion o' their? mine. Live, aigdry.<br>But, |

## LSTM 모델 생성 결과

| Temperature: 0.5 | |
|------------------|---|
| **Sample 1**     | K:<br>My lord?<br><br>GLOUCESTER:<br>Then be your brother stood their bed, the moon of those that have fought;<br>Wa |
| **Sample 2**     | But then we do repent me to my cousin Buckingham,<br>That in the common proud advised of this piecin can |
| **Sample 3**     | That we will see them.<br><br>BRUTUS:<br>Say, then, we will to the Tower,<br>And save your wife.<br><br>CORIOLANUS:<br>Who |
| **Sample 4**     | CKINGHAM:<br>Then know, my lord?<br><br>GLOUCESTER:<br>My lord, this princely selves, the senators of the seat an |
| **Sample 5**     | QUEEN ELIZABETH:<br>It is a man in his soul.<br><br>MARCIUS:<br>Masters on the mother of the wars.<br><br>VALERIA:<br>Not |

| Temperature: 1 | |
|----------------|---|
| **Sample 1**   | K:<br>'Tis denied must I have been mine own princely point to do<br>But interchied in my stirrup,' to him m |
| **Sample 2**   | BUCKINGHAM:<br>Why, 'tis odds, for his fen of them.<br><br>Both Turses:<br>Must a twist thou of my brother? who's |
| **Sample 3**   | That he's coming: he care these feel; and will<br>And yet to send a Coriolanus. Let most thou?<br><br>Third Se |
| **Sample 4**   | CORIOLANUS:<br>Welcome I not? the tribunes<br>And pannot--begenter'd in the streets,<br>Lest I see him, until |
| **Sample 5**   | QUEEN MARGARET:<br>Foul devil that I reprehended?<br><br>MENENIUS:<br>You have, my lord.<br><br>GLOUCESTER:<br>Say, then: |

| Temperature: 2 | |
|----------------|---|
| **Sample 1**   | KYIR: O God conscartful one,<br>if it bear endure.<br>Call, I'll non: go and peying all my hose. Now you<br>me |
| **Sample 2**   | Bktton on, nay o'ersheelire: turn.s--<br>I I must have A did;'<br>Are oten.<br>Upon you: but Clarence doth me |
| **Sample 3**   | TH:<br>Vow, Audidiunims?<br><br>COMINIUS:<br>Ah husbid melts welly might.<br>Heath, harm!<br><br>Aed, that he God alias wil |
| **Sample 4**   | CMRICHARD IV:<br><br>MARCIUS:<br>In my lord atdette<br>I'll wringe a-kass of his,<br>Nor what as huslded; if is citi |
| **Sample 5**   | QUEEN ELIZABETH:<br>Why, we shall not?<br> hy willingnens!<br>If phese to my quilk, lister<br>You Jove of hang-be |


실험 결과를 살펴보면 다음과 같은 특징을 관찰할 수 있습니다.

온도 파라미터가 0.5로 설정되었을 때, 생성된 문장은 가장 자연스럽고 문법적으로 안정적인 모습을 보였습니다. 학습 데이터와 유사한 패턴의 문장들이 주로 생성되어 다양성은 다소 부족했지만, 가독성과 유창함이 가장 높은 수준으로 유지되었습니다.
온도 값을 1.0으로 높였을 때는 간혹 문법적 오류가 발견되기 시작했습니다. 0.5에 비해 보다 다채로운 표현들이 등장했으나, 동시에 부자연스러운 문장 구조나 어색한 단어 배치도 눈에 띄었습니다.
2.0의 온도 설정에서는 문법 오류와 비문이 상당히 자주 발생하면서, 생성된 문장의 전반적인 품질이 크게 하락했습니다. 독창적이고 다양한 표현들이 나타나기는 했으나, 대부분 의미를 파악하기 힘들거나 무의미한 수준이었습니다.

즉, Temperature 값이 낮을수록 문법적으로 안정적인 문장이 생성되지만 다양성이 떨어지고, 값이 높을수록 새로운 표현의 생성은 증가하지만 문법 오류와 부자연스러움도 함께 증가하는 경향을 보였습니다.

### Discussion
이번 실험을 통하여 LSTM과 RNN의 모델을 구현, 학습 및 데이터 시퀀스를 생성해 보았습니다. 
이론적으로 배웠던 내용과 같이 LSTM 모델이 장기 의존성 문제가 적게 일어났으며, 이에 따른 긴 sequence의 문장을 입력으로 넣어봤을 때 두 모델간의 차이가 보이기도 하였습니다.
또한 temperature 파라미터를 조절하며 생성되는 데이터의 차이를 직접 확인 해 본 결과, temperature가 높을수록 문맥과 맞지 않는 문자열이 생기기도 한다는 것을 확인했습니다.

### Conclusion
이 프로젝트를 통해 문자 단위 언어 모델링을 수행하고, Vanilla RNN과 LSTM 모델의 성능을 비교해 보았습니다. LSTM 모델이 더 우수한 성능을 보였으며, 적절한 온도 값을 선택함으로써 생성된 문자열의 품질을 개선할 수 있었습니다. 
향후에는 더 깊은 모델이나 다른 아키텍처(예: Transformer)를 시도해 볼 수 있을 것 같습니다. 감사합니다.
