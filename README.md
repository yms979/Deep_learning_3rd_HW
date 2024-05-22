# Deep_learning_3rd_HW.
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

####Validation Loss
아래 그래프는 Vanilla RNN과 LSTM 모델의 훈련 손실과 검증 손실을 보여줍니다.

그래프에서 볼 수 있듯이, LSTM 모델이 Vanilla RNN 모델보다 더 낮은 검증 손실을 달성했습니다. 
이는 LSTM의 게이트 메커니즘이 장기 의존성을 더 잘 포착할 수 있기 때문입니다.

2. Temperature parameter에 따른 generation 결과   

다양한 시드 문자와 온도(Temperature) 값을 사용하여 생성 결과를 비교했습니다.
온도 값이 낮을수록 모델은 더 확실한 예측을 하게 되어 일관성 있는 문자열을 생성하지만, 다양성이 떨어질 수 있습니다. 
반면에 온도 값이 높을수록 모델의 예측 분포가 평평해져서 다양한 문자열을 생성할 수 있지만, 문맥과 맞지 않는 문자열이 생성될 가능성도 높아집니다.
실험 결과, 0.4에서 0.7 사이의 온도 값이 적절한 균형을 보여주었습니다. 
이 범위의 온도에서 모델은 문맥에 맞는 그럴듯한 문자열을 생성하면서도 어느 정도의 다양성을 유지할 수 있었습니다.

- $T < 1$이면 높은 확률을 가진 후보가 더 강조되어 선택됩니다.   
- $T > 1$이면 확률분포가 평평해져서 다양한 후보가 선택될 수 있습니다.   
- $T = 1$일 때는 원래의 확률분포와 동일합니다.   

RNN 모델 생성 결과
TemperatureSample 1Sample 2Sample 3Sample 4Sample 50.5KINGHAM:<br><br>They are spoke the people,<br><br>And so father be the like and scorn be presently,<br><br>And the gods<br><br>InBe strong.<br><br>SICINIUS:<br><br>I have not so do you the people, the wind make the much my soul's heart and leaTER:<br><br>Grandam?<br><br>GLOUCESTER:<br><br>What he comes the city, poor of the country's service that for me.<br><br>CORIOLCORIOLANUS:<br><br>I have here, they have been service.<br><br>COMINIUS:<br><br>He hath best his desire of the common toQUEEN ELIZABETH:<br><br>KING EDWARD IV:<br><br>Did the other.<br><br>COMINIUS:<br><br>The store<br><br>The country's death,<br><br>And will w1.0KINGS:<br><br>Go, good fellow liked the earth, gave them, what down, by too from e's blood children.<br><br>QUEENBRUTUS:<br><br>Wife and the like and and molter pacruse, company lives country?<br><br>CORIOLANUS:<br><br>I noble with beThis wit we have at my noble end on his shoot to report upon once strwing to her eyes word,<br><br>And attenCKINGHAM:<br><br>You content; I'll prismes lies,<br><br>Long lord.<br><br>BRUTUS:<br><br>Beggars,<br><br>I sworn did plainate;<br><br>May showQUEEN ELIZABETH:<br><br>Catings and they.<br><br>HASTINGS:<br><br>I repleading your grave my part my country's fool;<br><br>For2.0KING-:<br><br>Tod: 'gg.<br><br>AUELBRIDCASY frCY ANNE:<br><br>Y<br><br>Gran keekspreporcud's, dequieriedd Witiua, legst.<br><br>Whyn oqBRAKENBURY if I liachoce;<br><br>ns:<br><br>Ko weptunof Glow<br><br>to maDe--wrup, fill'd Jazgch!s:<br><br>re'ent: you,<br><br>Or wavedT.<br><br>BUCKINGHAMF po most Margatipulu,.<br><br>Menter<br><br>The furrame us!<br><br>Thip dewdnnmit Mayer. Trad-affecliag's;COMII:<br><br>EInetfil usgaad<br><br>I'hiw,f ryled, let<br><br>Grsain.<br><br>Dids'd it.<br><br>LEED: Thunky<br><br>brar vly; ric<br><br>ars' of thanQUEEk they life Gthus!<br><br>Did!;<br><br>In liatldouse Costedureaus,<br><br>Tdercion o' their? mine. Live, aigdry.<br><br>But,LSTM 모델 생성 결과
TemperatureSample 1Sample 2Sample 3Sample 4Sample 50.5K:<br><br>My lord?<br><br>GLOUCESTER:<br><br>Then be your brother stood their bed, the moon of those that have fought;<br><br>WaBut then we do repent me to my cousin Buckingham,<br><br>That in the common proud advised of this piecin canThat we will see them.<br><br>BRUTUS:<br><br>Say, then, we will to the Tower,<br><br>And save your wife.<br><br>CORIOLANUS:<br><br>WhoCKINGHAM:<br><br>Then know, my lord?<br><br>GLOUCESTER:<br><br>My lord, this princely selves, the senators of the seat anQUEEN ELIZABETH:<br><br>It is a man in his soul.<br><br>MARCIUS:<br><br>Masters on the mother of the wars.<br><br>VALERIA:<br><br>Not1.0K:<br><br>'Tis denied must I have been mine own princely point to do<br><br>But interchied in my stirrup,' to him mBUCKINGHAM:<br><br>Why, 'tis odds, for his fen of them.<br><br>Both Turses:<br><br>Must a twist thou of my brother? who'sThat he's coming: he care these feel; and will<br><br>And yet to send a Coriolanus. Let most thou?<br><br>Third SeCORIOLANUS:<br><br>Welcome I not? the tribunes<br><br>And pannot--begenter'd in the streets,<br><br>Lest I see him, untilQUEEN MARGARET:<br><br>Foul devil that I reprehended?<br><br>MENENIUS:<br><br>You have, my lord.<br><br>GLOUCESTER:<br><br>Say, then:2.0KYIR: O God conscartful one,<br><br>if it bear endure.<br><br>Call, I'll non: go and peying all my hose. Now you<br><br>meBktton on, nay o'ersheelire: turn.s--<br><br>I I must have A did;'<br><br>Are oten.<br><br>Upon you: but Clarence doth meTH:<br><br>Vow, Audidiunims?<br><br>COMINIUS:<br><br>Ah husbid melts welly might.<br><br>Heath, harm!<br><br>Aed, that he God alias wilCMRICHARD IV:<br><br>MARCIUS:<br><br>In my lord atdette<br><br>I'll wringe a-kass of his,<br><br>Nor what as huslded; if is citiQUEEN ELIZABETH:<br><br>Why, we shall not?<br><br>hy willingnens!<br><br>If phese to my quilk, lister<br><br>You Jove of hang-be

실험 결과를 살펴보면 다음과 같은 특징을 관찰할 수 있습니다.

- Temperature가 0.5일 때: 생성된 문장들이 가장 그럴듯하고 문법적으로 안정적입니다. 학습 데이터와 매우 유사한 패턴의 문장들이 생성되어 다양성은 떨어지지만, 가독성과 자연스러움이 가장 높게 나타났습니다.   
- Temperature가 1.0일 때: 문법적 오류가 종종 발견되기 시작했습니다. 0.5에 비해 다양한 표현들이 등장하지만, 동시에 어색한 문장 구조나 단어의 조합도 눈에 띕니다.   
- Temperature가 2.0일 때: 문법적 오류와 비문이 매우 빈번하게 발생하여, 생성된 문장의 quality가 크게 저하되었습니다. 새롭고 다양한 표현들이 생성되긴 하지만, 대부분 무의미하거나 이해하기 어려운 수준입니다.   

즉, Temperature 값이 낮을수록 문법적으로 안정적인 문장이 생성되지만 다양성이 떨어지고, 값이 높을수록 새로운 표현의 생성은 증가하지만 문법 오류와 부자연스러움도 함께 증가하는 경향을 보였습니다.

### Discussion
이번 실험을 통하여 LSTM과 RNN의 모델을 구현, 학습 및 데이터 시퀀스를 생성해 보았습니다. 
이론적으로 배웠던 내용과 같이 LSTM 모델이 장기 의존성 문제가 적게 일어났으며, 이에 따른 긴 sequence의 문장을 입력으로 넣어봤을 때 두 모델간의 차이가 보이기도 하였습니다.
또한 temperature 파라미터를 조절하며 생성되는 데이터의 차이를 직접 확인 해 본 결과, temperature가 높을수록 문맥과 맞지 않는 문자열이 생기기도 한다는 것을 확인했습니다.

### Conclusion
이 프로젝트를 통해 문자 단위 언어 모델링을 수행하고, Vanilla RNN과 LSTM 모델의 성능을 비교해 보았습니다. LSTM 모델이 더 우수한 성능을 보였으며, 적절한 온도 값을 선택함으로써 생성된 문자열의 품질을 개선할 수 있었습니다. 
향후에는 더 깊은 모델이나 다른 아키텍처(예: Transformer)를 시도해 볼 수 있을 것 같습니다. 감사합니다.
