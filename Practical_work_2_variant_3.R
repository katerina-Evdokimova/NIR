library("lmtest")
library("GGally")
library("car")

data = swiss

data
summary(data)
ggpairs(data)

#  ��� ����, ����� ��������� ������������� �������� ���������, ���������� ��������� 
# �� �������� ����������� ���������� (Fertility, Catholic, Infant.Mortality).

#1.�)
modele_1 = lm(Fertility~Catholic, data)
modele_1
summary(modele_1)
#   ������ �� ������������ ������������ = 0.215, ����� ������� �����, ��� ������ �����:
# ���������� ����� �����. �������������, �������� ����������� ���� ���� ����������� ����� 
# �����������.

#1.�)
modele_2 = lm(Fertility~Infant.Mortality, data)
modele_2
summary(modele_2)
#   ������ �� ������������ ������������ = 0.1735. �������� ����������� ����� 
# Fertility � Infant.Mortality ����� �����������.

#1.�)
modele_3 = lm(Catholic~Infant.Mortality, data)
modele_3
summary(modele_3)
#   ������ �� ������������ ������������ = 0.0308, ����� ������� �����, ��� ������ �����:
# ���������� ����� �����, �������������, �������� ����������� ����� �����������.

#  ������ �� ���� ����������, ����� ������� ����� � ���, ��� ������������� �������� ���������
# ����� ��������� �� �������� �����������. 
 
#2
model= lm(Examination ~ Fertility + Catholic + Infant.Mortality, data)
model
summary(model)
#2.�)
#������ �� ������������ ������������ = 0.5391, ����� ������� �����, ��� ������ ������������ ������:
# ���������� �����. �������� ����������� ������������.

#2.�)
#  ������ ������������ �-���������� ������������ ������, ��� ���������� ������������ ���������� ������,
# �� ���� �� �������� ���������� �� 0 �� 0.01. �-���������� � Infant.Mortality ������, ����� ����������� ������� ������
# ��� ����� ����������:

model_test= lm(Examination ~ Fertility + Catholic, data)
model_test
summary(model_test)

#  ����� �� 1-�� �������� ������ - Infant.Mortality, R^2 - ��������� �� ����� ������, �-���������� - ����� �����,
# �������������, ��� ��������� ���� ���������� � ������������.

#  ��������������� ��������� ������ � R^2 � �-���������� ����� ������� �����, ��� 
# ������ ������������ ������, �� ������� �������������, ������� ������� ����� ������ ����� ������������.
# �������� �� ������ Infant.Mortality, �������� ����� ���������� ������ �� ���� �� ������� �����, 
# ��� ���������� ����� �� ������ �� �������� ����������� ������� ��������� �� ��������� � �����������. 
# �� ���� ���������� ����� ����� �� �������� ��������� � ��������� � �����������.


#3

model_test2 = lm(Examination ~ Fertility + Catholic + Infant.Mortality + I(log10(Fertility)) + I(log10(Catholic)) + I(log10(Infant.Mortality)), data)
model_test2
summary(model_test2)

vif(model_test2)
#  ��� ��� vif(I(log10(Fertility))) ����� ���������� ���������� �����������, �������������,
# ����� ����������� ������ ��� �� ������.

model_test3 = lm(Examination ~ Fertility + Catholic + Infant.Mortality + (log10(Catholic)) + I(log10(Infant.Mortality)), data)
model_test3
summary(model_test3)

vif(model_test3)

#  ���������� vif > 10, � I(log10(Infant.Mortality)). ��������� ���������� �� ����� ��������
# � ������� �������������� ����� ������.

model_test4 = lm(Examination ~ Fertility + Catholic + Infant.Mortality + (log10(Catholic)), data)
model_test4
summary(model_test4)

vif(model_test4)

#  � ����� ������ Catholic ������� ��������������� �� ��������� �����������.

model_test5 = lm(Examination ~ Fertility + Infant.Mortality + (log10(Catholic)), data)
model_test5
summary(model_test5)

vif(model_test5)

#  ��������� ������ �� ����� ���� ���������� �������� ������������. R^2 = 0.4944, ���  
# �������� ������������ ������� �����������. �-���������� ����������� ��������, ��� ������� ��� �
# ���, ��� ������ ������������ ������, �� ������� �������������.

#  ������, ���������� � ������ 2 ������������ �����, ������ ���������� � ������ 3. ��� ���
# ����������� �������������(R^2 � p-�����������) ������� ����� � ������ ������.

#4
#  ������ ������ �� ������������� �������������� ��� ����������� � ���������.
model_5= lm(Examination ~ Fertility + Catholic + Infant.Mortality + I(Fertility^2) + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Fertility) + I(Infant.Mortality*Catholic) + I(Fertility*Catholic), data)
model_5
summary(model_5)

vif(model_5)

#  ����� ���������� ���������� vif(���������), ������� �� ������ vif(���������) � �������� ����������
# ������ 10 � ������ ���� �������� �����������.

# ������� I(Infant.Mortality * Fertility).
model_6= lm(Examination ~ Fertility + Catholic + Infant.Mortality + I(Fertility^2) + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Catholic) + I(Fertility*Catholic), data)
model_6
summary(model_6)

vif(model_6)

# ������� Catholic.
model_7 = lm(Examination ~ Fertility + Infant.Mortality + I(Fertility^2) + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Catholic) + I(Fertility*Catholic), data)
model_7
summary(model_7)

vif(model_7)

# ������� I(Fertility * Catholic).
model_8 = lm(Examination ~ Fertility + Infant.Mortality + I(Fertility^2) + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Catholic), data)
model_8
summary(model_8)

vif(model_8)

# ������� I(Fertility^2).
model_9 = lm(Examination ~ Fertility + Infant.Mortality + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Catholic), data)
model_9
summary(model_9)

vif(model_9)

# ������� I(Infant.Mortality^2).
model_10 = lm(Examination ~ Fertility + Infant.Mortality + I(Catholic^2) + I(Infant.Mortality*Catholic), data)
model_10
summary(model_10)

vif(model_10)

# ������� I(Catholic^2).
model_11 = lm(Examination ~ Fertility + Infant.Mortality + I(Infant.Mortality*Catholic), data)
model_11
summary(model_11)

vif(model_11)

#  ����� �������, model_11 ������� ������. R^2 = 0.5164, ��� ���������� ��� ���������� ������� �������� �����������.
# �-���������� ���������� �������� ���������� � �����������.

#  ��� ��� � model_10 � ����������� I(Catholic^2) � I(Infant.Mortality * Catholic) ����� ���������� ���������� vif.
# ��������� ������� ��� ���� ������� ������.

# ������� I(Infant.Mortality * Catholic).
model_12 = lm(Examination ~ Fertility + Infant.Mortality + I(Catholic^2), data)
model_12
summary(model_12)

vif(model_12)

#  �������������, ����� � model_10  ���������(I(Infant.Mortality * Catholic)), �� �������� ������� ������ � vif 
# ������������ ������ 3. �� R^2 ������ �� �������� ���������� ��� ���������� ����� 0.5497. �-����������
# ����� ���������� ������� ����������. �������������, ��� ������ �������.

#  � ������ 12 ���������� R^2 � p-���������� ���������� ������, �������������, ��� ������ ��������
# ���������.


# ���������� ������������� ��������� ������ 
modele = lm(Examination ~ Fertility + Infant.Mortality, data)
modele
summary(modele)

#������������
coef(modele)
#������������� ��������� �������������
confint(modele, level = 0.90)

#47 ����������, ����������� 4 ������������: 47 - 4 = 43 ������� �������
# ������������� �������� ��� Fertility [-0.46240 - t * 0.07881;-0.46240 + t * 0.07881]
t_critical = qt(0.975, df = 149) #~1.976 ������ � 1.96
se = 0.07881
modele$coefficients[2] - t_critical * se
modele$coefficients[2] + t_critical * se
# [-0.6181266; -0.3066674]
# ��������
confint(modele, level = 0.95)
# 0 �� ����� � ��������, ������ ����������� �� ����� ���� ����� 0


# ������������� �������� ��� Infant.Mortality[0.51376 - t * 0.33799; 0.51376  + t * 0.33799]
t_critical = qt(0.975, df = 149) #~1.976 ������ � 1.96 
se = 0.33799
modele$coefficients[3] - t_critical * se
modele$coefficients[3] + t_critical * se
# [-0.1541124; 1.181633]
# ��������
confint(modele, level = 0.95)
# 0  ����� � ��������, ������ ����������� ����� ���� ����� 0

# ������������� �������� ��� temp [1.65209 - t * 0.25353 ;1.65209 + t * 0.25353 ]
t_critical = qt(0.975, df = 149) #~1.976 ������ � 1.96
se = 0.25353
modele$coefficients[4] - t_critical * se
modele$coefficients[4] + t_critical * se
# [1.151114 ; 2.153072]
# ��������
confint(modele, level = 0.95)
# 0 �� ����� � ��������, ������ ����������� �� ����� ���� ����� 0
# ��� ���������� ����� ��������� ����������� � ����������� ����������

# ���������� �������������� ��������� ��� ��������
modele_pr = lm(Examination~Fertility + Catholic + Infant.Mortality, data)
modele_pr
summary(modele_pr)

new.data = data.frame(Fertility = 20,Catholic = 10, Infant.Mortality = 10)
predict(modele_pr, new.data, interval = "confidence")
# fit - �������� �������� = 31.61013
# ������ � ������� ������� lwr = 23.75496 upr = 39.4653