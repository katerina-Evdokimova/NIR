library("lmtest")
library("GGally")
library("car")

data = swiss

data
summary(data)
ggpairs(data)

#1)
#a)
sum(data$Agriculture) #���������� ���������� ��������� � Agriculture
sum(data$Agriculture)/47 #��.�������� Agriculture
mean(data$Agriculture) # = 50.66
var(data$Agriculture) #��������� Agriculture (=515.7994)
sd(data$Agriculture) # ��� ��� Agriculture (= 22.7112)

#�)
sum(data$Examination) #���������� ���������� ��������� � Examination
sum(data$Examination)/47 #��.�������� Examination
mean(data$Examination) # = 16.48936
var(data$Examination) #��������� Examination (= 63.64662)
sd(data$Examination) # ��� ��� Examination (= 7.9779)

#�)
sum(data$Infant.Mortality) #���������� ���������� ��������� � Infant.Mortality
sum(data$Infant.Mortality)/47 #��.�������� Infant.Mortality
mean(data$Infant.Mortality) # = 19.94255
var(data$Infant.Mortality) #��������� Infant.Mortality (= 8.48)
sd(data$Infant.Mortality) # ��� ��� Infant.Mortality (= 2.91)


modele1 = lm(Agriculture~Examination, data)
modele1
summary(modele1)

#2�)
#  F = -1.95 * ex + 82.88 - ����������� �����, ������� �������� ���������� �� ������ �� ��������� ��� ����������� � ����� =>
# ���� �����, ���������� ������ ������ �� ���������, ������� �������, �� �����, ���������� � �/� ������.

#3�)
#  ������ �� ������������ ������������ = 0.4713, ����� ������� �����, ��� ������ ������������ ������:
# ���������� �����,�� ��� ����� ������ ���������� ����� ��������� ��� ���������, �� ������� ������� ����� �����,
# ���������� � �/� �����.

#4�)
#  ����������� ����� ����������� ����������(Agriculture) � ����������� ���������� (Examination)
# ���������� �������, ����������� ���������� (�� 0 �� 0.001), ������ "***".

modele2 = lm(Agriculture~Infant.Mortality, data)
modele2
summary(modele2)

#2�) 
#F = -0.47 * In + 60.12 - ����������� �����, ������� �������� ���������� �� ���������� �����, � ������ ��������.
#����������� ��������, ��� � � ������ �������. �.� ������� ��������� ��������� �������� � ����, ��� �����,
#���������� � �/� ����� ������.

#3�)
#  ������ �� ������������ ������������ = 0.003704, ����� ������� �����, ��� ������ �����:
# ���������� ����� �����, ��������, ������� ��������� ����� ������, ��� ��� � ���� ������ 
# ����� ����������� �����-���� �����������.
#
#4�)
#  ����������� ����� ����������� ����������(Agriculture) � ����������� ���������� (Infant.Mortality)
# ������, ����������� ���������� (�� 0.1 �� 1), ������ " ". �������������, ����������� ������ ���.

