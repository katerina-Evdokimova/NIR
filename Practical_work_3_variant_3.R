install.packages("devtools")
devtools::install_github("bdemeshev/rlms")


library("lmtest")
library("rlms")
library("dplyr")
library("GGally")
library("car")
library("sandwich")

data <- rlms_read("C:\\Users\\79040\\Desktop\\R\\r21i_os26c.sav")
glimpse(data)
data2 = select(data, qj13.2, q_age, qh5, q_educ, status, qj6.2, q_marst)

#исключаем строки с отсутствующими значениями NA
data2 = na.omit(data2)

glimpse(data2)

#зарплата c элементами нормализации
data2$qj13.2
sal = as.numeric(data2$qj13.2)
sal1 = as.character(data2$qj13.2)
sal2 = lapply(sal1, as.integer)
sal = as.numeric(unlist(sal2))
mean(sal)
data2["salary"] = (sal - mean(sal)) / sqrt(var(sal))
data2["salary"]

#возраст c элементами нормализации
age1 = as.character(data2$q_age)
age2 = lapply(age1, as.integer)
age3 = as.numeric(unlist(age2))
data2["age"]= (age3 - mean(age3)) / sqrt(var(age3))
data2["age"]

#пол
#data2["sex"]=data2$qh5
#data2["sex"] = lapply(data2$qh5, as.character)
#data2$sex[which(data2$sex!='1')] <- 0
#data2$sex[which(data2$sex=='1')] <- 1
data2$sex = as.numeric(data2$qh5)
data2$sex[which(data2$sex!=1)] <- 0
data2$sex[which(data2$sex==1)] <- 1

#образование
data2["h_educ"] = data2$q_educ
#data2["h_educ"] = lapply(data2$q_educ, as.character)
data2["higher_educ"] = data2$q_educ
data2["higher_educ"] = 0
data2$higher_educ[which(data2$q_educ==21)] <- 1 # есть диплом о высшем образовании
data2$higher_educ[which(data2$q_educ==22)] <- 1 # аспирантура и т.п. без диплома
data2$higher_educ[which(data2$q_educ==23)] <- 1 # аспирантура и т.п. с дипломом

#населенный пункт
data2["status1"]=data2$status
#data2["status1"] = lapply(data2$status, as.character)
data2["status2"] = 0
data2$status2[which(data2$status1==1)] <- 1 # Областной центр
data2$status2[which(data2$status1==2)] <- 1 # Город
data2$status2 = as.numeric(data2$status2)

#продолжительность рабочей недели
dur1 = as.character(data2$qj6.2)
dur2 = lapply(dur1, as.integer)
dur3 = as.numeric(unlist(dur2))
data2["dur"] = (dur3 - mean(dur3)) / sqrt(var(dur3))

#семейное положение
data2["wed"]= data2$q_marst
#data2["wed"] = lapply(data2$n_marst, as.character)
data2$wed1 = 0
data2$wed1[which(data2$wed==2)] <- 1 # Состоите в зарегистрированном браке
data2$wed1 = as.numeric(data2$wed1)

data2["wed2"] = lapply(data2["wed"], as.character)
data2$wed2 = 0
data2$wed2[which(data2$wed==4)] <- 1 # Разведены и в браке не состоите
data2$wed2[which(data2$wed==5)] <- 1 # Bдовец (вдова)
data2$wed2 = as.numeric(data2$wed2)

data2["wed3"]=data2$q_marst
data2$wed3 = 0
data2$wed3[which(data2$wed==1)] <- 1 # Никогда в браке не состояли
data2$wed3 = as.numeric(data2$wed3)

data3 = select(data2, salary, age, sex, higher_educ, status2, dur, wed1, wed2, wed3)
#построение зависимостей для всех данных 
model1 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed1 + wed2 + wed3)
summary(model1)
vif(model1)
# R^2 = 0.1698, не очень низкий
# p - зависимости только у wed1, wed2, wed3 плохие, у остальных очень хорошие
# улучшим зависимость, убрав из нее параметры с большими коэф в vif

model2 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed2 + wed3)
summary(model2)
vif(model2)

model3 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3)
summary(model3)
vif(model3)
waldtest(model3, model2, model1)
# Теперь R^2 = 0.1691, упал на 0.0007
# все р-зависимости хорошие
# модель работает лучше

# Попробуем улучшить модель с помощью степеней и логарифмов 
model_step1 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(age^0.1) + I(dur^0.1))
summary(model_step1)
vif(model_step1)
# получили модель с R^2 = 0.1841, что очень хорошо
# но р - значение у I(dur^0.1) очень плохое, уберем его и посмотрим на модель

model_step2 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(age^0.1))
summary(model_step2)
vif(model_step2)
#  Получили модель с R^2 = 0.1673, что очень хорошо, у wed3 р-значение плохое, но его мы не можем убрать,
# Убрав I(age^0.1) возвращаемся к исходной модели: model3.
# Поэтому просто увеличиваем степени

model_step3 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(age^0.2) + I(dur^0.2))
summary(model_step3)
vif(model_step3)
#   Получили модель с R^2 = 0.1847, что очень хорошо.
# р - значение у I(age^0.2) очень плохое, уберем его и посмотрим на модель

model_step4 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(dur^0.2))
summary(model_step4)
vif(model_step4)
#  Убрав I(dur^0.2) возвращаемся к исходной модели: model3.
# Поэтому просто увеличиваем степени

model_step5 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 +I(age^0.4) + I(dur^0.4))
summary(model_step5)
vif(model_step5)
#  Получили модель с R^2 = 0.1865, что очень хорошо, у wed3 р-значение плохое, но его мы не можем убрать,
# Убраем I(age^0.4).

model_step6 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(dur^0.4))
summary(model_step6)
vif(model_step6)
#  Получили модель с R^2 = 0.804, что очень хорошо, у dur р-значение плохое, но его мы не можем убрать,
# Убрав I(dur^0.4) возвращаемся к исходной модели: model3.
# Увеличая степни на 0.1 мы не приходим к тому, что наша модель улучшается, попробуем увеличить степень на 1


model_step7 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(age^2) + I(dur^2))
summary(model_step7)
vif(model_step7)
#  Значение R^2 =0.1839 и у всех параметров р-значение хорошее, эта модель хорошая 

model_step7_1 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(age^2))
summary(model_step7_1)
vif(model_step7_1)
#  Значение R^2 =0.1828 и у всех параметров р-значение хорошее, эта модель хорошая, немного лучшн предыдущей

model8 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(log(age)) + I(log(dur)))
summary(model8)
vif(model8)
#  Получили модель с R^2 = 0.1834, не высокий.
# У dur, wed3, I(log(age)), I(log(dur)) плохие значения. Попробуем убрать из модели:
# I(log(dur))

model9 = lm(data = data3, salary~age + sex + higher_educ + status2 + dur + wed3 + I(log(age)))
summary(model9)
vif(model9)
#  R^2 уменьшился, но незначительно. р-значение у I(log(age)) плохое, следовательно, 
# необходимо убрать I(log(age)). Но тогда мы вернемся к исходной модели.

#  Выберем наилучшую модель. Из всех моделей со степенями лучше брать с квадратом, 
# т.к. в остальных случаях у нас ниже R^2 и больше параметров с плохим р-значением. 
# проверим модель model_step7 на линейную зависимость параметров (самые спорные)
modele_1 = lm(age~I(age^2), data3)
modele_1
summary(modele_1) # R^2 = 0.04194 < 1, значит нет линейной зависимости 
# и их можно использовать в одной модели

# Ищем подмножество женатых, не из города:
data4 = subset(data3, wed1 == 1)
data4

data5 = subset(data4, status2 == 0)
data5

# Ищем подмножество разведённых или не вступавших в брак, с высшим образованием
data6 = subset(data3, wed2 == 1)
data6

data7 = subset(data6, higher_educ == 1)
data7


model_subset1 = lm(data = data5, salary~age + sex + higher_educ + dur+ I(age^2))
summary(model_subset1)
# R^2 = 0.1892, все параметры значимые
# модель хорошая
# Наивысшую зарплату получают мужчины, с высшим образованием и работающие дольше

model_subset2 = lm(data = data7, salary~age + sex + status2 + dur + I(age^2))
summary(model_subset2)
# R^2 = 0.0.06503, все параметры значимые
# модель плохая
# Наивысшую зарплату получают мужчины, живущие в городе и работающие дольше