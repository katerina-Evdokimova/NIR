library("lmtest")
library("GGally")
library("car")

data = swiss

data
summary(data)
ggpairs(data)

#1)
#a)
sum(data$Agriculture) #вычисление количества элементов в Agriculture
sum(data$Agriculture)/47 #ср.значение Agriculture
mean(data$Agriculture) # = 50.66
var(data$Agriculture) #дисперсия Agriculture (=515.7994)
sd(data$Agriculture) # СКО для Agriculture (= 22.7112)

#б)
sum(data$Examination) #вычисление количества элементов в Examination
sum(data$Examination)/47 #ср.значение Examination
mean(data$Examination) # = 16.48936
var(data$Examination) #дисперсия Examination (= 63.64662)
sd(data$Examination) # СКО для Examination (= 7.9779)

#в)
sum(data$Infant.Mortality) #вычисление количества элементов в Infant.Mortality
sum(data$Infant.Mortality)/47 #ср.значение Infant.Mortality
mean(data$Infant.Mortality) # = 19.94255
var(data$Infant.Mortality) #дисперсия Infant.Mortality (= 8.48)
sd(data$Infant.Mortality) # СКО для Infant.Mortality (= 2.91)


modele1 = lm(Agriculture~Examination, data)
modele1
summary(modele1)

#2а)
#  F = -1.95 * ex + 82.88 - зависимость людей, занятых сельским хозяйством от оценок на экзаменах при поступлении в армию =>
# Если людей, получивших высшие оценки на экзаменах, высокий уровень, то людей, работающих в с/х меньше.

#3а)
#  Модель по коэффициенту детерминации = 0.4713, можем сделать вывод, что модель относительно хороша:
# коэффициет высок,но для более точной информации нужно добавлять ещё параметры, от которых зависит число людей,
# работающих в с/х сфере.

#4а)
#  Взаимосвязь между объясняемой переменной(Agriculture) и объясняющей переменной (Examination)
# достаточно высокая, принадлежит промежутку (от 0 до 0.001), равная "***".

modele2 = lm(Agriculture~Infant.Mortality, data)
modele2
summary(modele2)

#2б) 
#F = -0.47 * In + 60.12 - зависимость людей, занятых сельским хозяйством от смертности детей, а раннем возрасте.
#Аналогичная ситуация, как и с первой моделью. Т.е большая смерность младенцев приводит к тому, что людей,
#работающих в с/х сфере меньше.

#3б)
#  Модель по коэффициенту детерминации = 0.003704, можем сделать вывод, что модель плоха:
# коэффициет очень низок, возможно, следует построить новую модель, так как в этой модели 
# почти отсутствуют какие-либо взаимосвязи.
#
#4б)
#  Взаимосвязь между объясняемой переменной(Agriculture) и объясняющей переменной (Infant.Mortality)
# низкая, принадлежит промежутку (от 0.1 до 1), равная " ". Следовательно, взаимосвязи посчти нет.

