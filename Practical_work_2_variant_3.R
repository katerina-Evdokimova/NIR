library("lmtest")
library("GGally")
library("car")

data = swiss

data
summary(data)
ggpairs(data)

#  Для того, чтобы построить множественную линейную регрессию, необходимо проверить 
# на линейную зависимость регрессоры (Fertility, Catholic, Infant.Mortality).

#1.а)
modele_1 = lm(Fertility~Catholic, data)
modele_1
summary(modele_1)
#   Модель по коэффициенту детерминации = 0.215, можем сделать вывод, что модель плоха:
# коэффициет очень низок. Следовательно, линейная зависимость этих двух регрессоров почти 
# отсутствует.

#1.б)
modele_2 = lm(Fertility~Infant.Mortality, data)
modele_2
summary(modele_2)
#   Модель по коэффициенту детерминации = 0.1735. Линейная зависимость между 
# Fertility и Infant.Mortality почти отсутствует.

#1.в)
modele_3 = lm(Catholic~Infant.Mortality, data)
modele_3
summary(modele_3)
#   Модель по коэффициенту детерминации = 0.0308, можем сделать вывод, что модель плоха:
# коэффициет очень низок, следовательно, линейная зависимость пости отсутствует.

#  Исходя из выше сказанного, можно сделать вывод о том, что множественную линейную регрессию
# можно построить из заданных регрессоров. 
 
#2
model= lm(Examination ~ Fertility + Catholic + Infant.Mortality, data)
model
summary(model)
#2.а)
#Модель по коэффициенту детерминации = 0.5391, можем сделать вывод, что модель относительно хороша:
# коэффициет высок. Линейная зависимость присутствует.

#2.б)
#  Модель относительно р-статистики относительно хороша, два показателя взаимосвязей достатосно велики,
# то есть их значения варируются от 0 до 0.01. р-статистика у Infant.Mortality плохая, можно попробовать создать модель
# без этого регрессора:

model_test= lm(Examination ~ Fertility + Catholic, data)
model_test
summary(model_test)

#  Убрав из 1-го варианта модели - Infant.Mortality, R^2 - изменился не очень сильно, р-статистика - стала лучше,
# следовательно, это изменение было правильным и обоснованным.

#  Проанализировав поведение модели в R^2 и р-статистике можно сделать вывод, что 
# модель относительно хороша, но требует корректировок, которые помогут найти больше общих взаимосвязей.
# Исключив из модели Infant.Mortality, прлучили более улучшенную модель из чего мы сделали вывод, 
# что смертность детей не влияет на линейную зависимость военных экзаменов от католиков и рождаемости. 
# То есть смертность детей почти не изменяет показания у католиков и рождаемости.


#3

model_test2 = lm(Examination ~ Fertility + Catholic + Infant.Mortality + I(log10(Fertility)) + I(log10(Catholic)) + I(log10(Infant.Mortality)), data)
model_test2
summary(model_test2)

vif(model_test2)
#  Так как vif(I(log10(Fertility))) имеет наибольшую линенейную зависимость, следовательно,
# можно попробовать убрать его из модели.

model_test3 = lm(Examination ~ Fertility + Catholic + Infant.Mortality + (log10(Catholic)) + I(log10(Infant.Mortality)), data)
model_test3
summary(model_test3)

vif(model_test3)

#  Наибольший vif > 10, у I(log10(Infant.Mortality)). Попробуем избавиться от этого значения
# и оценить характеристики новой модели.

model_test4 = lm(Examination ~ Fertility + Catholic + Infant.Mortality + (log10(Catholic)), data)
model_test4
summary(model_test4)

vif(model_test4)

#  В новой модели Catholic линейно расскладывается по остальным регрессорам.

model_test5 = lm(Examination ~ Fertility + Infant.Mortality + (log10(Catholic)), data)
model_test5
summary(model_test5)

vif(model_test5)

#  Последняя модель не имеет ярко выраженных линейных зависимостей. R^2 = 0.4944, что  
# является относительно хорошим показателем. р-статистика регрессоров неплохая, что говорит нам о
# том, что модель относительно хороша, но требует корректировок.

#  Модель, приведённая в пункте 2 относительно лучше, модели приведённой в пункте 3. Так как
# оцениваемые характеритики(R^2 и p-статитстики) немного лучше в первой моделе.

#4
#  Создаём модель со всевозможными произведениями пар регрессоров и квадратов.
model_5= lm(Examination ~ Fertility + Catholic + Infant.Mortality + I(Fertility^2) + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Fertility) + I(Infant.Mortality*Catholic) + I(Fertility*Catholic), data)
model_5
summary(model_5)

vif(model_5)

#  Далее постепенно анализируя vif(регрессор), убираем из модели vif(регрессор) у которого показатель
# больше 10 и больше всех остаьных показателей.

# Убираем I(Infant.Mortality * Fertility).
model_6= lm(Examination ~ Fertility + Catholic + Infant.Mortality + I(Fertility^2) + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Catholic) + I(Fertility*Catholic), data)
model_6
summary(model_6)

vif(model_6)

# Убираем Catholic.
model_7 = lm(Examination ~ Fertility + Infant.Mortality + I(Fertility^2) + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Catholic) + I(Fertility*Catholic), data)
model_7
summary(model_7)

vif(model_7)

# Убираем I(Fertility * Catholic).
model_8 = lm(Examination ~ Fertility + Infant.Mortality + I(Fertility^2) + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Catholic), data)
model_8
summary(model_8)

vif(model_8)

# Убираем I(Fertility^2).
model_9 = lm(Examination ~ Fertility + Infant.Mortality + I(Catholic^2) + I(Infant.Mortality^2) + I(Infant.Mortality*Catholic), data)
model_9
summary(model_9)

vif(model_9)

# Убираем I(Infant.Mortality^2).
model_10 = lm(Examination ~ Fertility + Infant.Mortality + I(Catholic^2) + I(Infant.Mortality*Catholic), data)
model_10
summary(model_10)

vif(model_10)

# Убираем I(Catholic^2).
model_11 = lm(Examination ~ Fertility + Infant.Mortality + I(Infant.Mortality*Catholic), data)
model_11
summary(model_11)

vif(model_11)

#  Таким образом, model_11 хорошая модель. R^2 = 0.5164, что показывает нам достаточно хорошую линейную зависимость.
# р-статистика показывает неплохие результаты у регрессоров.

#  Так как в model_10 у регрессоров I(Catholic^2) и I(Infant.Mortality * Catholic) почти одинаковые показатели vif.
# Попробуем создать еще одну хорошую модель.

# Убираем I(Infant.Mortality * Catholic).
model_12 = lm(Examination ~ Fertility + Infant.Mortality + I(Catholic^2), data)
model_12
summary(model_12)

vif(model_12)

#  Действительно, убрав в model_10  регрессор(I(Infant.Mortality * Catholic)), мы получаем хорошую модель с vif 
# показателями меньше 3. По R^2 модель не уступает предыдущим его показатель равен 0.5497. р-статистика
# также показывает хорошие результаты. Следовательно, эта модель хорошая.

#  В модели 12 показатели R^2 и p-статистика достаточно хороши, следовательно, эта модель является
# наилучшей.


# Рассмотрим доверительные интервалы модели 
modele = lm(Examination ~ Fertility + Infant.Mortality, data)
modele
summary(modele)

#коэффициенты
coef(modele)
#доверительные интервалы коэффициентов
confint(modele, level = 0.90)

#47 наблюдений, оценивалось 4 коэффициента: 47 - 4 = 43 степени свободы
# Доверительный интервал для Fertility [-0.46240 - t * 0.07881;-0.46240 + t * 0.07881]
t_critical = qt(0.975, df = 149) #~1.976 близок к 1.96
se = 0.07881
modele$coefficients[2] - t_critical * se
modele$coefficients[2] + t_critical * se
# [-0.6181266; -0.3066674]
# проверка
confint(modele, level = 0.95)
# 0 не попал в интервал, значит коэффициент не может быть равен 0


# Доверительный интервал для Infant.Mortality[0.51376 - t * 0.33799; 0.51376  + t * 0.33799]
t_critical = qt(0.975, df = 149) #~1.976 близок к 1.96 
se = 0.33799
modele$coefficients[3] - t_critical * se
modele$coefficients[3] + t_critical * se
# [-0.1541124; 1.181633]
# проверка
confint(modele, level = 0.95)
# 0  попал в интервал, значит коэффициент может быть равен 0

# Доверительный интервал для temp [1.65209 - t * 0.25353 ;1.65209 + t * 0.25353 ]
t_critical = qt(0.975, df = 149) #~1.976 близок к 1.96
se = 0.25353
modele$coefficients[4] - t_critical * se
modele$coefficients[4] + t_critical * se
# [1.151114 ; 2.153072]
# проверка
confint(modele, level = 0.95)
# 0 не попал в интервал, значит коэффициент не может быть равен 0
# Все регрессоры имеют некоторую зависимость с объясняемой переменной

# Построение доверительного интервала для прогноза
modele_pr = lm(Examination~Fertility + Catholic + Infant.Mortality, data)
modele_pr
summary(modele_pr)

new.data = data.frame(Fertility = 20,Catholic = 10, Infant.Mortality = 10)
predict(modele_pr, new.data, interval = "confidence")
# fit - значение прогноза = 31.61013
# нижняя и верхняя граница lwr = 23.75496 upr = 39.4653