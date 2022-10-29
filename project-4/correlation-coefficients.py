import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
file = pd.read_csv("./beginner-repo/project-4/mycsv.csv")

li_pregs = file["pregs"]
li_plas = file["plas"]
li_pres = file["pres"]
li_skin = file["skin"]
li_test = file["test"]
li_bmi = file["BMI"]
li_pedi = file["pedi"]
li_age = file["Age"]
li_class = file["class"]
print("Means are:\n")
arr = []
#mean
arr.append(li_pregs.mean())
arr.append(li_plas.mean())
arr.append(li_pres.mean())
arr.append(li_skin.mean())
arr.append(li_test.mean())
arr.append(li_bmi.mean())
arr.append(li_pedi.mean())
arr.append(li_age.mean())
x = 0
while x<8:
    print(arr[x])
    x+=1
print("Medians are:\n")
#median
arr = []
arr.append(li_pregs.median())
arr.append(li_plas.median())
arr.append(li_pres.median())
arr.append(li_skin.median())
arr.append(li_test.median())
arr.append(li_bmi.median())
arr.append(li_pedi.median())
arr.append(li_age.median())
x = 0
while x<8:
    print(arr[x])
    x+=1
print("Modes are:\n")
#mode
arr = []
arr.append(st.mode(li_pregs))
arr.append(st.mode(li_plas))
arr.append(st.mode(li_pres))
arr.append(st.mode(li_skin))
arr.append(st.mode(li_test))
arr.append(st.mode(li_bmi))
arr.append(st.mode(li_pedi))
arr.append(st.mode(li_age))
x = 0
while x<8:
    print(arr[x])
    x+=1

print("The min. val in each field:\n")

#minimum
arr = []
arr.append(li_pregs.min())
arr.append(li_plas.min())
arr.append(li_pres.min())
arr.append(li_skin.min())
arr.append(li_test.min())
arr.append(li_bmi.min())
arr.append(li_pedi.min())
arr.append(li_age.min())
x = 0
while x<8:
    print(arr[x])
    x+=1

print("The maximum value in each field:\n")

#maximum
arr = []
arr.append(li_pregs.max())
arr.append(li_plas.max())
arr.append(li_pres.max())
arr.append(li_skin.max())
arr.append(li_test.max())
arr.append(li_bmi.max())
arr.append(li_pedi.max())
arr.append(li_age.max())
x = 0
while x<8:
    print(arr[x])
    x+=1

#Standard Dev
print("Standard Dev:\n")

arr = []
arr.append(li_pregs.std())
arr.append(li_plas.std())
arr.append(li_pres.std())
arr.append(li_skin.std())
arr.append(li_test.std())
arr.append(li_bmi.std())
arr.append(li_pedi.std())
arr.append(li_age.std())
x = 0
while x<8:
    print(arr[x])
    x+=1

#2(a)

plt.scatter(li_age, li_pregs, color = "red")
plt.title("Age vs. Number of times pregnant")
plt.xlabel('Age')
plt.ylabel('Number of times Pregnant')
plt.show()

plt.scatter(li_age, li_plas, color = "blue")
plt.title('Age vs. Plasma glucose concentration')
plt.xlabel('Age')
plt.ylabel('Plasma glucose concentration')
plt.show()

plt.scatter(li_age, li_pres, color = "green")
plt.title("Age vs. Diastolic blood pressure")
plt.xlabel('Age')
plt.ylabel('Diastolic blood pressure')
plt.show()

plt.scatter(li_age, li_skin, color = "green")
plt.title("Age vs. Triceps skin fold thickness")
plt.xlabel('Age')
plt.ylabel('Triceps skin fold thickness')
plt.show()

plt.scatter(li_age, li_test, color = "green")
plt.title("Age vs. 2-Hour serum insulin")
plt.xlabel('Age')
plt.ylabel('2-Hour serum insulin')
plt.show()

plt.scatter(li_age, li_bmi, color = "red")
plt.title("Age vs. BMI")
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()

plt.scatter(li_age, li_pedi, color = "blue")
plt.title("Age vs. Diabetes pedigree")
plt.xlabel('Age')
plt.ylabel('Diabetes pedigree')
plt.show()

#2(b)

plt.scatter(li_bmi, li_pregs, color = "red")
plt.title("BMI vs. Number of times pregnant")
plt.xlabel('BMI')
plt.ylabel('Number of times Pregnant')
plt.show()

plt.scatter(li_bmi, li_plas, color = "blue")
plt.title('BMI vs. Plasma glucose concentration')
plt.xlabel('BMI')
plt.ylabel('Plasma glucose concentration')
plt.show()

plt.scatter(li_bmi, li_pres, color = "green")
plt.title("BMI vs. Diastolic blood pressure")
plt.xlabel('BMI')
plt.ylabel('Diastolic blood pressure')
plt.show()

plt.scatter(li_bmi, li_skin, color = "green")
plt.title("BMI vs. Triceps skin fold thickness")
plt.xlabel('BMI')
plt.ylabel('Triceps skin fold thickness')
plt.show()

plt.scatter(li_bmi, li_test, color = "green")
plt.title("BMI vs. 2-Hour serum insulin")
plt.xlabel('BMI')
plt.ylabel('2-Hour serum insulin')
plt.show()

plt.scatter(li_bmi, li_age, color = "red")
plt.title("BMI vs. Age")
plt.xlabel('BMI')
plt.ylabel('Age')
plt.show()

plt.scatter(li_bmi, li_pedi, color = "blue")
plt.title("BMI vs. Diabetes pedigree")
plt.xlabel('BMI')
plt.ylabel('Diabetes pedigree')
plt.show()

#3
print("Correlation Coefficients:\n")
print(file.corr())

#4
#histograms

# file.hist("pregs",bins=20)
file.hist(column = "pregs")
file.hist(column = "skin")
plt.show()

#5
file["pregs"].hist(by = file["class"])
plt.show()
#6
ax = file[['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age']].plot(kind='box', title='Boxplots', showmeans=True)
plt.show()


