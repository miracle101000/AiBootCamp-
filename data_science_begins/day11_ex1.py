import matplotlib.pyplot as plt


# years = [2010, 2011, 2012, 2013]
# sales = [100, 120, 140, 160]
# # plt.plot(years, sales, label="Sales Trend",color="red", marker="o")
# # plt.title("Sales over Years")
# # plt.xlabel("Years")
# # plt.ylabel("Sales")
# # plt.legend()
# # plt.show()


# #Bar Chart
# categrories = ['Electronics', 'Clothing', 'Groceries']
# revenue = [250, 400, 150] 
# plt.bar(categrories, revenue, color="blue")
# plt.title("Revenue by Category")
# plt.show()

#Scatter Plot
hours_studied = [1, 2, 3, 4, 5]
exam_scores = [50, 55, 65, 70, 85]
plt.scatter(hours_studied, exam_scores, color="red")
plt.title("Exam Scores vs Hours Studied")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
# plt.show()