import matplotlib.pyplot as plt
# plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.962,0.952,0.95,0.952,0.948,0.946,0.947,0.931,0.928], 'ro')
# plt.axis([0, 1, 0.9, 1])
# t = plt.xlabel('Percentage of Testing Data', fontsize=14)
# t = plt.ylabel('Accuracy of Naive Bayes', fontsize=14)
# plt.show()

# plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.71, 0.782, 0.744, 0.735, 0.731, 0.692, 0.696, 0.65, 0.651], 'ro')
# plt.axis([0, 1, 0.6, 1])
# t = plt.xlabel('Percentage of Testing Data', fontsize=14)
# t = plt.ylabel('Accuracy of Decision Tree', fontsize=14)
# plt.show()

# plt.plot([1,5,10,15,20,25,30], [0.73,0.728,0.776,0.725,0.701,0.732,0.728], 'ro')
# plt.axis([0, 35, 0.6, 1])
# t = plt.xlabel('Max Depth of Decision Tree', fontsize=14)
# t = plt.ylabel('Accuracy', fontsize=14)
# plt.show()
import numpy as np

# plt.plot([3,5,7,9,11,13,15,17], [0.74,0.726,0.715,0.729,0.741,0.758,0.743,0.749], 'ro')
# plt.axis([0, 20, 0.6, 1])
# t = plt.xlabel('k of KNN', fontsize=14)
# t = plt.ylabel('Accuracy', fontsize=14)
# plt.xticks(np.arange(0, 20, 1.0))
# plt.show()

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# classifiers = ['Naive Bayes', 'KNN L1', 'KNN L2', 'KNN L_inf', 'Decision Tree']
# data = [23,17,35,29,12]
# ax.bar(classifiers,data)
# plt.show()

# plt.style.use('ggplot')
#
# x = ['Naive Bayes', 'KNN L1', 'KNN L2', 'KNN L_inf', 'Decision Tree']
# accuracy = [0.956,0.782,0.725,0.709,0.758]
#
# x_pos = [i for i, _ in enumerate(x)]
#
# plt.bar(x_pos, accuracy, color='green')
# plt.xlabel("Classifier Types")
# plt.ylabel("Accuracy Rate")
# plt.title("Accuracy of classifiers")
#
# plt.xticks(x_pos, x)
#
# plt.show()