import pandas as pd
from matplotlib import pyplot

dataset = pd.read_csv("healthcare-dataset-stroke-data.csv", index_col="id")

print("Size of raw dataset\n", dataset.shape)
print("5 first rows of raw dataset\n", dataset.head(5))
print("Summary of raw dataset\n", dataset.describe()
      )
# histogram of the variable
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()