import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('loss.xlsx')

colors = ['red', 'lightgreen', 'blue']
fig, ax = plt.subplots()

for i, column in enumerate(df.columns):
    ax.plot(df.index, df[column], label=column, color=colors[i])
ax.legend()

ax.set_title('Video Reconstruction Loss')
ax.set_xlabel('Epoches')
ax.set_ylabel('MSE')

plt.savefig('line_plot.png', dpi=600)

plt.show()