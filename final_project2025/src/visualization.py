import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_data(data):
    """Visualize the distribution of Information Types in the dataset."""

    plt.figure(figsize=(10, 6))

    sns.countplot(data=data,
                  x='Information Type',
                  order=data['Information Type'].value_counts().index)

    plt.title('Distribution of Information Types in Tweets')

    plt.xticks(rotation=45)

    plt.xlabel('Information Type')

    plt.ylabel('Count')

    plt.tight_layout()

    plt.show()
