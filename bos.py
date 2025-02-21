import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tkinter import Tk, Button, Frame, Label


# Step 1: Create or Load the Dataset
np.random.seed(42)  # For reproducibility
n_samples = 100

data = pd.DataFrame({
    "Price": np.random.uniform(200000, 800000, n_samples),
    "AirQuality": np.random.uniform(0, 100, n_samples),  # 0: Best, 100: Worst
    "Rooms": np.random.randint(2, 7, n_samples),
    "LotSize": np.random.uniform(500, 2000, n_samples),  # in square meters
    "Income": np.random.uniform(40000, 120000, n_samples),  # Annual income
    "SchoolQuality": np.random.uniform(1, 10, n_samples)  # 1: Worst, 10: Best
})

# Step 2: Define Features and Target Variable
X = data[["AirQuality", "Rooms", "LotSize", "Income", "SchoolQuality"]]
y = data["Price"]

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

# Step 7: Prepare Graphs
graphs = []

# Graph 1: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("graph1.png")
graphs.append("graph1.png")

# Graph 2: Predictions vs Actual Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predictions vs Actual Housing Prices")
plt.legend()
plt.savefig("graph2.png")
graphs.append("graph2.png")

# Graph 3: Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x="Coefficient", y="Feature", data=coefficients)
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.savefig("graph3.png")
graphs.append("graph3.png")


# Step 8: Build GUI for Navigation
class GraphViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Graph Viewer")
        self.graph_index = 0

        # Display Frame
        self.frame = Frame(master)
        self.frame.pack()

        # Label to Display Graph
        self.label = Label(self.frame)
        self.label.pack()

        # Navigation Buttons
        self.prev_button = Button(master, text="Previous", command=self.show_previous)
        self.prev_button.pack(side="left")

        self.next_button = Button(master, text="Next", command=self.show_next)
        self.next_button.pack(side="right")

        self.show_graph()

    def show_graph(self):
        image_path = graphs[self.graph_index]
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def show_next(self):
        if self.graph_index < len(graphs) - 1:
            self.graph_index += 1
            self.show_graph()

    def show_previous(self):
        if self.graph_index > 0:
            self.graph_index -= 1
            self.show_graph()


# Run the GUI
root = Tk()
app = GraphViewer(root)
root.mainloop()
