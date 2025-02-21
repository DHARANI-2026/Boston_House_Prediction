# Boston_House_Prediction
.

🏡 Housing Price Prediction with Graph Viewer
This project demonstrates a machine learning pipeline to predict housing prices based on various features like air quality, number of rooms, lot size, income, and school quality. It also includes a graphical user interface (GUI) built with Tkinter to navigate through generated visualizations.

🚀 Project Overview
Data Generation: Synthetic dataset of 100 samples with housing features.
Model Training: Linear Regression model to predict housing prices.
Evaluation: Performance metrics like Mean Squared Error (MSE) and R² score.
Visualization: Graphs for correlation heatmap, predictions vs. actual prices, and feature importance.
GUI: A user-friendly interface to navigate through generated graphs.
📦 Project Structure
bash
Copy
Edit
├── main.py          # Main script for data processing, model training, and GUI
├── graph1.png       # Correlation Heatmap
├── graph2.png       # Predicted vs Actual Prices
├── graph3.png       # Feature Importance
└── README.md        # Project documentation (this file)
⚙️ Installation & Setup
Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
Create a Virtual Environment (Optional but Recommended)
bash
Copy
Edit
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Create a requirements.txt by running:

bash
Copy
Edit
pip freeze > requirements.txt
Run the Application
bash
Copy
Edit
python main.py

🖼️ Graphs Generated
Correlation Heatmap: Shows relationships between features and target.
Predictions vs Actual Prices: Visualizes model performance.
Feature Importance: Displays regression coefficients as feature weights.
🛠️ Technologies Used
Programming Language: Python
Libraries:
Data Analysis: pandas, numpy
Visualization: matplotlib, seaborn
Machine Learning: scikit-learn
GUI: tkinter
📝 Example Output
Performance Metrics:

Mean Squared Error (MSE): varies based on random seed
R² Score: varies based on random seed
🤝 Contribution
Contributions are welcome!

Fork the project.
Create your branch (git checkout -b feature/new-feature).
Commit your changes (git commit -m "Add new feature").
Push to the branch (git push origin feature/new-feature).
Open a pull request.
