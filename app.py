from flask import Flask, render_template, request, redirect, url_for, session, flash
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import pickle
import pandas as pd
import plotly.express as px

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-secret"

USERS_FILE = "users.json"

# Load trained model and scaler
model = pickle.load(open("kmeans.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Cluster label mapping
cluster_names = {
    0: "High Value Customer",
    1: "Low Value Customer",
    2: "Target Customer",
    3: "Impulsive Buyer",
    4: "Average Customer"
}

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


@app.route('/')
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = load_users()

        if username and username in users:
            user_data = users[username]
            if check_password_hash(user_data["password"], password):
                session["user"] = username
                flash("Login successful", "success")
                return redirect(url_for("dashboard"))

        flash("Invalid username or password", "error")

    return render_template("login.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if "user" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        users = load_users()

        if not username or not password:
            flash("Username and password are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif username in users:
            flash("Username already exists.", "error")
        else:
            users[username] = {"password": generate_password_hash(password)}
            save_users(users)
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# Prediction
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    income = float(request.form['income'])
    score = float(request.form['score'])

    data = scaler.transform([[income, score]])
    cluster = model.predict(data)[0]
    label = cluster_names.get(cluster, "Unknown")

    result = f"Cluster {cluster} - {label}"
    return render_template("index.html", prediction=result, user=session.get("user"))


@app.route('/app')
@login_required
def app_page():
    return render_template("index.html", user=session.get("user"))

# Dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    df = pd.read_csv("Mall_Customers.csv")

    # Apply clustering with the trained model
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    df["Cluster"] = model.predict(scaler.transform(X))

    # Derived categories
    def spending_category(score):
        if score >= 70:
            return "High Spender"
        if score >= 40:
            return "Mid Spender"
        return "Low Spender"

    def income_group(value):
        if value < 40:
            return "Low Income (0-40k$)"
        if value < 80:
            return "Mid Income (40-80k$)"
        return "High Income (80k$+)"

    df["Spending Category"] = df["Spending Score (1-100)"].apply(spending_category)
    df["Income Group"] = df["Annual Income (k$)"].apply(income_group)

    # KPIs
    total_customers = len(df)
    avg_spending = round(df["Spending Score (1-100)"].mean(), 2)
    avg_income = round(df["Annual Income (k$)"].mean(), 2)
    age_range = f"{int(df['Age'].min())} - {int(df['Age'].max())}"
    high_spenders = int((df["Spending Category"] == "High Spender").sum())
    cluster_count = df["Cluster"].nunique()

    # Main scatter chart
    scatter_fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=df["Cluster"].astype(str),
        labels={
            "Annual Income (k$)": "Annual Income (k$)",
            "Spending Score (1-100)": "Spending Score"
        },
        title="Annual Income vs Spending Score",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        template="plotly_dark"
    )
    scatter_html = scatter_fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Spending category distribution
    pie_fig = px.pie(
        df,
        names="Spending Category",
        hole=0.45,
        title="Spending Category Distribution",
        color_discrete_sequence=["#facc15", "#22c55e", "#ef4444"]
    )
    pie_html = pie_fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Spending by income group
    income_fig = px.histogram(
        df,
        x="Income Group",
        color="Spending Category",
        barmode="stack",
        category_orders={
            "Income Group": ["Low Income (0-40k$)", "Mid Income (40-80k$)", "High Income (80k$+)"]
        },
        title="Spending Category by Income Group",
        template="plotly_dark",
        color_discrete_map={
            "High Spender": "#22c55e",
            "Mid Spender": "#facc15",
            "Low Spender": "#ef4444"
        }
    )
    income_html = income_fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Average income and spending by gender
    gender_df = df.groupby("Gender").agg({
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean"
    }).reset_index()
    gender_df["Annual Income (k$)"] = gender_df["Annual Income (k$)"].round(2)
    gender_df["Spending Score (1-100)"] = gender_df["Spending Score (1-100)"].round(2)

    gender_fig = px.bar(
        gender_df,
        x="Gender",
        y=["Spending Score (1-100)", "Annual Income (k$)"],
        barmode="group",
        title="Avg Income & Spending Score by Gender",
        labels={
            "value": "Average Value",
            "variable": "Metric"
        },
        template="plotly_dark",
        color_discrete_sequence=["#7c3aed", "#2dd4bf"]
    )
    gender_html = gender_fig.to_html(full_html=False, include_plotlyjs='cdn')

    cluster_info = [
        {"cluster": cluster, "label": label}
        for cluster, label in sorted(cluster_names.items())
    ]

    return render_template(
        "dashboard.html",
        total=total_customers,
        avg_spending=avg_spending,
        avg_income=avg_income,
        high_spenders=high_spenders,
        age_range=age_range,
        cluster_count=cluster_count,
        scatter_html=scatter_html,
        pie_html=pie_html,
        income_html=income_html,
        gender_html=gender_html,
        cluster_info=cluster_info,
        user=session.get("user")
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
