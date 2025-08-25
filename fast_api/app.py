

import io
import re
import pickle
import uvicorn
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.dates as mdates
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment



# Load the model and vectorizer from the model registry and local storage
# def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
#     # Set MLflow tracking URI to your server
#     mlflow.set_tracking_uri("http://34.155.138.211:5000/")
#     client = MlflowClient()
#     model_uri = f"models:/{model_name}/{model_version}"
#     model = mlflow.pyfunc.load_model(model_uri)
#     with open(vectorizer_path, 'rb') as file:
#         vectorizer = pickle.load(file)
   
#     return model, vectorizer



def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        return model, vectorizer
    except Exception as e:
        raise

# Initialize the model and vectorizer
# model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed

# Initialize the model and vectorizer
model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")  

@app.get("/")
def home():
    return {"message": "Welcome to our FastAPI service"}

@app.post("/predict_with_timestamps")
async def predict_with_timestamps(request: Request):
    data = await request.json()
    comments_data = data.get("comments")
    if not comments_data:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [item["text"] for item in comments_data]
        timestamps = [item["timestamp"] for item in comments_data]

        preprocessed = [preprocess_comment(c) for c in comments]
        dense_comments = vectorizer.transform(preprocessed).toarray()
        predictions = model.predict(dense_comments).tolist()

        response = [
            {"comment": c, "sentiment": str(p), "timestamp": t}
            for c, p, t in zip(comments, predictions, timestamps)
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    comments = data.get("comments")
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        dense_comments = vectorizer.transform(preprocessed).toarray()
        predictions = model.predict(dense_comments).tolist()

        response = [{"comment": c, "sentiment": p} for c, p in zip(comments, predictions)]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/generate_chart")
async def generate_chart(request: Request):
    try:
        data = await request.json()
        sentiment_counts = data.get("sentiment_counts")
        if not sentiment_counts:
            raise HTTPException(status_code=400, detail="No sentiment counts provided")

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, textprops={'color': 'w'})
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {e}")


@app.post("/generate_wordcloud")
async def generate_wordcloud(request: Request):
    try:
        data = await request.json()
        comments = data.get("comments")
        if not comments:
            raise HTTPException(status_code=400, detail="No comments provided")

        preprocessed = [preprocess_comment(c) for c in comments]
        text = " ".join(preprocessed)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {e}")


@app.post("/generate_trend_graph")
async def generate_trend_graph(request: Request):
    try:
        data = await request.json()
        sentiment_data = data.get("sentiment_data")
        if not sentiment_data:
            raise HTTPException(status_code=400, detail="No sentiment data provided")

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for s in [-1, 0, 1]:
            if s not in monthly_percentages.columns:
                monthly_percentages[s] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}
        for s in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[s],
                marker="o",
                linestyle="-",
                label=sentiment_labels[s],
                color=colors[s],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {e}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)