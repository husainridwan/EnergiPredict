from fastapi import FastAPI, Request, File, UploadFile, FileResponse, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from joblib import dump, load
import pickle

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_consumption(request: Request, message: str = None, error_message: str = None):
    return templates.TemplateResponse(
        "predict.html", 
        {"request": request, "message": message, "error_message": error_message}
    )

@app.post("/predict")
async def upload_and_process_file(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file, encoding='utf-8')
        # Loading the trained model
        model = pickle.load("models/stack.pkl")
        predictions = model.predict(df)

        output_df = pd.DataFrame({
            'Row Id': df.index,
            'Prediction': predictions
        })

        output_df = output_df.sort_values(by='Row Id', ascending=True)  # Ensure sorted rows

        output = "result.csv"
        output_df.to_csv(output, index=False)

        return FileResponse(output, media_type="application/octet-stream", filename=output)

    except UnicodeDecodeError as e:
        raise HTTPException(status_code=422, detail="Error Parsing File. Please Upload a Valid CSV File!") 
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"KeyError: {str(e)}. Ensure correct columns in your Data!")
    except pd.errors.ParserError:
        raise HTTPException(status_code=422, detail="Error parsing CSV file. Upload a valid CSV file with valid columns")
    except Exception as e:
        #logging.critical(e, exc_info=True) 
        raise HTTPException(status_code=500, detail=str(e))