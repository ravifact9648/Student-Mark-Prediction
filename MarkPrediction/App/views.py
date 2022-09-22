from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd

# Create your views here.
df = pd.DataFrame()
model=joblib.load(r"C:\Users\Ravi Prakash Yadav\Desktop\Projects\Student Mark Prediction\MarkPrediction\MarkPrediction_ML\student_mark_prediction.pkl")
def home(request):
    return render(request,'App/index.html')

def predict(request):
    global df
    input_features = [int(x) for x in request.POST.values()]
    features_value = np.array(input_features)
    
    #validate input hours
    
    if input_features[0] >12.7:
        if input_features[0] <0 or input_features[0] >24:
            return render(request,'App/index.html', context={'prediction_text':'Please enter valid hours between 1 to 24 if you live on the Earth'})
        return render(request,'App/index.html', context={'prediction_text':'It is recommended to Study below 12.5 hr a day '})
        
        

    output = model.predict([features_value])[0][0].round(2)

    # input and predicted value store in df then save in csv file
    df= pd.concat([df,pd.DataFrame({'Study Hours':input_features,'Predicted Output':[output]})],ignore_index=True)
    print(df)   
    df.to_csv('smp_data_from_app.csv')

    return render(request,'App/index.html', context={'prediction_text':'You will get [{}%] marks, when you do study [{}] hours per day '.format(output, int(features_value[0]))})

