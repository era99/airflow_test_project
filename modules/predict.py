import pandas as pd
import os
import json
import dill
from datetime import datetime

#/opt/airflow/project
model_path = os.environ.get('PROJECT_PATH', 'C:/Users/AKHMEERA/airflow_hw')
test_data_path = os.environ.get('PROJECT_PATH', 'C:/Users/AKHMEERA/airflow_hw')
predictions_path = os.environ.get('PROJECT_PATH', 'C:/Users/AKHMEERA/airflow_hw')


def load_model(filename):
    with open(filename, 'rb') as model_file:
        model = dill.load(model_file)

    return model


def load_test_data(test_dir):
    data_list = []
    for filename in os.listdir(test_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(test_dir, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
            data_list.append(pd.DataFrame([data]))

    return pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()


def save_predictions(predictions, output_path):
    os.makedirs(output_path, exist_ok=True)
    output_file = f'{output_path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    predictions.to_csv(output_file, index=False)
    print(f"Предсказания сохранены в {output_file}")


def predict():
    model = load_model(f'{model_path}/data/models/cars_pipe_202503310550.pkl')
    df_test = load_test_data(f'{test_data_path}/data/test/')
    if df_test.empty:
        print("Нет тестовых данных для предсказания.")
        return

    predictions = model.predict(df_test)
    df_test["prediction"] = predictions

    save_predictions(df_test[['id', 'prediction']], predictions_path)


if __name__ == '__main__':
    predict()
