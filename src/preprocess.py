import pandas as pd

def load_data(path):
    """
    CSV 파일을 불러옵니다.
    """
    df = pd.read_csv(path, index_col=0)
    return df

def tag_data_sufficiency(df, threshold=3):
    """
    각 행의 결측치 개수를 기준으로 'Sufficient' 또는 'Insufficient' 라벨을 붙입니다.
    threshold 이상 결측치가 있으면 'Insufficient'
    """
    df = df.copy()
    df["data_status"] = df.isnull().sum(axis=1).apply(lambda x: "Insufficient" if x >= threshold else "Sufficient")
    return df

def sample_balanced_train_data(df, sample_per_class=2500):
    """
    결측치가 없는 샘플 중에서 클래스 0과 1을 각각 sample_per_class개씩 샘플링하여 학습용 데이터 생성.
    """
    df_no_na = df[df.isnull().sum(axis=1) == 0]
    df_class_0 = df_no_na[df_no_na["SeriousDlqin2yrs"] == 0].sample(sample_per_class, random_state=42)
    df_class_1 = df_no_na[df_no_na["SeriousDlqin2yrs"] == 1].sample(sample_per_class, random_state=42)
    df_train = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42).copy()
    return df_train

def sample_test_data(df, df_train, test_size=1000, missing_size=50):
    """
    테스트 데이터는 결측치 포함 샘플 missing_size개 + 나머지 무작위 샘플로 구성.
    총 test_size개 반환.
    """
    df_test_missing = df[df.isnull().sum(axis=1) > 0].sample(missing_size, random_state=42)
    df_test_rest = df.drop(df_train.index).sample(test_size - missing_size, random_state=42)
    df_test = pd.concat([df_test_missing, df_test_rest]).sample(frac=1, random_state=42).copy()
    return df_test

def impute_test_data(df_test, df_train):
    """
    테스트 데이터의 결측치를 학습 데이터 기준 median으로 보간.
    """
    df_test = df_test.copy()
    for col in ["MonthlyIncome", "NumberOfDependents"]:
        median = df_train[col].median()
        df_test[col].fillna(median, inplace=True)
    return df_test
