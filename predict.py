import streamlit as st
import pandas as pd
import joblib

# 设置页面标题
st.title('Prevalence of myopia in 1 or 3 years')

# 用户输入
year = st.selectbox('Number of years', ['', '1 year', '3 years'], index=0, key='year')
grade = st.slider('Grade', 0, 100, 0, key='grade')
sex = st.selectbox('Sex', ['', 'Man', 'Woman'], index=0, key='sex')
resident = st.selectbox('Resident', ['', 'urban', 'village'], index=0, key='resident')
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1, key='bmi')
parental_myopia = st.selectbox('Do either of your parents have myopia?', ['', 'Yes', 'No'], index=0, key='parental_myopia')
parental_education = st.selectbox('Your parents educational background?', ['', 'High school and below', 'Bachelor', 'Master or above'], index=0, key='parental_education')
academic_pressure = st.selectbox('Do you have academic pressure?', ['', 'Yes', 'No'], index=0, key='academic_pressure')
bad_writing_habits = st.selectbox('How many bad writing habits do you have?', ['', '0', '1', '2', '3', '4', '5'], index=0, key='bad_writing_habits')
work_study_time_per_day = st.selectbox('Working/Studying time per day', ['', '<6h', '6-8h', '8-10h', '>10h'], index=0, key='work_study_time_per_day')
continuous_work_study_time_per_day = st.selectbox('Continuous working/studying time per day', ['', '<1h', '1-2h', '2-3h', '>3h'], index=0, key='continuous_work_study_time_per_day')
screen_time = st.selectbox('Screen time per day', ['', '<0.5h', '0.5-1h', '1-2h', '>2h'], index=0, key='screen_time')
sleep_time = st.selectbox('Sleeping time per day', ['', '<7h', '7-9h', '>9h'], index=0, key='sleep_time')
outdoor_time = st.selectbox('Outdoor time per day', ['', '<1h', '1-2h', '2-3h', '>3h'], index=0, key='outdoor_time')
frequency_of_sugary_snack = st.selectbox('Frequency of sugary snack', ['', 'less than once per month', 'monthly', 'weekly', 'daily'], index=0, key='frequency_of_sugary_snack')

# 创建映射字典
mapping = {
    'Year': {'1 year': 1, '3 years': 3},
    'Sex': {'Man': 1, 'Woman': 2},
    'Resident': {'urban': 1, 'village': 0},
    'Parental_myopia': {'Yes': 1, 'No': 0},
    'Parental_education': {'High school and below': 1, 'Bachelor': 2, 'Master or above': 3},
    'Academic_pressure': {'Yes': 1, 'No': 0},
    'Bad_writing_habits': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
    'Work_study_time_per_day': {'<6h': 1, '6-8h': 2, '8-10h': 3, '>10h': 4},
    'Continuous_work_study_time_per_day': {'<1h': 1, '1-2h': 2, '2-3h': 3, '>3h': 4},
    'Screen_time': {'<0.5h': 1, '0.5-1h': 2, '1-2h': 3, '>2h': 4},
    'Sleep_time': {'<7h': 1, '7-9h': 2, '>9h': 3},
    'Outdoor_time': {'<1h': 1, '1-2h': 2, '2-3h': 3, '>3h': 4},
    'Frequency_of_sugary_snack': {'less than once per month': 1, 'monthly': 2, 'weekly': 3, 'daily': 4}
}

# 检查所有必填项是否已填写
if st.button('Submit'):
    if '' in [year, sex, resident, parental_myopia, parental_education, academic_pressure, bad_writing_habits, work_study_time_per_day, continuous_work_study_time_per_day, screen_time, sleep_time, outdoor_time, frequency_of_sugary_snack]:
        st.error('Please fill in all required fields.')
    else:
        # 映射输入数据
        input_data = pd.DataFrame({
            'Year': [mapping['Year'][year]],
            'Grade': [grade],
            'Sex': [mapping['Sex'][sex]],
            'Resident': [mapping['Resident'][resident]],
            'BMI': [bmi],
            'Parental_myopia': [mapping['Parental_myopia'][parental_myopia]],
            'Parental_education': [mapping['Parental_education'][parental_education]],
            'Academic_pressure': [mapping['Academic_pressure'][academic_pressure]],
            'Bad_writing_habits': [mapping['Bad_writing_habits'][bad_writing_habits]],
            'Work_study_time_per_day': [mapping['Work_study_time_per_day'][work_study_time_per_day]],
            'Continuous_work_study_time_per_day': [mapping['Continuous_work_study_time_per_day'][continuous_work_study_time_per_day]],
            'Screen_time': [mapping['Screen_time'][screen_time]],
            'Sleep_time': [mapping['Sleep_time'][sleep_time]],
            'Outdoor_time': [mapping['Outdoor_time'][outdoor_time]],
            'Frequency_of_sugary_snack': [mapping['Frequency_of_sugary_snack'][frequency_of_sugary_snack]]
        })

        # 显示输入数据
        st.subheader('New Observations')
        st.write(input_data)

        # 加载模型
        if input_data['Year'].iloc[0] == 1:
            loaded_model = joblib.load('../model/5年发病率best_model.pkl')
        else:
            loaded_model = joblib.load('../model/3年发病率best_model.pkl')
        prediction = loaded_model.predict(input_data.iloc[:, 1:])
        st.subheader('Prediction')
        st.write(f'The predicted outcome is: {prediction[0]}')

