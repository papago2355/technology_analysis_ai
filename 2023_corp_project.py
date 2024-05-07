import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler,minmax_scale
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler,minmax_scale
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,mean_squared_error
import streamlit as st
import seaborn as sns
from scipy import stats
import time
import pickle
st.title('Demo for this project')

################################################################ Data Process
scaler = MinMaxScaler()
df = pd.read_csv('/Users/leejeewoong/Desktop/gibo/기보2023코드/10_16_all_in.csv')
df['ipc코드'] = df['ipc코드'].str[0:4]

le = preprocessing.LabelEncoder()
df['ipc코드'] = le.fit_transform(df['ipc코드'])
le = preprocessing.LabelEncoder()
df['업종 분류'] = le.fit_transform(df['업종 분류'])
le = preprocessing.LabelEncoder()
df['기술분야'] = le.fit_transform(df['기술분야'])

df.replace([np.inf, -np.inf],0, inplace=True)
df = df.drop(columns=['문서'])
df.dropna(inplace=True)
df = df.groupby('기술사업명칭').filter(lambda x : len(x)<=400)
df = df[['ipc코드', 'scores', '핵심성', '독창성', '부가가치',
       '파급성', '최종 평점2', '최종 평점3', '기술분야', '업종 분류', '인용 수', '피인용 수',
       '출원인 수', '패밀리 수', 'support', 'confidence', 'lift', 'leverage',
       'conviction', 'collective strength','기술사업명칭']]
df.columns = ['ipc', 'scores', 'core score', 'unique score', 'economy score', 'impact score', 'total2', 'total3',
       'tech area', 'tech class', 'citation', 'p citation', 'inventor', 'family', 'support',
       'confidence', 'lift', 'leverage', 'conviction', 'collective strength','tech name']
df[['core score', 'unique score', 'economy score', 'impact score', 'total2', 'total3',
       'citation', 'p citation', 'inventor', 'family', 'support',
       'confidence', 'lift', 'leverage', 'conviction', 'collective strength']] = scaler.fit_transform(df[['core score', 'unique score', 'economy score', 'impact score', 'total2', 'total3',
       'citation', 'p citation', 'inventor', 'family', 'support',
       'confidence', 'lift', 'leverage', 'conviction', 'collective strength']]) 
################################################################ Data Process

##### => Section1. Data plot end
st.header('1. Technology Analysis')
st.subheader('This section is for computing scores about the technology')
st.write('기술의 원천성 판단 모델 재학습 중................................')

#Section2. Train AI
y = df[['total3','core score','unique score','economy score','impact score']]
#X = df.drop(columns=['최종 평점2','초록','출원일자','출원번호','문서','embeddings','conviction','부가가치','독창성','핵심성','scores'])
X = df.drop(columns=['total3','total2','core score','unique score','economy score','impact score','tech name'])
# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

xgb_model = pickle.load(open('xgb_regressor_model.pkl', "rb"))
st.write('기술의 원천성 판단 모델 학습 완료! 이제 예측 시간을 도출합니다.')
start_time = time.time()
xgb_model.predict(df.drop(columns=['total3','total2','core score','unique score','economy score','impact score','tech name']))
st.write("---결과 도출까지 %s seconds ---" % (time.time() - start_time))
#Section3. Test Results
st.header('2. Result Analysis')
st.subheader('This section is for show the results of the technology you want to know')
option = st.selectbox(
    'select the tech you want to know',
    (df['tech name'].unique()))

tech1_group = df[df['tech name']==option]
tech1_group_org = tech1_group.copy()
st.subheader('2.1 Technology Area of similiary technologies')

temp_2_1 = df[df['tech area']==tech1_group['tech area'].unique()[0]]
temp_2_1_org = temp_2_1.copy()
temp = temp_2_1['tech name'].value_counts().mean()

temp_2_1 = pd.DataFrame(
   {"this tech": [len(tech1_group)], "area tech": [temp]}
)
#temp_2_1['Tech_same_area'] = temp

st.bar_chart(temp_2_1.T)

tech1_group = xgb_model.predict(tech1_group.drop(columns=['total3','total2','core score','unique score','economy score','impact score','tech name']))
tech2_group = xgb_model.predict(temp_2_1_org.drop(columns=['total3','total2','core score','unique score','economy score','impact score','tech name']))
mean_score = np.mean(tech1_group,axis=0)
mean_score2 = np.mean(tech2_group,axis=0)
st.markdown(f"최종 점수 : **{mean_score[0]}**")
st.markdown(f"핵심성 점수 : **{mean_score[1]}**")
st.markdown(f"독창성 점수 : **{mean_score[2]}**")
st.markdown(f"부가가치 창출 점수 : **{mean_score[3]}**")
st.markdown(f"파급성 점수 : **{mean_score[4]}**")

tech1_group = pd.DataFrame(tech1_group)
tech1_group.columns = ['final score','core score','unique score','economy score','impact score']

tech2_group = pd.DataFrame(tech2_group)
tech2_group.columns = ['final score','core score','unique score','economy score','impact score']
st.dataframe(tech1_group.describe())

st.subheader('기술의 점수 분포')

fig,ax = plt.subplots(figsize=(10, 4),ncols=2,nrows=2)

fig.subplots_adjust(hspace=1)
sns.kdeplot(tech1_group['core score'],ax=ax[0,0])
sns.kdeplot(tech1_group['unique score'],ax=ax[0,1])
sns.kdeplot(tech1_group['economy score'],ax=ax[1,0])
sns.kdeplot(tech1_group['impact score'],ax=ax[1,1])
st.pyplot(fig)

st.subheader('유사 기술 밀집도')
st.write('scores 점수가 100미만인 기술은 유사도가 높은 기술들입니다.')

tech1_group = pd.concat([tech1_group,tech1_group_org.drop(columns=['total3','total2','core score','unique score','economy score','impact score','tech name'])],axis=1)
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
sns.kdeplot(tech1_group['scores'], color="b",fill=True, ax=ax)  # Pass

# vertices
path = ax.collections[0].get_paths()[0]
vertices = path.vertices
codes = path.codes

# threshold
idx_th = np.where(vertices[:, 0] < 100)[0]
vertices_th = vertices[idx_th]
codes_th = codes[idx_th]
path.vertices = vertices_th
path.codes = codes_th
path.codes[0] = 1
path.codes[-1] = 79
sns.kdeplot(tech1_group['scores'], fill=False, color="k", ax=ax)

st.pyplot(fig)


temp = len(tech1_group[tech1_group['scores']>=100])
if temp == 0:
    temp = 1
    st.markdown(f"본 기술과 유사한 기술이 매우 많습니다.")
else:
    st.markdown(f"유사 기술에 해당하는 기술은 유사하지 않은 기술에 비해 **{len(tech1_group[tech1_group['scores']<100])/len(tech1_group[tech1_group['scores']>=100]):.2f}배**의 영역을 가집니다.")

mark1 = len(tech1_group[tech1_group['scores']<100]) < len(tech1_group[tech1_group['scores']>=100])


st.subheader('기술의 원천성 판단 결과')
if (tech1_group['final score'].mean()>0.75):
    st.markdown(f"본 **{option}**은 **기술의 원천성이 높다 판단됩니다.**")
else:
    st.markdown(f"본 **{option}**은 기술의 원천성이 높다고 보기는 어렵습니다.")

diff1 = (tech1_group['core score'].mean()-tech2_group['core score'].mean())/tech2_group['core score'].mean()*100
diff2 = (tech1_group['unique score'].mean()-tech2_group['unique score'].mean())/tech2_group['unique score'].mean()*100
diff3 = (tech1_group['impact score'].mean()-tech2_group['impact score'].mean())/tech2_group['impact score'].mean()*100
diff4 = (tech1_group['economy score'].mean()-tech2_group['economy score'].mean())/tech2_group['economy score'].mean()*100
st.markdown(f"본 기술과 동종 기술 영역 대비, 핵심성 점수가 {diff1:.2f} % 차이가 납니다.")

st.markdown(f"본 기술과 동종 기술 영역 대비, 독창성 점수가  {diff2:.2f} % 차이가 납니다.")

st.markdown(f"본 기술과 동종 기술 영역 대비, 파급성 점수가  {diff3:.2f} % 차이가 납니다.")

st.markdown(f"본 기술과 동종 기술 영역 대비, 부가가치 창출 점수가  {diff4:.2f} % 차이가 납니다.")
