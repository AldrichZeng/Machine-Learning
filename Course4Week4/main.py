from FaceRecognition import *

np.set_printoptions(threshold=np.nan)
# 获取模型
FRmodel=faceRecoModel(input_shape=(3,96,96))
# 打印模型的总参数数量
print("参数数量："+str(FRmodel.count_params()))

