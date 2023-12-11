import onnx
import onnxruntime
import numpy as np

from PIL import Image 

import torch

from torchvision import transforms

#load onnx [success]---------------------------
onnx_model = onnx.load("weights/best_403food_e200b150v2.onnx")
print(onnx.checker.check_model(onnx_model))
#print(onnx.helper.printable_graph(onnx_model.graph))

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#ONNX Runtime Test -------------------
ort_session = onnxruntime.InferenceSession("weights/best_403food_e200b150v2.onnx")

# # 모델 변환
# torch.onnx.export(torch_model,               # 실행될 모델
#                   x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
#                   "super_resolution.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
#                   export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
#                   opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
#                   do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
#                   input_names = ['input'],   # 모델의 입력값을 가리키는 이름
#                   output_names = ['output'], # 모델의 출력값을 가리키는 이름
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
#                                 'output' : {0 : 'batch_size'}})
# torch.onnx.export(model, img, f, verbose=False, opset_version=11,
#                   input_names=['images'], 
#                   output_names=['classes', 'boxes'])

to_tensor = transforms.ToTensor()
img = Image.open("data/samples/11_256.jpg")
ort_inputs = {ort_session.get_inputs()[0].name:[to_tensor(img)]}

#output_classe = {ort_session.get_outputs()[0] : classes}
ort_classe, ort_boxes = ort_session.run(None, ort_inputs)

print(f" ---------Classe[{len(ort_classe)}][{ort_classe.shape}]-------- \n {ort_classe} : ")
print(f" ---------Boxes[{len(ort_boxes)}][{ort_boxes.shape}]-------- \n {ort_boxes}")

arr = ort_classe.flatten()
arr = np.unique(arr)
print(f"합 : {np.sum(arr)}")

arrMax = np.argmax(ort_classe, axis=-1)
print(np.sum(ort_classe))
print(arrMax[-2])


# 배열에서 가장 많이 나온 값을 찾음
most_common_value = np.bincount(arrMax).argmax()

# 가장 많이 나온 값의 모든 인덱스를 찾음
indices = np.where(arrMax == most_common_value)[0]
print(f"가장 많이 나온 값 : {most_common_value}")


# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # ONNX 런타임에서 계산된 결과값
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")