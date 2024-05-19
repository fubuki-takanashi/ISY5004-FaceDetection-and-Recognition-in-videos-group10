import glob
import cv2
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN


IMG_PATH = './data_tbbt/face_images'
DATA_PATH = './data_tbbt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def adjust_box(box, img):
    img_size = img.shape
    box = [
        int(max(box[0], 0)),
        int(max(box[1], 0)),
        int(min(box[2], img_size[1])),
        int(min(box[3], img_size[0])),
    ]
    return box


def trans(img):
    img = img.resize((160, 160))

    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

threshold = [0.6, 0.7, 0.7]
mtcnn = MTCNN(threshold ,keep_all=True, device = device)
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device).eval()
# vggface2

embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr) + '/*.jpg'):
        # img = Image.open(file)
        img = cv2.imread(file)

        with torch.no_grad():
            bounding_boxes, _ = mtcnn.detect(img)
            if bounding_boxes is not None:
                # 只有一个脸
                face_position = bounding_boxes[0]
                face_position = face_position.astype(int)

                # 裁剪出人脸区域作为第二个模型的输入
                face_position = adjust_box(face_position, img)
                cropped = img[face_position[1]:face_position[3], face_position[0]:face_position[2], :]
                # print(cropped.shape)
                scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)
                img = Image.fromarray(scaled)

        embed = model(trans(img).to(device).unsqueeze(0))
        embeds.append(embed)


    if len(embeds) == 0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True)

    embeddings.append(embedding)
    names.append(usr)

embeddings = torch.cat(embeddings)  # [n,512]
names = np.array(names)

torch.save(embeddings, DATA_PATH + "/facesDataset.pth")
np.save(DATA_PATH + "/usernames", names)
print('Update Completed! There are {0} people in facesDataset'.format(names.shape[0]))