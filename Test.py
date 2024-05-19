import cv2
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

# Parameters
threshold = [0.6, 0.7, 0.7]  # MTCNN's threshold
tolerance = 3  # Threshold for recognizing face
image_size = 160
DATA_PATH = './data_tbbt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 30, (1280, 720))
output_lmm = cv2.VideoWriter('lmm_output.avi', fourcc, 30, (1280, 720))
output_al = cv2.VideoWriter('al_output.avi', fourcc, 30, (1280, 720))
if not output_movie.isOpened():
    print("Error: Output movie file could not be opened.")
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def load_facesDataset():
    embeds = torch.load(DATA_PATH+'/facesDataset.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names

mtcnn = MTCNN(threshold, keep_all=True, device=device)
model = InceptionResnetV1(classify=False, pretrained="casia-webface").to(device).eval()

# Encoding known faces
embeds, names = load_facesDataset()

# Open video file
video_capture = cv2.VideoCapture("tbbt.mp4")
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Dictionary to store timestamps where each person appears
face_timestamps = {name: [] for name in names}

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect bounding boxes and landmarks
    bounding_boxes, _, points = mtcnn.detect(frame, landmarks=True)
    face_names = []

    if bounding_boxes is not None:
        for face_position in bounding_boxes:
            face_position = face_position.astype(int)
            face_position = adjust_box(face_position, frame)
            cropped = frame[face_position[1]:face_position[3], face_position[0]:face_position[2], :]
            scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
            face_img = Image.fromarray(scaled)

            embeds_set = []
            embeds_set.append(model(trans(face_img).to(device).unsqueeze(0)))
            detect_embeds = torch.cat(embeds_set)  # [1,512]
            norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(embeds, 0, 1).unsqueeze(0)
            norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)  # (1,n)

            min_dist, embed_idx = torch.min(norm_score, dim=1)
            if min_dist * pow(10, 6) > tolerance:
                continue
            else:
                name = names[embed_idx]
                face_names.append(name)
                timestamp = frame_count / fps
                face_timestamps[name].append(timestamp)

            cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (255, 255, 0), 2)
            cv2.putText(frame, name, (face_position[0], face_position[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), thickness=2, lineType=2)

    frame_count += 1

    output_movie.write(frame)
    if face_names.count("penny") > 0:
        output_lmm.write(frame)
    if face_names.count("sheldon") > 0:
        output_al.write(frame)

    print("Writing frame {} / {}".format(frame_count, length))

    cv2.imshow('detect', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Output timestamps for each person
for name, timestamps in face_timestamps.items():
    print(f"Timestamps for {name}:")
    for timestamp in timestamps:
        print(f"{timestamp:.2f} seconds")

# Optionally, save timestamps to a file
with open('face_timestamps.txt', 'w') as f:
    for name, timestamps in face_timestamps.items():
        f.write(f"Timestamps for {name}:\n")
        for timestamp in timestamps:
            f.write(f"{timestamp:.2f} seconds\n")
        f.write("\n")
