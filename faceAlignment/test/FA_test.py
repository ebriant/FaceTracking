import face_alignment

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu')
b = fa.get_landmarks('test/assets/a_001.jpg')
print(b)
