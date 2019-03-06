import unittest
import face_alignment


class Tester(unittest.TestCase):
    def test_predict_points(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='gpu')
        fa.get_landmarks('test/assets/a_001.png')
        fa.get_landmarks('test/assets/b_001.png')
        fa.get_landmarks('test/assets/c_001.png')
        fa.get_landmarks('test/assets/d_001.png')
        fa.get_landmarks('test/assets/e_001.png')
        fa.get_landmarks('test/assets/f_001.png')
if __name__ == '__main__':
    unittest.main()
