import os
import face_recognition as fr


class CustomersDict:
    def __init__(self, pic_folder):
        self.pic_folder = pic_folder
        return

    def set_pic_folder(self, new_pic_folder):
        self.pic_folder = new_pic_folder
        return

    @staticmethod
    def get_file_names(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield file

    def create_image_dict(self):
        X = get_file_names(self.pic_folder)
        list_of_pics = [x for x in X]
        list_of_names = [j.replace(".jpeg", "") for j in list_of_pics]
        new_path = self.pic_folder + r'/{}'
        picture_objects = [fr.face_encodings(fr.load_image_file(new_path.format(x)))[0] for x in list_of_pics]
        return list_of_names, picture_objects


def get_file_names(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file