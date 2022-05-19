import clip
import cv2
import numpy as np
import torch
from PIL import Image
import subprocess
import os
import sys
from itertools import islice
from pathlib import Path
from vlm.clip_api import CLIP_API


class CLIP_OBJECT_DETECTOR:
    def __init__(self):
        # clip_version = "ViT-L/14@336px"
        self.clip_feat_dim = 768
        clip_api = CLIP_API('vit')
        self.model, self.preprocess = clip_api.get_model()
        if torch.cuda.is_available():
            self.model.cuda().eval()
        else:
            self.model.cpu().eval()
        self.img_size = self.model.visual.input_resolution
        self.clip_api = clip_api
        self.place_feats, self.place_texts = self.load_place_feats()
        self.object_feats, self.object_texts = self.load_people_feats()
        #self.object_feats, self.object_texts = self.load_object_feats(self.place_texts)
        print("Done loading texts")

    def patch_frames_v(self, frame):
        h, w, channels = frame.shape
        half2 = h//2
        top = frame[:half2, :]
        bottom = frame[half2:, :]
        # saving all the images
        # cv2.imwrite() function will save the image 
        # into your pc
        return(top, bottom)
    
    def patch_frames_h(self, frame):
        h, w, channels = frame.shape
        half = w//2
        # this will be the first column
        left_part = frame[:, :half] 

        right_part = frame[:, half:]  

        # [:,half:] means al the rows and all
        # saving all the images
        # cv2.imwrite() function will save the image 
        # into your pc
        return(right_part, left_part)
       

    def get_text_feats(self, in_text, batch_size=64):
        if torch.cuda.is_available():
            text_tokens = clip.tokenize(in_text).cuda()
        else: 
            text_tokens = clip.tokenize(in_text).cpu()
        text_id = 0
        text_feats = np.zeros((len(in_text), self.clip_feat_dim), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)
            text_batch = text_tokens[text_id:text_id+batch_size]
            with torch.no_grad():
                batch_feats = self.model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            text_feats[text_id:text_id+batch_size, :] = batch_feats
            text_id += batch_size
        return(text_feats)

    def get_img_feats(self, img):
        img_pil = Image.fromarray(np.uint8(img))
        img_in = self.preprocess(img_pil)[None, ...]
        if torch.cuda.is_available():
            img_in_c = img_in.cuda()
        else:
            img_in_c= img_in.cpu()
        with torch.no_grad():
            img_feats = self.model.encode_image(img_in_c).float()
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_feats = np.float32(img_feats.cpu())
        return(img_feats)

    def get_nn_text(self, raw_texts, text_feats, img_feats):
        scores = text_feats @ img_feats.T
        scores = scores.squeeze()
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
        high_to_low_scores = np.sort(scores).squeeze()[::-1]
        return(high_to_low_texts, high_to_low_scores)

    def load_place_feats(self):
        # Load scene categories from Places365.
        if not os.path.exists('categories_places365.txt'):
            subprocess.run(["/usr/bin/wget", "https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_places365.txt"])
        place_categories = np.loadtxt('categories_places365.txt', dtype=str)
        place_texts = []
        for place in place_categories[:, 0]:
            place = place.split('/')[2:]
            if len(place) > 1:
                place = place[1] + ' ' + place[0]
            else:
                place = place[0]
            place = place.replace('_', ' ')
            place_texts.append(place)
        prmt = [f'They are located in {p}' for p in place_texts]
        place_feats = self.get_text_feats(prmt)
        return(place_feats, place_texts)

    def load_people_feats(self):
        # Load scene categories from Places365.
        people_triplets = np.load('triplets', allow_pickle=True)
        person_texts = []
        for person in people_triplets:
            #print(person)
            person_texts.append(person)
        #prmt = [f'They are located in {p}' for p in place_texts]
        place_feats = self.get_text_feats(person_texts)
        return(place_feats, person_texts)

    def load_object_feats(self, place_texts):
        if not os.path.exists('dictionary_and_semantic_hierarchy.txt'):
            subprocess.run(["/usr/bin/wget", "https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierarchy.txt"])
        with open('dictionary_and_semantic_hierarchy.txt') as fid:
            object_categories = fid.readlines()
        object_texts = []
#pf = ProfanityFilter()
        for object_text in object_categories[1:]:
            object_text = object_text.strip()
            object_text = object_text.split('\t')[3]
            safe_list = ''
            for variant in object_text.split(','):
                text = variant.strip()
                #safe_list += f'{text}, '
            #safe_list = safe_list[:-2]
            if len(text) > 0:
                object_texts.append(text)
        object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
        #print(object_texts)
        prmt = [f'Picture of a {o}' for o in object_texts]
        #prmt = [f'This is a {o} in the photo' for o in object_texts]
        object_feats = self.get_text_feats(prmt)
        return(object_feats, object_texts)

    def mdf_selection(self, frame):
        img_feats = self.get_img_feats(frame)
        frame_texts = ['a low quality blurry image', 'a high quality sharp image']
        frame_feats = self.get_text_feats([f'{p}.' for p in frame_texts])
        sorted_frame_texts, frame_scores = self.get_nn_text(frame_texts, frame_feats, img_feats)
        #print(sorted_frame_texts[0], " ", frame_scores[0])
        places = 0
        #if ppl_result == 'people':
        if sorted_frame_texts[0] == 'a high quality sharp image':
            frame_texts = ['many different objects in image', 'blurry backgound in image']
            frame_feats = self.get_text_feats([f'{p}.' for p in frame_texts])
            sorted_frame_texts, frame_scores = self.get_nn_text(frame_texts, frame_feats, img_feats)
            if sorted_frame_texts[0] == 'blurry backgound in image':
                places = 0
            else:
                places = 1
            ppl_texts = ['no people', 'people']
            ppl_feats = self.get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])
            sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats)
            ppl_result = sorted_ppl_texts[0]
            if ppl_result == 'people':
                people = 1
            else:
                people = 0
            return(1, places, people)
        return(0)

    def clip_expert(self, frame, place_topk, obj_topk):
        img_feats = self.get_img_feats(frame)
        # Zero-shot VLM: classify number of people.
        ppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
        ppl_feats = self.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
        sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats)
        ppl_result = sorted_ppl_texts[0]
        # Zero-shot VLM: classify places.
        #place_feats = self.get_text_feats([f'Scene of a {p}.' for p in place_texts ])
        sorted_places, places_scores = self.get_nn_text(self.place_texts, self.place_feats, img_feats)
        # Zero-shot VLM: classify objects.
        sorted_obj_texts, obj_scores = self.get_nn_text(self.object_texts, self.object_feats, img_feats)
        #object_list = ''
        #for i in range(obj_topk):
            #object_list += f'{sorted_obj_texts[i]}, '
            #print(object_list)
        #object_list = object_list[:-2]
        return(sorted_places[:place_topk], [sorted_obj_texts[:obj_topk], obj_scores[:obj_topk]], [ppl_result])

    def clip_experts_for_moive(self, movie_id, scene_element):
        movie_info, fps, fn = self.clip_api.download_and_get_minfo(movie_id, to_print=True)
        if (fn):
            remote_api = self.clip_api.nre
            metadata = remote_api.get_movie_info(movie_id)
            mdfs = metadata['mdfs'][scene_element]
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                locations_ = []
                objects_ = []
                for mdf in range(mdfs[0], mdfs[2]):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, _frame_ = cap.read() # Read the frame
                    #for _frame__ in self.patch_frames_v(_frame_):
                        #for frame_ in self.patch_frames_h(_frame__):
                    scale_down_x = 0.25
                    scale_down_y = 0.25
                    frame_res = cv2.cvtColor(_frame_, cv2.COLOR_BGR2RGB)
                    #frame_res = cv2.resize(frame_rgb, None, fx= scale_down_x, fy= scale_down_y, interpolation= cv2.INTER_LINEAR)
                    if not ret:
                        print("File not found")
                    else:
                        #mdf_experts = self.clip_expert(frame_rgb, 3, 10)
                        good_frame = self.mdf_selection(frame_res)
                        if good_frame  == 1:
                            mdf_experts = self.clip_expert(frame_res, 10, 10)
                            for object_ in mdf_experts[1][0]:
                                objects_.append(object_)
                            for location_ in mdf_experts[0]:
                                locations_.append(location_)
                        #else:
                        #    print("Bad Frame")

                counts = {item: objects_.count(item) for item in objects_}
                sorted_objects = dict(sorted(counts.items(), key=lambda item: item[1], reverse = True))
                counts = {item: locations_.count(item) for item in locations_}
                sorted_locations = dict(sorted(counts.items(), key=lambda item: item[1], reverse = True))
                print(sorted_objects)
                return(list(islice(sorted_objects, 15)), list(islice(sorted_locations, 3)) )


def main():
    cod = CLIP_OBJECT_DETECTOR()
    #clip.clip_encode_video('/home/dimas/0028_The_Crying_Game_00_53_53_876-00_53_55_522.mp4','Movies/114207205',0)
    res = cod.clip_experts_for_moive('Movies/114206691', 0)
    #res = cod.load_people_feats()
    print(res)


if __name__ == "__main__":
    main()