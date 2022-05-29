import clip
import cv2
import numpy as np
import torch
from PIL import Image
import subprocess
import os
import sys
from tqdm import tqdm
from itertools import islice
from pathlib import Path
from vlm.clip_api import CLIP_API
from torchvision.datasets import CIFAR100
from nebula3_database.playground_db import PLAYGROUND_DB
from nebula3_database.movie_db import MOVIE_DB


class VLM_DETECTOR_API:
    def __init__(self):
        # clip_version = "ViT-L/14@336px"
        self.clip_feat_dim = 768
        clip_api = CLIP_API('vit')
        self.pg_api = PLAYGROUND_DB()
        self.movie_db = MOVIE_DB()
        self.model, self.preprocess = clip_api.get_model()
        if torch.cuda.is_available():
            self.model.cuda().eval()
        else:
            self.model.cpu().eval()
        self.img_size = self.model.visual.input_resolution
        print("Loading CLIP")
        self.clip_api = clip_api
        print("Loading places")
        self.place_feats, self.place_texts = self.load_place_feats()
        print("Loading people")
        self.people_feats, self.people_texts = self.load_people_feats()
        print("Done loading texts")

    def patch_frames_v(self, frame):
        h, w, channels = frame.shape
        half2 = h//2
        top = frame[:half2, :]
        bottom = frame[half2:, :]
        return(top, bottom)
    
    def patch_frames_h(self, frame):
        h, w, channels = frame.shape
        half = w//2
        left_part = frame[:, :half] 
        right_part = frame[:, half:]  
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

    def get_nn_text(self, raw_texts, text_feats, img_feats, score):
        scores = text_feats @ img_feats.T
        scores = scores.squeeze()
        high_to_low_texts = []
        high_to_low_scores = []
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        high_to_low_texts_ = [raw_texts[i] for i in high_to_low_ids]
        high_to_low_scores_ = np.sort(scores).squeeze()[::-1]
        for t, s in zip(high_to_low_texts_, high_to_low_scores_):
            if s > score:
                high_to_low_texts.append(t)
                high_to_low_scores.append(s)
        return(high_to_low_texts, high_to_low_scores)

    def load_place_feats(self):
        # Load scene categories from Places365.
        if not os.path.exists('categories_places365.txt'):
            subprocess.run(["/usr/bin/wget", "https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_places365.txt"])
        place_categories = np.loadtxt('categories_places365.txt', dtype=str)
        place_texts = []
        for place in tqdm(place_categories[:, 0]):
            place = place.split('/')[2:]
            if len(place) > 1:
                place = place[1] + ' ' + place[0]
            else:
                place = place[0]
            place = place.replace('_', ' ')
            place_texts.append(place)
        prmt = [f'a movie frame of the {p}' for p in place_texts]
        place_feats = self.get_text_feats(prmt)
        return(place_feats, place_texts)
    
    def load_place_categories_feats(self):
        place_categories = np.loadtxt('categories_places365.txt', dtype=str)
        place_texts = []
        for place in tqdm(place_categories[:, 0]):
            place = place.split('/')[3:]
            if len(place) > 0:
                #print(place)
                place_texts.append(place[0])
        place_texts = list( dict.fromkeys(place_texts))
        print(place_texts)

    def load_people_feats(self):
        # Load scene categories from Places365.
        people_triplets = np.load('triplets', allow_pickle=True)
        person_texts = []
        for person in tqdm(people_triplets):
            person_texts.append(person)
        person_feats = self.get_text_feats(person_texts)
        return(person_feats, person_texts)

    def mdf_selection(self, frame):
        img_feats = self.get_img_feats(frame)
        frame_texts = ['a low quality blurry image','image is also insanely blurry','blurry photo','a high quality sharp image']
        frame_feats = self.get_text_feats([f'{p}.' for p in frame_texts])
        sorted_frame_texts, frame_scores = self.get_nn_text(frame_texts, frame_feats, img_feats, 0)
        #print(sorted_frame_texts[0], " ", frame_scores[0])
        #if ppl_result == 'people':
        if sorted_frame_texts[0] == 'a high quality sharp image':
            frame_texts = ['outdoor', 'hockey', 'performance', 'rodeo', 'public', 
            'shop', 'exterior', 'interior', 'indoor', 'natural', 'urban', 'sand', 'vegetation',
            'door', 'cultivated', 'wild', 'broadleaf', 'water', 'baseball', 'football', 
            'soccer', 'platform', 'asia', 'ocean_deep', 'undefined place']
            #frame_texts = ['indoor', 'outdoor', 'undefined place']
            frame_feats = self.get_text_feats([f'{p}.' for p in frame_texts])
            sorted_frame_texts, frame_scores = self.get_nn_text(frame_texts, frame_feats, img_feats, 0)
            # if sorted_frame_texts[0] != 'undefined place':
            #     places = 0
            # else:
            #     places = 1
            places = sorted_frame_texts[0]
            ppl_texts = ['no people', 'people', 'a lot of persons', 'man', 'woman']
            ppl_feats = self.get_text_feats([f'There are {p} in this image.' for p in ppl_texts])
            sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats, 0)
            ppl_result = sorted_ppl_texts[0]
            # if ppl_result == 'no people':
            #     people = 0
            # else:
            #     people = 1
            people = ppl_result
            # ppl_texts = ['a lot of objects', 'no sharp objects']
            # ppl_feats = self.get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])
            # sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats, 0)
            # ppl_result = sorted_ppl_texts[0]
            # if ppl_result == 'no sharp objects':
            #     objects = 0
            # else:
            #     objects = 1
            return(1, places, people)
        return(0, 'undefined place', 'no people')

    def clip_persons_expert(self, frame, topk):
        img_feats = self.get_img_feats(frame)
        # Zero-shot VLM: classify number of people.
        ppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
        ppl_feats = self.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
        sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats, 0)
        ppl_result = sorted_ppl_texts[0]
        sorted_ppl_texts, ppl_scores = self.get_nn_text(self.people_texts, self.people_feats, img_feats, 0.16)
        return(sorted_ppl_texts[:topk], [ppl_result])

    def clip_location_expert(self, frame, topk):
        img_feats = self.get_img_feats(frame)
        sorted_places, places_scores = self.get_nn_text(self.place_texts, \
                                                        self.place_feats, img_feats, 0.14)
        return(sorted_places[:topk])
    
    def get_frames(self, movie_id):
        movie_info, fps, fn = self.clip_api.download_and_get_minfo(movie_id, to_print=True)
        all_frames = {}
        if (fn):
            remote_api = self.clip_api.nre
            metadata = remote_api.get_movie_info(movie_id)
           
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                locations_ = []
                scene_persons = {}
                scene_number_of_ppl = {}            
                for scene_element, data in enumerate(metadata['scene_elements']):
                    mdf_frames = []
                    print("Scene element: ", scene_element)
                    #mdfs = metadata['mdfs'][scene_element]
                    objects_ = []
                    persons_ = []
                    number_of_ppl = []
                    first_frame = metadata['scene_elements'][scene_element][0]
                    last_frame = metadata['scene_elements'][scene_element][1]
                    for mdf in tqdm(range(first_frame, last_frame)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                        ret, _frame_ = cap.read() # Read the frame    
                        scale_down_x = 0.5
                        scale_down_y = 0.5
                        #
                        #frame_resise = cv2.resize(_frame_, None, fx= scale_down_x, fy= scale_down_y, interpolation= cv2.INTER_AREA)
                        frame_res = cv2.cvtColor(_frame_, cv2.COLOR_BGR2RGB)

                        if not ret:
                            print("File not found")
                        else:
                            mdf_frames.append(frame_res)
                    all_frames[scene_element] = mdf_frames
        return(all_frames)

    def clip_experts_for_scene_element(self, movie_id):
        movie_info, fps, fn = self.clip_api.download_and_get_minfo(movie_id, to_print=True)
        if (fn):
            remote_api = self.clip_api.nre
            metadata = remote_api.get_movie_info(movie_id)
           
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            first_frame = 0
            last_frame = metadata['scene_elements'][-1][1]
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                
                scene_persons = {}
                scene_number_of_ppl = {}  
                scene_location = {}          
                for scene_element, data in enumerate(metadata['scene_elements']):
                    #print("Scene element: ", scene_element)
                    mdfs = metadata['mdfs'][scene_element]
                    objects_ = []
                    persons_ = []
                    locations_ = []
                    number_of_ppl = []
                    for mdf in range(mdfs[0], mdfs[2]):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                        ret, _frame_ = cap.read() # Read the frame    
                        scale_down_x = 0.30
                        scale_down_y = 0.20
                        dim = (336, 336)
                        #
                        #frame_resise = cv2.resize(_frame_, None, fx= scale_down_x, fy= scale_down_y, interpolation= cv2.INTER_AREA)
                        frame_resise = cv2.resize(_frame_, dim , interpolation= cv2.INTER_AREA)
                        frame_res = cv2.cvtColor(frame_resise, cv2.COLOR_BGR2RGB)

                        if not ret:
                            print("File not found")
                        else:
                            good_frame, pcl, ppl = self.mdf_selection(frame_res)
                            #if good_frame  == 1:
                            if pcl != 'undefined place':
                                #print(pcl)
                                for loc in self.clip_location_expert(frame_res, 20):
                                    locations_.append(loc)
                            #if ppl != 'no people':
                            persons, number_of = self.clip_persons_expert(frame_res, 20)
                            for pers in persons:
                                persons_.append(pers)
                            for nbr in number_of:
                                number_of_ppl.append(nbr)
                    counts = {item: persons_.count(item) for item in persons_}
                    sorted_persons = dict(sorted(counts.items(), key=lambda item: item[1], reverse = True))
                    sorted_persons = list(islice(sorted_persons, 4))
                    scene_persons[scene_element] = sorted_persons

                    counts = {item: number_of_ppl.count(item) for item in number_of_ppl}
                    sorted_number_persons = dict(sorted(counts.items(), key=lambda item: item[1], reverse = True))
                    sorted_number_persons = list(islice(sorted_number_persons, 1))
                    scene_number_of_ppl[scene_element] = sorted_number_persons
                    
                    counts = {item: locations_.count(item) for item in locations_}
                    sorted_locations = dict(sorted(counts.items(), key=lambda item: item[1], reverse = True))
                    sorted_locations = list(islice(sorted_locations, 1))
                    scene_location[scene_element] = sorted_locations
                return(scene_location, scene_number_of_ppl, scene_persons)
       

    def get_s2_bases(self, movie_id):
        output = {}
        query = 'FOR doc IN s2_clsmdc FILTER doc.movie_id == "{}" RETURN doc'.format(movie_id)
        cursor = self.pg_api.db.aql.execute(query)
        for data in cursor:
            output[data['scene_element']] = data['base']
        return(output)

def main():
    cod = VLM_DETECTOR_API()
    #clip.clip_encode_video('/home/dimas/0028_The_Crying_Game_00_53_53_876-00_53_55_522.mp4','Movies/114207205',0) Movies/114208196
    movie_idx = cod.movie_db.get_all_movies()
    for idx in movie_idx:
        res = cod.clip_experts_for_scene_element(idx)
    #res = cod.load_people_feats()
    #cod.load_place_categories_feats()
        print(res)
    #s2_movies = cod.get_s2_movies()
    movie_idx = cod.movie_db.get_all_movies()
    # for idx in movie_idx:
    #     bases = cod.get_s2_bases(idx)
    #     frames = cod.get_frames(idx)
    #     for base in bases:
            
    #         base_feats = cod.get_text_feats([bases[base]])
    #         for frame in frames[base]:
    #             frame_feats = cod.get_img_feats(frame)
    #             print(bases[base])
    #             print(len(frame_feats[0]))
    #             print(len(base_feats))
    #             print(len(base_feats[0]))
    #             sorted_frames, frames_scores = cod.get_nn_text([bases[base]], \
    #                                                     base_feats, frame_feats, 0.11)
                #print(sorted_frames, frames_scores)






if __name__ == "__main__":
    main()